"""Matched bounded LNS postprocessing for each locked road-aware neural model."""

from __future__ import annotations

import argparse
import hashlib
import json
import signal
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import default_collate


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
FORMAL = HERE / "formal_p1_v1"
LNS_DIR = ROOT / "experiments" / "hard4_lns_20261106"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(LNS_DIR))

from problem_mcvrptw_v2 import Config, MCVRPPDTW
from road_audit import audit_routes
from road_route_engine import RoadRouteEngine
from search import solve
from smoke_matrix_aware_decode import MODEL_KEYS, build_model, split_actions
from train_matrix_aware_finetune import STATE_KWARGS


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-pool", type=Path, required=True)
    parser.add_argument("--test-manifest", type=Path, required=True)
    parser.add_argument("--model-lock", type=Path, required=True)
    parser.add_argument("--model-seed", type=int, required=True)
    parser.add_argument("--lns-seconds", type=float, default=3.0)
    parser.add_argument("--lns-seed", type=int, default=1234)
    parser.add_argument("--max-minutes", type=float, default=12.0)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output}")

    test = torch.load(args.test_pool, map_location="cpu", weights_only=True)
    manifest = json.loads(args.test_manifest.read_text())
    lock = json.loads(args.model_lock.read_text())
    if test["meta"].get("purpose") != "formal_test":
        raise ValueError("Refusing to evaluate development data as formal test")
    if manifest.get("sha256") != digest(args.test_pool):
        raise ValueError("Test-pool hash differs from manifest")
    if test["meta"].get("model_lock_sha256") != digest(args.model_lock):
        raise ValueError("Test was not generated after this model lock")
    matches = [row for row in lock["models"] if row["initial_model_seed"] == args.model_seed]
    if len(matches) != 1:
        raise ValueError("Requested model is not uniquely present in lock")
    spec = matches[0]
    checkpoint_path = ROOT / spec["checkpoint"]
    if digest(checkpoint_path) != spec["checkpoint_sha256"]:
        raise ValueError("Locked checkpoint hash mismatch")

    plan = json.loads((FORMAL / "plan.json").read_text())
    Config.set_normalization_profile(50, plan["normalization_profile"])
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = build_model(checkpoint)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    model.set_decode_type("greedy")

    args.output.mkdir(parents=True)
    report = {
        "status": "running",
        "scope": "locked road-aware neural construction plus matched fixed-budget LNS",
        "test_pool_sha256": digest(args.test_pool),
        "test_manifest_sha256": digest(args.test_manifest),
        "model_lock_sha256": digest(args.model_lock),
        "model_seed": args.model_seed,
        "checkpoint_sha256": spec["checkpoint_sha256"],
        "checkpoint_epoch": spec["checkpoint_epoch"],
        "decoder_semantics_version": 2,
        "evaluation_semantics_version": 2,
        "lns_seconds_per_instance": args.lns_seconds,
        "lns_seed": args.lns_seed,
        "max_minutes": args.max_minutes,
        "instance_limit": args.limit,
        "timing_boundary": (
            "neural construction and LNS search timed separately; integrated time is their sum; "
            "model loading and independent final audits are excluded"
        ),
        "source_sha256": {
            str(Path(__file__).resolve().relative_to(ROOT)): digest(Path(__file__)),
            "state_mcvrptw_v2.py": digest(ROOT / "state_mcvrptw_v2.py"),
            "nets/attention_model.py": digest(ROOT / "nets" / "attention_model.py"),
            str((HERE / "road_route_engine.py").relative_to(ROOT)): digest(HERE / "road_route_engine.py"),
            str((HERE / "road_audit.py").relative_to(ROOT)): digest(HERE / "road_audit.py"),
            str((LNS_DIR / "search.py").relative_to(ROOT)): digest(LNS_DIR / "search.py"),
        },
        "rows": [],
    }
    started = time.monotonic()
    signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError("neural-LNS timeout")))
    signal.alarm(max(1, int(args.max_minutes * 60)))
    try:
        selected_data = test["data"] if args.limit is None else test["data"][:args.limit]
        if not selected_data:
            raise ValueError("At least one instance is required")
        for index, sample in enumerate(selected_data):
            batch = default_collate([{key: sample[key] for key in MODEL_KEYS}])
            decode_started = time.monotonic()
            with torch.no_grad():
                _, _, actions = model(batch, return_pi=True, state_kwargs=STATE_KWARGS)
            executable, forced = MCVRPPDTW._sanitize_underfilled_dispatches(actions, 50)
            neural_seconds = time.monotonic() - decode_started
            routes = split_actions(executable[0].tolist(), 50)
            engine = RoadRouteEngine(sample, plan["physical_config"], plan["normalization_profile"])
            neural_cost = engine.cost(routes)
            neural_audit = audit_routes(sample, routes, plan["physical_config"])
            if neural_cost is None or not neural_audit["valid"]:
                raise AssertionError((index, "neural", neural_audit))

            search_started = time.monotonic()
            result = solve(
                engine,
                routes,
                seed=args.lns_seed,
                seconds=args.lns_seconds,
                max_iterations=1000,
            )
            search_wall_seconds = time.monotonic() - search_started
            final_audit = audit_routes(sample, result["routes"], plan["physical_config"])
            if not final_audit["valid"] or result["cost"]["objective"] > neural_cost["objective"] + 1e-8:
                raise AssertionError((index, "neural_lns", final_audit))
            row = {
                "test_index": index,
                "neural": {
                    "cost": neural_cost,
                    "audit": neural_audit,
                    "seconds": neural_seconds,
                    "forced_underfilled_declines": int(forced[0]),
                    "routes": routes,
                },
                "neural_lns": {
                    "cost": result["cost"],
                    "audit": final_audit,
                    "search_seconds": result["seconds"],
                    "search_wall_seconds": search_wall_seconds,
                    "integrated_seconds": neural_seconds + search_wall_seconds,
                    "iterations": result["iterations"],
                    "partial_last_iteration_discarded": result["partial_last_iteration_discarded"],
                    "route_evaluations": result["route_evaluations"],
                },
            }
            report["rows"].append(row)
            (args.output / f"instance_{index:03d}.json").write_text(json.dumps(row, indent=2) + "\n")
            report["elapsed_seconds"] = time.monotonic() - started
            (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
            print(
                f"ROAD_NEURAL_LNS seed={args.model_seed} {index + 1}/{len(selected_data)} "
                f"neural={neural_cost['completed']} hybrid={result['cost']['completed']} "
                f"J={result['cost']['objective']:.3f} elapsed_s={report['elapsed_seconds']:.1f}",
                flush=True,
            )
        report["status"] = "completed"
    except BaseException as error:
        report.update(status="failed", error=repr(error))
        raise
    finally:
        signal.alarm(0)
        report["elapsed_seconds"] = time.monotonic() - started
        (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({
        "status": report["status"],
        "model_seed": args.model_seed,
        "instances": len(report["rows"]),
        "elapsed_seconds": report["elapsed_seconds"],
    }, indent=2))


if __name__ == "__main__":
    main()
