"""Strict insertion and bounded LNS on the frozen Shenzhen road test."""

from __future__ import annotations

import argparse
import hashlib
import json
import signal
import sys
import time
from pathlib import Path

import torch


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
FORMAL = HERE / "formal_p1_v1"
LNS_DIR = ROOT / "experiments" / "hard4_lns_20261106"
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(LNS_DIR))

from greedy_hard4 import make_start
from road_audit import audit_routes
from road_route_engine import RoadRouteEngine
from search import solve


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-pool", type=Path, required=True)
    parser.add_argument("--test-manifest", type=Path, required=True)
    parser.add_argument("--model-lock", type=Path, required=True)
    parser.add_argument("--lns-seconds", type=float, default=3.0)
    parser.add_argument("--lns-seed", type=int, default=1234)
    parser.add_argument("--max-minutes", type=float, default=30.0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output}")
    if args.lns_seconds <= 0 or args.max_minutes <= 0:
        raise ValueError("Finite positive time limits are required")

    test = torch.load(args.test_pool, map_location="cpu", weights_only=True)
    manifest = json.loads(args.test_manifest.read_text())
    if test["meta"].get("purpose") != "formal_test":
        raise ValueError("Refusing to evaluate development data as formal test")
    if manifest.get("sha256") != digest(args.test_pool):
        raise ValueError("Test-pool hash differs from manifest")
    if test["meta"].get("model_lock_sha256") != digest(args.model_lock):
        raise ValueError("Test was not generated after the frozen model lock")

    plan = json.loads((FORMAL / "plan.json").read_text())
    args.output.mkdir(parents=True)
    report = {
        "status": "running",
        "scope": "strict insertion and fixed-budget LNS on untouched road-native formal test",
        "test_pool_sha256": digest(args.test_pool),
        "test_manifest_sha256": digest(args.test_manifest),
        "model_lock_sha256": digest(args.model_lock),
        "lns_seconds_per_instance": args.lns_seconds,
        "lns_seed": args.lns_seed,
        "max_minutes": args.max_minutes,
        "timing_boundary": (
            "insertion reports complete construction time; LNS integrated time includes a fresh "
            "insertion start plus bounded search; audits are reported separately"
        ),
        "source_sha256": {
            str(Path(__file__).resolve().relative_to(ROOT)): digest(Path(__file__)),
            str((HERE / "greedy_hard4.py").relative_to(ROOT)): digest(HERE / "greedy_hard4.py"),
            str((HERE / "road_route_engine.py").relative_to(ROOT)): digest(HERE / "road_route_engine.py"),
            str((HERE / "road_audit.py").relative_to(ROOT)): digest(HERE / "road_audit.py"),
            str((LNS_DIR / "search.py").relative_to(ROOT)): digest(LNS_DIR / "search.py"),
        },
        "rows": [],
    }
    started = time.monotonic()
    timeout_seconds = max(1, int(args.max_minutes * 60))
    signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError("control timeout")))
    signal.alarm(timeout_seconds)
    try:
        for index, sample in enumerate(test["data"]):
            engine = RoadRouteEngine(sample, plan["physical_config"], plan["normalization_profile"])
            insertion_started = time.monotonic()
            initial = make_start(engine)
            insertion_seconds = time.monotonic() - insertion_started
            initial_cost = engine.cost(initial)
            initial_audit = audit_routes(sample, initial, plan["physical_config"])
            if initial_cost is None or not initial_audit["valid"]:
                raise AssertionError((index, "strict_insertion", initial_audit))

            search_started = time.monotonic()
            result = solve(
                engine,
                initial,
                seed=args.lns_seed,
                seconds=args.lns_seconds,
                max_iterations=1000,
            )
            search_wall_seconds = time.monotonic() - search_started
            final_audit = audit_routes(sample, result["routes"], plan["physical_config"])
            if not final_audit["valid"] or result["cost"]["objective"] > initial_cost["objective"] + 1e-8:
                raise AssertionError((index, "lns", final_audit))
            row = {
                "test_index": index,
                "insertion": {
                    "cost": initial_cost,
                    "audit": initial_audit,
                    "seconds": insertion_seconds,
                },
                "lns": {
                    "cost": result["cost"],
                    "audit": final_audit,
                    "search_seconds": result["seconds"],
                    "search_wall_seconds": search_wall_seconds,
                    "integrated_seconds": insertion_seconds + search_wall_seconds,
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
                f"ROAD_CONTROL {index + 1}/{len(test['data'])} "
                f"insert={initial_cost['completed']} lns={result['cost']['completed']} "
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
        "instances": len(report["rows"]),
        "elapsed_seconds": report["elapsed_seconds"],
    }, indent=2))


if __name__ == "__main__":
    main()
