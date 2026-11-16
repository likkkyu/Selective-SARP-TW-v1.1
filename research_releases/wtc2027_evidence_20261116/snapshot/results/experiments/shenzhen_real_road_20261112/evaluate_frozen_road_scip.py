"""Bounded hard-four SCIP reference on the frozen Shenzhen road test."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import signal
import sys
import time
from pathlib import Path

import torch


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
MIP_DIR = ROOT / "experiments" / "mip_scip_20261106"
LNS_DIR = ROOT / "experiments" / "hard4_lns_20261106"
sys.path[:0] = [str(MIP_DIR), str(HERE), str(LNS_DIR)]

from greedy_hard4 import make_start
from road_audit import audit_routes
from road_route_engine import RoadRouteEngine
from scip_selective import SelectiveSCIP


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-pool", type=Path, required=True)
    parser.add_argument("--test-manifest", type=Path, required=True)
    parser.add_argument("--model-lock", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=16)
    parser.add_argument("--seconds", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--max-minutes", type=float, default=8.0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output}")
    if not 1 <= args.samples <= 128 or args.seconds <= 0 or args.max_minutes <= 0:
        raise ValueError("Invalid finite SCIP protocol")

    test = torch.load(args.test_pool, map_location="cpu", weights_only=True)
    manifest = json.loads(args.test_manifest.read_text())
    if test["meta"].get("purpose") != "formal_test":
        raise ValueError("Refusing non-formal data")
    if manifest.get("sha256") != digest(args.test_pool):
        raise ValueError("Test-pool hash differs from manifest")
    if test["meta"].get("model_lock_sha256") != digest(args.model_lock):
        raise ValueError("Test was not generated after the frozen model lock")

    plan = json.loads((HERE / "formal_p1_v1" / "plan.json").read_text())
    config = plan["physical_config"]
    profile = plan["normalization_profile"]
    args.output.mkdir(parents=True)
    report = {
        "status": "running",
        "scope": "bounded hard-four SCIP reference on first predeclared frozen road-test instances",
        "interpretation": "finite-time MIP incumbent/reference; not an exact optimum unless SCIP reports optimal",
        "test_indices": list(range(args.samples)),
        "selection_rule": "contiguous indices 0..samples-1 fixed before launching this SCIP run",
        "test_pool_sha256": digest(args.test_pool),
        "test_manifest_sha256": digest(args.test_manifest),
        "model_lock_sha256": digest(args.model_lock),
        "seconds_per_instance": args.seconds,
        "threads": 1,
        "solver_seed": args.seed,
        "minimum_completed_orders_per_dispatch": 4,
        "distance_semantics": "directed road shortest-path matrices",
        "warm_start": "strict deterministic insertion; no neural warm start",
        "hardware": {"platform": platform.platform(), "processor": platform.processor()},
        "source_sha256": {
            str(Path(__file__).relative_to(ROOT)): digest(Path(__file__)),
            str((MIP_DIR / "scip_selective.py").relative_to(ROOT)): digest(MIP_DIR / "scip_selective.py"),
            str((HERE / "greedy_hard4.py").relative_to(ROOT)): digest(HERE / "greedy_hard4.py"),
            str((HERE / "road_route_engine.py").relative_to(ROOT)): digest(HERE / "road_route_engine.py"),
            str((HERE / "road_audit.py").relative_to(ROOT)): digest(HERE / "road_audit.py"),
        },
        "rows": [],
    }
    started = time.monotonic()
    signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError("SCIP protocol timeout")))
    signal.alarm(max(1, int(args.max_minutes * 60)))
    try:
        for index in report["test_indices"]:
            sample = test["data"][index]
            engine = RoadRouteEngine(sample, config, profile)
            insertion_started = time.monotonic()
            warm_routes = make_start(engine)
            insertion_seconds = time.monotonic() - insertion_started
            warm_cost = engine.cost(warm_routes)
            warm_audit = audit_routes(sample, warm_routes, config)
            if warm_cost is None or not warm_audit["valid"]:
                raise AssertionError((index, "warm_start", warm_audit))

            solver = SelectiveSCIP(
                config,
                profile,
                vehicles=engine.k,
                time_limit=args.seconds,
                gap=0.0,
                threads=1,
                seed=args.seed,
                verbose=False,
                min_completed_per_dispatch=4,
            )
            solved = solver.solve(
                sample,
                warm_actions=engine.actions(warm_routes),
                logfile=args.output / f"instance_{index:03d}.log",
            )
            if solved.get("actions") is None:
                raise AssertionError((index, "SCIP returned no incumbent"))
            final_audit = audit_routes(sample, solved["routes"], config)
            final_cost = engine.cost(solved["routes"])
            if not final_audit["valid"] or final_cost is None:
                raise AssertionError((index, "SCIP audit", final_audit))
            replay_error = abs(final_cost["objective"] - solved["replayed"]["objective"])
            if replay_error > 1e-6:
                raise AssertionError((index, "road replay mismatch", replay_error))
            row = {
                "test_index": index,
                "insertion": {"seconds": insertion_seconds, "cost": warm_cost, "audit": warm_audit},
                "scip": solved,
                "road_engine_cost": final_cost,
                "road_audit": final_audit,
                "road_replay_error": replay_error,
            }
            report["rows"].append(row)
            report["elapsed_seconds"] = time.monotonic() - started
            (args.output / f"instance_{index:03d}.json").write_text(json.dumps(row, indent=2) + "\n")
            (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
            solver.model.freeProb()
            print(
                f"ROAD_SCIP {index + 1}/{args.samples} status={solved['status']} "
                f"served={solved['completed']} J={solved['objective']:.3f} "
                f"bound={solved['dual_bound']:.3g} elapsed_s={report['elapsed_seconds']:.1f}",
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
    print(json.dumps({"status": report["status"], "instances": len(report["rows"]), "elapsed_seconds": report["elapsed_seconds"]}, indent=2))


if __name__ == "__main__":
    main()
