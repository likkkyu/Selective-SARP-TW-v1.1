"""Run the frozen P1 insertion plus strong-LNS controls on CPU."""
import hashlib
import json
import signal
import sys
import time
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
BASE = HERE / "formal_p1_v1"
OUT = BASE / "controls_v1"
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "experiments/hard4_lns_20261106"))

from greedy_hard4 import make_start
from road_audit import audit_routes
from road_route_engine import RoadRouteEngine
from search import solve


ARMS = ("straight_fixed25", "road_fixed25", "road_osm_speed")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def arm_sample(sample, arm, speed):
    result = dict(sample)
    if arm == "straight_fixed25":
        result["road_distance_matrix_km"] = sample["straight_distance_matrix_km"]
        result["road_travel_time_matrix_hours"] = sample["straight_distance_matrix_km"] / speed
    elif arm == "road_fixed25":
        result["road_travel_time_matrix_hours"] = sample["road_distance_matrix_km"] / speed
    elif arm != "road_osm_speed":
        raise ValueError(arm)
    return result


def main():
    if OUT.exists():
        raise FileExistsError(f"Refusing to overwrite {OUT}")
    OUT.mkdir()
    plan = json.loads((BASE / "plan.json").read_text())
    if plan["status"] != "prepared" or digest(BASE / "road_cases.pt") != plan["road_cases_sha256"]:
        raise ValueError("Frozen P1 data failed its manifest")
    for relative, expected in plan["source_sha256"].items():
        if digest(ROOT / relative) != expected:
            raise ValueError(f"Frozen pre-generation source changed: {relative}")
    bundle = torch.load(BASE / "road_cases.pt", map_location="cpu", weights_only=True)
    data = bundle["data"]
    if len(data) != plan["samples"]:
        raise ValueError("Unexpected case count")
    report = {
        "status": "running",
        "scope": plan["scope"],
        "road_cases_sha256": plan["road_cases_sha256"],
        "source_sha256": {str(Path(__file__).relative_to(ROOT)): digest(__file__), str((HERE / "road_audit.py").relative_to(ROOT)): digest(HERE / "road_audit.py")},
        "arms": ARMS,
        "rows": [],
    }
    started = time.monotonic()
    signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError("Finite 1800-second control timeout")))
    signal.alarm(1800)
    try:
        for instance, sample in enumerate(data):
            entry = {"instance": instance, "arms": {}}
            for arm in ARMS:
                candidate = arm_sample(sample, arm, plan["physical_config"]["VEHICLE_SPEED"])
                engine = RoadRouteEngine(candidate, plan["physical_config"], plan["normalization_profile"])
                begin = time.monotonic()
                initial = make_start(engine)
                initial_seconds = time.monotonic() - begin
                initial_cost = engine.cost(initial)
                initial_audit = audit_routes(candidate, initial, plan["physical_config"])
                if initial_cost is None or not initial_audit["valid"]:
                    raise AssertionError((instance, arm, "invalid initial", initial_audit))
                result = solve(engine, initial, seed=plan["search_seed"], seconds=plan["strong_lns_seconds"], max_iterations=1000)
                integrated_seconds = time.monotonic() - begin
                final_audit = audit_routes(candidate, result["routes"], plan["physical_config"])
                if not final_audit["valid"] or result["cost"]["objective"] > initial_cost["objective"] + 1e-8:
                    raise AssertionError((instance, arm, final_audit))
                entry["arms"][arm] = {
                    "initial_cost": initial_cost, "initial_audit": initial_audit,
                    "initialization_seconds": initial_seconds,
                    "lns_cost": result["cost"], "lns_routes": [list(route) for route in result["routes"]],
                    "lns_audit": final_audit, "lns_search_seconds": result["seconds"],
                    "integrated_seconds": integrated_seconds, "iterations": result["iterations"],
                    "route_evaluations": result["route_evaluations"],
                }
                print("CONTROL", instance + 1, "/", len(data), arm, "completed", result["cost"]["completed"], "vehicles", result["cost"]["vehicles"], "J", round(result["cost"]["objective"], 4), flush=True)
            report["rows"].append(entry)
            (OUT / f"instance_{instance:03d}.json").write_text(json.dumps(entry, indent=2) + "\n")
            report["elapsed_seconds"] = time.monotonic() - started
            (OUT / "report.json").write_text(json.dumps(report, indent=2) + "\n")
        report["status"] = "completed"
    except BaseException as error:
        report.update(status="failed", error=repr(error))
        raise
    finally:
        signal.alarm(0)
        report["elapsed_seconds"] = time.monotonic() - started
        (OUT / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"status": report["status"], "instances": len(report["rows"]), "elapsed_seconds": report["elapsed_seconds"]}, indent=2))


if __name__ == "__main__":
    main()
