"""Frozen three-seed neural construction plus pre-execution road repair."""
import gc
import hashlib
import json
import random
import signal
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import default_collate

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
BASE = HERE / "formal_p1_v1"
OUT = BASE / "neural_transfer_v1"
sys.path.insert(0, str(ROOT / "experiments/dispatch_clock_fix_20261106"))
from bootstrap import Config, KW
sys.path.insert(0, str(ROOT / "experiments/hard4_lns_20261106"))
from route_engine import RouteEngine, split_actions
from search import SearchTimeout, repair, solve
sys.path.insert(0, str(ROOT / "experiments/hard4_speed_20261106"))
sys.path.insert(0, str(ROOT / "experiments/neural_diagnosis_20261107"))
sys.path.insert(0, str(HERE))
from road_audit import audit_routes
from road_route_engine import RoadRouteEngine


MODEL_KEYS = ("depot", "loc", "node_type", "demand_passenger", "demand_cargo", "time_windows", "pickup_delivery_pairs", "pd_pair_mask", "n_orders")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def model_sample(sample):
    return {key: sample[key] for key in MODEL_KEYS}


def primary_road_sample(sample, speed):
    result = dict(sample)
    result["road_travel_time_matrix_hours"] = sample["road_distance_matrix_km"] / speed
    return result


def main():
    if OUT.exists():
        raise FileExistsError(f"Refusing to overwrite {OUT}")
    OUT.mkdir()
    plan = json.loads((BASE / "plan.json").read_text())
    if digest(BASE / "road_cases.pt") != plan["road_cases_sha256"]:
        raise ValueError("Road cases do not match their frozen manifest")
    data = torch.load(BASE / "road_cases.pt", map_location="cpu", weights_only=True)["data"]
    protocol = {
        "status": "frozen_before_results",
        "scope": "three unchanged learned-policy training seeds; coordinate-based neural construction followed by pre-execution primary road_fixed25 repair and matched 0.5-second LNS",
        "road_cases_sha256": plan["road_cases_sha256"],
        "models": plan["models"],
        "repair_timeout_seconds_per_case": 5.0,
        "lns_seconds": plan["neural_repair_lns_seconds"],
        "search_seed": plan["search_seed"],
        "hard_minimum": 4,
        "interpretation_guard": "Not a road-matrix-aware neural policy and not retraining. Rejected requests are declined before execution; invalid provisional road routes are never executed.",
    }
    (OUT / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")

    from clock_patch import install as install_clock
    from timing_candidate import install as install_timing
    from completion_guard import install as install_completion
    from admission_targeted import install as install_admission
    from physical_guard_fast import install as install_physical
    from run_v3 import install_policy
    from depot_noop import install as install_depot
    from evaluate_model import build_model_from_checkpoint
    install_clock(); install_timing(); install_completion(); install_admission(); install_physical(); install_policy("learned"); install_depot()
    Config.set_normalization_profile(50, plan["normalization_profile"])

    report = {"status": "running", "protocol": protocol, "models": []}
    started = time.monotonic()
    signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError("Finite 1800-second neural-transfer timeout")))
    signal.alarm(1800)
    try:
        for spec in plan["models"]:
            path = Path(spec["checkpoint"])
            if digest(path) != spec["sha256"]:
                raise ValueError(f"Checkpoint hash mismatch for {spec}")
            checkpoint = torch.load(path, map_location="cpu", weights_only=False)
            model = build_model_from_checkpoint(checkpoint, torch.device("cpu"))
            model.load_state_dict(checkpoint["model_state_dict"])
            model.eval(); model.set_decode_type("greedy")
            rows = []
            label = f"learned_{spec['seed']}"
            for instance, sample in enumerate(data):
                batch = default_collate([model_sample(sample)])
                begin = time.monotonic()
                with torch.no_grad():
                    cost, log_likelihood, pi = model(batch, return_pi=True, state_kwargs=KW)
                inference_seconds = time.monotonic() - begin
                if not torch.isfinite(cost).all() or not torch.isfinite(log_likelihood).all():
                    raise ValueError((label, instance, "non-finite neural output"))
                routes = split_actions(pi[0].tolist(), 50)
                proxy = RouteEngine(model_sample(sample), plan["physical_config"], plan["normalization_profile"]).cost(routes)
                if proxy is None:
                    raise AssertionError((label, instance, "neural proxy route invalid"))
                road_sample = primary_road_sample(sample, plan["physical_config"]["VEHICLE_SPEED"])
                engine = RoadRouteEngine(road_sample, plan["physical_config"], plan["normalization_profile"])
                before = engine.cost(routes)
                invalid_routes = sum(engine.route(route) is None for route in routes)
                served = {node for route in routes for node in route if node <= engine.n}
                pool = set(range(1, engine.n + 1)) - served
                repair_begin = time.monotonic()
                try:
                    repaired = repair(engine, routes, pool, random.Random(plan["search_seed"]), repair_begin + protocol["repair_timeout_seconds_per_case"], regret=True)
                    result = solve(engine, repaired, seed=plan["search_seed"], seconds=protocol["lns_seconds"], max_iterations=1000)
                except SearchTimeout as error:
                    raise TimeoutError((label, instance, "road repair timeout")) from error
                repair_seconds = time.monotonic() - repair_begin
                audit = audit_routes(road_sample, result["routes"], plan["physical_config"])
                if result["cost"] is None or not audit["valid"]:
                    raise AssertionError((label, instance, audit))
                row = {
                    "instance": instance,
                    "neural_proxy_cost": proxy,
                    "neural_inference_seconds": inference_seconds,
                    "road_valid_before_repair": before is not None,
                    "road_cost_before_repair": before,
                    "invalid_road_routes_before_repair": invalid_routes,
                    "road_repair_and_lns_seconds": repair_seconds,
                    "road_cost_after_repair": result["cost"],
                    "road_audit_after_repair": audit,
                    "road_routes_after_repair": [list(route) for route in result["routes"]],
                    "lns_iterations": result["iterations"],
                }
                rows.append(row)
                (OUT / f"{label}.json").write_text(json.dumps({"model": spec, "rows": rows}, indent=2) + "\n")
                print("NEURAL", label, instance + 1, "/", len(data), "proxy", proxy["completed"], "road_valid", before is not None, "repaired", result["cost"]["completed"], "J", round(result["cost"]["objective"], 4), flush=True)
            summary = {
                "model": label, "seed": spec["seed"], "samples": len(rows),
                "direct_road_valid": sum(row["road_valid_before_repair"] for row in rows),
                "mean_proxy_completed": sum(row["neural_proxy_cost"]["completed"] for row in rows) / len(rows),
                "mean_repaired_completed": sum(row["road_cost_after_repair"]["completed"] for row in rows) / len(rows),
                "hard_invalid_after_repair": sum(not row["road_audit_after_repair"]["valid"] for row in rows),
                "mean_inference_seconds": sum(row["neural_inference_seconds"] for row in rows) / len(rows),
                "mean_repair_seconds": sum(row["road_repair_and_lns_seconds"] for row in rows) / len(rows),
            }
            report["models"].append(summary)
            report["elapsed_seconds"] = time.monotonic() - started
            (OUT / "report.json").write_text(json.dumps(report, indent=2) + "\n")
            del model, checkpoint; gc.collect()
        report["status"] = "completed"
    except BaseException as error:
        report.update(status="failed", error=repr(error))
        raise
    finally:
        signal.alarm(0)
        report["elapsed_seconds"] = time.monotonic() - started
        (OUT / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"status": report["status"], "elapsed_seconds": report["elapsed_seconds"], "models": report["models"]}, indent=2))


if __name__ == "__main__":
    main()
