"""Four-case CPU preflight of a frozen neural constructor on road matrices.

The checkpoint is unchanged. Neural construction still sees node coordinates;
road feasibility is evaluated independently and repaired before execution.
"""
import hashlib
import json
import random
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import default_collate

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "experiments/dispatch_clock_fix_20261106"))
from bootstrap import Config, KW, MCVRPPDTW, SNAP

sys.path.insert(0, str(ROOT / "experiments/hard4_lns_20261106"))
from route_engine import RouteEngine, split_actions
from search import SearchTimeout, repair, solve

sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "experiments/hard4_speed_20261106"))
sys.path.insert(0, str(ROOT / "experiments/neural_diagnosis_20261107"))
from road_route_engine import RoadRouteEngine


MODEL_KEYS = (
    "depot", "loc", "node_type", "demand_passenger", "demand_cargo",
    "time_windows", "pickup_delivery_pairs", "pd_pair_mask", "n_orders",
)


def tensor_sample(sample):
    result = {}
    for key in MODEL_KEYS:
        value = sample[key]
        if key == "pd_pair_mask":
            result[key] = torch.as_tensor(value, dtype=torch.bool)
        elif key in {"pickup_delivery_pairs", "n_orders"}:
            result[key] = torch.as_tensor(value, dtype=torch.long)
        else:
            result[key] = torch.as_tensor(value, dtype=torch.float32)
    return result


def main():
    output = HERE / "neural_transfer_preflight_v1.json"
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    confirmation = json.loads((ROOT / "experiments/neural_confirmation_20261107/confirmation_v1/plan.json").read_text())
    spec = next(model for model in confirmation["models"] if model["policy"] == "learned" and model["seed"] == 1234)
    path = Path(spec["checkpoint"])
    if hashlib.sha256(path.read_bytes()).hexdigest() != spec["sha256"]:
        raise ValueError("Checkpoint hash mismatch")
    profile = confirmation["normalization_profile"]
    Config.set_normalization_profile(50, profile)

    from clock_patch import install as install_clock
    from timing_candidate import install as install_timing
    from completion_guard import install as install_completion
    from admission_targeted import install as install_admission
    from physical_guard_fast import install as install_physical
    from run_v3 import install_policy
    from depot_noop import install as install_depot
    from evaluate_model import build_model_from_checkpoint

    install_clock(); install_timing(); install_completion(); install_admission(); install_physical()
    install_policy("learned"); install_depot()
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    model = build_model_from_checkpoint(checkpoint, torch.device("cpu"))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval(); model.set_decode_type("greedy")

    rows = []
    for instance in range(4):
        source = HERE / f"outputs/preflight_instance{instance:03d}_v3/road_instance.pt"
        bundle = torch.load(source, map_location="cpu", weights_only=True)
        road_sample = bundle["sample"]
        model_sample = tensor_sample(road_sample)
        batch = default_collate([model_sample])
        begin = time.monotonic()
        with torch.no_grad():
            cost, log_likelihood, pi = model(batch, return_pi=True, state_kwargs=KW)
        inference_seconds = time.monotonic() - begin
        if not torch.isfinite(cost).all() or not torch.isfinite(log_likelihood).all():
            raise ValueError("Non-finite neural output")
        actions = pi[0].tolist()
        routes = split_actions(actions, 50)
        straight_engine = RouteEngine(model_sample, confirmation["physical_config"], profile)
        straight_cost = straight_engine.cost(routes)
        if straight_cost is None:
            raise ValueError("Frozen neural plan is invalid even under its coordinate-distance proxy")
        road_engine = RoadRouteEngine(road_sample, confirmation["physical_config"], profile)
        before = road_engine.cost(routes)
        invalid_routes = [index for index, route in enumerate(routes) if road_engine.route(route) is None]
        served = {node for route in routes for node in route if node <= road_engine.n}
        pool = set(range(1, road_engine.n + 1)) - served
        repair_begin = time.monotonic()
        try:
            repaired = repair(road_engine, routes, pool, random.Random(1234 + instance), repair_begin + 5.0, regret=True)
            repaired_result = solve(road_engine, repaired, seed=1234, seconds=0.5, max_iterations=1000)
        except SearchTimeout as error:
            raise TimeoutError(f"Road repair exceeded its finite preflight budget on instance {instance}") from error
        repair_seconds = time.monotonic() - repair_begin
        after = repaired_result["cost"]
        if after is None or after["completed"] < 4 * after["vehicles"]:
            raise AssertionError("Road repair failed the hard-four certificate")
        rows.append({
            "instance": instance,
            "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            "neural_inference_seconds": inference_seconds,
            "neural_proxy_cost": straight_cost,
            "road_valid_before_repair": before is not None,
            "road_cost_before_repair": before,
            "invalid_road_routes_before_repair": invalid_routes,
            "road_repair_and_lns_seconds": repair_seconds,
            "road_cost_after_repair": after,
            "road_routes_after_repair": [list(route) for route in repaired_result["routes"]],
        })
        print("TRANSFER", instance, "proxy_completed", straight_cost["completed"], "road_valid", before is not None,
              "invalid_routes", len(invalid_routes), "repaired_completed", after["completed"], flush=True)
    report = {
        "status": "pass",
        "classification": "four-case CPU transfer preflight; not formal paper evidence",
        "checkpoint": spec,
        "samples": len(rows),
        "road_valid_before_repair": sum(row["road_valid_before_repair"] for row in rows),
        "mean_proxy_completed": sum(row["neural_proxy_cost"]["completed"] for row in rows) / len(rows),
        "mean_repaired_completed": sum(row["road_cost_after_repair"]["completed"] for row in rows) / len(rows),
        "hard_invalid_after_repair": 0,
        "rows": rows,
        "interpretation": "Tests transfer plus pre-execution road-feasibility repair; it is not a road-matrix-aware neural policy and not retraining.",
    }
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: report[key] for key in ("status", "samples", "road_valid_before_repair", "mean_proxy_completed", "mean_repaired_completed", "hard_invalid_after_repair")}, indent=2))


if __name__ == "__main__":
    main()
