"""One-case CPU smoke test for matrix-aware neural decoding (no retraining)."""

import hashlib
import json
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import default_collate


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
FORMAL = HERE / "formal_p1_v1"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

from nets.attention_model import AttentionModel
from problem_mcvrptw_v2 import Config, MCVRPPDTW
from road_audit import audit_routes


MODEL_KEYS = (
    "depot", "loc", "node_type", "demand_passenger", "demand_cargo",
    "time_windows", "pickup_delivery_pairs", "pd_pair_mask", "n_orders",
    "road_distance_matrix_km", "road_travel_time_matrix_hours",
)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def build_model(checkpoint):
    args = checkpoint.get("args", {}) or {}
    if not isinstance(args, dict):
        args = vars(args)
    return AttentionModel(
        embedding_dim=args.get("embedding_dim", 256),
        hidden_dim=args.get("hidden_dim", 256),
        problem=MCVRPPDTW,
        n_encode_layers=args.get("n_encode_layers", 6),
        n_heads=args.get("n_heads", 8),
        tanh_clipping=args.get("tanh_clipping", 10.0),
        normalization=args.get("normalization", "batch"),
        shrink_size=None,
        decode_pickup_urgency_bias=args.get("decode_pickup_urgency_bias", 0.0),
        decode_pickup_urgency_horizon_hours=args.get("decode_pickup_urgency_horizon_hours", 1.0),
    )


def split_actions(actions, n_orders):
    routes, current = [], []
    reject_index = 2 * n_orders + 1
    for action in actions:
        if action == reject_index:
            continue
        if action == 0:
            if current:
                routes.append(current)
                current = []
            continue
        if 1 <= action <= 2 * n_orders:
            current.append(action)
    if current:
        routes.append(current)
    return routes


def main():
    plan = json.loads((FORMAL / "plan.json").read_text())
    spec = plan["models"][0]
    if digest(spec["checkpoint"]) != spec["sha256"]:
        raise ValueError("checkpoint hash mismatch")
    sample = torch.load(FORMAL / "road_cases.pt", map_location="cpu", weights_only=True)["data"][0]
    road_sample = dict(sample)
    road_sample["road_travel_time_matrix_hours"] = (
        sample["road_distance_matrix_km"] / plan["physical_config"]["VEHICLE_SPEED"]
    )
    batch = default_collate([{key: road_sample[key] for key in MODEL_KEYS}])
    Config.set_normalization_profile(50, plan["normalization_profile"])
    checkpoint = torch.load(spec["checkpoint"], map_location="cpu", weights_only=False)
    model = build_model(checkpoint)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    model.set_decode_type("greedy")
    state_kwargs = {
        "max_concurrent_open_orders": 6,
        "min_orders_per_dispatch": 4,
        "enable_delivery_viability": True,
        "enable_viability_fallback": False,
        "relax_pickup_commitment_trip_time": False,
    }
    started = time.monotonic()
    with torch.no_grad():
        cost, log_likelihood, pi = model(batch, return_pi=True, state_kwargs=state_kwargs)
    elapsed = time.monotonic() - started
    if not torch.isfinite(cost).all() or not torch.isfinite(log_likelihood).all():
        raise AssertionError("non-finite matrix-aware decode")
    raw_routes = split_actions(pi[0].tolist(), 50)
    raw_audit = audit_routes(road_sample, raw_routes, plan["physical_config"])
    executable_pi, forced_declines = MCVRPPDTW._sanitize_underfilled_dispatches(pi, 50)
    routes = split_actions(executable_pi[0].tolist(), 50)
    audit = audit_routes(road_sample, routes, plan["physical_config"])
    if not audit["valid"]:
        raise AssertionError({"audit": audit, "routes": routes, "action_tail": pi[0].tolist()[-40:]})
    result = {
        "status": "pass",
        "scope": "runtime smoke only; unchanged Euclidean-trained checkpoint",
        "checkpoint_seed": spec["seed"],
        "instance": 0,
        "decode_seconds_cpu": elapsed,
        "finite_cost": float(cost[0]),
        "raw_candidate_valid": raw_audit["valid"],
        "forced_underfilled_declines": int(forced_declines[0]),
        "completed_orders": audit["completed"],
        "vehicles": audit["vehicles"],
        "minimum_completed_per_trip": audit["minimum_completed_per_trip"],
        "hard_violations": len(audit["violations"]),
        "interpretation_guard": "This verifies the matrix-aware code path, not model quality or road-aware learning.",
    }
    (HERE / "matrix_aware_decode_smoke_report.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
