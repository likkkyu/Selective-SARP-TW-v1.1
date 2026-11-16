"""Evaluate one locked neural checkpoint on every frozen formal road case."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import torch


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
FORMAL = HERE / "formal_p1_v1"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

from problem_mcvrptw_v2 import Config, MCVRPPDTW
from road_audit import audit_routes
from road_route_engine import RoadRouteEngine
from smoke_matrix_aware_decode import build_model, split_actions
from train_matrix_aware_finetune import STATE_KWARGS, batches, select_batch


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-pool", type=Path, required=True)
    parser.add_argument("--test-manifest", type=Path, required=True)
    parser.add_argument("--model-lock", type=Path, required=True)
    parser.add_argument("--model-seed", type=int, required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output}")

    test = torch.load(args.test_pool, map_location="cpu", weights_only=True)
    manifest = json.loads(args.test_manifest.read_text())
    lock = json.loads(args.model_lock.read_text())
    if test["meta"].get("purpose") != "formal_test":
        raise ValueError("Refusing to treat a development pool as formal test")
    if manifest.get("sha256") != digest(args.test_pool):
        raise ValueError("Test-pool hash differs from manifest")
    if test["meta"].get("model_lock_sha256") != digest(args.model_lock):
        raise ValueError("Test was not generated after this model lock")
    if lock.get("status") != "locked":
        raise ValueError("Model lock is not complete")
    matches = [row for row in lock["models"] if row["initial_model_seed"] == args.model_seed]
    if len(matches) != 1:
        raise ValueError(f"Model seed {args.model_seed} not uniquely present in lock")
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

    rows = []
    started = time.monotonic()
    indices = list(range(len(test["data"])))
    with torch.no_grad():
        for group in batches(indices, args.batch_size):
            batch = select_batch(test["data"], group)
            cost, _, actions = model(batch, return_pi=True, state_kwargs=STATE_KWARGS)
            executable, forced = MCVRPPDTW._sanitize_underfilled_dispatches(actions, 50)
            for local, index in enumerate(group):
                routes = split_actions(executable[local].tolist(), 50)
                audit = audit_routes(test["data"][index], routes, plan["physical_config"])
                engine = RoadRouteEngine(
                    test["data"][index],
                    plan["physical_config"],
                    plan["normalization_profile"],
                )
                comparable_cost = engine.cost(routes)
                if comparable_cost is None:
                    raise AssertionError(
                        f"Independent scalar engine rejected test instance {index} for seed {args.model_seed}"
                    )
                rows.append({
                    "test_index": index,
                    "objective_cost": float(comparable_cost["objective"]),
                    "decoder_candidate_objective_cost": float(cost[local]),
                    "raw_cost": float(comparable_cost["raw_cost"]),
                    "distance_km": float(comparable_cost["distance"]),
                    "completed": int(audit["completed"]),
                    "service_rate_percent": 2.0 * float(audit["completed"]),
                    "vehicles": int(audit["vehicles"]),
                    "minimum_completed_per_trip": audit["minimum_completed_per_trip"],
                    "forced_underfilled_declines": int(forced[local]),
                    "valid": bool(audit["valid"]),
                    "violations": audit["violations"],
                })
    invalid = sum(not row["valid"] for row in rows)
    report = {
        "status": "pass" if invalid == 0 else "fail",
        "scope": "locked neural checkpoint on untouched road-native formal test",
        "decoder_semantics_version": 2,
        "evaluation_semantics_version": 2,
        "decoder_semantics": (
            "hard-constraint tolerance aligned to independent audit; depot return "
            "forbidden with open pickup commitments; unsafe all-masked fallback rejected"
        ),
        "state_code_sha256": digest(ROOT / "state_mcvrptw_v2.py"),
        "attention_model_code_sha256": digest(ROOT / "nets" / "attention_model.py"),
        "route_engine_code_sha256": digest(HERE / "road_route_engine.py"),
        "feasibility_tolerance_hours": 1e-4,
        "test_pool_sha256": digest(args.test_pool),
        "test_manifest_sha256": digest(args.test_manifest),
        "model_lock_sha256": digest(args.model_lock),
        "model_seed": args.model_seed,
        "checkpoint_sha256": spec["checkpoint_sha256"],
        "checkpoint_epoch": spec["checkpoint_epoch"],
        "instances": len(rows),
        "mean_objective_cost": sum(row["objective_cost"] for row in rows) / len(rows),
        "mean_completed": sum(row["completed"] for row in rows) / len(rows),
        "service_rate_percent": sum(row["service_rate_percent"] for row in rows) / len(rows),
        "mean_vehicles": sum(row["vehicles"] for row in rows) / len(rows),
        "hard_invalid": invalid,
        "forced_underfilled_declines": sum(row["forced_underfilled_declines"] for row in rows),
        "decode_and_audit_seconds": time.monotonic() - started,
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: report[key] for key in (
        "status", "model_seed", "instances", "service_rate_percent",
        "mean_objective_cost", "mean_vehicles", "hard_invalid",
        "decode_and_audit_seconds",
    )}, indent=2))
    if invalid:
        raise AssertionError(f"{invalid} formal-test instances failed route audit")


if __name__ == "__main__":
    main()
