"""Reload and independently audit a road-aware checkpoint on a frozen split."""

from __future__ import annotations

import argparse
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

from problem_mcvrptw_v2 import Config, MCVRPPDTW
from road_audit import audit_routes
from smoke_matrix_aware_decode import MODEL_KEYS, build_model, split_actions
from train_matrix_aware_finetune import STATE_KWARGS, batches, select_batch


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pool", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--training-protocol", type=Path, required=True)
    parser.add_argument("--split", choices=("training", "validation"), default="validation")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output}")

    plan = json.loads((FORMAL / "plan.json").read_text())
    protocol = json.loads(args.training_protocol.read_text())
    pool = torch.load(args.pool, map_location="cpu", weights_only=True)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if digest(args.pool) != protocol["pool_sha256"]:
        raise ValueError("Pool hash differs from the frozen training protocol")
    if pool["meta"]["formal_test_road_cases_sha256"] != protocol["formal_test_sha256_excluded"]:
        raise ValueError("Formal-test exclusion metadata mismatch")
    indices = protocol[f"{args.split}_indices"]
    Config.set_normalization_profile(50, plan["normalization_profile"])
    model = build_model(checkpoint)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    model.set_decode_type("greedy")

    rows = []
    started = time.monotonic()
    with torch.no_grad():
        for group in batches(indices, args.batch_size):
            batch = select_batch(pool["data"], group)
            cost, _, pi = model(batch, return_pi=True, state_kwargs=STATE_KWARGS)
            executable, forced = MCVRPPDTW._sanitize_underfilled_dispatches(pi, 50)
            for local, data_index in enumerate(group):
                routes = split_actions(executable[local].tolist(), 50)
                audit = audit_routes(pool["data"][data_index], routes, plan["physical_config"])
                rows.append(
                    {
                        "pool_index": int(data_index),
                        "objective_cost": float(cost[local]),
                        "completed": int(audit["completed"]),
                        "service_rate_percent": 2.0 * float(audit["completed"]),
                        "vehicles": int(audit["vehicles"]),
                        "minimum_completed_per_trip": audit["minimum_completed_per_trip"],
                        "forced_underfilled_declines": int(forced[local]),
                        "valid": bool(audit["valid"]),
                        "violations": audit["violations"],
                    }
                )
    invalid = sum(not row["valid"] for row in rows)
    report = {
        "status": "pass" if invalid == 0 else "fail",
        "scope": f"independent checkpoint reload on frozen {args.split} split",
        "pool_sha256": digest(args.pool),
        "checkpoint_sha256": digest(args.checkpoint),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "training_protocol_sha256": digest(args.training_protocol),
        "split": args.split,
        "instances": len(rows),
        "mean_objective_cost": sum(row["objective_cost"] for row in rows) / len(rows),
        "mean_completed": sum(row["completed"] for row in rows) / len(rows),
        "service_rate_percent": sum(row["service_rate_percent"] for row in rows) / len(rows),
        "hard_invalid": invalid,
        "forced_underfilled_declines": sum(row["forced_underfilled_declines"] for row in rows),
        "decode_and_audit_seconds": time.monotonic() - started,
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: report[key] for key in (
        "status", "instances", "mean_objective_cost", "mean_completed",
        "service_rate_percent", "hard_invalid", "forced_underfilled_declines",
        "decode_and_audit_seconds",
    )}, indent=2))
    if invalid:
        raise AssertionError(f"{invalid} independently audited instances are invalid")


if __name__ == "__main__":
    main()
