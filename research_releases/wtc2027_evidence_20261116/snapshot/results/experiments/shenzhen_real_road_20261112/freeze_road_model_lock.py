"""Freeze the three road-aware neural checkpoints before formal-test creation."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--run", type=Path, action="append", required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output}")
    if len(args.run) != 3:
        raise ValueError("Exactly three independently trained runs are required")

    models = []
    seeds = set()
    pool_hashes = set()
    split_hashes = set()
    for run in args.run:
        protocol_path = run / "protocol.json"
        report_path = run / "report.json"
        evaluation_path = run / "independent_validation_best.json"
        checkpoint_path = run / "best.pt"
        for path in (protocol_path, report_path, evaluation_path, checkpoint_path):
            if not path.is_file():
                raise FileNotFoundError(path)
        protocol = json.loads(protocol_path.read_text())
        report = json.loads(report_path.read_text())
        evaluation = json.loads(evaluation_path.read_text())
        if protocol.get("status") != "completed" or report.get("status") != "completed":
            raise ValueError(f"Incomplete run: {run}")
        if evaluation.get("status") != "pass" or evaluation.get("hard_invalid") != 0:
            raise ValueError(f"Independent validation failed: {run}")
        seed = int(protocol["initial_model_seed"])
        if seed in seeds:
            raise ValueError(f"Duplicate initial model seed: {seed}")
        seeds.add(seed)
        pool_hashes.add(protocol["pool_sha256"])
        split_hashes.add(protocol["validation_split_source_sha256"])
        models.append({
            "initial_model_seed": seed,
            "training_seed": int(protocol["seed"]),
            "checkpoint": str(checkpoint_path.resolve().relative_to(ROOT)),
            "checkpoint_sha256": digest(checkpoint_path),
            "checkpoint_epoch": int(evaluation["checkpoint_epoch"]),
            "validation_service_rate_percent": evaluation["service_rate_percent"],
            "validation_mean_objective_cost": evaluation["mean_objective_cost"],
            "validation_hard_invalid": evaluation["hard_invalid"],
            "training_protocol": str(protocol_path.resolve().relative_to(ROOT)),
            "training_protocol_sha256": digest(protocol_path),
            "independent_validation": str(evaluation_path.resolve().relative_to(ROOT)),
            "independent_validation_sha256": digest(evaluation_path),
        })
    if len(pool_hashes) != 1 or len(split_hashes) != 1:
        raise ValueError("Runs do not share the same development pool and validation split")

    payload = {
        "status": "locked",
        "locked_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "neural model choices frozen before road-native formal-test generation",
        "development_pool_sha256": next(iter(pool_hashes)),
        "validation_split_source_sha256": next(iter(split_hashes)),
        "checkpoint_selection_rule": (
            "zero hard invalid, then highest validation service rate, "
            "then lowest validation mean cost"
        ),
        "models": sorted(models, key=lambda row: row["initial_model_seed"]),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({
        "status": payload["status"],
        "models": len(payload["models"]),
        "output": str(args.output),
        "sha256": digest(args.output),
    }, indent=2))


if __name__ == "__main__":
    main()
