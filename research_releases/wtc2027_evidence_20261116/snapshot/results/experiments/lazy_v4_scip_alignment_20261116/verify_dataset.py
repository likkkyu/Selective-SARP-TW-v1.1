"""Prove that seed regeneration reproduces the frozen SCIP scenario exactly."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import torch


ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT = Path(__file__).resolve().parent
PLAN = json.loads((EXPERIMENT / "PLAN.json").read_text(encoding="utf-8"))
SNAPSHOT = ROOT / "experiments/evaluation_alignment_20261103/snapshot"
sys.path.insert(0, str(SNAPSHOT))

from problem_mcvrptw_v2 import MCVRPPDTWDataset  # noqa: E402


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    scenario = ROOT / PLAN["scenario_path"]
    if sha256(scenario) != PLAN["scenario_sha256"]:
        raise RuntimeError("Frozen scenario hash mismatch")
    saved = torch.load(scenario, map_location="cpu", weights_only=False)
    generated = MCVRPPDTWDataset(
        num_samples=PLAN["samples"],
        graph_size=PLAN["graph_size"],
        seed=PLAN["dataset_seed"],
    )
    if len(saved["data"]) != len(generated) or len(generated) != PLAN["samples"]:
        raise RuntimeError("Dataset length mismatch")
    tensors = 0
    scalars = 0
    for instance, (left, right) in enumerate(zip(saved["data"], generated)):
        if set(left) != set(right):
            raise RuntimeError(f"Key mismatch at instance {instance}")
        for key in left:
            if torch.is_tensor(left[key]):
                tensors += 1
                if not torch.equal(left[key], right[key]):
                    raise RuntimeError(f"Tensor mismatch at instance {instance}, key {key}")
            else:
                scalars += 1
                if left[key] != right[key]:
                    raise RuntimeError(f"Value mismatch at instance {instance}, key {key}")
    report = {
        "status": "passed",
        "scenario_path": PLAN["scenario_path"],
        "scenario_sha256": PLAN["scenario_sha256"],
        "dataset_seed": PLAN["dataset_seed"],
        "graph_size": PLAN["graph_size"],
        "instances": len(generated),
        "tensor_fields_compared": tensors,
        "non_tensor_fields_compared": scalars,
        "comparison": "exact key equality and torch.equal for every tensor",
    }
    (EXPERIMENT / "DATASET_EQUIVALENCE.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
