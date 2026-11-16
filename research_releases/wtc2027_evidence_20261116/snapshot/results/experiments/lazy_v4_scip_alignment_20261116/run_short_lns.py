"""Apply the preselected 0.5-second LNS to each current neural start."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import signal
import sys
import time
import traceback


ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT = Path(__file__).resolve().parent
for folder in reversed(
    (
        ROOT / "experiments/drl_canonical_audit_20261115",
        ROOT / "experiments/hard4_lns_20261106",
        ROOT / "experiments/min_dispatch_hard_20261106",
        ROOT / "experiments/mip_scip_20261106",
    )
):
    sys.path.insert(0, str(folder))

import torch  # noqa: E402

torch.set_num_threads(1)
import audit_common as common  # noqa: E402
from evaluate_lazy_v4_canonical import install_lazy_patches  # noqa: E402
from problem_mcvrptw_v2 import MCVRPPDTWDataset  # noqa: E402
from route_engine import RouteEngine, split_actions  # noqa: E402
from search import solve  # noqa: E402


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    plan_path = args.plan.resolve()
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    if plan.get("status") != "frozen_before_short_lns_alignment":
        raise ValueError("Short-LNS alignment protocol is not frozen")
    for relative, expected in plan["source_sha256"].items():
        observed = sha256(ROOT / relative)
        if observed != expected:
            raise ValueError(
                {"source_hash_mismatch": relative, "expected": expected, "observed": observed}
            )
    inputs = {}
    for record in plan["input_reports"]:
        path = ROOT / record["path"]
        if sha256(path) != record["sha256"]:
            raise ValueError(f"Input evidence changed: {record['label']}")
        report = json.loads(path.read_text(encoding="utf-8"))
        if report.get("status") != "passed" or len(report.get("rows", [])) != plan["samples"]:
            raise ValueError(f"Incomplete input: {record['label']}")
        inputs[record["label"]] = report

    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(output)
    install_lazy_patches()
    normalization = common.profile(plan["graph_size"])
    common.Config.set_normalization_profile(plan["graph_size"], normalization)
    config = common.parameters(common.Config)
    dataset = MCVRPPDTWDataset(
        num_samples=plan["samples"],
        graph_size=plan["graph_size"],
        seed=plan["dataset_seed"],
    )
    started = time.monotonic()
    report = {
        "status": "running",
        "plan": str(plan_path.relative_to(ROOT)),
        "plan_sha256": sha256(plan_path),
        "rows": [],
    }

    def save() -> None:
        report["elapsed_seconds"] = time.monotonic() - started
        output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    signal.signal(
        signal.SIGALRM,
        lambda *_: (_ for _ in ()).throw(TimeoutError("Short-LNS alignment timeout")),
    )
    signal.alarm(plan["total_timeout_seconds"])
    save()
    try:
        labels = [f"neural_{seed}" for seed in plan["training_seeds"]]
        for instance in range(plan["samples"]):
            sample = dataset[instance]
            methods = {}
            for label in labels:
                source = inputs[label]["rows"][instance]
                if source["instance"] != instance:
                    raise RuntimeError(f"Instance mismatch for {label} at {instance}")
                actions = source["actions"]
                start = common.verify_actions(
                    sample, actions, plan["graph_size"], config, normalization
                )
                engine = RouteEngine(sample, config, normalization)
                result = solve(
                    engine,
                    split_actions(actions, plan["graph_size"]),
                    seed=plan["short_lns"]["search_seed"],
                    seconds=plan["short_lns"]["seconds_per_instance"],
                    max_iterations=plan["short_lns"]["max_iterations"],
                )
                final = common.verify_actions(
                    sample, result["actions"], plan["graph_size"], config, normalization
                )
                if final["cost"]["objective"] > start["cost"]["objective"] + 1e-8:
                    raise RuntimeError(f"LNS worsened {label} at instance {instance}")
                methods[label] = {
                    "start": start,
                    "short_lns": final,
                    "short_lns_actions": result["actions"],
                    "search_seconds": result["seconds"],
                    "iterations": result["iterations"],
                    "partial_last_iteration_discarded": result[
                        "partial_last_iteration_discarded"
                    ],
                }
            report["rows"].append({"instance": instance, "methods": methods})
            save()
            print(
                "ALIGNMENT_LNS",
                instance + 1,
                "/",
                plan["samples"],
                [methods[label]["short_lns"]["audit"]["completed"] for label in labels],
                flush=True,
            )
        report["status"] = "passed"
        report["summary"] = {
            "instances": len(report["rows"]),
            "hard_violations": 0,
            "unfulfilled": 0,
            "all_trips_at_least_four": all(
                all(
                    completed >= 4
                    for completed in row["methods"][label]["short_lns"][
                        "completed_per_trip"
                    ]
                )
                for row in report["rows"]
                for label in labels
            ),
        }
        if not report["summary"]["all_trips_at_least_four"]:
            raise RuntimeError("An underfilled trip survived the independent audit")
    except BaseException as exc:
        report.update(status="failed", error=repr(exc), traceback=traceback.format_exc())
        raise
    finally:
        save()
        signal.alarm(0)


if __name__ == "__main__":
    main()
