"""Run the predeclared 0.5 s LNS confirmation from frozen neural/insertion starts."""

from __future__ import annotations

import argparse
import hashlib
import json
import signal
import sys
import time
import traceback
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
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
    plan = json.loads(plan_path.read_text())
    if plan.get("status") != "frozen_before_hybrid_evaluation":
        raise ValueError("hybrid confirmation protocol is not frozen")
    for relative, expected in plan["source_sha256"].items():
        observed = sha256(ROOT / relative)
        if observed != expected:
            raise ValueError(
                {"source_hash_mismatch": relative, "expected": expected, "observed": observed}
            )
    for record in plan["input_reports"]:
        if sha256(Path(record["path"])) != record["sha256"]:
            raise ValueError(f"input evidence changed: {record['label']}")

    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    install_lazy_patches()
    n = plan["graph_size"]
    normalization = common.profile(n)
    common.Config.set_normalization_profile(n, normalization)
    config = common.parameters(common.Config)
    dataset = MCVRPPDTWDataset(
        num_samples=plan["samples"], graph_size=n, seed=plan["dataset_seed"]
    )

    inputs = {record["label"]: json.loads(Path(record["path"]).read_text()) for record in plan["input_reports"]}
    neural_labels = [f"neural_{seed}" for seed in plan["neural_seeds"]]
    baseline = inputs["strict_insertion"]
    for label in neural_labels:
        if inputs[label]["status"] != "passed" or len(inputs[label]["rows"]) != plan["samples"]:
            raise ValueError(f"incomplete neural input: {label}")
    if baseline["status"] != "passed" or len(baseline["rows"]) != plan["samples"]:
        raise ValueError("incomplete baseline input")

    started = time.monotonic()
    report = {
        "status": "running",
        "plan": str(plan_path),
        "plan_sha256": sha256(plan_path),
        "rows": [],
    }

    def save() -> None:
        report["elapsed_seconds"] = time.monotonic() - started
        (output / "REPORT.json").write_text(json.dumps(report, indent=2) + "\n")

    signal.signal(
        signal.SIGALRM,
        lambda *_: (_ for _ in ()).throw(TimeoutError("hybrid confirmation timeout")),
    )
    signal.alarm(plan["total_timeout_seconds"])
    save()
    try:
        for index in range(plan["samples"]):
            sample = dataset[index]
            starts = {
                label: inputs[label]["rows"][index]["actions"] for label in neural_labels
            }
            starts["strict_insertion"] = baseline["rows"][index]["insertion_actions"]
            methods = {}
            for label in neural_labels + ["strict_insertion"]:
                actions = starts[label]
                start_audit = common.verify_actions(sample, actions, n, config, normalization)
                engine = RouteEngine(sample, config, normalization)
                result = solve(
                    engine,
                    split_actions(actions, n),
                    seed=plan["short_lns"]["search_seed"],
                    seconds=plan["short_lns"]["seconds_per_instance"],
                    max_iterations=plan["short_lns"]["max_iterations"],
                )
                final_audit = common.verify_actions(
                    sample, result["actions"], n, config, normalization
                )
                if final_audit["cost"]["objective"] > start_audit["cost"]["objective"] + 1e-8:
                    raise RuntimeError(f"short LNS worsened {label} on instance {index}")
                methods[label] = {
                    "start": start_audit,
                    "start_actions_sha256": hashlib.sha256(
                        json.dumps(actions, separators=(",", ":")).encode()
                    ).hexdigest(),
                    "short_lns": final_audit,
                    "short_lns_actions": result["actions"],
                    "search_seconds": result["seconds"],
                    "iterations": result["iterations"],
                    "partial_last_iteration_discarded": result[
                        "partial_last_iteration_discarded"
                    ],
                }
            report["rows"].append({"instance": index, "methods": methods})
            save()
            print(
                "HYBRID",
                index + 1,
                "/",
                plan["samples"],
                "neural_completed",
                [methods[label]["short_lns"]["audit"]["completed"] for label in neural_labels],
                "insertion_completed",
                methods["strict_insertion"]["short_lns"]["audit"]["completed"],
                flush=True,
            )
        report["status"] = "passed"
        report["summary"] = {
            "instances": len(report["rows"]),
            "methods": {},
            "hard_violations": 0,
            "unfulfilled": 0,
            "all_trips_at_least_four": True,
        }
        for label in neural_labels + ["strict_insertion"]:
            records = [row["methods"][label] for row in report["rows"]]
            report["summary"]["methods"][label] = {
                "start_service_rate": sum(x["start"]["audit"]["completed"] for x in records)
                / (len(records) * n),
                "short_lns_service_rate": sum(
                    x["short_lns"]["audit"]["completed"] for x in records
                )
                / (len(records) * n),
                "start_mean_objective": sum(x["start"]["cost"]["objective"] for x in records)
                / len(records),
                "short_lns_mean_objective": sum(
                    x["short_lns"]["cost"]["objective"] for x in records
                )
                / len(records),
                "mean_search_seconds": sum(x["search_seconds"] for x in records) / len(records),
            }
    except BaseException as exc:
        report.update(status="failed", error=repr(exc), traceback=traceback.format_exc())
        raise
    finally:
        save()
        signal.alarm(0)


if __name__ == "__main__":
    main()
