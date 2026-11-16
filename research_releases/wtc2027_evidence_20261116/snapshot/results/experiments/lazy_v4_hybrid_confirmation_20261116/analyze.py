"""Paired analysis of the frozen neural/insertion 0.5 s LNS confirmation."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from pathlib import Path


BOOTSTRAP_SEED = 20261116
BOOTSTRAP_REPLICATES = 20_000


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def quantile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    position = probability * (len(ordered) - 1)
    lower, upper = math.floor(position), math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def paired_summary(differences: list[float], rng: random.Random) -> dict:
    count = len(differences)
    bootstrap = [
        sum(differences[rng.randrange(count)] for _ in range(count)) / count
        for _ in range(BOOTSTRAP_REPLICATES)
    ]
    tolerance = 1e-12
    return {
        "mean_difference": sum(differences) / count,
        "paired_instance_bootstrap_95pct_ci": [
            quantile(bootstrap, 0.025),
            quantile(bootstrap, 0.975),
        ],
        "negative_instances": sum(value < -tolerance for value in differences),
        "ties": sum(abs(value) <= tolerance for value in differences),
        "positive_instances": sum(value > tolerance for value in differences),
        "instances": count,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--baseline-report", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    report = json.loads(args.report.read_text())
    baseline = json.loads(args.baseline_report.read_text())
    if report.get("status") != "passed" or len(report.get("rows", [])) != plan["samples"]:
        raise ValueError("hybrid report is incomplete")
    if report.get("plan_sha256") != sha256(args.plan):
        raise ValueError("hybrid report was not generated from the supplied plan")
    if baseline.get("status") != "passed" or len(baseline.get("rows", [])) != plan["samples"]:
        raise ValueError("strong-LNS baseline report is incomplete")

    neural_labels = [f"neural_{seed}" for seed in plan["neural_seeds"]]
    metrics = {
        "service_rate": lambda audit: audit["audit"]["completed"] / plan["graph_size"],
        "objective": lambda audit: audit["cost"]["objective"],
        "raw_cost_cny": lambda audit: audit["cost"]["raw_cost"],
        "distance_km": lambda audit: audit["cost"]["distance"],
        "vehicle_count": lambda audit: audit["cost"]["vehicles"],
    }
    series: dict[str, dict[str, list[float]]] = {}
    for metric, getter in metrics.items():
        neural_start = [
            sum(getter(row["methods"][label]["start"]) for label in neural_labels)
            / len(neural_labels)
            for row in report["rows"]
        ]
        neural_short = [
            sum(getter(row["methods"][label]["short_lns"]) for label in neural_labels)
            / len(neural_labels)
            for row in report["rows"]
        ]
        insertion_start = [getter(row["methods"]["strict_insertion"]["start"]) for row in report["rows"]]
        insertion_short = [
            getter(row["methods"]["strict_insertion"]["short_lns"]) for row in report["rows"]
        ]
        strong = [getter(row["lns"]) for row in baseline["rows"]]
        series[metric] = {
            "neural_start_seed_averaged": neural_start,
            "neural_plus_short_lns_seed_averaged": neural_short,
            "strict_insertion": insertion_start,
            "insertion_plus_short_lns": insertion_short,
            "strong_lns_10s": strong,
        }

    methods = {
        method: {
            metric: sum(values[method]) / len(values[method]) for metric, values in series.items()
        }
        for method in next(iter(series.values()))
    }
    rng = random.Random(BOOTSTRAP_SEED)
    comparisons = [
        ("neural_short_minus_neural_start", "neural_plus_short_lns_seed_averaged", "neural_start_seed_averaged"),
        ("insertion_short_minus_insertion_start", "insertion_plus_short_lns", "strict_insertion"),
        ("neural_short_minus_insertion_start", "neural_plus_short_lns_seed_averaged", "strict_insertion"),
        ("neural_short_minus_insertion_short", "neural_plus_short_lns_seed_averaged", "insertion_plus_short_lns"),
        ("neural_short_minus_strong_lns", "neural_plus_short_lns_seed_averaged", "strong_lns_10s"),
    ]
    paired = {}
    for prefix, left, right in comparisons:
        for metric, values in series.items():
            paired[f"{prefix}_{metric}"] = paired_summary(
                [a - b for a, b in zip(values[left], values[right])], rng
            )

    neural_reports = {
        record["label"]: json.loads(Path(record["path"]).read_text())
        for record in plan["input_reports"]
        if record["label"].startswith("neural_")
    }
    neural_inference_seconds = [
        sum(neural_reports[label]["rows"][index]["inference_seconds"] for label in neural_labels)
        / len(neural_labels)
        for index in range(plan["samples"])
    ]
    neural_search_seconds = [
        sum(report["rows"][index]["methods"][label]["search_seconds"] for label in neural_labels)
        / len(neural_labels)
        for index in range(plan["samples"])
    ]
    insertion_build_seconds = [row["insertion_seconds"] for row in baseline["rows"]]
    insertion_short_seconds = [
        row["methods"]["strict_insertion"]["search_seconds"] for row in report["rows"]
    ]
    methods["neural_start_seed_averaged"]["recorded_inference_seconds"] = sum(neural_inference_seconds) / 100
    methods["neural_plus_short_lns_seed_averaged"]["recorded_component_seconds"] = (
        sum(neural_inference_seconds) + sum(neural_search_seconds)
    ) / 100
    methods["strict_insertion"]["recorded_build_seconds"] = sum(insertion_build_seconds) / 100
    methods["insertion_plus_short_lns"]["recorded_component_seconds"] = (
        sum(insertion_build_seconds) + sum(insertion_short_seconds)
    ) / 100
    methods["strong_lns_10s"]["recorded_component_seconds"] = sum(
        row["insertion_seconds"] + row["lns_search_seconds"] for row in baseline["rows"]
    ) / 100

    result = {
        "status": "passed",
        "scope": "paired confirmation on the same closed 100-instance N100 final set",
        "experimental_unit": "paired N100 instance; neural seeds averaged within instance",
        "bootstrap": {
            "seed": BOOTSTRAP_SEED,
            "replicates": BOOTSTRAP_REPLICATES,
            "method": "nonparametric percentile bootstrap over paired instances",
        },
        "methods": methods,
        "paired": paired,
        "audit": report["summary"],
        "timing_boundary": {
            "publication_grade_cross_method_ratio_available": False,
            "reason": "neural inference was recorded on the AutoDL GPU node, whereas insertion and both LNS runs were recorded on the local CPU; quality comparisons are paired and valid, but end-to-end cross-method time ratios require a same-machine benchmark",
            "within_local_short_lns_budget_is_matched": True,
        },
        "claim_boundary": "the 0.5-second LNS materially improves neural construction, but superiority over insertion or 10-second LNS may be claimed only if the paired estimates support it",
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
