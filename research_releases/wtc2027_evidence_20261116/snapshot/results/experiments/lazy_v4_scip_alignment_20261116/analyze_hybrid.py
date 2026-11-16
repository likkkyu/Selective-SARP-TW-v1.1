"""Paired analysis of current neural+0.5s LNS versus insertion/SCIP."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import random
import statistics


ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT = Path(__file__).resolve().parent
PLAN = json.loads((EXPERIMENT / "PLAN.json").read_text(encoding="utf-8"))
SHORT_PLAN_PATH = EXPERIMENT / "SHORT_LNS_PLAN.json"
SHORT_PLAN = json.loads(SHORT_PLAN_PATH.read_text(encoding="utf-8"))
SHORT_REPORT_PATH = EXPERIMENT / "results/SHORT_LNS_REPORT.json"
SCIP_ROOT = ROOT / PLAN["comparison_source"]
OUTPUT = EXPERIMENT / "HYBRID_PAIRED_ANALYSIS.json"
REPLICATES = 20_000
SEED = 20261117


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def quantile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    position = probability * (len(ordered) - 1)
    low, high = math.floor(position), math.ceil(position)
    if low == high:
        return ordered[low]
    fraction = position - low
    return ordered[low] * (1 - fraction) + ordered[high] * fraction


def paired(values: list[float], rng: random.Random) -> dict:
    n = len(values)
    bootstrap = [
        statistics.fmean(values[rng.randrange(n)] for _ in range(n))
        for _ in range(REPLICATES)
    ]
    return {
        "mean_difference": statistics.fmean(values),
        "paired_instance_bootstrap_95pct_ci": [
            quantile(bootstrap, 0.025),
            quantile(bootstrap, 0.975),
        ],
        "negative_instances": sum(value < -1e-12 for value in values),
        "ties": sum(abs(value) <= 1e-12 for value in values),
        "positive_instances": sum(value > 1e-12 for value in values),
        "instances": n,
    }


def mean_metrics(costs: list[dict]) -> dict:
    return {
        "service_rate": statistics.fmean(cost["completed"] / PLAN["graph_size"] for cost in costs),
        "objective": statistics.fmean(cost["objective"] for cost in costs),
        "raw_cost": statistics.fmean(cost["raw_cost"] for cost in costs),
        "distance": statistics.fmean(cost["distance"] for cost in costs),
        "vehicles": statistics.fmean(cost["vehicles"] for cost in costs),
    }


def main() -> None:
    short = json.loads(SHORT_REPORT_PATH.read_text(encoding="utf-8"))
    if short.get("status") != "passed" or len(short.get("rows", [])) != PLAN["samples"]:
        raise RuntimeError("Short-LNS report is incomplete")
    if short.get("plan_sha256") != sha256(SHORT_PLAN_PATH):
        raise RuntimeError("Short-LNS plan hash mismatch")
    if (
        short["summary"]["hard_violations"]
        or short["summary"]["unfulfilled"]
        or not short["summary"]["all_trips_at_least_four"]
    ):
        raise RuntimeError("Short-LNS hard audit failed")

    labels = [f"neural_{seed}" for seed in PLAN["training_seeds"]]
    scip_rows = []
    for instance in range(PLAN["samples"]):
        path = SCIP_ROOT / f"scip_n100_{instance:02d}/report.json"
        report = json.loads(path.read_text(encoding="utf-8"))
        if report.get("status") != "completed" or report.get("scenario_sha256") != PLAN["scenario_sha256"]:
            raise RuntimeError(f"Invalid SCIP report {instance}")
        row = report["rows"][0]
        if row["instance"] != instance:
            raise RuntimeError(f"SCIP index mismatch {instance}")
        scip_rows.append(row)

    metrics = {
        "service_rate": lambda audited: audited["audit"]["completed"] / PLAN["graph_size"],
        "objective": lambda audited: audited["cost"]["objective"],
        "raw_cost": lambda audited: audited["cost"]["raw_cost"],
        "distance": lambda audited: audited["cost"]["distance"],
        "vehicles": lambda audited: audited["cost"]["vehicles"],
    }
    series: dict[str, dict[str, list[float]]] = {}
    for metric, getter in metrics.items():
        raw = [
            statistics.fmean(getter(row["methods"][label]["start"]) for label in labels)
            for row in short["rows"]
        ]
        hybrid = [
            statistics.fmean(getter(row["methods"][label]["short_lns"]) for label in labels)
            for row in short["rows"]
        ]
        insertion = [getter(row["initial"]) for row in scip_rows]
        incumbent = [getter(row["final"]) for row in scip_rows]
        series[metric] = {
            "current_neural_seed_averaged": raw,
            "neural_plus_0_5s_lns_seed_averaged": hybrid,
            "strict_insertion": insertion,
            "scip_300s_incumbent": incumbent,
        }

    methods = {
        method: {
            metric: statistics.fmean(values[method]) for metric, values in series.items()
        }
        for method in next(iter(series.values()))
    }
    raw_reports = {
        314159: EXPERIMENT / "results/seed_314159/report.json",
        271828: EXPERIMENT / "results/seed_271828_serial_retry1/report.json",
        161803: EXPERIMENT / "results/seed_161803/report.json",
    }
    loaded = {
        seed: json.loads(path.read_text(encoding="utf-8")) for seed, path in raw_reports.items()
    }
    per_instance_inference = [
        statistics.fmean(loaded[seed]["rows"][instance]["inference_seconds"] for seed in PLAN["training_seeds"])
        for instance in range(PLAN["samples"])
    ]
    per_instance_lns = [
        statistics.fmean(short["rows"][instance]["methods"][label]["search_seconds"] for label in labels)
        for instance in range(PLAN["samples"])
    ]
    insertion_seconds = [row["insertion_seconds"] for row in scip_rows]
    scip_pipeline_seconds = [
        row["insertion_seconds"] + row["solver_external_seconds"] for row in scip_rows
    ]
    methods["current_neural_seed_averaged"]["mean_inference_seconds_local_cpu"] = statistics.fmean(per_instance_inference)
    methods["neural_plus_0_5s_lns_seed_averaged"]["mean_end_to_end_seconds_local_cpu"] = statistics.fmean(
        inference + search for inference, search in zip(per_instance_inference, per_instance_lns)
    )
    methods["strict_insertion"]["mean_build_seconds_local_cpu"] = statistics.fmean(insertion_seconds)
    methods["scip_300s_incumbent"]["mean_pipeline_seconds_local_cpu"] = statistics.fmean(scip_pipeline_seconds)
    hybrid_seconds = methods["neural_plus_0_5s_lns_seed_averaged"][
        "mean_end_to_end_seconds_local_cpu"
    ]
    timing_ratios = {
        "insertion_over_neural_hybrid": methods["strict_insertion"][
            "mean_build_seconds_local_cpu"
        ]
        / hybrid_seconds,
        "scip_pipeline_over_neural_hybrid": methods["scip_300s_incumbent"][
            "mean_pipeline_seconds_local_cpu"
        ]
        / hybrid_seconds,
    }

    rng = random.Random(SEED)
    comparisons = {}
    for name, left, right in (
        ("hybrid_minus_raw", "neural_plus_0_5s_lns_seed_averaged", "current_neural_seed_averaged"),
        ("hybrid_minus_insertion", "neural_plus_0_5s_lns_seed_averaged", "strict_insertion"),
        ("hybrid_minus_scip_incumbent", "neural_plus_0_5s_lns_seed_averaged", "scip_300s_incumbent"),
    ):
        comparisons[name] = {
            metric: paired(
                [left_value - right_value for left_value, right_value in zip(values[left], values[right])],
                rng,
            )
            for metric, values in series.items()
        }

    seed_metrics = {}
    for label, seed in zip(labels, PLAN["training_seeds"]):
        starts = [row["methods"][label]["start"]["cost"] for row in short["rows"]]
        finals = [row["methods"][label]["short_lns"]["cost"] for row in short["rows"]]
        seed_metrics[str(seed)] = {
            "start": mean_metrics(starts),
            "short_lns": mean_metrics(finals),
        }

    result = {
        "status": "passed",
        "scope": SHORT_PLAN["scope"],
        "claim_boundary": SHORT_PLAN["claim_boundary"],
        "experimental_unit": "paired N100 instance; training seeds averaged within each instance",
        "samples": PLAN["samples"],
        "training_seeds": PLAN["training_seeds"],
        "methods": methods,
        "seed_metrics": seed_metrics,
        "paired": comparisons,
        "scip_solver": {
            "time_limit_instances": sum(row["result"]["status"] == "timelimit" for row in scip_rows),
            "optimal_instances": sum(row["result"]["status"] == "optimal" for row in scip_rows),
            "improved_warm_start_instances": sum(row["objective_improvement"] > 1e-9 for row in scip_rows),
            "median_gap": statistics.median(row["result"]["gap"] for row in scip_rows),
        },
        "audit": short["summary"],
        "timing_ratios": timing_ratios,
        "timing_boundary": "Same Apple M2 host and one-thread evaluator, but runs occurred in separate sessions; ratios are hardware-specific deployment observations.",
        "bootstrap": {
            "replicates": REPLICATES,
            "seed": SEED,
            "method": "paired nonparametric percentile bootstrap over instances",
        },
    }
    OUTPUT.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "status": result["status"],
                "methods": result["methods"],
                "paired": {
                    key: {
                        "service_rate": value["service_rate"],
                        "objective": value["objective"],
                    }
                    for key, value in result["paired"].items()
                },
                "scip_solver": result["scip_solver"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
