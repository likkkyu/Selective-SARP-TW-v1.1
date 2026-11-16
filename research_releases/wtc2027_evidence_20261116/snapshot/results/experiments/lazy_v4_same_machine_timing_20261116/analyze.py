"""Analyze the frozen same-node constructor timing benchmark."""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path


def quantile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    position = probability * (len(ordered) - 1)
    lower, upper = math.floor(position), math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def describe(values: list[float]) -> dict:
    return {
        "n": len(values),
        "mean_seconds": sum(values) / len(values),
        "median_seconds": quantile(values, 0.5),
        "p95_seconds": quantile(values, 0.95),
        "min_seconds": min(values),
        "max_seconds": max(values),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    protocol = json.loads(args.protocol.read_text())
    report = json.loads(args.report.read_text())
    indices = protocol["selection"]["selected_indices"]
    if report.get("status") != "passed":
        raise ValueError("timing report did not pass")
    seeds = [str(seed) for seed in protocol["neural_seeds"]]
    for seed in seeds:
        rows = report["neural"][seed]["rows"]
        if [row["instance"] for row in rows] != indices:
            raise ValueError(f"neural timing coverage mismatch for {seed}")
    if [row["instance"] for row in report["insertion"]["rows"]] != indices:
        raise ValueError("insertion timing coverage mismatch")

    neural_by_seed = {
        seed: [row["seconds"] for row in report["neural"][seed]["rows"]] for seed in seeds
    }
    neural = [
        sum(neural_by_seed[seed][position] for seed in seeds) / len(seeds)
        for position in range(len(indices))
    ]
    insertion = [row["seconds"] for row in report["insertion"]["rows"]]
    neural_short = [value + 0.5 for value in neural]
    insertion_short = [value + 0.5 for value in insertion]
    insertion_strong = [value + 10.0 for value in insertion]

    rng = random.Random(protocol["statistics"]["bootstrap_seed"])
    replicates = protocol["statistics"]["bootstrap_replicates"]
    ratios = {"insertion_over_neural": [], "insertion_short_over_neural_short": [], "strong_pipeline_over_neural_short": []}
    differences = []
    for _ in range(replicates):
        positions = [rng.randrange(len(indices)) for _ in indices]
        n_mean = sum(neural[i] for i in positions) / len(positions)
        i_mean = sum(insertion[i] for i in positions) / len(positions)
        ns_mean = sum(neural_short[i] for i in positions) / len(positions)
        is_mean = sum(insertion_short[i] for i in positions) / len(positions)
        strong_mean = sum(insertion_strong[i] for i in positions) / len(positions)
        differences.append(i_mean - n_mean)
        ratios["insertion_over_neural"].append(i_mean / n_mean)
        ratios["insertion_short_over_neural_short"].append(is_mean / ns_mean)
        ratios["strong_pipeline_over_neural_short"].append(strong_mean / ns_mean)

    result = {
        "status": "passed",
        "scope": "same-node latency only; full 100-instance reports remain the quality evidence",
        "environment": report["environment"],
        "selected_indices": indices,
        "methods": {
            "neural_by_seed": {seed: describe(values) for seed, values in neural_by_seed.items()},
            "neural_seed_averaged": describe(neural),
            "strict_insertion": describe(insertion),
            "neural_plus_0.5s_lns_derived": describe(neural_short),
            "insertion_plus_0.5s_lns_derived": describe(insertion_short),
            "insertion_plus_10s_lns_derived": describe(insertion_strong),
        },
        "paired_bootstrap": {
            "replicates": replicates,
            "seed": protocol["statistics"]["bootstrap_seed"],
            "insertion_minus_neural_mean_seconds_95pct_ci": [quantile(differences, 0.025), quantile(differences, 0.975)],
            **{
                f"{name}_ratio_of_means_95pct_ci": [quantile(values, 0.025), quantile(values, 0.975)]
                for name, values in ratios.items()
            },
        },
        "point_ratios": {
            "insertion_over_neural": (sum(insertion) / len(insertion)) / (sum(neural) / len(neural)),
            "insertion_short_over_neural_short": (sum(insertion_short) / len(insertion_short)) / (sum(neural_short) / len(neural_short)),
            "strong_pipeline_over_neural_short": (sum(insertion_strong) / len(insertion_strong)) / (sum(neural_short) / len(neural_short)),
        },
        "claim_boundary": protocol["claim_boundary"],
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
