"""Fail-closed paired analysis of current N100 models versus insertion/SCIP."""

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
SCIP_ROOT = ROOT / PLAN["comparison_source"]
OUTPUT = EXPERIMENT / "PAIRED_ANALYSIS.json"
BOOTSTRAP_REPLICATES = 20_000
BOOTSTRAP_SEED = 20261116


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def percentile_ci(values: list[float], seed: int) -> list[float]:
    rng = random.Random(seed)
    n = len(values)
    means = sorted(
        sum(values[rng.randrange(n)] for _ in range(n)) / n
        for _ in range(BOOTSTRAP_REPLICATES)
    )
    return [means[500], means[19_500]]


def report_for_seed(seed: int) -> dict:
    candidates = [
        EXPERIMENT / f"results/seed_{seed}/report.json",
        EXPERIMENT / f"results/seed_{seed}_serial_retry1/report.json",
    ]
    passed = []
    for path in candidates:
        if not path.exists():
            continue
        report = json.loads(path.read_text(encoding="utf-8"))
        if report.get("status") == "passed":
            passed.append((path, report))
    if len(passed) != 1:
        raise RuntimeError(
            f"Seed {seed} requires exactly one passed report, found {len(passed)}"
        )
    path, report = passed[0]
    checkpoint = next(item for item in PLAN["checkpoints"] if item["seed"] == seed)
    if report["checkpoint_sha256"] != checkpoint["sha256"]:
        raise RuntimeError(f"Checkpoint hash mismatch for seed {seed}")
    if (
        report["dataset_seed"] != PLAN["dataset_seed"]
        or report["graph_size"] != PLAN["graph_size"]
        or report["samples"] != PLAN["samples"]
        or len(report["rows"]) != PLAN["samples"]
    ):
        raise RuntimeError(f"Protocol mismatch for seed {seed}")
    summary = report["summary"]
    if (
        summary["hard_violations"]
        or summary["unfulfilled"]
        or not summary["all_trips_at_least_four"]
        or not summary["all_objectives_finite"]
    ):
        raise RuntimeError(f"Hard audit failed for seed {seed}")
    return {"path": str(path.relative_to(ROOT)), "report": report}


def load_scip() -> dict[int, dict]:
    rows: dict[int, dict] = {}
    for instance in range(PLAN["samples"]):
        path = SCIP_ROOT / f"scip_n100_{instance:02d}/report.json"
        report = json.loads(path.read_text(encoding="utf-8"))
        if (
            report.get("status") != "completed"
            or report.get("scenario_sha256") != PLAN["scenario_sha256"]
            or report.get("graph_size") != PLAN["graph_size"]
            or len(report.get("rows", [])) != 1
        ):
            raise RuntimeError(f"Invalid SCIP report for instance {instance}")
        row = report["rows"][0]
        if row["instance"] != instance:
            raise RuntimeError(f"SCIP instance index mismatch at {instance}")
        for label in ("initial", "final"):
            audited = row[label]
            if audited["audit"]["violations"] or audited["audit"]["unfulfilled"]:
                raise RuntimeError(f"Hard-invalid {label} plan at instance {instance}")
            if any(completed < 4 for completed in audited["completed_per_trip"]):
                raise RuntimeError(f"Underfilled {label} trip at instance {instance}")
        if not math.isfinite(row["final"]["cost"]["objective"]):
            raise RuntimeError(f"Non-finite SCIP objective at instance {instance}")
        rows[instance] = row
    return rows


def summarize(costs: list[dict]) -> dict:
    return {
        "service_rate": statistics.fmean(cost["completed"] / PLAN["graph_size"] for cost in costs),
        "objective": statistics.fmean(cost["objective"] for cost in costs),
        "raw_cost": statistics.fmean(cost["raw_cost"] for cost in costs),
        "distance": statistics.fmean(cost["distance"] for cost in costs),
        "vehicles": statistics.fmean(cost["vehicles"] for cost in costs),
    }


def main() -> None:
    equivalence = json.loads(
        (EXPERIMENT / "DATASET_EQUIVALENCE.json").read_text(encoding="utf-8")
    )
    if equivalence.get("status") != "passed":
        raise RuntimeError("Dataset equivalence gate did not pass")
    neural = {seed: report_for_seed(seed) for seed in PLAN["training_seeds"]}
    scip = load_scip()

    seed_metrics = {}
    for seed, item in neural.items():
        rows = sorted(item["report"]["rows"], key=lambda row: row["instance"])
        if [row["instance"] for row in rows] != list(range(PLAN["samples"])):
            raise RuntimeError(f"Missing or duplicate instances for seed {seed}")
        seed_metrics[str(seed)] = {
            **summarize([row["cost"] for row in rows]),
            "mean_inference_seconds": statistics.fmean(row["inference_seconds"] for row in rows),
            "report": item["path"],
        }

    per_instance = []
    for instance in range(PLAN["samples"]):
        neural_costs = [
            neural[seed]["report"]["rows"][instance]["cost"]
            for seed in PLAN["training_seeds"]
        ]
        neural_mean = {
            key: statistics.fmean(cost[key] for cost in neural_costs)
            for key in ("completed", "objective", "raw_cost", "distance", "vehicles")
        }
        insertion = scip[instance]["initial"]["cost"]
        incumbent = scip[instance]["final"]["cost"]
        per_instance.append(
            {
                "instance": instance,
                "neural_seed_mean": neural_mean,
                "insertion": {key: insertion[key] for key in neural_mean},
                "scip_incumbent": {key: incumbent[key] for key in neural_mean},
                "scip_status": scip[instance]["result"]["status"],
                "scip_gap": scip[instance]["result"]["gap"],
                "scip_improvement": scip[instance]["objective_improvement"],
            }
        )

    neural_costs = [row["neural_seed_mean"] for row in per_instance]
    insertion_costs = [row["insertion"] for row in per_instance]
    scip_costs = [row["scip_incumbent"] for row in per_instance]
    service_diff = [
        (row["neural_seed_mean"]["completed"] - row["insertion"]["completed"])
        / PLAN["graph_size"]
        for row in per_instance
    ]
    objective_diff = [
        row["neural_seed_mean"]["objective"] - row["insertion"]["objective"]
        for row in per_instance
    ]
    report = {
        "status": "passed",
        "scope": PLAN["scope"],
        "claim_boundary": PLAN["claim_boundary"],
        "experimental_unit": "paired N100 instance; three training seeds averaged within each instance",
        "samples": PLAN["samples"],
        "training_seeds": PLAN["training_seeds"],
        "dataset_equivalence": equivalence,
        "seed_metrics": seed_metrics,
        "methods": {
            "current_neural_seed_averaged": summarize(neural_costs),
            "strict_insertion": summarize(insertion_costs),
            "scip_incumbent": summarize(scip_costs),
        },
        "paired_neural_minus_insertion": {
            "service_rate_difference": statistics.fmean(service_diff),
            "service_rate_difference_95pct_ci": percentile_ci(service_diff, BOOTSTRAP_SEED),
            "objective_difference": statistics.fmean(objective_diff),
            "objective_difference_95pct_ci": percentile_ci(objective_diff, BOOTSTRAP_SEED + 1),
            "neural_better_objective_instances": sum(value < 0 for value in objective_diff),
            "ties": sum(abs(value) < 1e-9 for value in objective_diff),
            "insertion_better_objective_instances": sum(value > 0 for value in objective_diff),
        },
        "scip_solver": {
            "time_limit_instances": sum(row["scip_status"] == "timelimit" for row in per_instance),
            "optimal_instances": sum(row["scip_status"] == "optimal" for row in per_instance),
            "improved_warm_start_instances": sum(row["scip_improvement"] > 1e-9 for row in per_instance),
            "median_gap": statistics.median(row["scip_gap"] for row in per_instance),
        },
        "bootstrap": {
            "replicates": BOOTSTRAP_REPLICATES,
            "seed": BOOTSTRAP_SEED,
            "method": "paired nonparametric percentile bootstrap over instances",
        },
        "per_instance": per_instance,
    }
    OUTPUT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({key: report[key] for key in ("status", "methods", "paired_neural_minus_insertion", "scip_solver")}, indent=2))


if __name__ == "__main__":
    main()
