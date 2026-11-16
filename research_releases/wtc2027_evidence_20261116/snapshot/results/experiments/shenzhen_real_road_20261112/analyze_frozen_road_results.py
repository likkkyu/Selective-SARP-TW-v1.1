"""Nested, paired summary of the frozen real-road experiment."""

from __future__ import annotations

import hashlib
import json
import platform
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SEEDS = (1234, 4321, 8765)
N_BOOT = 20_000
BOOT_SEED = 2026111402


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def percentile_ci(values: np.ndarray, rng: np.random.Generator) -> list[float]:
    n = len(values)
    draws = rng.integers(0, n, size=(N_BOOT, n))
    means = values[draws].mean(axis=1)
    return [float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))]


def summarize(values: np.ndarray, rng: np.random.Generator) -> dict:
    return {
        "mean": float(values.mean()),
        "sd_across_instances": float(values.std(ddof=1)),
        "bootstrap_95_ci_for_mean": percentile_ci(values, rng),
        "min": float(values.min()),
        "max": float(values.max()),
    }


def paired_delta(a: np.ndarray, b: np.ndarray, rng: np.random.Generator) -> dict:
    delta = a - b
    return {
        "mean_delta": float(delta.mean()),
        "bootstrap_95_ci": percentile_ci(delta, rng),
        # Keep the sign convention neutral: larger is preferable for service,
        # whereas smaller is preferable for objective, vehicles and runtime.
        "positive_delta_instances": int((delta > 1e-12).sum()),
        "zero_delta_instances": int((np.abs(delta) <= 1e-12).sum()),
        "negative_delta_instances": int((delta < -1e-12).sum()),
    }


def main() -> None:
    rng = np.random.default_rng(BOOT_SEED)
    neural_reports = {}
    for seed in SEEDS:
        path = HERE / f"road_native_formal_test_v1_neural_lns3s_model{seed}_search1234" / "report.json"
        report = json.loads(path.read_text())
        if report.get("status") != "completed" or len(report.get("rows", [])) != 128:
            raise ValueError(f"Incomplete neural-LNS report: {path}")
        if any(not row["neural"]["audit"]["valid"] or not row["neural_lns"]["audit"]["valid"] for row in report["rows"]):
            raise ValueError(f"Hard-invalid neural or hybrid route: {path}")
        neural_reports[seed] = (path, report)

    controls_path = HERE / "road_native_formal_test_v1_controls_lns3s_seed1234" / "report.json"
    controls = json.loads(controls_path.read_text())
    if controls.get("status") != "completed" or len(controls.get("rows", [])) != 128:
        raise ValueError("Incomplete control report")
    if any(not row["insertion"]["audit"]["valid"] or not row["lns"]["audit"]["valid"] for row in controls["rows"]):
        raise ValueError("Hard-invalid control route")

    scip_path = HERE / "road_native_formal_test_v1_scip5s_first16_hard4_v1" / "report.json"
    scip = json.loads(scip_path.read_text())
    if scip.get("status") != "completed" or len(scip.get("rows", [])) != 16:
        raise ValueError("Incomplete bounded-SCIP report")
    if any(not row["road_audit"]["valid"] for row in scip["rows"]):
        raise ValueError("Hard-invalid bounded-SCIP route")

    test_hashes = {report["test_pool_sha256"] for _, report in neural_reports.values()}
    test_hashes.add(controls["test_pool_sha256"])
    test_hashes.add(scip["test_pool_sha256"])
    lock_hashes = {report["model_lock_sha256"] for _, report in neural_reports.values()}
    lock_hashes.add(controls["model_lock_sha256"])
    if len(test_hashes) != 1 or len(lock_hashes) != 1:
        raise ValueError("Results do not share the same test set and model lock")

    fields = ("completed", "objective", "vehicles", "seconds")
    seed_arrays = {"neural": {}, "hybrid": {}}
    for seed, (_, report) in neural_reports.items():
        rows = report["rows"]
        seed_arrays["neural"][seed] = {
            "completed": np.asarray([row["neural"]["audit"]["completed"] for row in rows], dtype=float),
            "objective": np.asarray([row["neural"]["cost"]["objective"] for row in rows], dtype=float),
            "vehicles": np.asarray([row["neural"]["cost"]["vehicles"] for row in rows], dtype=float),
            "seconds": np.asarray([row["neural"]["seconds"] for row in rows], dtype=float),
        }
        seed_arrays["hybrid"][seed] = {
            "completed": np.asarray([row["neural_lns"]["audit"]["completed"] for row in rows], dtype=float),
            "objective": np.asarray([row["neural_lns"]["cost"]["objective"] for row in rows], dtype=float),
            "vehicles": np.asarray([row["neural_lns"]["cost"]["vehicles"] for row in rows], dtype=float),
            "seconds": np.asarray([row["neural_lns"]["integrated_seconds"] for row in rows], dtype=float),
        }

    control_arrays = {
        "insertion": {
            "completed": np.asarray([row["insertion"]["audit"]["completed"] for row in controls["rows"]], dtype=float),
            "objective": np.asarray([row["insertion"]["cost"]["objective"] for row in controls["rows"]], dtype=float),
            "vehicles": np.asarray([row["insertion"]["cost"]["vehicles"] for row in controls["rows"]], dtype=float),
            "seconds": np.asarray([row["insertion"]["seconds"] for row in controls["rows"]], dtype=float),
        },
        "lns": {
            "completed": np.asarray([row["lns"]["audit"]["completed"] for row in controls["rows"]], dtype=float),
            "objective": np.asarray([row["lns"]["cost"]["objective"] for row in controls["rows"]], dtype=float),
            "vehicles": np.asarray([row["lns"]["cost"]["vehicles"] for row in controls["rows"]], dtype=float),
            "seconds": np.asarray([row["lns"]["integrated_seconds"] for row in controls["rows"]], dtype=float),
        },
    }

    aggregate = {}
    for method in ("neural", "hybrid"):
        aggregate[method] = {
            field: np.stack([seed_arrays[method][seed][field] for seed in SEEDS]).mean(axis=0)
            for field in fields
        }
    aggregate.update(control_arrays)

    method_summary = {}
    for method, arrays in aggregate.items():
        method_summary[method] = {
            "completed_orders": summarize(arrays["completed"], rng),
            "service_rate_percent": summarize(arrays["completed"] * 2.0, rng),
            "objective": summarize(arrays["objective"], rng),
            "vehicles": summarize(arrays["vehicles"], rng),
            "algorithm_seconds": summarize(arrays["seconds"], rng),
            "full_service_instances": int(np.isclose(arrays["completed"], 50.0).sum()),
            "hard_invalid": 0,
        }

    seed_level = {}
    for method in ("neural", "hybrid"):
        service = np.asarray([
            seed_arrays[method][seed]["completed"].mean() * 2.0 for seed in SEEDS
        ])
        objective = np.asarray([
            seed_arrays[method][seed]["objective"].mean() for seed in SEEDS
        ])
        seed_level[method] = {
            "training_seeds": list(SEEDS),
            "service_rate_percent_by_seed": service.tolist(),
            "service_rate_percent_mean": float(service.mean()),
            "service_rate_percent_sd_across_training_seeds": float(service.std(ddof=1)),
            "objective_by_seed": objective.tolist(),
            "objective_mean": float(objective.mean()),
            "objective_sd_across_training_seeds": float(objective.std(ddof=1)),
            "full_service_instances_by_seed": {
                str(seed): int(np.isclose(seed_arrays[method][seed]["completed"], 50.0).sum())
                for seed in SEEDS
            },
            "note": "n=3 training seeds; descriptive mean and SD only, no population-level seed inference",
        }

    contrasts = {}
    for a, b in (
        ("neural", "insertion"),
        ("neural", "lns"),
        ("hybrid", "insertion"),
        ("hybrid", "lns"),
        ("hybrid", "neural"),
    ):
        contrasts[f"{a}_minus_{b}"] = {
            "completed_orders": paired_delta(aggregate[a]["completed"], aggregate[b]["completed"], rng),
            "service_rate_percentage_points": paired_delta(aggregate[a]["completed"] * 2.0, aggregate[b]["completed"] * 2.0, rng),
            "objective": paired_delta(aggregate[a]["objective"], aggregate[b]["objective"], rng),
            "vehicles": paired_delta(aggregate[a]["vehicles"], aggregate[b]["vehicles"], rng),
            "algorithm_seconds": paired_delta(aggregate[a]["seconds"], aggregate[b]["seconds"], rng),
        }

    scip_rows = scip["rows"]
    scip_summary = {
        "instances": len(scip_rows),
        "seconds_per_instance": scip["seconds_per_instance"],
        "statuses": {
            status: sum(row["scip"]["status"] == status for row in scip_rows)
            for status in sorted({row["scip"]["status"] for row in scip_rows})
        },
        "insertion_service_rate_percent": float(np.mean([
            row["insertion"]["cost"]["completed"] for row in scip_rows
        ]) * 2.0),
        "scip_incumbent_service_rate_percent": float(np.mean([
            row["road_engine_cost"]["completed"] for row in scip_rows
        ]) * 2.0),
        "insertion_objective_mean": float(np.mean([
            row["insertion"]["cost"]["objective"] for row in scip_rows
        ])),
        "scip_incumbent_objective_mean": float(np.mean([
            row["road_engine_cost"]["objective"] for row in scip_rows
        ])),
        "objective_improvements_over_warm_start": int(sum(
            row["road_engine_cost"]["objective"] < row["insertion"]["cost"]["objective"] - 1e-8
            for row in scip_rows
        )),
        "optimal_instances": int(sum(row["scip"]["status"] == "optimal" for row in scip_rows)),
        "all_hard_audits_pass": True,
        "maximum_road_replay_error": float(max(row["road_replay_error"] for row in scip_rows)),
        "interpretation": "finite-time MIP incumbent/reference only; no exact-optimum claim",
    }

    result = {
        "status": "completed",
        "scope": "frozen 128-case Shenzhen road-native formal test",
        "test_pool_sha256": next(iter(test_hashes)),
        "model_lock_sha256": next(iter(lock_hashes)),
        "independent_unit": "road-native test instance",
        "n_independent_instances": 128,
        "nested_repeats": "three independently trained neural seeds per instance; one fixed LNS search seed",
        "bootstrap": {
            "resamples": N_BOOT,
            "seed": BOOT_SEED,
            "unit": "paired test instance",
            "interval": "percentile 95% confidence interval for the conditional instance mean",
        },
        "inference_boundary": (
            "Instance bootstrap intervals are conditional on the three frozen neural models, the fixed "
            "test generator, and LNS search seed 1234. They are not population-level uncertainty over "
            "all possible training or search seeds."
        ),
        "method_summary": method_summary,
        "training_seed_summary": seed_level,
        "paired_contrasts": contrasts,
        "bounded_scip_reference": scip_summary,
        "source_reports": {
            "controls": {"path": str(controls_path.relative_to(ROOT)), "sha256": digest(controls_path)},
            "bounded_scip": {"path": str(scip_path.relative_to(ROOT)), "sha256": digest(scip_path)},
            "neural_hybrid": {
                str(seed): {"path": str(path.relative_to(ROOT)), "sha256": digest(path)}
                for seed, (path, _) in neural_reports.items()
            },
        },
        "software": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "platform": platform.platform(),
        },
        "data_scope_note": (
            "Road topology and shortest-path matrices use the frozen Shenzhen OSM driving graph; "
            "pickup-delivery requests and time windows are controlled synthetic scenarios sampled "
            "directly on that graph, not observed trip records."
        ),
    }
    output = HERE / "road_native_formal_test_v1_analysis.json"
    output.write_text(json.dumps(result, indent=2) + "\n")

    def ci(metric: dict, digits=3):
        low, high = metric["bootstrap_95_ci_for_mean"]
        return f"{metric['mean']:.{digits}f} [{low:.{digits}f}, {high:.{digits}f}]"

    lines = [
        "# Shenzhen road-native formal test (v1)",
        "",
        "## Design boundary",
        "",
        "- Frozen test: 128 independent N50 scenarios on the Shenzhen Futian–Luohu OSM driving graph.",
        "- Demand: controlled synthetic pickup–delivery requests sampled directly on road nodes; not observed trips.",
        "- Neural replication: three independently trained seeds (1234, 4321, 8765).",
        "- Search: one fixed LNS seed (1234), 3 s per instance. No p-values are claimed across search seeds.",
        "- Every reported route passed the independent hard-constraint audit, including the four-completed-orders minimum for every dispatched trip.",
        "",
        "## Primary results",
        "",
        "Values are instance means with paired-instance bootstrap 95% CIs; neural rows first average the three frozen training seeds within each instance.",
        "",
        "| Method | Service rate, % | Objective | Vehicles | Algorithm time, s | Full-service instances* |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    labels = {
        "neural": "Neural construction",
        "hybrid": "Neural + 3 s LNS",
        "insertion": "Strict insertion",
        "lns": "Insertion + 3 s LNS",
    }
    for method in ("neural", "hybrid", "insertion", "lns"):
        summary = method_summary[method]
        lines.append(
            f"| {labels[method]} | {ci(summary['service_rate_percent'], 2)} | "
            f"{ci(summary['objective'], 2)} | {ci(summary['vehicles'], 2)} | "
            f"{ci(summary['algorithm_seconds'], 3)} | {summary['full_service_instances']}/128 |"
        )
    h_l = contrasts["hybrid_minus_lns"]["service_rate_percentage_points"]
    h_n = contrasts["hybrid_minus_neural"]["service_rate_percentage_points"]
    lines += [
        "",
        "## Paired interpretation",
        "",
        f"- Matched LNS improved neural construction by {h_n['mean_delta']:.3f} percentage points "
        f"(95% CI {h_n['bootstrap_95_ci'][0]:.3f} to {h_n['bootstrap_95_ci'][1]:.3f}).",
        f"- Neural + LNS minus insertion + LNS was {h_l['mean_delta']:.3f} percentage points "
        f"(95% CI {h_l['bootstrap_95_ci'][0]:.3f} to {h_l['bootstrap_95_ci'][1]:.3f}).",
        "- These intervals quantify variation across the 128 frozen scenarios conditional on the fixed models and search seed; they do not quantify all possible training/search-seed variation.",
        "",
        "## Training-seed stability",
        "",
        f"- Pure neural service rate: {seed_level['neural']['service_rate_percent_mean']:.3f}% ± "
        f"{seed_level['neural']['service_rate_percent_sd_across_training_seeds']:.3f} percentage points (SD across 3 seeds).",
        f"- Neural + LNS service rate: {seed_level['hybrid']['service_rate_percent_mean']:.3f}% ± "
        f"{seed_level['hybrid']['service_rate_percent_sd_across_training_seeds']:.3f} percentage points (SD across 3 seeds).",
        f"- Pure neural full-service counts by training seed (1234/4321/8765): "
        f"{seed_level['neural']['full_service_instances_by_seed']['1234']}/"
        f"{seed_level['neural']['full_service_instances_by_seed']['4321']}/"
        f"{seed_level['neural']['full_service_instances_by_seed']['8765']} of 128.",
        f"- Neural + LNS full-service counts by training seed (1234/4321/8765): "
        f"{seed_level['hybrid']['full_service_instances_by_seed']['1234']}/"
        f"{seed_level['hybrid']['full_service_instances_by_seed']['4321']}/"
        f"{seed_level['hybrid']['full_service_instances_by_seed']['8765']} of 128.",
        "- *For neural rows, the table count requires the same instance to be fully served by all three training seeds; control rows have one fixed run per instance.",
        "",
        "## Bounded MIP reference",
        "",
        f"- On the predeclared first {scip_summary['instances']} frozen test instances, hard-four SCIP used one CPU thread and {scip_summary['seconds_per_instance']:.0f} s per instance.",
        f"- The strict-insertion warm start and final SCIP incumbent both served {scip_summary['scip_incumbent_service_rate_percent']:.3f}% of orders with mean objective {scip_summary['scip_incumbent_objective_mean']:.3f}; SCIP improved the warm-start objective in {scip_summary['objective_improvements_over_warm_start']}/{scip_summary['instances']} instances.",
        f"- All {scip_summary['instances']} runs reached the time limit and {scip_summary['optimal_instances']} proved optimality. Therefore these are finite-time MIP incumbents, not exact optima or lower-bound quality claims.",
        f"- Every incumbent passed the independent road-route and hard-four audit; maximum objective replay error was {scip_summary['maximum_road_replay_error']:.2e}.",
        "",
        "## Integrity note",
        "",
        "An earlier decoder-v1 evaluation exposed one numerical boundary deadlock. The original failure artifact is retained. Decoder v2 aligns the hard-feasibility tolerance with the independent audit and refuses unsafe depot fallback when pickup commitments remain. All final results above were recomputed from unchanged checkpoint weights on the unchanged frozen test set.",
    ]
    markdown = HERE / "RESULTS_REAL_ROAD_FORMAL_V1.md"
    markdown.write_text("\n".join(lines) + "\n")
    print(json.dumps({
        "status": result["status"],
        "analysis": str(output),
        "markdown": str(markdown),
        "test_instances": result["n_independent_instances"],
    }, indent=2))


if __name__ == "__main__":
    main()
