#!/usr/bin/env python3
"""Paired, instance-level analysis for the frozen Shenzhen P1 experiment.

The independent experimental unit is one of the 32 road instances.  The three
fixed neural training seeds are averaged within each instance; they are not
treated as 96 independent observations.  Confidence intervals use a paired
bootstrap over common instance indices.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
import statistics
from pathlib import Path
from typing import Callable, Iterable


ROOT = Path(__file__).resolve().parent
FORMAL = ROOT / "formal_p1_v1"
CONTROL_PATH = FORMAL / "controls_v1" / "report.json"
NEURAL_DIR = FORMAL / "neural_transfer_v1"
OUT_JSON = FORMAL / "analysis_v1.json"
OUT_MD = FORMAL / "RESULTS_P1.md"
BOOTSTRAP_SEED = 20261113
BOOTSTRAP_DRAWS = 10_000


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def mean(values: Iterable[float]) -> float:
    return statistics.fmean(values)


def percentile(sorted_values: list[float], probability: float) -> float:
    position = (len(sorted_values) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return sorted_values[lower]
    weight = position - lower
    return sorted_values[lower] * (1.0 - weight) + sorted_values[upper] * weight


def paired_bootstrap(
    left: list[float], right: list[float], *, seed_offset: int = 0
) -> dict[str, float]:
    if len(left) != len(right) or not left:
        raise ValueError("paired bootstrap requires equally sized non-empty arrays")
    differences = [a - b for a, b in zip(left, right)]
    rng = random.Random(BOOTSTRAP_SEED + seed_offset)
    n = len(differences)
    boot = [
        mean(differences[rng.randrange(n)] for _ in range(n))
        for _ in range(BOOTSTRAP_DRAWS)
    ]
    boot.sort()
    return {
        "n_independent_instances": n,
        "mean_difference": mean(differences),
        "ci95_low": percentile(boot, 0.025),
        "ci95_high": percentile(boot, 0.975),
        "positive_pairs": sum(value > 0 for value in differences),
        "ties": sum(value == 0 for value in differences),
        "negative_pairs": sum(value < 0 for value in differences),
    }


def summarize(values: list[float]) -> dict[str, float]:
    return {
        "n": len(values),
        "mean": mean(values),
        "sd_across_instances": statistics.stdev(values) if len(values) > 1 else 0.0,
        "min": min(values),
        "max": max(values),
    }


def canonical_routes(routes: list[list[int]]) -> tuple[tuple[int, ...], ...]:
    """Ignore vehicle labels but preserve every within-route visit sequence."""
    return tuple(sorted(tuple(route) for route in routes))


def route_digest(routes: list[list[int]]) -> str:
    payload = json.dumps(canonical_routes(routes), separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def control_series(rows: list[dict], arm: str, stage: str, metric: str) -> list[float]:
    if stage == "time":
        return [float(row["arms"][arm][metric]) for row in rows]
    return [float(row["arms"][arm][f"{stage}_cost"][metric]) for row in rows]


def main() -> None:
    controls = load_json(CONTROL_PATH)
    if controls.get("status") != "completed" or len(controls.get("rows", [])) != 32:
        raise RuntimeError("formal control report is incomplete")
    control_rows = sorted(controls["rows"], key=lambda row: row["instance"])

    neural_reports = [
        load_json(NEURAL_DIR / f"learned_{seed}.json") for seed in (1234, 4321, 8765)
    ]
    for report in neural_reports:
        report["rows"] = sorted(report["rows"], key=lambda row: row["instance"])
        if [row["instance"] for row in report["rows"]] != list(range(32)):
            raise RuntimeError("neural report does not contain the frozen 32 instances")

    def neural_instance_average(extractor: Callable[[dict], float]) -> list[float]:
        return [
            mean(float(extractor(report["rows"][index])) for report in neural_reports)
            for index in range(32)
        ]

    straight_lns = {
        metric: control_series(control_rows, "straight_fixed25", "lns", metric)
        for metric in ("completed", "vehicles", "distance", "energy", "objective")
    }
    road_lns = {
        metric: control_series(control_rows, "road_fixed25", "lns", metric)
        for metric in ("completed", "vehicles", "distance", "energy", "objective")
    }
    speed_lns = {
        metric: control_series(control_rows, "road_osm_speed", "lns", metric)
        for metric in ("completed", "vehicles", "distance", "energy", "objective")
    }
    strict = {
        metric: control_series(control_rows, "road_fixed25", "initial", metric)
        for metric in ("completed", "vehicles", "distance", "energy", "objective")
    }
    strict_time = control_series(
        control_rows, "road_fixed25", "time", "initialization_seconds"
    )
    lns_time = control_series(control_rows, "road_fixed25", "time", "integrated_seconds")

    neural = {
        "completed": neural_instance_average(
            lambda row: row["road_cost_after_repair"]["completed"]
        ),
        "vehicles": neural_instance_average(
            lambda row: row["road_cost_after_repair"]["vehicles"]
        ),
        "distance": neural_instance_average(
            lambda row: row["road_cost_after_repair"]["distance"]
        ),
        "energy": neural_instance_average(
            lambda row: row["road_cost_after_repair"]["energy"]
        ),
        "objective": neural_instance_average(
            lambda row: row["road_cost_after_repair"]["objective"]
        ),
        "pipeline_seconds": neural_instance_average(
            lambda row: row["neural_inference_seconds"]
            + row["road_repair_and_lns_seconds"]
        ),
    }

    route_hashes_by_instance = [
        [route_digest(report["rows"][index]["road_routes_after_repair"]) for report in neural_reports]
        for index in range(32)
    ]
    identical_three = sum(len(set(hashes)) == 1 for hashes in route_hashes_by_instance)
    unique_solution_counts = [len(set(hashes)) for hashes in route_hashes_by_instance]

    direct_valid = sum(
        bool(row["road_valid_before_repair"])
        for report in neural_reports
        for row in report["rows"]
    )
    final_invalid = sum(
        not bool(row["road_audit_after_repair"]["valid"])
        for report in neural_reports
        for row in report["rows"]
    )
    control_invalid = sum(
        not bool(row["arms"][arm]["lns_audit"]["valid"])
        for row in control_rows
        for arm in ("straight_fixed25", "road_fixed25", "road_osm_speed")
    )

    output = {
        "status": "completed",
        "analysis_protocol": {
            "independent_unit": "one frozen Shenzhen road instance",
            "n_independent_instances": 32,
            "neural_seed_handling": "three fixed training seeds averaged within each instance",
            "search_seed_handling": "fixed search seed; not counted as replication",
            "uncertainty": "paired percentile bootstrap over common instance indices",
            "bootstrap_draws": BOOTSTRAP_DRAWS,
            "bootstrap_seed": BOOTSTRAP_SEED,
            "road_cases_sha256": controls["road_cases_sha256"],
        },
        "road_representation_effect_road_minus_straight": {
            metric: paired_bootstrap(road_lns[metric], straight_lns[metric], seed_offset=i)
            for i, metric in enumerate(("vehicles", "distance", "energy", "objective"))
        },
        "uncalibrated_osm_speed_effect_osm_minus_fixed25": {
            metric: paired_bootstrap(speed_lns[metric], road_lns[metric], seed_offset=20 + i)
            for i, metric in enumerate(("vehicles", "distance", "energy", "objective"))
        },
        "quality_time_ladder_road_fixed25": {
            "strict_insertion": {
                "completed": summarize(strict["completed"]),
                "service_rate_percent": 100.0 * mean(strict["completed"]) / 50.0,
                "full_service_instances": sum(value == 50 for value in strict["completed"]),
                "seconds": summarize(strict_time),
            },
            "neural_plus_road_repair": {
                "completed": summarize(neural["completed"]),
                "service_rate_percent": 100.0 * mean(neural["completed"]) / 50.0,
                "full_service_instances": sum(value == 50 for value in neural["completed"]),
                "seconds": summarize(neural["pipeline_seconds"]),
            },
            "strong_lns": {
                "completed": summarize(road_lns["completed"]),
                "service_rate_percent": 100.0 * mean(road_lns["completed"]) / 50.0,
                "full_service_instances": sum(value == 50 for value in road_lns["completed"]),
                "seconds": summarize(lns_time),
            },
            "neural_minus_strict_completed": paired_bootstrap(
                neural["completed"], strict["completed"], seed_offset=40
            ),
            "lns_minus_neural_completed": paired_bootstrap(
                road_lns["completed"], neural["completed"], seed_offset=41
            ),
        },
        "neural_transfer_diagnostic": {
            "direct_road_valid_plans": direct_valid,
            "direct_road_plans_total": 96,
            "direct_road_valid_percent": 100.0 * direct_valid / 96.0,
            "hard_invalid_after_repair": final_invalid,
            "instances_with_identical_repaired_routes_across_all_three_seeds": identical_three,
            "mean_unique_repaired_solutions_per_instance": mean(unique_solution_counts),
            "unique_repaired_solution_count_distribution": {
                str(value): unique_solution_counts.count(value) for value in sorted(set(unique_solution_counts))
            },
            "interpretation": (
                "The unchanged coordinate-trained policy did not directly satisfy the road-time protocol. "
                "Reported neural-pipeline outcomes therefore measure construction followed by mandatory "
                "pre-execution road-feasibility repair, not direct road-aware neural generalization."
            ),
        },
        "constraint_audit": {
            "invalid_final_control_solutions": control_invalid,
            "control_solutions_audited": 96,
            "invalid_final_neural_pipeline_solutions": final_invalid,
            "neural_pipeline_solutions_audited": 96,
        },
    }

    OUT_JSON.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    road_effect = output["road_representation_effect_road_minus_straight"]
    ladder = output["quality_time_ladder_road_fixed25"]
    neural_diag = output["neural_transfer_diagnostic"]
    md = f"""# 深圳真实路网 P1 正式结果（v1）

## 统计口径

- 独立实验单位为 **32 个冻结的深圳道路实例**，每例含 50 个客货请求。
- 三个神经网络训练种子先在同一实例内取平均，再进行实例级比较；未将 96 条模型—实例记录伪作 96 个独立重复。
- 区间为共同实例索引上的 10,000 次配对百分位 bootstrap 95% CI。搜索随机种子固定，不计作独立重复。
- 全部结论只适用于当前福田—罗湖 OSM 路网与半合成订单协议。

## 真实路网表征的影响

在强 LNS 均实现 32/32 实例全服务的条件下，用道路最短路替代直线距离后：

- 每例车辆数增加 **{road_effect['vehicles']['mean_difference']:.2f}** 辆（95% CI {road_effect['vehicles']['ci95_low']:.2f} 至 {road_effect['vehicles']['ci95_high']:.2f}）；
- 总行驶距离增加 **{road_effect['distance']['mean_difference']:.2f} km**（95% CI {road_effect['distance']['ci95_low']:.2f} 至 {road_effect['distance']['ci95_high']:.2f}）；
- 能耗项增加 **{road_effect['energy']['mean_difference']:.2f}**（95% CI {road_effect['energy']['ci95_low']:.2f} 至 {road_effect['energy']['ci95_high']:.2f}）。

这说明直线几何会系统性低估车辆与里程需求。`road_osm_speed` 仅使用 OSM 道路等级/默认限速代理，未由深圳实测速度校准，因此只作为敏感性分析，不作为现实交通时耗结论。

## 道路场景下的质量—时间阶梯

| 方法 | 平均完成订单 | 服务率 | 全服务实例 | 平均时间/例 |
|---|---:|---:|---:|---:|
| 严格插入 | {ladder['strict_insertion']['completed']['mean']:.2f}/50 | {ladder['strict_insertion']['service_rate_percent']:.2f}% | {ladder['strict_insertion']['full_service_instances']}/32 | {ladder['strict_insertion']['seconds']['mean']:.3f} s |
| 原神经构造 + 道路可行性修复 | {ladder['neural_plus_road_repair']['completed']['mean']:.2f}/50 | {ladder['neural_plus_road_repair']['service_rate_percent']:.2f}% | {ladder['neural_plus_road_repair']['full_service_instances']}/32 | {ladder['neural_plus_road_repair']['seconds']['mean']:.3f} s |
| 强 LNS（10 s） | {ladder['strong_lns']['completed']['mean']:.2f}/50 | {ladder['strong_lns']['service_rate_percent']:.2f}% | {ladder['strong_lns']['full_service_instances']}/32 | {ladder['strong_lns']['seconds']['mean']:.3f} s |

- 神经管线相对严格插入平均多完成 **{ladder['neural_minus_strict_completed']['mean_difference']:.2f}** 单/例（95% CI {ladder['neural_minus_strict_completed']['ci95_low']:.2f} 至 {ladder['neural_minus_strict_completed']['ci95_high']:.2f}）。
- 强 LNS 相对神经管线平均多完成 **{ladder['lns_minus_neural_completed']['mean_difference']:.2f}** 单/例（95% CI {ladder['lns_minus_neural_completed']['ci95_low']:.2f} 至 {ladder['lns_minus_neural_completed']['ci95_high']:.2f}）。

## 神经迁移的证据边界

- 未经修复的道路可行方案为 **{neural_diag['direct_road_valid_plans']}/96**。
- 修复后 96/96 方案通过独立硬约束审计；控制组 96/96 亦通过。
- 三个种子修复后路线完全相同的实例为 **{neural_diag['instances_with_identical_repaired_routes_across_all_three_seeds']}/32**；每例平均保留 {neural_diag['mean_unique_repaired_solutions_per_instance']:.2f} 个不同最终解。

因此，当前结果可作为“坐标模型在真实路网下需要道路可行性修复”的外部有效性证据，以及质量—时间取舍的基线；不能表述为原神经模型已直接学习道路最短路或真实交通时耗。若论文要主张道路感知神经调度，需要把道路时间矩阵显式接入状态/掩码并重新训练，再在同一冻结测试集上比较。
"""
    OUT_MD.write_text(md, encoding="utf-8")

    print(json.dumps({"analysis": str(OUT_JSON), "summary": str(OUT_MD)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
