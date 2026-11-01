"""
局部双单拼车机会密度分析脚本
Rideshare opportunity density analysis under the current stable hard constraints.

This script is intentionally analysis-only:
- it does NOT modify the current stable training/state semantics;
- it does NOT use model checkpoints;
- it measures structural two-order opportunities in the generated data distribution.

Key definitions:
- consolidation opportunity:
    at least one precedence-valid two-order single-trip interleaving is feasible
- overlap rideshare opportunity:
    at least one feasible interleaving picks up the second order before the first delivery
- distance-saving overlap opportunity:
    overlap-feasible and shorter than serving the two orders separately
"""

import argparse
import json
import math
import random
from collections import Counter, defaultdict
from itertools import combinations

import torch

from baseline_utils import evaluate_node_routes
from problem_mcvrptw_v2 import Config, MCVRPPDTWDataset


INTERLEAVINGS = [
    ("serial_ab", (("P", 0), ("D", 0), ("P", 1), ("D", 1)), False),
    ("overlap_ab_ab", (("P", 0), ("P", 1), ("D", 0), ("D", 1)), True),
    ("overlap_ab_ba", (("P", 0), ("P", 1), ("D", 1), ("D", 0)), True),
    ("serial_ba", (("P", 1), ("D", 1), ("P", 0), ("D", 0)), False),
    ("overlap_ba_ba", (("P", 1), ("P", 0), ("D", 1), ("D", 0)), True),
    ("overlap_ba_ab", (("P", 1), ("P", 0), ("D", 0), ("D", 1)), True),
]
BLOCKER_PRIORITY = ["capacity", "pickup_tw", "ride_time", "trip_time", "operation_end"]
PICKUP_DISTANCE_BUCKETS_KM = [0.0, 2.0, 4.0, 6.0, 8.0, math.inf]
PICKUP_TW_OVERLAP_BUCKETS_MIN = [0.0, 15.0, 30.0, math.inf]
DIRECT_PD_TIME_BUCKETS_MIN = [0.0, 15.0, 30.0, 45.0, math.inf]
EPS = 1e-5


def parse_args():
    parser = argparse.ArgumentParser(description="局部双单拼车机会密度分析")
    parser.add_argument("--graph-size", type=int, default=25, help="订单数")
    parser.add_argument("--num-samples", type=int, default=64, help="样本数")
    parser.add_argument("--seed", type=int, default=12345, help="随机种子")
    parser.add_argument(
        "--max-pairs-per-instance",
        type=int,
        default=None,
        help="每个样本最多分析多少个订单对；默认分析全部",
    )
    parser.add_argument(
        "--cross-check-routes",
        type=int,
        default=20,
        help="抽样多少条可行共享路线用 get_costs 做交叉验证",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default=None,
        help="可选：将汇总结果写入 JSON 文件",
    )
    parser.add_argument(
        "--self-check",
        action="store_true",
        help="仅运行内置语义自检，不执行正式分析",
    )
    return parser.parse_args()


def _bucket_label(value, edges, unit):
    for low, high in zip(edges[:-1], edges[1:]):
        if value < high - EPS:
            if math.isinf(high):
                return f">={low:.0f}{unit}"
            return f"[{low:.0f},{high:.0f}){unit}"
    low = edges[-2]
    return f">={low:.0f}{unit}"


def _safe_rate(num, denom):
    return float(num) / float(denom) if denom else 0.0


def _to_float_pair(coord):
    return float(coord[0].item()), float(coord[1].item())


def _euclidean_km(coord_a, coord_b):
    dx = coord_a[0] - coord_b[0]
    dy = coord_a[1] - coord_b[1]
    return math.sqrt(dx * dx + dy * dy) * Config.AREA_SIZE


def _travel_time_hours(coord_a, coord_b):
    return _euclidean_km(coord_a, coord_b) / Config.VEHICLE_SPEED


def _order_meta(sample, order_idx):
    n_orders = int(sample["n_orders"])
    pickup_coord = _to_float_pair(sample["loc"][order_idx])
    delivery_coord = _to_float_pair(sample["loc"][order_idx + n_orders])
    pickup_tw = tuple(float(x.item()) for x in sample["time_windows"][order_idx])
    delivery_tw = tuple(float(x.item()) for x in sample["time_windows"][order_idx + n_orders])
    type_flag = int(sample["node_type"][order_idx].item())
    passenger_demand = float(sample["demand_passenger"][order_idx].item())
    cargo_demand = float(sample["demand_cargo"][order_idx].item())
    return {
        "order_idx": int(order_idx),
        "pickup_node": int(order_idx + 1),
        "delivery_node": int(order_idx + n_orders + 1),
        "type": "passenger" if type_flag == 1 else "cargo",
        "pickup_coord": pickup_coord,
        "delivery_coord": delivery_coord,
        "pickup_tw": pickup_tw,
        "delivery_tw": delivery_tw,
        "pickup_demand_p": passenger_demand,
        "pickup_demand_c": cargo_demand,
        "delivery_demand_p": -passenger_demand,
        "delivery_demand_c": -cargo_demand,
        "direct_distance_km": _euclidean_km(pickup_coord, delivery_coord),
        "direct_travel_time_h": _travel_time_hours(pickup_coord, delivery_coord),
    }


def _pair_type(meta_a, meta_b):
    kinds = sorted([meta_a["type"], meta_b["type"]])
    if kinds == ["cargo", "cargo"]:
        return "cc"
    if kinds == ["cargo", "passenger"]:
        return "pc"
    return "pp"


def _pair_features(meta_a, meta_b):
    pickup_overlap_h = max(
        0.0,
        min(meta_a["pickup_tw"][1], meta_b["pickup_tw"][1]) - max(meta_a["pickup_tw"][0], meta_b["pickup_tw"][0]),
    )
    direct_pd_max_min = max(meta_a["direct_travel_time_h"], meta_b["direct_travel_time_h"]) * 60.0
    return {
        "pickup_pickup_distance_km": _euclidean_km(meta_a["pickup_coord"], meta_b["pickup_coord"]),
        "pickup_tw_overlap_min": pickup_overlap_h * 60.0,
        "pickup_tw_center_gap_min": abs(
            ((meta_a["pickup_tw"][0] + meta_a["pickup_tw"][1]) * 0.5)
            - ((meta_b["pickup_tw"][0] + meta_b["pickup_tw"][1]) * 0.5)
        ) * 60.0,
        "direct_pd_time_max_min": direct_pd_max_min,
        "separate_distance_km": _separate_service_distance(meta_a, meta_b),
    }


def _separate_service_distance(meta_a, meta_b):
    depot = (0.5, 0.5)
    # This helper is overwritten in analyze_pair with the real depot; placeholder kept for defensive use.
    return 0.0 if depot is None else None


def _separate_service_distance_with_depot(depot_coord, meta_a, meta_b):
    def single(meta):
        return (
            _euclidean_km(depot_coord, meta["pickup_coord"])
            + meta["direct_distance_km"]
            + _euclidean_km(meta["delivery_coord"], depot_coord)
        )

    return single(meta_a) + single(meta_b)


def _dominant_blocker(blocker_counts):
    if not blocker_counts:
        return "unknown"
    return min(
        blocker_counts.keys(),
        key=lambda key: (-blocker_counts[key], BLOCKER_PRIORITY.index(key) if key in BLOCKER_PRIORITY else 10),
    )


def simulate_interleaving(sample, order_pair, interleaving):
    depot_coord = _to_float_pair(sample["depot"])
    metas = [_order_meta(sample, order_idx) for order_idx in order_pair]
    current_coord = depot_coord
    current_time = Config.OPERATION_START
    trip_start_time = Config.OPERATION_START
    used_capacity_p = 0.0
    used_capacity_c = 0.0
    passenger_pickup_finish = {}
    total_distance_km = 0.0

    for action, pair_pos in interleaving:
        meta = metas[pair_pos]
        is_pickup = action == "P"
        target_coord = meta["pickup_coord"] if is_pickup else meta["delivery_coord"]
        target_tw = meta["pickup_tw"] if is_pickup else meta["delivery_tw"]
        travel_distance_km = _euclidean_km(current_coord, target_coord)
        travel_time_h = travel_distance_km / Config.VEHICLE_SPEED
        arrival_time = current_time + travel_time_h

        if is_pickup and meta["type"] == "passenger" and Config.HARD_PASSENGER_PICKUP_TIMEWINDOW:
            if arrival_time > target_tw[1] + EPS:
                return {
                    "feasible": False,
                    "blocker": "pickup_tw",
                    "failure_step": f"pickup_{meta['order_idx']}",
                }

        service_start = max(arrival_time, target_tw[0])
        service_finish = service_start + Config.SERVICE_TIME
        next_capacity_p = used_capacity_p + (meta["pickup_demand_p"] if is_pickup else meta["delivery_demand_p"])
        next_capacity_c = used_capacity_c + (meta["pickup_demand_c"] if is_pickup else meta["delivery_demand_c"])
        if next_capacity_p > 1.0 + EPS or next_capacity_c > 1.0 + EPS:
            return {
                "feasible": False,
                "blocker": "capacity",
                "failure_step": f"{'pickup' if is_pickup else 'delivery'}_{meta['order_idx']}",
            }

        if (not is_pickup) and meta["type"] == "passenger" and Config.HARD_PASSENGER_MAX_RIDE_TIME:
            pickup_finish = passenger_pickup_finish.get(meta["order_idx"])
            if pickup_finish is None:
                return {
                    "feasible": False,
                    "blocker": "ride_time",
                    "failure_step": f"delivery_{meta['order_idx']}",
                }
            ride_time_h = arrival_time - pickup_finish
            excess_ride_time_h = max(ride_time_h - meta["direct_travel_time_h"], 0.0)
            if (
                ride_time_h > Config.PASSENGER_MAX_RIDE_TIME_MINUTES / 60.0 + EPS
                or excess_ride_time_h > Config.PASSENGER_MAX_EXCESS_RIDE_TIME_MINUTES / 60.0 + EPS
            ):
                return {
                    "feasible": False,
                    "blocker": "ride_time",
                    "failure_step": f"delivery_{meta['order_idx']}",
                }

        predicted_finish_with_return = service_finish + _travel_time_hours(target_coord, depot_coord)
        if Config.HARD_MAX_TRIP_TIME and predicted_finish_with_return - trip_start_time > Config.MAX_TRIP_TIME + EPS:
            return {
                "feasible": False,
                "blocker": "trip_time",
                "failure_step": f"{'pickup' if is_pickup else 'delivery'}_{meta['order_idx']}",
            }
        if Config.HARD_OPERATION_END and predicted_finish_with_return > Config.OPERATION_END + EPS:
            return {
                "feasible": False,
                "blocker": "operation_end",
                "failure_step": f"{'pickup' if is_pickup else 'delivery'}_{meta['order_idx']}",
            }

        if is_pickup and meta["type"] == "passenger":
            passenger_pickup_finish[meta["order_idx"]] = service_finish

        used_capacity_p = max(next_capacity_p, 0.0)
        used_capacity_c = max(next_capacity_c, 0.0)
        current_coord = target_coord
        current_time = service_finish
        total_distance_km += travel_distance_km

    return_distance_km = _euclidean_km(current_coord, depot_coord)
    total_distance_km += return_distance_km
    final_time = current_time + return_distance_km / Config.VEHICLE_SPEED
    return {
        "feasible": True,
        "blocker": None,
        "failure_step": None,
        "total_distance_km": total_distance_km,
        "final_time": final_time,
    }


def analyze_pair(sample, order_i, order_j):
    depot_coord = _to_float_pair(sample["depot"])
    meta_i = _order_meta(sample, order_i)
    meta_j = _order_meta(sample, order_j)
    pair_type = _pair_type(meta_i, meta_j)
    pair_features = _pair_features(meta_i, meta_j)
    pair_features["separate_distance_km"] = _separate_service_distance_with_depot(depot_coord, meta_i, meta_j)

    interleaving_results = []
    blocker_counts = Counter()
    feasible_any = False
    feasible_overlap = False
    best_overlap_distance = None
    best_any_distance = None
    best_sequence = None

    for name, pattern, is_overlap in INTERLEAVINGS:
        result = simulate_interleaving(sample, (order_i, order_j), pattern)
        result.update({"name": name, "is_overlap": is_overlap, "pattern": pattern})
        interleaving_results.append(result)
        if result["feasible"]:
            feasible_any = True
            if best_any_distance is None or result["total_distance_km"] < best_any_distance:
                best_any_distance = result["total_distance_km"]
                best_sequence = name
            if is_overlap:
                feasible_overlap = True
                if best_overlap_distance is None or result["total_distance_km"] < best_overlap_distance:
                    best_overlap_distance = result["total_distance_km"]
        else:
            blocker_counts[result["blocker"]] += 1

    return {
        "order_pair": [int(order_i), int(order_j)],
        "pair_type": pair_type,
        "feasible_any": feasible_any,
        "feasible_overlap": feasible_overlap,
        "distance_saving_overlap": bool(
            feasible_overlap and best_overlap_distance is not None and best_overlap_distance + EPS < pair_features["separate_distance_km"]
        ),
        "best_any_distance_km": best_any_distance,
        "best_overlap_distance_km": best_overlap_distance,
        "best_sequence": best_sequence,
        "dominant_blocker": None if feasible_any else _dominant_blocker(blocker_counts),
        "blocker_counts": dict(blocker_counts),
        "interleavings": interleaving_results,
        **pair_features,
    }


def _empty_counter_block():
    return {"pairs": 0, "feasible_any": 0, "feasible_overlap": 0, "distance_saving_overlap": 0}


def _add_bucket_result(bucket_dict, bucket, pair_result):
    bucket_stats = bucket_dict[bucket]
    bucket_stats["pairs"] += 1
    bucket_stats["feasible_any"] += int(pair_result["feasible_any"])
    bucket_stats["feasible_overlap"] += int(pair_result["feasible_overlap"])
    bucket_stats["distance_saving_overlap"] += int(pair_result["distance_saving_overlap"])


def _summarize_bucket_rates(bucket_dict):
    summary = {}
    for bucket, stats in bucket_dict.items():
        summary[bucket] = {
            **stats,
            "feasible_any_rate": _safe_rate(stats["feasible_any"], stats["pairs"]),
            "feasible_overlap_rate": _safe_rate(stats["feasible_overlap"], stats["pairs"]),
            "distance_saving_overlap_rate": _safe_rate(stats["distance_saving_overlap"], stats["pairs"]),
        }
    return dict(summary)


def cross_check_feasible_routes(cross_check_candidates, requested):
    if requested <= 0 or not cross_check_candidates:
        return {
            "requested": int(requested),
            "checked": 0,
            "max_distance_error_km": 0.0,
            "mismatched_routes": 0,
        }

    checked = 0
    max_distance_error = 0.0
    mismatched_routes = 0
    for sample, node_route, expected_distance in cross_check_candidates[:requested]:
        _, details, _ = evaluate_node_routes(sample, [node_route])
        actual_distance = float(details["total_distance"][0].item())
        distance_error = abs(actual_distance - expected_distance)
        max_distance_error = max(max_distance_error, distance_error)
        violations_ok = (
            float(details["passenger_pickup_hard_violations"][0].item()) == 0.0
            and float(details["passenger_total_ride_time_violations"][0].item()) == 0.0
            and float(details["passenger_excess_ride_time_violations"][0].item()) == 0.0
            and float(details["trip_overtime_penalty"][0].item()) == 0.0
            and float(details["started_not_completed_orders"][0].item()) == 0.0
        )
        if distance_error > 1e-4 or (not violations_ok):
            mismatched_routes += 1
        checked += 1

    return {
        "requested": int(requested),
        "checked": int(checked),
        "max_distance_error_km": max_distance_error,
        "mismatched_routes": int(mismatched_routes),
    }


def run_analysis(args):
    dataset = MCVRPPDTWDataset(num_samples=args.num_samples, graph_size=args.graph_size, seed=args.seed)
    rng = random.Random(args.seed)

    overall = _empty_counter_block()
    by_type = defaultdict(_empty_counter_block)
    blockers = Counter()
    blockers_by_type = defaultdict(Counter)
    distance_buckets = defaultdict(_empty_counter_block)
    tw_overlap_buckets = defaultdict(_empty_counter_block)
    direct_pd_time_buckets = defaultdict(_empty_counter_block)
    passenger_order_counts = []
    cargo_order_counts = []
    pair_examples = []
    cross_check_candidates = []

    for sample_idx, sample in enumerate(dataset):
        n_orders = int(sample["n_orders"])
        passenger_orders = int((sample["node_type"][:n_orders] == 1).sum().item())
        passenger_order_counts.append(passenger_orders)
        cargo_order_counts.append(n_orders - passenger_orders)

        pair_list = list(combinations(range(n_orders), 2))
        if args.max_pairs_per_instance is not None and len(pair_list) > args.max_pairs_per_instance:
            pair_list = rng.sample(pair_list, args.max_pairs_per_instance)

        for order_i, order_j in pair_list:
            pair_result = analyze_pair(sample, order_i, order_j)
            overall["pairs"] += 1
            overall["feasible_any"] += int(pair_result["feasible_any"])
            overall["feasible_overlap"] += int(pair_result["feasible_overlap"])
            overall["distance_saving_overlap"] += int(pair_result["distance_saving_overlap"])

            pair_type = pair_result["pair_type"]
            by_type[pair_type]["pairs"] += 1
            by_type[pair_type]["feasible_any"] += int(pair_result["feasible_any"])
            by_type[pair_type]["feasible_overlap"] += int(pair_result["feasible_overlap"])
            by_type[pair_type]["distance_saving_overlap"] += int(pair_result["distance_saving_overlap"])

            if not pair_result["feasible_any"] and pair_result["dominant_blocker"] is not None:
                blockers[pair_result["dominant_blocker"]] += 1
                blockers_by_type[pair_type][pair_result["dominant_blocker"]] += 1

            distance_bucket = _bucket_label(pair_result["pickup_pickup_distance_km"], PICKUP_DISTANCE_BUCKETS_KM, "km")
            tw_overlap_bucket = _bucket_label(pair_result["pickup_tw_overlap_min"], PICKUP_TW_OVERLAP_BUCKETS_MIN, "min")
            direct_pd_time_bucket = _bucket_label(pair_result["direct_pd_time_max_min"], DIRECT_PD_TIME_BUCKETS_MIN, "min")
            _add_bucket_result(distance_buckets, distance_bucket, pair_result)
            _add_bucket_result(tw_overlap_buckets, tw_overlap_bucket, pair_result)
            _add_bucket_result(direct_pd_time_buckets, direct_pd_time_bucket, pair_result)

            if pair_result["feasible_overlap"] and len(pair_examples) < 10:
                pair_examples.append({
                    "sample_index": int(sample_idx),
                    "order_pair": pair_result["order_pair"],
                    "pair_type": pair_type,
                    "best_sequence": pair_result["best_sequence"],
                    "pickup_pickup_distance_km": pair_result["pickup_pickup_distance_km"],
                    "pickup_tw_overlap_min": pair_result["pickup_tw_overlap_min"],
                    "best_overlap_distance_km": pair_result["best_overlap_distance_km"],
                    "separate_distance_km": pair_result["separate_distance_km"],
                })

            if pair_result["feasible_overlap"] and len(cross_check_candidates) < max(args.cross_check_routes * 2, 10):
                best_overlap = min(
                    (r for r in pair_result["interleavings"] if r["feasible"] and r["is_overlap"]),
                    key=lambda item: item["total_distance_km"],
                )
                node_route = []
                n_orders_total = int(sample["n_orders"])
                for action, pair_pos in best_overlap["pattern"]:
                    order_idx = pair_result["order_pair"][pair_pos]
                    node_route.append(order_idx + 1 if action == "P" else order_idx + n_orders_total + 1)
                cross_check_candidates.append((sample, node_route, best_overlap["total_distance_km"]))

    cross_check = cross_check_feasible_routes(cross_check_candidates, args.cross_check_routes)
    infeasible_pairs = overall["pairs"] - overall["feasible_any"]

    summary = {
        "config": {
            "graph_size": int(args.graph_size),
            "num_samples": int(args.num_samples),
            "seed": int(args.seed),
            "max_pairs_per_instance": args.max_pairs_per_instance,
            "note": "Measures structural pair-level rideshare opportunity under current hard service constraints; it intentionally does not apply the stable anti-rideshare pickup blanket gate.",
        },
        "dataset_summary": {
            "avg_passenger_orders": _safe_rate(sum(passenger_order_counts), len(passenger_order_counts)),
            "avg_cargo_orders": _safe_rate(sum(cargo_order_counts), len(cargo_order_counts)),
            "avg_passenger_ratio": _safe_rate(sum(passenger_order_counts), sum(passenger_order_counts) + sum(cargo_order_counts)),
        },
        "overall": {
            **overall,
            "feasible_any_rate": _safe_rate(overall["feasible_any"], overall["pairs"]),
            "feasible_overlap_rate": _safe_rate(overall["feasible_overlap"], overall["pairs"]),
            "distance_saving_overlap_rate": _safe_rate(overall["distance_saving_overlap"], overall["pairs"]),
            "infeasible_pairs": infeasible_pairs,
        },
        "by_type": {
            pair_type: {
                **stats,
                "feasible_any_rate": _safe_rate(stats["feasible_any"], stats["pairs"]),
                "feasible_overlap_rate": _safe_rate(stats["feasible_overlap"], stats["pairs"]),
                "distance_saving_overlap_rate": _safe_rate(stats["distance_saving_overlap"], stats["pairs"]),
            }
            for pair_type, stats in by_type.items()
        },
        "blockers": {
            blocker: {
                "count": int(count),
                "rate_among_infeasible": _safe_rate(count, infeasible_pairs),
            }
            for blocker, count in blockers.items()
        },
        "blockers_by_type": {
            pair_type: {
                blocker: {
                    "count": int(count),
                    "rate_among_infeasible_type": _safe_rate(count, by_type[pair_type]["pairs"] - by_type[pair_type]["feasible_any"]),
                }
                for blocker, count in blocker_counter.items()
            }
            for pair_type, blocker_counter in blockers_by_type.items()
        },
        "feature_slices": {
            "pickup_pickup_distance_km": _summarize_bucket_rates(distance_buckets),
            "pickup_tw_overlap_min": _summarize_bucket_rates(tw_overlap_buckets),
            "direct_pd_time_max_min": _summarize_bucket_rates(direct_pd_time_buckets),
        },
        "cross_check": cross_check,
        "pair_examples": pair_examples,
    }
    return summary


def _print_header(args):
    print("=" * 72)
    print("局部双单拼车机会密度分析")
    print("=" * 72)
    print(f"  graph_size            : {args.graph_size}")
    print(f"  num_samples           : {args.num_samples}")
    print(f"  seed                  : {args.seed}")
    print(f"  max_pairs_per_instance: {args.max_pairs_per_instance if args.max_pairs_per_instance is not None else 'all'}")
    print("  semantic note         : analyze structural hard-constraint opportunities, not current decoder pickup blanket")
    print("-" * 72)


def _print_type_table(by_type):
    print("【按类型拆分】")
    print(f"{'类型':<8} {'pairs':>8} {'any-feasible':>14} {'overlap':>12} {'saving-overlap':>16}")
    print("-" * 72)
    for pair_type in ["pp", "pc", "cc"]:
        stats = by_type.get(pair_type, {"pairs": 0, "feasible_any_rate": 0.0, "feasible_overlap_rate": 0.0, "distance_saving_overlap_rate": 0.0})
        print(
            f"{pair_type:<8}"
            f"{stats.get('pairs', 0):>8}"
            f"{stats.get('feasible_any_rate', 0.0) * 100:>13.2f}%"
            f"{stats.get('feasible_overlap_rate', 0.0) * 100:>11.2f}%"
            f"{stats.get('distance_saving_overlap_rate', 0.0) * 100:>15.2f}%"
        )
    print()


def _print_blockers(summary):
    print("【主阻塞约束（仅统计无机会的 pair）】")
    if not summary["blockers"]:
        print("  所有 pair 都存在至少一种可行 interleaving，没有阻塞统计。")
        print()
        return
    for blocker, stats in sorted(summary["blockers"].items(), key=lambda item: -item[1]["count"]):
        print(f"  {blocker:<14}: {stats['count']:>6} ({stats['rate_among_infeasible'] * 100:>6.2f}%)")
    print()


def _print_slice(title, slice_summary):
    print(f"【{title}】")
    print(f"{'bucket':<18} {'pairs':>8} {'any-feasible':>14} {'overlap':>12}")
    print("-" * 60)
    for bucket, stats in slice_summary.items():
        print(
            f"{bucket:<18}"
            f"{stats['pairs']:>8}"
            f"{stats['feasible_any_rate'] * 100:>13.2f}%"
            f"{stats['feasible_overlap_rate'] * 100:>11.2f}%"
        )
    print()


def print_summary(summary, args):
    _print_header(args)
    dataset_summary = summary["dataset_summary"]
    overall = summary["overall"]
    print("【数据集概况】")
    print(f"  Avg passenger orders : {dataset_summary['avg_passenger_orders']:.2f}")
    print(f"  Avg cargo orders     : {dataset_summary['avg_cargo_orders']:.2f}")
    print(f"  Avg passenger ratio  : {dataset_summary['avg_passenger_ratio'] * 100:.2f}%")
    print()

    print("【总体机会率】")
    print(f"  Total pairs analyzed            : {overall['pairs']}")
    print(f"  Consolidation opportunity rate  : {overall['feasible_any_rate'] * 100:.2f}%")
    print(f"  True overlap rideshare rate     : {overall['feasible_overlap_rate'] * 100:.2f}%")
    print(f"  Distance-saving overlap rate    : {overall['distance_saving_overlap_rate'] * 100:.2f}%")
    print(f"  Infeasible pairs                : {overall['infeasible_pairs']}")
    print()

    _print_type_table(summary["by_type"])
    _print_blockers(summary)
    _print_slice("按 pickup-pickup 距离分桶", summary["feature_slices"]["pickup_pickup_distance_km"])
    _print_slice("按 pickup 时间窗重叠分桶", summary["feature_slices"]["pickup_tw_overlap_min"])
    _print_slice("按最大 direct PD 时间分桶", summary["feature_slices"]["direct_pd_time_max_min"])

    cross_check = summary["cross_check"]
    print("【canonical get_costs 交叉验证】")
    print(f"  Checked routes        : {cross_check['checked']} / requested {cross_check['requested']}")
    print(f"  Max distance error km : {cross_check['max_distance_error_km']:.6f}")
    print(f"  Mismatched routes     : {cross_check['mismatched_routes']}")
    print()

    if summary["pair_examples"]:
        print("【可行 overlap 示例（前 10 个）】")
        for example in summary["pair_examples"]:
            print(
                f"  sample={example['sample_index']:>3} pair={tuple(example['order_pair'])} type={example['pair_type']} "
                f"seq={example['best_sequence']} dist={example['best_overlap_distance_km']:.2f}km "
                f"sep={example['separate_distance_km']:.2f}km tw_overlap={example['pickup_tw_overlap_min']:.1f}min"
            )
        print()


def build_test_sample(order_specs, depot=(0.5, 0.5)):
    n_orders = len(order_specs)
    depot_tensor = torch.tensor(depot, dtype=torch.float)
    pickup_locs = []
    delivery_locs = []
    pickup_types = []
    delivery_types = []
    demand_p = torch.zeros(2 * n_orders, dtype=torch.float)
    demand_c = torch.zeros(2 * n_orders, dtype=torch.float)
    time_windows = torch.zeros(2 * n_orders, 2, dtype=torch.float)

    for idx, spec in enumerate(order_specs):
        pickup_locs.append(torch.tensor(spec["pickup"], dtype=torch.float))
        delivery_locs.append(torch.tensor(spec["delivery"], dtype=torch.float))
        type_flag = 1.0 if spec["type"] == "passenger" else 0.0
        pickup_types.append(type_flag)
        delivery_types.append(type_flag)
        if spec["type"] == "passenger":
            demand = float(spec.get("demand", 1.0)) / Config.PASSENGER_CAPACITY
            demand_p[idx] = demand
            demand_p[idx + n_orders] = -demand
        else:
            demand = float(spec.get("demand", 1.0)) / Config.CARGO_CAPACITY
            demand_c[idx] = demand
            demand_c[idx + n_orders] = -demand
        time_windows[idx] = torch.tensor(spec.get("pickup_tw", (Config.OPERATION_START, Config.OPERATION_END)), dtype=torch.float)
        time_windows[idx + n_orders] = torch.tensor(spec.get("delivery_tw", (Config.OPERATION_START, Config.OPERATION_END)), dtype=torch.float)

    return {
        "depot": depot_tensor,
        "loc": torch.cat([torch.stack(pickup_locs), torch.stack(delivery_locs)], dim=0),
        "node_type": torch.tensor(pickup_types + delivery_types, dtype=torch.float),
        "demand_passenger": demand_p,
        "demand_cargo": demand_c,
        "time_windows": time_windows,
        "pickup_delivery_pairs": torch.arange(n_orders),
        "pd_pair_mask": torch.zeros(2 * n_orders + 1, 2 * n_orders + 1, dtype=torch.bool),
        "n_orders": n_orders,
    }


def run_self_check():
    checks = []

    feasible_sample = build_test_sample([
        {"type": "passenger", "pickup": (0.25, 0.25), "delivery": (0.27, 0.34), "pickup_tw": (10.0, 16.0)},
        {"type": "passenger", "pickup": (0.28, 0.26), "delivery": (0.30, 0.36), "pickup_tw": (10.0, 16.0)},
    ], depot=(0.20, 0.20))
    checks.append((simulate_interleaving(feasible_sample, (0, 1), INTERLEAVINGS[1][1]), True, None, "feasible overlap"))

    pickup_tw_sample = build_test_sample([
        {"type": "passenger", "pickup": (0.21, 0.21), "delivery": (0.22, 0.26), "pickup_tw": (10.0, 16.0)},
        {"type": "passenger", "pickup": (0.95, 0.95), "delivery": (0.96, 0.96), "pickup_tw": (10.0, 10.20)},
    ], depot=(0.20, 0.20))
    checks.append((simulate_interleaving(pickup_tw_sample, (0, 1), INTERLEAVINGS[1][1]), False, "pickup_tw", "pickup TW blocker"))

    ride_time_sample = build_test_sample([
        {"type": "passenger", "pickup": (0.10, 0.10), "delivery": (0.12, 0.12), "pickup_tw": (10.0, 16.0)},
        {"type": "passenger", "pickup": (0.90, 0.90), "delivery": (0.88, 0.88), "pickup_tw": (10.0, 16.0)},
    ], depot=(0.10, 0.10))
    checks.append((simulate_interleaving(ride_time_sample, (0, 1), INTERLEAVINGS[1][1]), False, "ride_time", "ride time blocker"))

    trip_time_sample = build_test_sample([
        {"type": "cargo", "pickup": (0.21, 0.21), "delivery": (0.22, 0.23), "pickup_tw": (10.0, 16.0)},
        {"type": "cargo", "pickup": (0.22, 0.22), "delivery": (0.23, 0.24), "pickup_tw": (13.30, 16.0)},
    ], depot=(0.20, 0.20))
    checks.append((simulate_interleaving(trip_time_sample, (0, 1), INTERLEAVINGS[4][1]), False, "trip_time", "trip time blocker"))

    capacity_sample = build_test_sample([
        {"type": "cargo", "pickup": (0.25, 0.25), "delivery": (0.26, 0.28), "pickup_tw": (10.0, 16.0), "demand": 12.0},
        {"type": "cargo", "pickup": (0.27, 0.26), "delivery": (0.28, 0.29), "pickup_tw": (10.0, 16.0), "demand": 12.0},
    ], depot=(0.20, 0.20))
    checks.append((simulate_interleaving(capacity_sample, (0, 1), INTERLEAVINGS[1][1]), False, "capacity", "capacity blocker"))

    for result, expected_feasible, expected_blocker, label in checks:
        assert result["feasible"] == expected_feasible, f"{label} failed: feasible mismatch {result}"
        if not expected_feasible:
            assert result["blocker"] == expected_blocker, f"{label} failed: blocker mismatch {result}"

    print("Self-check passed: feasible overlap / pickup_tw / ride_time / trip_time / capacity cases all match expectations.")


def main():
    args = parse_args()
    if args.self_check:
        run_self_check()
        return

    summary = run_analysis(args)
    print_summary(summary, args)
    if args.output_json:
        with open(args.output_json, "w", encoding="utf-8") as fh:
            json.dump(summary, fh, indent=2, ensure_ascii=False)
        print(f"结果已写入: {args.output_json}")


if __name__ == "__main__":
    main()
