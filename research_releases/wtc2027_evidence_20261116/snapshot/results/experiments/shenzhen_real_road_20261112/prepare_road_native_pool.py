"""Build Shenzhen road-native training cases without coordinate snapping.

Pickup, delivery and depot terminals are sampled directly from the largest
strongly connected OSM driving graph.  Passenger pickup-delivery distances are
stratified on shortest-path distance, and time windows are recomputed from the
same road matrices used by the model.  The resulting cases are development
data only; a separate seed must be frozen after all model choices are locked.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import sys
import time
from pathlib import Path

import networkx as nx
import numpy as np
import torch
from scipy.spatial import cKDTree


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
GRAPH = HERE / "data" / "futian_luohu_current.json"
FORMAL = HERE / "formal_p1_v1"
sys.path.insert(0, str(ROOT / "experiments/dispatch_clock_fix_20261106"))
from bootstrap import SNAP
sys.path.insert(0, str(SNAP / "experiments/sensitivity_20261030"))
from scenarios import make
sys.path.insert(0, str(ROOT))
from problem_mcvrptw_v2 import Config
sys.path.insert(0, str(HERE))
from road_network import load_overpass


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def to_python(sample):
    return {
        key: value.detach().cpu().tolist() if hasattr(value, "detach") else value
        for key, value in sample.items()
    }


def to_tensor(sample):
    result = {}
    for key, value in sample.items():
        if key in {"road_node_ids", "pickup_delivery_pairs", "n_orders"}:
            result[key] = torch.as_tensor(value, dtype=torch.long)
        elif key == "pd_pair_mask":
            result[key] = torch.as_tensor(value, dtype=torch.bool)
        elif key == "road_lonlat":
            result[key] = torch.as_tensor(value, dtype=torch.float64)
        else:
            result[key] = torch.as_tensor(value, dtype=torch.float32)
    return result


def graph_terminals(
    graph: nx.DiGraph,
    bbox: dict[str, float],
    base_sample: dict,
    rng: random.Random,
):
    eligible = [
        int(node)
        for node, attrs in graph.nodes(data=True)
        if bbox["west"] <= attrs["lon"] <= bbox["east"]
        and bbox["south"] <= attrs["lat"] <= bbox["north"]
    ]
    if len(eligible) < 1000:
        raise ValueError(f"Only {len(eligible)} eligible road nodes")
    coords = np.asarray(
        [(graph.nodes[node]["lon"], graph.nodes[node]["lat"]) for node in eligible],
        dtype=float,
    )
    mean_lat = math.radians(float(coords[:, 1].mean()))
    scale = np.asarray([111.320 * math.cos(mean_lat), 110.574])
    projected = coords * scale
    tree = cKDTree(projected)
    used: set[int] = set()

    center = np.asarray(
        [(bbox["west"] + bbox["east"]) / 2, (bbox["south"] + bbox["north"]) / 2]
    ) * scale
    depot_candidates = tree.query_ball_point(center, r=2.0)
    if not depot_candidates:
        depot_candidates = [int(tree.query(center, k=1)[1])]
    depot = eligible[rng.choice(depot_candidates)]
    used.add(depot)

    n_orders = 50
    cluster_centers = rng.sample(eligible, 3)
    cluster_projected = {
        node: np.asarray([graph.nodes[node]["lon"], graph.nodes[node]["lat"]]) * scale
        for node in cluster_centers
    }
    clustered_indices = set(rng.sample(range(n_orders), n_orders // 2))
    pickups: list[int] = []
    for order in range(n_orders):
        if order not in clustered_indices:
            while True:
                candidate = rng.choice(eligible)
                if candidate not in used:
                    break
        else:
            center_node = rng.choice(cluster_centers)
            target = cluster_projected[center_node] + np.asarray(
                [rng.gauss(0.0, 0.8), rng.gauss(0.0, 0.8)]
            )
            _, positions = tree.query(target, k=64)
            candidate = next(
                eligible[int(position)]
                for position in np.atleast_1d(positions)
                if eligible[int(position)] not in used
            )
        pickups.append(candidate)
        used.add(candidate)

    node_type = np.asarray(base_sample["node_type"][:n_orders], dtype=float)
    passenger_orders = [i for i in range(n_orders) if node_type[i] == 1.0]
    cargo_orders = [i for i in range(n_orders) if node_type[i] == 0.0]
    rng.shuffle(passenger_orders)
    long_count = round(len(passenger_orders) * Config.PASSENGER_PD_LONG_RATIO)
    mid_count = round(len(passenger_orders) * Config.PASSENGER_PD_MID_RATIO)
    passenger_bands = {}
    for idx, order in enumerate(passenger_orders):
        if idx < long_count:
            passenger_bands[order] = (
                Config.PASSENGER_PD_LONG_MIN_KM,
                Config.PASSENGER_PD_LONG_MAX_KM,
                "passenger_long",
            )
        elif idx < long_count + mid_count:
            passenger_bands[order] = (
                Config.PASSENGER_PD_MID_MIN_KM,
                Config.PASSENGER_PD_MID_MAX_KM,
                "passenger_mid",
            )
        else:
            passenger_bands[order] = (
                Config.PASSENGER_PD_SHORT_MIN_KM,
                Config.PASSENGER_PD_SHORT_MAX_KM,
                "passenger_short",
            )
    rng.shuffle(cargo_orders)
    cargo_short_count = round(len(cargo_orders) * Config.PD_SHORT_DISTANCE_RATIO)
    cargo_bands = {
        order: (
            0.3,
            Config.PD_SHORT_DISTANCE_KM if idx < cargo_short_count else Config.PD_LONG_DISTANCE_KM,
            "cargo_short" if idx < cargo_short_count else "cargo_long",
        )
        for idx, order in enumerate(cargo_orders)
    }

    deliveries: list[int] = []
    pair_info = []
    pickup_resamples = 0
    eligible_set = set(eligible)
    for order, pickup in enumerate(pickups):
        low, high, band = passenger_bands.get(order, cargo_bands.get(order))
        distance_map = {}
        candidates = []
        for attempt in range(101):
            distance_map = nx.single_source_dijkstra_path_length(
                graph, pickup, cutoff=float(high), weight="length_km"
            )
            candidates = [
                node for node, distance in distance_map.items()
                if node in eligible_set and node not in used and low <= distance <= high
            ]
            if candidates:
                break
            if attempt == 100:
                raise RuntimeError(
                    f"No strict delivery candidate for order {order} band {band} after 100 pickup resamples"
                )
            # A long-distance band may be impossible from a boundary pickup.
            # Replace that pickup before any vehicle is dispatched; never
            # silently change the requested road-distance stratum.
            used.remove(pickup)
            while True:
                replacement = rng.choice(eligible)
                if replacement not in used:
                    break
            pickup = replacement
            pickups[order] = pickup
            used.add(pickup)
            pickup_resamples += 1
        delivery = rng.choice(candidates)
        used.add(delivery)
        deliveries.append(delivery)
        pair_info.append(
            {
                "order": order,
                "band": band,
                "distance_km": float(distance_map[delivery]),
                "band_fallback": False,
            }
        )

    terminals = [depot] + pickups + deliveries
    size = len(terminals)
    road_distance = np.empty((size, size), dtype=float)
    for row, source in enumerate(terminals):
        # Pair-selection maps above are deliberately cutoff-limited.  The
        # terminal matrix must instead cover every directed terminal pair.
        distances = nx.single_source_dijkstra_path_length(graph, source, weight="length_km")
        for column, target in enumerate(terminals):
            road_distance[row, column] = distances[target]
    if not np.isfinite(road_distance).all():
        raise ValueError("Non-finite road matrix")
    road_time = road_distance / float(Config.VEHICLE_SPEED)
    lonlat = [(graph.nodes[node]["lon"], graph.nodes[node]["lat"]) for node in terminals]
    normalized = [
        [
            (lon - bbox["west"]) / (bbox["east"] - bbox["west"]),
            (lat - bbox["south"]) / (bbox["north"] - bbox["south"]),
        ]
        for lon, lat in lonlat
    ]
    if not all(0.0 <= value <= 1.0 for point in normalized for value in point):
        raise AssertionError("Road-native normalized coordinate outside [0, 1]")
    return terminals, lonlat, normalized, road_distance, road_time, pair_info, pickup_resamples


def recompute_time_windows(sample, road_distance: np.ndarray, rng: random.Random):
    n_orders = 50
    node_type = np.asarray(sample["node_type"][:n_orders], dtype=float)
    windows = np.zeros((2 * n_orders, 2), dtype=np.float32)
    for order in range(n_orders):
        passenger = node_type[order] == 1.0
        weights = (
            Config.PASSENGER_TW_PERIOD_WEIGHTS if passenger
            else Config.CARGO_TW_PERIOD_WEIGHTS
        )
        bounds = (
            Config.PASSENGER_TW_PERIOD_BOUNDS if passenger
            else Config.CARGO_TW_PERIOD_BOUNDS
        )
        bucket = rng.choices(range(3), weights=weights, k=1)[0]
        width = Config.PASSENGER_TW_WIDTH if passenger else Config.CARGO_TW_WIDTH
        earliest = Config.OPERATION_START + road_distance[0, 1 + order] / Config.VEHICLE_SPEED
        feasible_low = max(Config.OPERATION_START, float(earliest))
        feasible_high = Config.OPERATION_END - width
        low = max(feasible_low, bounds[bucket][0])
        high = min(feasible_high, bounds[bucket][1])
        if high > low:
            start = rng.uniform(low, high)
        elif feasible_high > feasible_low:
            start = rng.uniform(feasible_low, feasible_high)
        else:
            start = feasible_low
        start = min(max(start, Config.OPERATION_START), Config.OPERATION_END - width)
        windows[order] = (start, start + width)
        direct = road_distance[1 + order, 1 + n_orders + order] / Config.VEHICLE_SPEED
        delivery_earliest = min(start + Config.SERVICE_TIME + direct, Config.OPERATION_END)
        delivery_latest = min(start + width + Config.SERVICE_TIME + direct + width, Config.OPERATION_END)
        windows[n_orders + order] = (delivery_earliest, max(delivery_earliest, delivery_latest))
    return windows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=4)
    parser.add_argument("--seed", type=int, default=2026111303)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--purpose",
        choices=("development", "formal_test"),
        default="development",
        help="Formal-test cases require a pre-existing frozen model lock.",
    )
    parser.add_argument("--model-lock", type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output}")
    if args.samples < 1:
        raise ValueError("At least one case is required")
    if args.purpose == "formal_test":
        if args.model_lock is None or not args.model_lock.is_file():
            raise ValueError("--purpose=formal_test requires an existing --model-lock")
        model_lock = json.loads(args.model_lock.read_text())
        if model_lock.get("status") != "locked" or len(model_lock.get("models", [])) != 3:
            raise ValueError("Formal-test generation requires a complete three-model lock")
        model_lock_sha256 = digest(args.model_lock)
    elif args.model_lock is not None:
        raise ValueError("--model-lock is only valid with --purpose=formal_test")
    else:
        model_lock_sha256 = None

    protocol = json.loads((HERE / "protocol.json").read_text())
    formal_plan = json.loads((FORMAL / "plan.json").read_text())
    bbox = protocol["bbox_wgs84"]
    graph = load_overpass(GRAPH)
    generated = make(n=50, samples=args.samples, seed=args.seed, ratio=None, tw=1.0, excess=1.0)
    road_cases = []
    case_metadata = []
    started = time.monotonic()
    for index, original in enumerate(generated["data"]):
        sample = to_python(original)
        rng = random.Random(args.seed * 1_000_003 + index)
        terminals, lonlat, normalized, distance, travel, pair_info, pickup_resamples = graph_terminals(
            graph, bbox, sample, rng
        )
        sample.update(
            depot=normalized[0],
            loc=normalized[1:],
            time_windows=recompute_time_windows(sample, distance, rng).tolist(),
            road_node_ids=terminals,
            road_lonlat=lonlat,
            road_distance_matrix_km=distance.tolist(),
            road_travel_time_matrix_hours=travel.tolist(),
        )
        road_cases.append(to_tensor(sample))
        pair_distances = np.asarray([entry["distance_km"] for entry in pair_info])
        case_metadata.append(
            {
                "instance": index,
                "unique_road_nodes": len(set(terminals)),
                "band_fallbacks": 0,
                "pickup_resamples_for_strict_band": pickup_resamples,
                "pair_distance_km_mean": float(pair_distances.mean()),
                "pair_distance_km_max": float(pair_distances.max()),
                "pair_info": pair_info,
            }
        )
        print("ROAD_NATIVE", index + 1, "/", args.samples, "elapsed_s", round(time.monotonic() - started, 1), flush=True)

    payload = {
        "meta": {
            "scope": (
                "frozen road-native formal test generated after all neural model choices were locked"
                if args.purpose == "formal_test"
                else "road-native road-aware development only; never formal testing"
            ),
            "purpose": args.purpose,
            "generator_version": 3,
            "seed": args.seed,
            "samples": args.samples,
            "graph_size": 50,
            "graph_sha256": digest(GRAPH),
            "formal_test_road_cases_sha256": formal_plan["road_cases_sha256"],
            "formal_test_seed": formal_plan["seed"],
            "model_lock_sha256": model_lock_sha256,
            "model_lock_path": (
                str(args.model_lock.resolve().relative_to(ROOT))
                if args.model_lock is not None else None
            ),
            "matrix_speed_kph": Config.VEHICLE_SPEED,
            "terminal_sampling": "directly on largest strongly connected OSM driving graph",
            "time_window_basis": "road shortest-path distance at constant protocol speed",
            "elapsed_seconds": time.monotonic() - started,
        },
        "data": road_cases,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, args.output)
    manifest = {
        **payload["meta"],
        "status": "prepared",
        "output": str(args.output.resolve().relative_to(ROOT)),
        "sha256": digest(args.output),
        "cases": case_metadata,
    }
    args.output.with_suffix(".json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({key: manifest[key] for key in ("status", "samples", "elapsed_seconds", "sha256")}, indent=2))


if __name__ == "__main__":
    main()
