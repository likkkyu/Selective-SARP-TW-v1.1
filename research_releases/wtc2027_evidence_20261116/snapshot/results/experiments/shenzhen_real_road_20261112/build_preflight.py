"""Build one paired Euclidean/real-road N50 CPU preflight."""
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "experiments/hard4_lns_20261106"))

from greedy_hard4 import make_start
from road_network import haversine_km, load_overpass, matrix_diagnostics, normalized_to_lonlat, snap_points, terminal_matrices
from road_route_engine import RoadRouteEngine
from search import solve


DEFAULT_GRAPH = HERE / "data/futian_luohu_current.json"
DEFAULT_SCENARIO = ROOT / "experiments/neural_confirmation_20261107/confirmation_v1/common_n50_seed20711369.pt"
DEFAULT_PLAN = ROOT / "experiments/neural_confirmation_20261107/confirmation_v1/plan.json"


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def python_sample(sample):
    return {key: value.detach().cpu().tolist() if hasattr(value, "detach") else value for key, value in sample.items()}


def evaluate(engine, lns_seconds, seed):
    begin = time.monotonic()
    initial = make_start(engine)
    initial_seconds = time.monotonic() - begin
    initial_cost = engine.cost(initial)
    result = solve(engine, initial, seed=seed, seconds=lns_seconds, max_iterations=1000)
    return {
        "initial": {"routes": [list(route) for route in initial], "cost": initial_cost, "seconds": initial_seconds},
        "lns": {
            "routes": [list(route) for route in result["routes"]],
            "cost": result["cost"],
            "search_seconds": result["seconds"],
            "iterations": result["iterations"],
            "route_evaluations": result["route_evaluations"],
            "partial_last_iteration_discarded": result["partial_last_iteration_discarded"],
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--graph", type=Path, default=DEFAULT_GRAPH)
    parser.add_argument("--scenario", type=Path, default=DEFAULT_SCENARIO)
    parser.add_argument("--plan", type=Path, default=DEFAULT_PLAN)
    parser.add_argument("--instance", type=int, default=0)
    parser.add_argument("--lns-seconds", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--output", type=Path, default=HERE / "outputs/preflight_instance000")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output directory already exists; preflights are immutable")
    protocol = json.loads((HERE / "protocol.json").read_text())
    plan = json.loads(args.plan.read_text())
    archive = torch.load(args.scenario, map_location="cpu", weights_only=True)
    data = archive["data"] if isinstance(archive, dict) and "data" in archive else archive
    if not 0 <= args.instance < len(data):
        parser.error("Instance index is outside the scenario archive")
    sample = python_sample(data[args.instance])
    points = [sample["depot"]] + sample["loc"]
    lonlat = normalized_to_lonlat(points, protocol["bbox_wgs84"])

    graph_begin = time.monotonic()
    graph = load_overpass(args.graph)
    graph_seconds = time.monotonic() - graph_begin
    terminal_ids, snap_offsets = snap_points(graph, lonlat, unique=True)
    snapped_lonlat = [(graph.nodes[node]["lon"], graph.nodes[node]["lat"]) for node in terminal_ids]
    matrix_begin = time.monotonic()
    road_distance, road_travel = terminal_matrices(graph, terminal_ids)
    matrix_seconds = time.monotonic() - matrix_begin
    straight_distance = np.asarray([[haversine_km(a, b) for b in snapped_lonlat] for a in snapped_lonlat])
    straight_travel = straight_distance / float(plan["physical_config"]["VEHICLE_SPEED"])
    bbox = protocol["bbox_wgs84"]
    normalized_snapped = [
        [(lon - bbox["west"]) / (bbox["east"] - bbox["west"]), (lat - bbox["south"]) / (bbox["north"] - bbox["south"])]
        for lon, lat in snapped_lonlat
    ]
    sample["depot"] = normalized_snapped[0]
    sample["loc"] = normalized_snapped[1:]
    sample["road_distance_matrix_km"] = road_distance.tolist()
    sample["road_travel_time_matrix_hours"] = road_travel.tolist()
    sample["road_node_ids"] = terminal_ids
    sample["road_lonlat"] = snapped_lonlat

    straight_sample = dict(sample)
    straight_sample["road_distance_matrix_km"] = straight_distance.tolist()
    straight_sample["road_travel_time_matrix_hours"] = straight_travel.tolist()
    road_fixed_sample = dict(sample)
    road_fixed_sample["road_travel_time_matrix_hours"] = (road_distance / float(plan["physical_config"]["VEHICLE_SPEED"])).tolist()
    straight_engine = RoadRouteEngine(straight_sample, plan["physical_config"], plan["normalization_profile"])
    road_fixed_engine = RoadRouteEngine(road_fixed_sample, plan["physical_config"], plan["normalization_profile"])
    road_speed_engine = RoadRouteEngine(sample, plan["physical_config"], plan["normalization_profile"])
    straight_result = evaluate(straight_engine, args.lns_seconds, args.seed)
    road_fixed_result = evaluate(road_fixed_engine, args.lns_seconds, args.seed)
    road_speed_result = evaluate(road_speed_engine, args.lns_seconds, args.seed)
    mask = straight_distance > 1e-10
    ratio = road_distance[mask] / straight_distance[mask]
    report = {
        "status": "pass",
        "scope": "one unused N50 scenario mapped to current Shenzhen OSM; CPU preflight, not formal evidence",
        "instance": args.instance,
        "scenario_sha256": sha256(args.scenario),
        "graph_sha256": sha256(args.graph),
        "graph": {
            "nodes_largest_strong_component": graph.number_of_nodes(),
            "edges_largest_strong_component": graph.number_of_edges(),
            "load_seconds": graph_seconds,
            "matrix_seconds": matrix_seconds,
            "unique_snapped_nodes": len(set(terminal_ids)),
            "snap_offset_km_mean": float(np.mean(snap_offsets)),
            "snap_offset_km_max": float(np.max(snap_offsets)),
        },
        "road_matrix": matrix_diagnostics(road_distance, road_travel, snapped_lonlat),
        "paired_geometry": {
            "road_to_original_euclidean_median": float(np.median(ratio)),
            "road_to_original_euclidean_p95": float(np.quantile(ratio, 0.95)),
        },
        "straight_fixed25": straight_result,
        "road_fixed25": road_fixed_result,
        "road_osm_speed": road_speed_result,
        "frozen": {
            "physical_config": plan["physical_config"],
            "normalization_profile": plan["normalization_profile"],
            "lns_seconds": args.lns_seconds,
            "lns_seed": args.seed,
            "hard_minimum": 4,
        },
    }
    for arm in (report["straight_fixed25"], report["road_fixed25"], report["road_osm_speed"]):
        if arm["initial"]["cost"]["completed"] < 4 * arm["initial"]["cost"]["vehicles"]:
            raise AssertionError("Hard-four certificate failed for insertion")
        if arm["lns"]["cost"]["completed"] < 4 * arm["lns"]["cost"]["vehicles"]:
            raise AssertionError("Hard-four certificate failed for LNS")
    args.output.mkdir(parents=True)
    torch.save({"sample": sample, "metadata": {"graph_sha256": report["graph_sha256"], "scenario_sha256": report["scenario_sha256"]}}, args.output / "road_instance.pt")
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({
        "output": str(args.output),
        "graph": report["graph"],
        "geometry": report["paired_geometry"],
        "straight_fixed25_lns": report["straight_fixed25"]["lns"]["cost"],
        "road_fixed25_lns": report["road_fixed25"]["lns"]["cost"],
        "road_osm_speed_lns": report["road_osm_speed"]["lns"]["cost"],
    }, indent=2))


if __name__ == "__main__":
    main()
