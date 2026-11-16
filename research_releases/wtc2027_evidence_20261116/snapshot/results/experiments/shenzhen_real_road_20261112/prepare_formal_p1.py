"""Freeze fresh P1 cases before any formal road-network result is observed."""
import hashlib
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUT = HERE / "formal_p1_v1"
GRAPH = HERE / "data/futian_luohu_current.json"
SEED = 2026111201
SAMPLES = 32

sys.path.insert(0, str(ROOT / "experiments/dispatch_clock_fix_20261106"))
from bootstrap import SNAP
sys.path.insert(0, str(SNAP / "experiments/sensitivity_20261030"))
from scenarios import make, save
sys.path.insert(0, str(HERE))
from road_network import haversine_km, load_overpass, normalized_to_lonlat, snap_points, terminal_matrices


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def to_python(sample):
    return {key: value.detach().cpu().tolist() if hasattr(value, "detach") else value for key, value in sample.items()}


def to_tensor(sample):
    result = {}
    for key, value in sample.items():
        if key in {"road_node_ids"}:
            result[key] = torch.as_tensor(value, dtype=torch.long)
        elif key in {"pd_pair_mask"}:
            result[key] = torch.as_tensor(value, dtype=torch.bool)
        elif key in {"pickup_delivery_pairs", "n_orders"}:
            result[key] = torch.as_tensor(value, dtype=torch.long)
        elif key == "road_lonlat":
            result[key] = torch.as_tensor(value, dtype=torch.float64)
        else:
            result[key] = torch.as_tensor(value, dtype=torch.float32)
    return result


def main():
    if OUT.exists():
        raise FileExistsError(f"Refusing to overwrite {OUT}")
    OUT.mkdir()
    confirmation = json.loads((ROOT / "experiments/neural_confirmation_20261107/confirmation_v1/plan.json").read_text())
    models = [model for model in confirmation["models"] if model["policy"] == "learned"]
    protocol = json.loads((HERE / "protocol.json").read_text())
    sources = [
        HERE / "prepare_formal_p1.py", HERE / "road_network.py", HERE / "road_route_engine.py",
        HERE / "greedy_hard4.py", ROOT / "experiments/hard4_lns_20261106/search.py",
    ]
    plan = {
        "status": "selection_frozen_before_generation",
        "scope": "fresh 32-case P1 Shenzhen current-road semi-synthetic external validation; no tuning on these cases",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "seed": SEED, "samples": SAMPLES, "graph_size": 50,
        "primary_arm": "road_fixed25",
        "geometry_control": "straight_fixed25_on_identical_snapped_terminals",
        "uncalibrated_sensitivity": "road_osm_speed",
        "models": models,
        "physical_config": confirmation["physical_config"],
        "normalization_profile": confirmation["normalization_profile"],
        "hard_minimum_completed_orders_per_trip": 4,
        "strong_lns_seconds": 10.0,
        "neural_repair_lns_seconds": 0.5,
        "search_seed": 1234,
        "graph_sha256": digest(GRAPH),
        "source_sha256": {str(path.relative_to(ROOT)): digest(path) for path in sources},
        "host": platform.platform(),
        "interpretation_guard": "P1 isolates road representation with synthetic requests. OSM class speeds are a proxy, not observed traffic speeds.",
    }
    (OUT / "plan_before_generation.json").write_text(json.dumps(plan, indent=2) + "\n")

    scenario_path = OUT / "fresh_synthetic_seed2026111201.pt"
    save(make(n=50, samples=SAMPLES, seed=SEED, ratio=None, tw=1.0, excess=1.0), scenario_path)
    scenario = torch.load(scenario_path, map_location="cpu", weights_only=False)
    graph = load_overpass(GRAPH)
    road_cases, diagnostics = [], []
    bbox = protocol["bbox_wgs84"]
    for index, original in enumerate(scenario["data"]):
        sample = to_python(original)
        target_lonlat = normalized_to_lonlat([sample["depot"]] + sample["loc"], bbox)
        node_ids, offsets = snap_points(graph, target_lonlat, unique=True)
        snapped_lonlat = [(graph.nodes[node]["lon"], graph.nodes[node]["lat"]) for node in node_ids]
        road_distance, road_travel = terminal_matrices(graph, node_ids)
        straight_distance = np.asarray([[haversine_km(a, b) for b in snapped_lonlat] for a in snapped_lonlat])
        normalized = [[(lon - bbox["west"]) / (bbox["east"] - bbox["west"]), (lat - bbox["south"]) / (bbox["north"] - bbox["south"])] for lon, lat in snapped_lonlat]
        sample.update(
            depot=normalized[0], loc=normalized[1:], road_node_ids=node_ids, road_lonlat=snapped_lonlat,
            straight_distance_matrix_km=straight_distance.tolist(),
            road_distance_matrix_km=road_distance.tolist(),
            road_travel_time_matrix_hours=road_travel.tolist(),
        )
        road_cases.append(to_tensor(sample))
        mask = straight_distance > 1e-10
        diagnostics.append({
            "instance": index,
            "passenger_orders": int(sum(sample["node_type"][:50])),
            "cargo_orders": int(50 - sum(sample["node_type"][:50])),
            "unique_road_nodes": len(set(node_ids)),
            "synthetic_target_to_node_km_mean": float(np.mean(offsets)),
            "synthetic_target_to_node_km_max": float(np.max(offsets)),
            "road_to_straight_median": float(np.median(road_distance[mask] / straight_distance[mask])),
        })
        print("MATRIX", index + 1, "/", SAMPLES, "unique", len(set(node_ids)), "ratio", round(diagnostics[-1]["road_to_straight_median"], 3), flush=True)
    torch.save({
        "meta": {"graph_size": 50, "samples": SAMPLES, "seed": SEED, "graph_sha256": plan["graph_sha256"], "coordinate_order": "lon_lat_wgs84"},
        "data": road_cases,
    }, OUT / "road_cases.pt")
    (OUT / "mapping_diagnostics.json").write_text(json.dumps(diagnostics, indent=2) + "\n")
    plan.update(
        status="prepared",
        scenario_sha256=digest(scenario_path),
        road_cases_sha256=digest(OUT / "road_cases.pt"),
        passenger_orders_total=sum(row["passenger_orders"] for row in diagnostics),
        cargo_orders_total=sum(row["cargo_orders"] for row in diagnostics),
        graph_nodes_largest_strong_component=graph.number_of_nodes(),
        graph_edges_largest_strong_component=graph.number_of_edges(),
    )
    (OUT / "plan.json").write_text(json.dumps(plan, indent=2) + "\n")
    print(json.dumps({key: plan[key] for key in ("status", "samples", "scenario_sha256", "road_cases_sha256", "passenger_orders_total", "cargo_orders_total")}, indent=2))


if __name__ == "__main__":
    main()
