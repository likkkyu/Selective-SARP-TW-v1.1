"""Prepare road-matrix training/validation cases disjoint from frozen P1 tests."""

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
GRAPH = HERE / "data" / "futian_luohu_current.json"
FORMAL = HERE / "formal_p1_v1"
sys.path.insert(0, str(ROOT / "experiments/dispatch_clock_fix_20261106"))
from bootstrap import SNAP
sys.path.insert(0, str(SNAP / "experiments/sensitivity_20261030"))
from scenarios import make
sys.path.insert(0, str(HERE))
from road_network import load_overpass, normalized_to_lonlat, snap_points, terminal_matrices


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def to_python(sample):
    return {key: value.detach().cpu().tolist() if hasattr(value, "detach") else value for key, value in sample.items()}


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=128)
    parser.add_argument("--seed", type=int, default=2026111301)
    parser.add_argument("--output", type=Path, default=HERE / "road_training_pool_seed2026111301.pt")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output}")
    if args.samples < 2:
        raise ValueError("at least two cases are required")

    protocol = json.loads((HERE / "protocol.json").read_text())
    formal_plan = json.loads((FORMAL / "plan.json").read_text())
    generated = make(n=50, samples=args.samples, seed=args.seed, ratio=None, tw=1.0, excess=1.0)
    graph = load_overpass(GRAPH)
    bbox = protocol["bbox_wgs84"]
    road_cases = []
    mapping = []
    started = time.monotonic()
    for index, original in enumerate(generated["data"]):
        sample = to_python(original)
        target_lonlat = normalized_to_lonlat([sample["depot"]] + sample["loc"], bbox)
        node_ids, offsets = snap_points(graph, target_lonlat, unique=True)
        snapped_lonlat = [(graph.nodes[node]["lon"], graph.nodes[node]["lat"]) for node in node_ids]
        road_distance, _ = terminal_matrices(graph, node_ids)
        normalized = [
            [
                (lon - bbox["west"]) / (bbox["east"] - bbox["west"]),
                (lat - bbox["south"]) / (bbox["north"] - bbox["south"]),
            ]
            for lon, lat in snapped_lonlat
        ]
        sample.update(
            depot=normalized[0],
            loc=normalized[1:],
            road_node_ids=node_ids,
            road_lonlat=snapped_lonlat,
            road_distance_matrix_km=road_distance.tolist(),
            road_travel_time_matrix_hours=(road_distance / formal_plan["physical_config"]["VEHICLE_SPEED"]).tolist(),
        )
        road_cases.append(to_tensor(sample))
        mapping.append(
            {
                "instance": index,
                "unique_road_nodes": len(set(node_ids)),
                "snap_offset_km_mean": float(np.mean(offsets)),
                "snap_offset_km_max": float(np.max(offsets)),
            }
        )
        print(
            "POOL",
            index + 1,
            "/",
            args.samples,
            "elapsed_s",
            round(time.monotonic() - started, 1),
            flush=True,
        )

    payload = {
        "meta": {
            "scope": "road-aware neural training/validation only; never formal P1 testing",
            "seed": args.seed,
            "samples": args.samples,
            "graph_size": 50,
            "graph_sha256": digest(GRAPH),
            "formal_test_road_cases_sha256": formal_plan["road_cases_sha256"],
            "formal_test_seed": formal_plan["seed"],
            "matrix_speed_kph": formal_plan["physical_config"]["VEHICLE_SPEED"],
            "elapsed_seconds": time.monotonic() - started,
        },
        "data": road_cases,
    }
    torch.save(payload, args.output)
    try:
        output_label = str(args.output.resolve().relative_to(ROOT))
    except ValueError:
        output_label = str(args.output.resolve())
    manifest = {
        **payload["meta"],
        "status": "prepared",
        "output": output_label,
        "sha256": digest(args.output),
        "mapping": mapping,
    }
    args.output.with_suffix(".json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({key: manifest[key] for key in ("status", "samples", "elapsed_seconds", "sha256")}, indent=2))


if __name__ == "__main__":
    main()
