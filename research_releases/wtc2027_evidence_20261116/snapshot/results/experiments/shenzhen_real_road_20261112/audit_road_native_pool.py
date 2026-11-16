"""Independent structural audit for a frozen road-native scenario pool."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pool", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--model-lock", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output}")

    pool = torch.load(args.pool, map_location="cpu", weights_only=True)
    manifest = json.loads(args.manifest.read_text())
    meta = pool.get("meta", {})
    cases = pool.get("data", [])
    errors: list[str] = []
    per_case = []

    def require(condition: bool, message: str) -> None:
        if not condition:
            errors.append(message)

    require(meta.get("purpose") == "formal_test", "pool purpose is not formal_test")
    require(manifest.get("purpose") == "formal_test", "manifest purpose is not formal_test")
    require(manifest.get("sha256") == digest(args.pool), "pool hash differs from manifest")
    require(meta.get("model_lock_sha256") == digest(args.model_lock), "model lock hash mismatch")
    require(len(cases) == int(meta.get("samples", -1)), "case count differs from pool metadata")
    require(len(cases) == len(manifest.get("cases", [])), "case count differs from manifest")

    speed = float(meta.get("matrix_speed_kph", 0.0))
    require(speed > 0.0, "matrix speed must be positive")
    for index, sample in enumerate(cases):
        loc = sample["loc"].detach().cpu().numpy()
        depot = sample["depot"].detach().cpu().numpy()
        node_ids = sample["road_node_ids"].detach().cpu().numpy()
        lonlat = sample["road_lonlat"].detach().cpu().numpy()
        distance = sample["road_distance_matrix_km"].detach().cpu().numpy()
        travel = sample["road_travel_time_matrix_hours"].detach().cpu().numpy()
        windows = sample["time_windows"].detach().cpu().numpy()
        n = int(sample["n_orders"].item())
        size = 2 * n + 1
        case_errors = []

        def case_require(condition: bool, message: str) -> None:
            if not condition:
                case_errors.append(message)

        case_require(n == 50, f"n_orders={n}, expected 50")
        case_require(loc.shape == (2 * n, 2), f"loc shape {loc.shape}")
        case_require(depot.shape == (2,), f"depot shape {depot.shape}")
        case_require(node_ids.shape == (size,), f"road_node_ids shape {node_ids.shape}")
        case_require(lonlat.shape == (size, 2), f"road_lonlat shape {lonlat.shape}")
        case_require(distance.shape == (size, size), f"distance shape {distance.shape}")
        case_require(travel.shape == (size, size), f"travel shape {travel.shape}")
        case_require(windows.shape == (2 * n, 2), f"time-window shape {windows.shape}")
        case_require(len(set(map(int, node_ids.tolist()))) == size, "road terminals are not unique")
        case_require(np.isfinite(distance).all() and (distance >= 0).all(), "invalid distance matrix")
        case_require(np.isfinite(travel).all() and (travel >= 0).all(), "invalid travel matrix")
        case_require(np.allclose(np.diag(distance), 0.0, atol=1e-6), "nonzero distance diagonal")
        case_require(np.allclose(np.diag(travel), 0.0, atol=1e-6), "nonzero travel diagonal")
        if speed > 0:
            case_require(np.allclose(travel, distance / speed, rtol=1e-5, atol=1e-6), "travel != distance/speed")
        case_require(np.isfinite(loc).all() and np.isfinite(depot).all(), "non-finite normalized coordinate")
        case_require(((loc >= 0) & (loc <= 1)).all() and ((depot >= 0) & (depot <= 1)).all(), "normalized coordinate outside [0,1]")
        case_require(np.isfinite(windows).all(), "non-finite time window")
        case_require((windows[:, 1] >= windows[:, 0]).all(), "negative time-window width")

        case_meta = manifest["cases"][index]
        case_require(case_meta.get("instance") == index, "manifest instance index mismatch")
        case_require(case_meta.get("unique_road_nodes") == size, "manifest unique-node count mismatch")
        case_require(case_meta.get("band_fallbacks") == 0, "distance-band fallback present")
        pair_info = case_meta.get("pair_info", [])
        case_require(len(pair_info) == n, f"pair_info count {len(pair_info)}")
        for pair in pair_info:
            order = int(pair["order"])
            if not 0 <= order < n:
                case_errors.append(f"invalid pair order {order}")
                continue
            direct = float(distance[1 + order, 1 + n + order])
            case_require(abs(direct - float(pair["distance_km"])) <= 2e-4, f"pair {order} distance mismatch")
            case_require(not bool(pair.get("band_fallback")), f"pair {order} used band fallback")

        if case_errors:
            errors.extend(f"case {index}: {message}" for message in case_errors)
        per_case.append({
            "instance": index,
            "valid": not case_errors,
            "unique_road_nodes": int(len(set(map(int, node_ids.tolist())))),
            "mean_direct_pair_distance_km": float(np.mean([distance[1 + i, 1 + n + i] for i in range(n)])),
            "mean_directed_terminal_distance_km": float(distance.mean()),
            "asymmetric_terminal_pairs": int((np.abs(distance - distance.T) > 1e-4).sum()),
        })

    result = {
        "status": "passed" if not errors else "failed",
        "scope": "independent structural audit of frozen Shenzhen road-native formal-test pool",
        "pool": str(args.pool.resolve().relative_to(ROOT)),
        "pool_sha256": digest(args.pool),
        "manifest": str(args.manifest.resolve().relative_to(ROOT)),
        "manifest_sha256": digest(args.manifest),
        "model_lock_sha256": digest(args.model_lock),
        "cases": len(cases),
        "all_cases_valid": not errors,
        "band_fallbacks": int(sum(case.get("band_fallbacks", 0) for case in manifest.get("cases", []))),
        "minimum_unique_road_nodes": min((row["unique_road_nodes"] for row in per_case), default=0),
        "mean_direct_pair_distance_km": float(np.mean([row["mean_direct_pair_distance_km"] for row in per_case])),
        "mean_directed_terminal_distance_km": float(np.mean([row["mean_directed_terminal_distance_km"] for row in per_case])),
        "cases_with_asymmetric_road_distances": int(sum(row["asymmetric_terminal_pairs"] > 0 for row in per_case)),
        "errors": errors,
        "per_case": per_case,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({key: result[key] for key in (
        "status", "cases", "all_cases_valid", "band_fallbacks",
        "minimum_unique_road_nodes", "mean_direct_pair_distance_km",
        "cases_with_asymmetric_road_distances",
    )}, indent=2))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
