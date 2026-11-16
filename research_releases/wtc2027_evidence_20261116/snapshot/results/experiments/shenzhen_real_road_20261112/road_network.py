"""Minimal OSM/Overpass road graph and shortest-path matrix utilities.

The implementation deliberately avoids osmnx so the preflight runs in the
project's existing CPU environment. Coordinates are always (lon, lat), WGS84.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import networkx as nx
import numpy as np
from scipy.spatial import cKDTree


EXCLUDED_HIGHWAYS = {
    "bridleway", "construction", "corridor", "cycleway", "footway",
    "path", "pedestrian", "proposed", "raceway", "steps",
}

DEFAULT_SPEED_KPH = {
    "motorway": 80.0,
    "motorway_link": 50.0,
    "trunk": 60.0,
    "trunk_link": 45.0,
    "primary": 50.0,
    "primary_link": 40.0,
    "secondary": 40.0,
    "secondary_link": 35.0,
    "tertiary": 35.0,
    "tertiary_link": 30.0,
    "residential": 25.0,
    "living_street": 15.0,
    "service": 15.0,
    "unclassified": 25.0,
    "road": 25.0,
}


def haversine_km(a: Sequence[float], b: Sequence[float]) -> float:
    lon1, lat1 = map(math.radians, a)
    lon2, lat2 = map(math.radians, b)
    dlon, dlat = lon2 - lon1, lat2 - lat1
    value = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    return 6371.0088 * 2 * math.asin(min(1.0, math.sqrt(value)))


def _first(value):
    return value[0] if isinstance(value, list) and value else value


def parse_maxspeed_kph(value, fallback: float) -> float:
    value = _first(value)
    if value is None:
        return fallback
    text = str(value).strip().lower()
    try:
        number = float(text.split()[0])
        if "mph" in text:
            number *= 1.609344
        return number if 3.0 <= number <= 160.0 else fallback
    except (TypeError, ValueError):
        return fallback


def _add_min_edge(graph: nx.DiGraph, u: int, v: int, length_km: float, speed_kph: float, highway: str) -> None:
    candidate = {
        "length_km": float(length_km),
        "speed_kph": float(speed_kph),
        "travel_time_h": float(length_km / speed_kph),
        "highway": highway,
    }
    old = graph.get_edge_data(u, v)
    if old is None or candidate["travel_time_h"] < old["travel_time_h"]:
        graph.add_edge(u, v, **candidate)


def graph_from_overpass(payload: Mapping) -> nx.DiGraph:
    elements = payload.get("elements", [])
    nodes = {
        int(e["id"]): (float(e["lon"]), float(e["lat"]))
        for e in elements if e.get("type") == "node" and "lon" in e and "lat" in e
    }
    graph = nx.DiGraph()
    for node_id, (lon, lat) in nodes.items():
        graph.add_node(node_id, lon=lon, lat=lat)
    for way in (e for e in elements if e.get("type") == "way"):
        tags = way.get("tags", {})
        highway = str(_first(tags.get("highway", "")))
        if not highway or highway in EXCLUDED_HIGHWAYS:
            continue
        if tags.get("access") in {"no", "private"} or tags.get("motor_vehicle") in {"no", "private"}:
            continue
        ids = [int(v) for v in way.get("nodes", []) if int(v) in nodes]
        if len(ids) < 2:
            continue
        fallback = DEFAULT_SPEED_KPH.get(highway, 25.0)
        speed = parse_maxspeed_kph(tags.get("maxspeed"), fallback)
        oneway = str(tags.get("oneway", "")).lower()
        reverse_only = oneway == "-1"
        forward_only = oneway in {"yes", "1", "true", "-1"} or tags.get("junction") == "roundabout"
        for u, v in zip(ids, ids[1:]):
            length = haversine_km(nodes[u], nodes[v])
            if reverse_only:
                _add_min_edge(graph, v, u, length, speed, highway)
            else:
                _add_min_edge(graph, u, v, length, speed, highway)
                if not forward_only:
                    _add_min_edge(graph, v, u, length, speed, highway)
    isolates = list(nx.isolates(graph))
    graph.remove_nodes_from(isolates)
    if graph.number_of_nodes() == 0:
        raise ValueError("Overpass response contains no driveable connected road nodes")
    components = list(nx.strongly_connected_components(graph))
    largest = max(components, key=len)
    return graph.subgraph(largest).copy()


def load_overpass(path: Path | str) -> nx.DiGraph:
    with Path(path).open() as handle:
        return graph_from_overpass(json.load(handle))


def normalized_to_lonlat(points: Sequence[Sequence[float]], bbox: Mapping[str, float]) -> list[tuple[float, float]]:
    west, east = float(bbox["west"]), float(bbox["east"])
    south, north = float(bbox["south"]), float(bbox["north"])
    result = []
    for x, y in points:
        if not (0.0 <= float(x) <= 1.0 and 0.0 <= float(y) <= 1.0):
            raise ValueError("Normalized coordinates must be within [0, 1]")
        result.append((west + float(x) * (east - west), south + float(y) * (north - south)))
    return result


def snap_points(graph: nx.DiGraph, points_lonlat: Sequence[Sequence[float]], unique: bool = False) -> tuple[list[int], list[float]]:
    ids = list(graph.nodes)
    coords = np.asarray([(graph.nodes[i]["lon"], graph.nodes[i]["lat"]) for i in ids], dtype=float)
    mean_lat = math.radians(float(coords[:, 1].mean()))
    scale = np.asarray([111.320 * math.cos(mean_lat), 110.574])
    tree = cKDTree(coords * scale)
    targets = np.asarray(points_lonlat, dtype=float) * scale
    if not unique:
        distances, positions = tree.query(targets, k=1)
        return [ids[int(i)] for i in positions], [float(v) for v in distances]
    if len(points_lonlat) > len(ids):
        raise ValueError("Cannot assign unique road nodes: more terminals than graph nodes")
    chosen, offsets, used = [], [], set()
    k = min(16, len(ids))
    for target in targets:
        while True:
            distances, positions = tree.query(target, k=k)
            distances = np.atleast_1d(distances)
            positions = np.atleast_1d(positions)
            available = [(float(d), int(p)) for d, p in zip(distances, positions) if int(p) not in used]
            if available:
                distance, position = min(available)
                used.add(position)
                chosen.append(ids[position])
                offsets.append(distance)
                break
            if k == len(ids):
                raise RuntimeError("Unique road-node assignment failed")
            k = min(len(ids), k * 2)
    return chosen, offsets


def terminal_matrices(graph: nx.DiGraph, terminal_ids: Sequence[int]) -> tuple[np.ndarray, np.ndarray]:
    terminals = list(map(int, terminal_ids))
    size = len(terminals)
    distance = np.full((size, size), np.inf, dtype=float)
    travel = np.full((size, size), np.inf, dtype=float)
    unique = sorted(set(terminals))
    distance_maps = {
        u: nx.single_source_dijkstra_path_length(graph, u, weight="length_km") for u in unique
    }
    travel_maps = {
        u: nx.single_source_dijkstra_path_length(graph, u, weight="travel_time_h") for u in unique
    }
    for i, u in enumerate(terminals):
        for j, v in enumerate(terminals):
            distance[i, j] = distance_maps[u].get(v, math.inf)
            travel[i, j] = travel_maps[u].get(v, math.inf)
    if not np.isfinite(distance).all() or not np.isfinite(travel).all():
        raise ValueError("At least one snapped terminal pair is disconnected")
    return distance, travel


def matrix_diagnostics(distance: np.ndarray, travel: np.ndarray, points_lonlat: Sequence[Sequence[float]]) -> dict:
    straight = np.asarray([
        [haversine_km(a, b) for b in points_lonlat] for a in points_lonlat
    ])
    mask = straight > 1e-8
    ratios = distance[mask] / straight[mask]
    speed = distance[travel > 1e-10] / travel[travel > 1e-10]
    return {
        "terminals": int(distance.shape[0]),
        "unique_locations": int(np.unique(np.asarray(points_lonlat), axis=0).shape[0]),
        "distance_km_min": float(distance[mask].min()) if mask.any() else 0.0,
        "distance_km_mean": float(distance[mask].mean()) if mask.any() else 0.0,
        "distance_km_max": float(distance.max()),
        "road_to_straight_median": float(np.median(ratios)) if ratios.size else 1.0,
        "road_to_straight_p95": float(np.quantile(ratios, 0.95)) if ratios.size else 1.0,
        "implied_speed_kph_median": float(np.median(speed)) if speed.size else 0.0,
        "asymmetric_distance_pairs": int(np.count_nonzero(np.abs(distance - distance.T) > 1e-8) // 2),
        "asymmetric_time_pairs": int(np.count_nonzero(np.abs(travel - travel.T) > 1e-8) // 2),
    }

