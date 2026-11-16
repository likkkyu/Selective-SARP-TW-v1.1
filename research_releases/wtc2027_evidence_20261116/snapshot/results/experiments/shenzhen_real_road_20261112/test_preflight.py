"""CPU-only deterministic checks for road parsing and route semantics."""
import json
import math
from pathlib import Path

from road_network import graph_from_overpass, matrix_diagnostics, snap_points, terminal_matrices
from road_audit import audit_routes
from road_route_engine import RoadRouteEngine


def fixture():
    nodes = [
        {"type": "node", "id": 1, "lon": 114.000, "lat": 22.500},
        {"type": "node", "id": 2, "lon": 114.010, "lat": 22.500},
        {"type": "node", "id": 3, "lon": 114.010, "lat": 22.510},
        {"type": "node", "id": 4, "lon": 114.000, "lat": 22.510},
        {"type": "node", "id": 5, "lon": 114.005, "lat": 22.505},
    ]
    ways = [
        {"type": "way", "id": 10, "nodes": [1, 2, 3, 4, 1], "tags": {"highway": "primary", "maxspeed": "36"}},
        {"type": "way", "id": 11, "nodes": [1, 5, 3], "tags": {"highway": "residential", "oneway": "yes", "maxspeed": "18"}},
        {"type": "way", "id": 12, "nodes": [2, 5, 4], "tags": {"highway": "residential", "maxspeed": "18"}},
    ]
    return {"elements": nodes + ways}


def sample(distance, travel):
    # Four passenger orders. Pickup nodes 1..4, delivery nodes 5..8.
    return {
        "depot": [0.0, 0.0],
        "loc": [[0.0, 0.0]] * 8,
        "node_type": [1] * 8,
        "demand_passenger": [1 / 15] * 4 + [-1 / 15] * 4,
        "demand_cargo": [0.0] * 8,
        "time_windows": [[10.0, 15.0]] * 4 + [[10.0, 16.0]] * 4,
        "road_distance_matrix_km": distance,
        "road_travel_time_matrix_hours": travel,
    }


CONFIG = {
    "OPERATION_START": 10.0, "OPERATION_END": 16.0, "MAX_TRIP_TIME": 3.0, "SERVICE_TIME": 0.05,
    "PASSENGER_MAX_RIDE_TIME_MINUTES": 70.0, "PASSENGER_MAX_EXCESS_RIDE_TIME_MINUTES": 30.0,
    "HARD_CARGO_PICKUP_TIMEWINDOW": True, "PASSENGER_CAPACITY": 15, "CARGO_CAPACITY": 20,
    "PASSENGER_WEIGHT_KG": 65.0, "CARGO_UNIT_WEIGHT_KG": 1.0, "ENERGY_BASE": 0.18,
    "ENERGY_WEIGHT_REF": 10000.0, "ELECTRICITY_PRICE": 1.0, "PASSENGER_DELAY_COST": 0.6,
    "CARGO_DELAY_COST": 0.06, "CARGO_DELAY_TIER1_MIN": 30.0, "CARGO_DELAY_TIER2_MIN": 60.0,
    "CARGO_DELAY_COST_30_60": 0.18, "CARGO_DELAY_COST_60_PLUS": 0.36, "VEHICLE_COST": 20.0,
    "ALPHA_ENERGY": 1.0, "ALPHA_DELAY": 2.5, "ALPHA_VEHICLE": 3.0, "ALPHA_REJECT": 575.0,
}
PROFILE = {"energy_cost_raw": 98.93, "passenger_delivery_delay_cost_raw": 576.49, "cargo_delay_cost_raw": 42.86, "vehicle_cost_norm": 180.0}


def main():
    graph = graph_from_overpass(fixture())
    assert graph.number_of_nodes() == 5
    assert graph.has_edge(1, 5) and not graph.has_edge(5, 1), "oneway semantics were lost"
    points = [(114.0001, 22.5001), (114.0099, 22.5001), (114.0099, 22.5099), (114.0001, 22.5099)]
    snapped, offsets = snap_points(graph, points)
    assert snapped == [1, 2, 3, 4]
    distance4, travel4 = terminal_matrices(graph, snapped)
    diag = matrix_diagnostics(distance4, travel4, points)
    assert diag["road_to_straight_median"] >= 1.0
    # Expand depot + eight service nodes by repeating connected terminals.
    terminals = [1, 1, 2, 3, 4, 3, 4, 1, 2]
    distance, travel = terminal_matrices(graph, terminals)
    engine = RoadRouteEngine(sample(distance.tolist(), travel.tolist()), CONFIG, PROFILE, vehicles=1)
    route = (1, 2, 3, 4, 5, 6, 7, 8)
    metrics = engine.route(route)
    assert metrics is not None and metrics["orders"] == (1, 2, 3, 4)
    assert engine.cost([route]) is not None
    audit = audit_routes(sample(distance.tolist(), travel.tolist()), [route], CONFIG, vehicles=1)
    assert audit["valid"] and audit["completed"] == 4
    assert engine.cost([(1, 5)]) is None, "hard minimum of four completed orders was not enforced"
    short_audit = audit_routes(sample(distance.tolist(), travel.tolist()), [(1, 5)], CONFIG, vehicles=1)
    assert not short_audit["valid"] and any(value["kind"] == "min_completed_orders" for value in short_audit["violations"])
    bad = sample(distance.tolist(), travel.tolist())
    # Slow the actually executed 4->5 arc while keeping the direct 1->5
    # reference unchanged; order 1 must then violate its ride-time ceiling.
    bad["road_travel_time_matrix_hours"][4][5] = 2.0
    bad_engine = RoadRouteEngine(bad, CONFIG, PROFILE, vehicles=1)
    assert bad_engine.route(route) is None, "matrix travel time did not affect passenger feasibility"
    report = {
        "status": "pass",
        "graph_nodes": graph.number_of_nodes(),
        "graph_edges": graph.number_of_edges(),
        "snap_offsets_km_max": max(offsets),
        "matrix": diag,
        "route_distance_km": metrics["distance"],
    }
    output = Path(__file__).with_name("preflight_unit_report.json")
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
