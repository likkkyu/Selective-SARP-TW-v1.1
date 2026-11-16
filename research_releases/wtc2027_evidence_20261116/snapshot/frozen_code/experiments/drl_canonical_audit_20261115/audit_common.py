"""Solver-free canonical route audit helpers.

This module intentionally duplicates the small deterministic cost/audit adapter
used by the SCIP comparison, but does not import PySCIPOpt or any optimizer.
The formulas remain cross-checked against the tensor scorer and independent
route engine in ``verify_actions``.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
import sys

import torch


ROOT = Path(__file__).resolve().parents[2]
SNAP = ROOT / "experiments/evaluation_alignment_20261103/snapshot"
for folder in reversed(
    (
        ROOT / "experiments/hard4_confirmation_20261106",
        ROOT / "experiments/dispatch_clock_fix_20261106",
        ROOT / "experiments/hard4_speed_20261106",
        ROOT / "experiments/hard4_lns_20261106",
        SNAP / "experiments/reject_ablation_20261029",
        SNAP / "experiments/sensitivity_20261030",
        SNAP,
    )
):
    sys.path.insert(0, str(folder))

torch.set_num_threads(1)
from baseline_utils import sample_to_batch  # noqa: E402
from problem_mcvrptw_v2 import Config, MCVRPPDTW  # noqa: E402
from route_audit import audit  # noqa: E402


def lists(sample):
    return {
        key: value.detach().cpu().tolist() if hasattr(value, "detach") else value
        for key, value in sample.items()
    }


def parameters(config):
    names = (
        "OPERATION_START",
        "OPERATION_END",
        "AREA_SIZE",
        "VEHICLE_SPEED",
        "MAX_TRIP_TIME",
        "SERVICE_TIME",
        "PASSENGER_MAX_RIDE_TIME_MINUTES",
        "PASSENGER_MAX_EXCESS_RIDE_TIME_MINUTES",
        "HARD_CARGO_PICKUP_TIMEWINDOW",
        "PASSENGER_CAPACITY",
        "CARGO_CAPACITY",
        "PASSENGER_WEIGHT_KG",
        "CARGO_UNIT_WEIGHT_KG",
        "ENERGY_BASE",
        "ENERGY_WEIGHT_REF",
        "ELECTRICITY_PRICE",
        "PASSENGER_DELAY_COST",
        "CARGO_DELAY_COST",
        "CARGO_DELAY_TIER1_MIN",
        "CARGO_DELAY_TIER2_MIN",
        "CARGO_DELAY_COST_30_60",
        "CARGO_DELAY_COST_60_PLUS",
        "VEHICLE_COST",
        "ALPHA_ENERGY",
        "ALPHA_DELAY",
        "ALPHA_VEHICLE",
        "ALPHA_REJECT",
        "ALPHA_UNFULFILLED",
        "ALPHA_TRIP_OVERTIME",
    )
    return {name: getattr(config, name) for name in names}


def profile(graph_size):
    return json.loads(
        (SNAP / f"outputs/normalization_profiles/normalization_n{graph_size}.json").read_text()
    )


def _arc_matrices(sample, config):
    n = len(sample["loc"]) // 2
    nodes = list(range(2 * n + 1))
    if "road_distance_matrix_km" in sample and "road_travel_time_matrix_hours" in sample:
        distances = sample["road_distance_matrix_km"]
        travel = sample["road_travel_time_matrix_hours"]
        expected = 2 * n + 1
        if (
            len(distances) != expected
            or len(travel) != expected
            or any(len(row) != expected for row in distances + travel)
        ):
            raise ValueError("Road matrices must be square and include every node")
        distance_dict = {
            (i, j): float(distances[i][j]) for i in nodes for j in nodes if i != j
        }
        travel_dict = {(i, j): float(travel[i][j]) for i in nodes for j in nodes if i != j}
        if any(
            not math.isfinite(value) or value < 0
            for value in (*distance_dict.values(), *travel_dict.values())
        ):
            raise ValueError("Road matrices must be finite and nonnegative")
        return distance_dict, travel_dict
    coordinates = [sample["depot"]] + sample["loc"]
    distance_dict = {
        (i, j): math.dist(coordinates[i], coordinates[j]) * config["AREA_SIZE"]
        for i in nodes
        for j in nodes
        if i != j
    }
    return distance_dict, {
        edge: value / config["VEHICLE_SPEED"] for edge, value in distance_dict.items()
    }


def scalar_cost(sample, actions, config, normalization_profile):
    sample = lists(sample)
    n = len(sample["loc"]) // 2
    distance_matrix, travel_matrix = _arc_matrices(sample, config)
    current = 0
    current_time = config["OPERATION_START"]
    passenger_load = cargo_load = 0.0
    energy = distance = passenger_delay = cargo_delay = 0.0
    dispatches = rejected = 0
    for node in actions:
        if node == 2 * n + 1:
            rejected += 1
            continue
        if node == 0 and current == 0:
            continue
        km = distance_matrix[current, node]
        travel = travel_matrix[current, node]
        energy += (
            km
            * config["ENERGY_BASE"]
            * (
                1
                + (
                    passenger_load
                    * config["PASSENGER_CAPACITY"]
                    * config["PASSENGER_WEIGHT_KG"]
                    + cargo_load * config["CARGO_CAPACITY"] * config["CARGO_UNIT_WEIGHT_KG"]
                )
                / config["ENERGY_WEIGHT_REF"]
            )
            * config["ELECTRICITY_PRICE"]
        )
        distance += km
        if current == 0:
            dispatches += 1
            current_time = max(
                config["OPERATION_START"], sample["time_windows"][node - 1][0] - travel
            )
        arrival = current_time + travel
        if node == 0:
            current = 0
            current_time = config["OPERATION_START"]
            passenger_load = cargo_load = 0.0
            continue
        index = node - 1
        if node > n:
            late = max(0.0, arrival - sample["time_windows"][index][1]) * 60
            if sample["node_type"][index] == 1:
                passenger_delay += late
            else:
                cargo_delay += late
        current_time = max(arrival, sample["time_windows"][index][0]) + config["SERVICE_TIME"]
        passenger_load += sample["demand_passenger"][index]
        cargo_load += sample["demand_cargo"][index]
        current = node
    passenger_cost = passenger_delay * config["PASSENGER_DELAY_COST"]
    tier1 = config["CARGO_DELAY_TIER1_MIN"]
    tier2 = config["CARGO_DELAY_TIER2_MIN"]
    cargo_cost = (
        min(cargo_delay, tier1) * config["CARGO_DELAY_COST"]
        + max(0.0, min(cargo_delay, tier2) - tier1) * config["CARGO_DELAY_COST_30_60"]
        + max(0.0, cargo_delay - tier2) * config["CARGO_DELAY_COST_60_PLUS"]
    )
    vehicle_cost = dispatches * config["VEHICLE_COST"]
    reject_penalty = rejected * config["ALPHA_REJECT"]
    objective = (
        config["ALPHA_ENERGY"] * energy / normalization_profile["energy_cost_raw"]
        + config["ALPHA_DELAY"]
        * (
            passenger_cost / normalization_profile["passenger_delivery_delay_cost_raw"]
            + cargo_cost / normalization_profile["cargo_delay_cost_raw"]
        )
        + config["ALPHA_VEHICLE"]
        * vehicle_cost
        / max(1.0, normalization_profile["vehicle_cost_norm"])
        + reject_penalty
    )
    return {
        "objective": objective,
        "raw_cost": energy + passenger_cost + cargo_cost + vehicle_cost + reject_penalty,
        "energy": energy,
        "distance": distance,
        "passenger_delay_cost": passenger_cost,
        "cargo_delay_cost": cargo_cost,
        "vehicle_cost": vehicle_cost,
        "reject_penalty": reject_penalty,
    }


def audit_config(config, graph_size, vehicles):
    return {
        "start": config["OPERATION_START"],
        "end": config["OPERATION_END"],
        "area": config["AREA_SIZE"],
        "speed": config["VEHICLE_SPEED"],
        "trip": config["MAX_TRIP_TIME"],
        "service": config["SERVICE_TIME"],
        "min_dispatch": 4,
        "hard_cargo": config["HARD_CARGO_PICKUP_TIMEWINDOW"],
        "ride": config["PASSENGER_MAX_RIDE_TIME_MINUTES"] / 60,
        "excess": config["PASSENGER_MAX_EXCESS_RIDE_TIME_MINUTES"] / 60,
        "max_open": 6,
        "vehicles": vehicles,
    }


_PATCHES_INSTALLED = False


def install_deployment_patches():
    global _PATCHES_INSTALLED
    if _PATCHES_INSTALLED:
        return
    from admission_targeted import install as install_admission
    from clock_patch import install as install_clock
    from completion_guard import install as install_completion
    from depot_noop import install as install_depot_noop
    from timing_candidate import install as install_timing

    install_clock()
    install_timing()
    install_completion()
    install_admission()
    install_depot_noop()
    _PATCHES_INSTALLED = True


def verify_actions(sample, actions, graph_size, config, normalization_profile):
    vehicles = math.ceil(graph_size / 6)
    route_audit = audit(sample, actions, audit_config(config, graph_size, vehicles))
    if route_audit["violations"] or route_audit["unfulfilled"]:
        raise RuntimeError({"audit": route_audit})

    from route_engine import RouteEngine, split_actions

    routes = split_actions(actions, graph_size)
    independent = RouteEngine(sample, config, normalization_profile, vehicles=vehicles).cost(routes)
    if independent is None:
        raise RuntimeError("Independent route engine rejected plan")
    replay = scalar_cost(sample, actions, config, normalization_profile)
    action_tensor = torch.tensor([actions], dtype=torch.long)
    with torch.no_grad():
        tensor_objective, details = MCVRPPDTW.get_costs(
            sample_to_batch(sample), action_tensor, return_details=True
        )
    errors = {
        "scalar_vs_route_objective": abs(replay["objective"] - independent["objective"]),
        "tensor_vs_route_objective": abs(float(tensor_objective[0]) - independent["objective"]),
        "tensor_vs_route_raw": abs(float(details["total_cost_raw"][0]) - independent["raw_cost"]),
    }
    if max(errors.values()) > 0.02:
        raise RuntimeError(("three-way cost mismatch", errors, replay, independent))
    return {
        "audit": route_audit,
        "cost": independent,
        "scalar_cost": replay,
        "cost_errors": errors,
        "completed_per_trip": [len(route) // 2 for route in routes],
    }
