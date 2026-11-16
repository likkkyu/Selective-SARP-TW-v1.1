"""CPU regression tests for the optional road-matrix neural environment path."""

import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from problem_mcvrptw_v2 import MCVRPPDTW
from state_mcvrptw_v2 import StateMCVRPPDTW


def dataset(with_road_matrices):
    n_orders = 4
    nodes = 2 * n_orders + 1
    value = {
        "depot": torch.tensor([[0.0, 0.0]]),
        "loc": torch.tensor([[[0.01 * index, 0.0] for index in range(1, nodes)]]),
        "node_type": torch.zeros(1, 2 * n_orders),
        "demand_passenger": torch.zeros(1, 2 * n_orders),
        "demand_cargo": torch.tensor([[0.05] * n_orders + [-0.05] * n_orders]),
        "time_windows": torch.tensor([[[10.0, 16.0]] * (2 * n_orders)]),
    }
    if with_road_matrices:
        distance = torch.full((1, nodes, nodes), 2.0)
        travel = torch.full((1, nodes, nodes), 0.1)
        diagonal = torch.arange(nodes)
        distance[:, diagonal, diagonal] = 0.0
        travel[:, diagonal, diagonal] = 0.0
        # Candidate node 1 looks nearby in coordinate space but cannot complete a
        # depot-node-depot trip within the three-hour hard limit on the road.
        travel[:, 0, 1] = 2.0
        travel[:, 1, 0] = 2.0
        value["road_distance_matrix_km"] = distance
        value["road_travel_time_matrix_hours"] = travel
    return value


def main():
    euclidean = StateMCVRPPDTW.initialize(dataset(False), min_orders_per_dispatch=4)
    road = StateMCVRPPDTW.initialize(dataset(True), min_orders_per_dispatch=4)
    assert not euclidean.uses_explicit_road_matrices
    assert road.uses_explicit_road_matrices

    euclidean_mask = euclidean.get_mask(skip_pickup_commitment=True)
    road_mask = road.get_mask(skip_pickup_commitment=True)
    assert not bool(euclidean_mask[0, 0, 1]), "Euclidean control unexpectedly blocks the nearby pickup"
    assert bool(road_mask[0, 0, 1]), "Road-time matrix did not block the impossible pickup"

    selected = torch.tensor([2])
    updated = road.update(selected, current_mask=road_mask)
    assert abs(float(updated.lengths[0, 0]) - 2.0) < 1e-6
    expected_time = StateMCVRPPDTW.OPERATION_START + 0.1 + StateMCVRPPDTW.SERVICE_TIME
    assert abs(float(updated.current_time[0, 0]) - expected_time) < 1e-5

    data = dataset(True)
    pi = torch.tensor([[0, 2, 6, 0]])
    loc_with_depot = torch.cat((data["depot"][:, None, :], data["loc"]), dim=1)
    _, segment_distance, segment_time, total_distance = MCVRPPDTW._compute_distance_energy(
        data, pi, loc_with_depot
    )
    assert torch.allclose(segment_distance, torch.tensor([[2.0, 2.0, 2.0]]))
    assert torch.allclose(segment_time, torch.tensor([[0.1, 0.1, 0.1]]))
    assert torch.allclose(total_distance, torch.tensor([6.0]))

    # A two-order candidate trip is cancelled before dispatch under the hard
    # four-order rule; both touched orders become explicit pre-departure declines.
    short = torch.tensor([[1, 5, 2, 6, 0]])
    sanitized, forced = MCVRPPDTW._sanitize_underfilled_dispatches(short, 4)
    assert torch.equal(sanitized, torch.zeros_like(short))
    assert float(forced[0]) == 2.0

    print(
        {
            "status": "pass",
            "road_candidate_1_masked": True,
            "matrix_update_distance_km": float(updated.lengths[0, 0]),
            "matrix_update_time_hours": float(updated.current_time[0, 0]),
            "matrix_cost_distance_km": float(total_distance[0]),
            "underfilled_trip_forced_declines": int(forced[0]),
        }
    )


if __name__ == "__main__":
    main()
