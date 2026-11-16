"""Replay frozen road-LNS routes through the neural state transition system."""

import json
import sys
from pathlib import Path

import torch


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
FORMAL = HERE / "formal_p1_v1"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

from state_mcvrptw_v2 import StateMCVRPPDTW
from problem_mcvrptw_v2 import MCVRPPDTW


CORE_KEYS = (
    "depot",
    "loc",
    "node_type",
    "demand_passenger",
    "demand_cargo",
    "time_windows",
    "road_distance_matrix_km",
    "road_travel_time_matrix_hours",
)


def batched_road_fixed25(sample, speed=25.0):
    result = {key: sample[key].unsqueeze(0) for key in CORE_KEYS}
    result["road_travel_time_matrix_hours"] = result["road_distance_matrix_km"] / speed
    return result


def main():
    bundle = torch.load(FORMAL / "road_cases.pt", map_location="cpu", weights_only=True)
    report = json.loads((FORMAL / "controls_v1" / "report.json").read_text())
    assert report["status"] == "completed" and len(bundle["data"]) == len(report["rows"]) == 32

    max_distance_error = 0.0
    max_training_cost_distance_error = 0.0
    replayed_actions = 0
    for instance, (sample, row) in enumerate(zip(bundle["data"], report["rows"])):
        state = StateMCVRPPDTW.initialize(
            batched_road_fixed25(sample),
            allow_reject=True,
            max_concurrent_open_orders=6,
            min_orders_per_dispatch=4,
            enable_delivery_viability=True,
        )
        routes = row["arms"]["road_fixed25"]["lns_routes"]
        action_sequence = []
        for route in routes:
            for action in list(route) + [0]:
                action_sequence.append(action)
                mask = state.get_mask()
                if bool(mask[0, 0, action]):
                    raise AssertionError(
                        f"instance={instance} action={action} was masked after {replayed_actions} replay actions"
                    )
                state = state.update(torch.tensor([action]), current_mask=mask)
                replayed_actions += 1
        if not bool(state.all_finished()):
            raise AssertionError(f"instance={instance} did not terminate after replay")
        expected = float(row["arms"]["road_fixed25"]["lns_cost"]["distance"])
        observed = float(state.lengths[0, 0])
        max_distance_error = max(max_distance_error, abs(observed - expected))
        if abs(observed - expected) > 2e-3:
            raise AssertionError((instance, expected, observed))
        _, details = MCVRPPDTW.get_costs(
            batched_road_fixed25(sample),
            torch.tensor([action_sequence], dtype=torch.long),
            return_details=True,
        )
        cost_distance = float(details["total_distance"][0])
        max_training_cost_distance_error = max(
            max_training_cost_distance_error, abs(cost_distance - expected)
        )
        if abs(cost_distance - expected) > 2e-3:
            raise AssertionError((instance, "training_cost_distance", expected, cost_distance))
        if int(details["completed_orders"][0]) != 50 or int(details["used_vehicles"][0]) != len(routes):
            raise AssertionError((instance, "count mismatch"))

    result = {
        "status": "pass",
        "instances": 32,
        "replayed_actions": replayed_actions,
        "masked_planned_actions": 0,
        "max_distance_error_km": max_distance_error,
        "max_training_cost_distance_error_km": max_training_cost_distance_error,
        "explicit_road_matrix_semantics": True,
    }
    output = HERE / "real_road_state_crosscheck_report.json"
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
