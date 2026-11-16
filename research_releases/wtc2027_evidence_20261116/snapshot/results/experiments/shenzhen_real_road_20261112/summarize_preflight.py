"""Aggregate the immutable four-instance CPU road preflight."""
import json
import statistics
from pathlib import Path


HERE = Path(__file__).resolve().parent
INPUTS = [HERE / f"outputs/preflight_instance{i:03d}_v3/report.json" for i in range(4)]
OUTPUT = HERE / "preflight_summary_v1.json"
ARMS = ("straight_fixed25", "road_fixed25", "road_osm_speed")


def mean(rows, key):
    return statistics.mean(row[key] for row in rows)


def main():
    if OUTPUT.exists():
        raise FileExistsError(f"Refusing to overwrite {OUTPUT}")
    reports = [json.loads(path.read_text()) for path in INPUTS]
    if any(report["status"] != "pass" for report in reports):
        raise ValueError("A source preflight did not pass")
    arms = {}
    for arm in ARMS:
        rows = [report[arm]["lns"]["cost"] for report in reports]
        arms[arm] = {
            "instances": len(rows),
            "mean_completed": mean(rows, "completed"),
            "full_service_instances": sum(row["completed"] == 50 for row in rows),
            "mean_vehicles": mean(rows, "vehicles"),
            "mean_distance_km": mean(rows, "distance"),
            "mean_energy_cost": mean(rows, "energy"),
            "mean_objective": mean(rows, "objective"),
        }
    paired = []
    for report in reports:
        straight = report["straight_fixed25"]["lns"]["cost"]
        road_fixed = report["road_fixed25"]["lns"]["cost"]
        road_speed = report["road_osm_speed"]["lns"]["cost"]
        paired.append({
            "instance": report["instance"],
            "road_fixed25_minus_straight_completed": road_fixed["completed"] - straight["completed"],
            "road_osm_speed_minus_road_fixed25_completed": road_speed["completed"] - road_fixed["completed"],
            "road_fixed25_minus_straight_distance_km": road_fixed["distance"] - straight["distance"],
            "road_osm_speed_minus_road_fixed25_distance_km": road_speed["distance"] - road_fixed["distance"],
        })
    output = {
        "status": "pass",
        "classification": "CPU development preflight; not formal paper evidence",
        "instances": len(reports),
        "graph_sha256": reports[0]["graph_sha256"],
        "graph_nodes_largest_strong_component": reports[0]["graph"]["nodes_largest_strong_component"],
        "graph_edges_largest_strong_component": reports[0]["graph"]["edges_largest_strong_component"],
        "all_unique_terminal_assignments": all(report["graph"]["unique_snapped_nodes"] == 101 for report in reports),
        "road_to_straight_median_mean": statistics.mean(report["paired_geometry"]["road_to_original_euclidean_median"] for report in reports),
        "arms": arms,
        "paired": paired,
        "interpretation_guard": "OSM class speeds are free-flow/default proxies until calibrated with Shenzhen taxi observations; no causal or field-performance claim.",
    }
    OUTPUT.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
