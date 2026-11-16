"""Independent road-matrix hard-constraint audit (no search or model imports)."""
import math


def _python(value):
    return value.detach().cpu().tolist() if hasattr(value, "detach") else value


def audit_routes(sample, routes, config, vehicles=None):
    s = {key: _python(value) for key, value in sample.items()}
    c = dict(config)
    n = len(s["loc"]) // 2
    limit = vehicles or math.ceil(n / 6)
    distance = s["road_distance_matrix_km"]
    travel = s["road_travel_time_matrix_hours"]
    violations, all_completed = [], []
    if len([route for route in routes if route]) > limit:
        violations.append({"kind": "vehicle_limit", "observed": len(routes), "limit": limit})
    for route_id, route in enumerate(route for route in routes if route):
        if len(route) != len(set(route)):
            violations.append({"kind": "duplicate_node", "route": route_id})
        depart = c["OPERATION_START"]
        if route and isinstance(route[0], int) and 1 <= route[0] <= 2 * n:
            depart = max(depart, s["time_windows"][route[0] - 1][0] - travel[0][route[0]])
        current_time, current = depart, 0
        passenger_load = cargo_load = 0.0
        picked, completed, opened = set(), set(), {}
        for position, node in enumerate(route):
            if not isinstance(node, int) or not 1 <= node <= 2 * n:
                violations.append({"kind": "node_range", "route": route_id, "position": position, "node": node})
                continue
            j, order, pickup = node - 1, (node - 1) % n + 1, node <= n
            arrival = current_time + travel[current][node]
            start = max(arrival, s["time_windows"][j][0])
            finish = start + c["SERVICE_TIME"]
            if pickup:
                if order in picked:
                    violations.append({"kind": "duplicate_pickup", "route": route_id, "order": order})
                if (s["node_type"][j] == 1 or c["HARD_CARGO_PICKUP_TIMEWINDOW"]) and start > s["time_windows"][j][1] + 1e-4:
                    violations.append({"kind": "pickup_time_window", "route": route_id, "order": order})
                picked.add(order); opened[order] = finish
            else:
                if order not in opened:
                    violations.append({"kind": "precedence_or_cross_trip", "route": route_id, "order": order})
                else:
                    if s["node_type"][j] == 1:
                        ride = start - opened[order]
                        if ride > c["PASSENGER_MAX_RIDE_TIME_MINUTES"] / 60 + 1e-4:
                            violations.append({"kind": "passenger_ride_time", "route": route_id, "order": order})
                        if ride - travel[order][node] > c["PASSENGER_MAX_EXCESS_RIDE_TIME_MINUTES"] / 60 + 1e-4:
                            violations.append({"kind": "passenger_excess_ride_time", "route": route_id, "order": order})
                    del opened[order]; completed.add(order)
            passenger_load += s["demand_passenger"][j]
            cargo_load += s["demand_cargo"][j]
            if not -1e-4 <= passenger_load <= 1 + 1e-4:
                violations.append({"kind": "passenger_capacity", "route": route_id, "position": position})
            if not -1e-4 <= cargo_load <= 1 + 1e-4:
                violations.append({"kind": "cargo_capacity", "route": route_id, "position": position})
            if len(opened) > 6:
                violations.append({"kind": "open_order_limit", "route": route_id, "position": position})
            if finish > c["OPERATION_END"] + 1e-4:
                violations.append({"kind": "operation_end_at_service", "route": route_id, "position": position})
            current_time, current = finish, node
        returned = current_time + travel[current][0]
        if opened or picked != completed:
            violations.append({"kind": "unfulfilled", "route": route_id, "orders": sorted(opened)})
        if returned > c["OPERATION_END"] + 1e-4:
            violations.append({"kind": "operation_end_return", "route": route_id})
        if returned - depart > c["MAX_TRIP_TIME"] + 1e-4:
            violations.append({"kind": "trip_time", "route": route_id})
        if len(completed) < 4:
            violations.append({"kind": "min_completed_orders", "route": route_id, "completed": len(completed)})
        all_completed.extend(completed)
    if len(all_completed) != len(set(all_completed)):
        violations.append({"kind": "order_served_by_multiple_routes"})
    return {
        "valid": not violations,
        "violations": violations,
        "completed": len(set(all_completed)),
        "vehicles": len([route for route in routes if route]),
        "minimum_completed_per_trip": min((len({(node - 1) % n + 1 for node in route if node > n}) for route in routes if route), default=0),
    }

