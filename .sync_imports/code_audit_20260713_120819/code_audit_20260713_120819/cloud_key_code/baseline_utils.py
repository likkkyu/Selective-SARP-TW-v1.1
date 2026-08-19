"""Shared baseline utilities for SA / GA / Gurobi re-evaluation."""

from copy import deepcopy

import numpy as np
import torch

from problem_mcvrptw_v2 import Config, MCVRPPDTW, node_routes_to_pi


def default_num_vehicles(n_orders):
    return Config.get_default_num_vehicles(n_orders)


def sample_to_batch(sample):
    return {
        key: value.unsqueeze(0) if torch.is_tensor(value) else value
        for key, value in sample.items()
    }


def order_routes_to_node_routes(order_routes, n_orders):
    node_routes = []
    for route in order_routes:
        nodes = []
        for order_id in route:
            pickup_node = int(order_id) + 1
            delivery_node = int(order_id) + n_orders + 1
            nodes.extend([pickup_node, delivery_node])
        if nodes:
            node_routes.append(nodes)
    return node_routes


def balanced_split(sequence, num_vehicles):
    if not sequence:
        return []
    return [list(chunk) for chunk in np.array_split(sequence, num_vehicles) if len(chunk) > 0]


def repair_order_routes(order_routes, n_orders, num_vehicles=None):
    num_vehicles = num_vehicles or default_num_vehicles(n_orders)
    sanitized_routes = []
    seen_orders = set()

    for route in order_routes:
        cleaned_route = []
        for order_id in route:
            if 0 <= int(order_id) < n_orders and int(order_id) not in seen_orders:
                seen_orders.add(int(order_id))
                cleaned_route.append(int(order_id))
        if cleaned_route:
            sanitized_routes.append(cleaned_route)

    missing_orders = [order_id for order_id in range(n_orders) if order_id not in seen_orders]
    if missing_orders:
        if not sanitized_routes:
            sanitized_routes = [[] for _ in range(num_vehicles)]
        for index, order_id in enumerate(missing_orders):
            sanitized_routes[index % max(len(sanitized_routes), 1)].append(order_id)

    sanitized_routes = [route for route in sanitized_routes if route]
    if not sanitized_routes and n_orders > 0:
        sanitized_routes = balanced_split(list(range(n_orders)), num_vehicles)

    return sanitized_routes[:num_vehicles]


def evaluate_node_routes(sample, node_routes):
    n_orders = int(sample['n_orders'])
    batch = sample_to_batch(sample)
    pi = node_routes_to_pi(node_routes, n_orders)
    with torch.no_grad():
        objective_cost, details = MCVRPPDTW.get_costs(batch, pi, return_details=True)
    return objective_cost.item(), details, pi


def evaluate_order_routes(sample, order_routes, num_vehicles=None):
    n_orders = int(sample['n_orders'])
    repaired_routes = repair_order_routes(order_routes, n_orders, num_vehicles=num_vehicles)
    node_routes = order_routes_to_node_routes(repaired_routes, n_orders)
    objective_cost, details, pi = evaluate_node_routes(sample, node_routes)
    return objective_cost, details, pi, repaired_routes, node_routes


def build_solution_info(objective_cost, details, node_routes, algorithm, extra=None):
    info = {
        'algorithm': algorithm,
        'objective_cost': float(objective_cost),
        # v6: raw_total_cost (元) 是论文报告的主指标
        'raw_total_cost': float(details['total_cost_raw'].item()),
        'energy_cost': float(details['energy_cost_raw'].item()),
        'passenger_delivery_delay_cost': float(details['passenger_delivery_delay_cost_raw'].item()),
        'cargo_delay': float(details['cargo_delay_cost_raw'].item()),
        'trip_overtime_penalty': float(details['trip_overtime_penalty'].item()),
        'reject_penalty': float(details['reject_penalty'].item()),
        'unfulfilled_penalty': float(details['unfulfilled_penalty'].item()),
        'rejected_orders': float(details['rejected_orders'].item()),
        'unfulfilled_orders': float(details['unfulfilled_orders'].item()),
        'vehicle_cost': float(details['vehicle_cost_raw'].item()),
        'total_distance': float(details['total_distance'].item()),
        'used_vehicles': int(round(details['used_vehicles'].item())),
        'routes': node_routes,
    }
    if extra:
        info.update(extra)
    return info


def clone_routes(routes):
    return [list(route) for route in routes]


def ensure_minimum_routes(order_routes, num_vehicles):
    routes = clone_routes(order_routes)
    while len(routes) < num_vehicles:
        routes.append([])
    return routes


def random_initial_order_routes(n_orders, num_vehicles, rng):
    order_ids = list(range(n_orders))
    rng.shuffle(order_ids)
    return balanced_split(order_ids, num_vehicles)


def summarize_results(results):
    if not results:
        return None
    return {
        'num_instances': len(results),
        'avg_objective_cost': float(np.mean([result['objective_cost'] for result in results])),
        'avg_raw_total_cost': float(np.mean([result['raw_total_cost'] for result in results])),
        'avg_energy_cost': float(np.mean([result['energy_cost'] for result in results])),
        'avg_passenger_delivery_delay_cost': float(np.mean([result['passenger_delivery_delay_cost'] for result in results])),
        'avg_cargo_delay': float(np.mean([result['cargo_delay'] for result in results])),
        'avg_trip_overtime_penalty': float(np.mean([result['trip_overtime_penalty'] for result in results])),
        'avg_reject_penalty': float(np.mean([result['reject_penalty'] for result in results])),
        'avg_unfulfilled_penalty': float(np.mean([result['unfulfilled_penalty'] for result in results])),
        'avg_rejected_orders': float(np.mean([result['rejected_orders'] for result in results])),
        'avg_unfulfilled_orders': float(np.mean([result['unfulfilled_orders'] for result in results])),
        'avg_vehicle_cost': float(np.mean([result['vehicle_cost'] for result in results])),
        'avg_total_distance': float(np.mean([result['total_distance'] for result in results])),
        'avg_used_vehicles': float(np.mean([result['used_vehicles'] for result in results])),
        'results': results,
    }
