"""Shared baseline utilities for SA / GA / Gurobi re-evaluation."""

import numpy as np
import torch

from problem_mcvrptw_v2 import Config, MCVRPPDTW, node_routes_to_pi


LEGACY_PROTOCOL = 'legacy_unaligned_v0'
ALIGNED_PROTOCOL = 'drl_aligned_v1'

PENALTY_HARD_CONSTRAINT_MODE = 'penalty'
STRICT_HARD_CONSTRAINT_MODE = 'strict'
HARD_CONSTRAINT_MODES = (PENALTY_HARD_CONSTRAINT_MODE, STRICT_HARD_CONSTRAINT_MODE)


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


def repair_order_routes(order_routes, n_orders, num_vehicles=None, fill_missing_orders=True):
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

    if fill_missing_orders:
        missing_orders = [order_id for order_id in range(n_orders) if order_id not in seen_orders]
        if missing_orders:
            if not sanitized_routes:
                sanitized_routes = [[] for _ in range(num_vehicles)]
            for index, order_id in enumerate(missing_orders):
                sanitized_routes[index % max(len(sanitized_routes), 1)].append(order_id)

    sanitized_routes = [route for route in sanitized_routes if route]
    if not sanitized_routes and n_orders > 0 and fill_missing_orders:
        sanitized_routes = balanced_split(list(range(n_orders)), num_vehicles)

    return sanitized_routes[:num_vehicles]




def prune_order_routes_for_strict(sample, order_routes, n_orders, num_vehicles=None):
    """Greedily rebuild routes under strict hard constraints.

    仅用于 baseline strict 搜索辅助：
    - passenger pickup latest TW
    - operation end
    - max trip time
    - simple capacity bound (normalized demand)
    """
    num_vehicles = num_vehicles or default_num_vehicles(n_orders)
    sanitized_routes = repair_order_routes(
        order_routes,
        n_orders,
        num_vehicles=num_vehicles,
        fill_missing_orders=False,
    )

    demand_p_per_order = sample['demand_passenger'][:n_orders]
    demand_c_per_order = sample['demand_cargo'][:n_orders]

    area_size = Config.AREA_SIZE
    vehicle_speed = Config.VEHICLE_SPEED
    service_time = Config.SERVICE_TIME
    max_trip_time = Config.MAX_TRIP_TIME
    op_start = Config.OPERATION_START
    op_end = Config.OPERATION_END

    loc = sample['loc']
    depot = sample['depot'].squeeze()
    time_windows = sample['time_windows']

    def get_loc(node_idx):
        if int(node_idx) == 0:
            return depot
        return loc[int(node_idx) - 1]

    def try_append(state, order_id):
        order_id = int(order_id)
        dp = float(demand_p_per_order[order_id].item())
        dc = float(demand_c_per_order[order_id].item())

        if (state['load_p'] + max(dp, 0.0)) > 1.0 + 1e-6:
            return None
        if (state['load_c'] + max(dc, 0.0)) > 1.0 + 1e-6:
            return None

        pickup_node = order_id + 1
        delivery_node = order_id + n_orders + 1

        pickup_travel = (get_loc(state['last_node']) - get_loc(pickup_node)).norm(p=2).item() * area_size
        pickup_arrival = state['cur_time'] + pickup_travel / vehicle_speed
        pickup_tw_start = float(time_windows[order_id, 0].item())
        pickup_tw_end = float(time_windows[order_id, 1].item())
        pickup_start = max(pickup_arrival, pickup_tw_start)
        pickup_finish = pickup_start + service_time

        delivery_travel = (get_loc(pickup_node) - get_loc(delivery_node)).norm(p=2).item() * area_size
        delivery_arrival = pickup_finish + delivery_travel / vehicle_speed
        delivery_tw_start = float(time_windows[order_id + n_orders, 0].item())
        delivery_start = max(delivery_arrival, delivery_tw_start)
        delivery_finish = delivery_start + service_time

        depot_travel = (get_loc(delivery_node) - get_loc(0)).norm(p=2).item() * area_size
        depot_arrival = delivery_finish + depot_travel / vehicle_speed
        trip_time = depot_arrival - op_start

        violate_hard = (
            pickup_start > pickup_tw_end + 1e-9
            or pickup_start > op_end + 1e-9
            or delivery_start > op_end + 1e-9
            or trip_time > max_trip_time + 1e-9
        )
        if violate_hard:
            return None

        return {
            'route': state['route'] + [order_id],
            'load_p': state['load_p'] + max(dp, 0.0),
            'load_c': state['load_c'] + max(dc, 0.0),
            'cur_time': delivery_finish,
            'last_node': delivery_node,
        }

    flat_orders = [int(order_id) for route in sanitized_routes[:num_vehicles] for order_id in route]

    states = []
    dropped_orders = []

    for order_id in flat_orders:
        best_index = None
        best_state = None
        best_finish_time = float('inf')

        for idx in range(len(states)):
            next_state = try_append(states[idx], order_id)
            if next_state is None:
                continue
            if next_state['cur_time'] < best_finish_time:
                best_finish_time = next_state['cur_time']
                best_state = next_state
                best_index = idx

        if len(states) < num_vehicles:
            fresh_state = {
                'route': [],
                'load_p': 0.0,
                'load_c': 0.0,
                'cur_time': op_start,
                'last_node': 0,
            }
            next_state = try_append(fresh_state, order_id)
            if next_state is not None and next_state['cur_time'] < best_finish_time:
                best_finish_time = next_state['cur_time']
                best_state = next_state
                best_index = None

        if best_state is None:
            dropped_orders.append(order_id)
            continue

        if best_index is None:
            states.append(best_state)
        else:
            states[best_index] = best_state

    pruned_routes = [state['route'] for state in states if state['route']]
    return pruned_routes[:num_vehicles], dropped_orders




def _detail_scalar(details, key, default=0.0):
    value = details.get(key, default)
    if torch.is_tensor(value):
        return float(value.item())
    return float(value)


def _build_eval_meta(
    details,
    objective_cost,
    eval_protocol,
    hard_violation_penalty_weight,
    hard_constraint_mode=PENALTY_HARD_CONSTRAINT_MODE,
    strict_infeasible_cost=1e12,
):
    objective_legacy = float(objective_cost)
    raw_legacy = _detail_scalar(details, 'total_cost_raw', objective_legacy)

    pickup_hard_violations = _detail_scalar(details, 'passenger_pickup_hard_violations', 0.0)
    total_ride_hard_violations = _detail_scalar(details, 'passenger_total_ride_time_violations', 0.0)
    excess_ride_hard_violations = _detail_scalar(details, 'passenger_excess_ride_time_violations', 0.0)
    hard_violation_count = pickup_hard_violations + total_ride_hard_violations + excess_ride_hard_violations

    comparable = eval_protocol == ALIGNED_PROTOCOL
    hard_constraint_mode = hard_constraint_mode if hard_constraint_mode in HARD_CONSTRAINT_MODES else PENALTY_HARD_CONSTRAINT_MODE
    applied_penalty_weight = float(hard_violation_penalty_weight) if comparable else 0.0
    hard_violation_penalty = hard_violation_count * applied_penalty_weight

    is_hard_feasible = hard_violation_count <= 1e-9
    strict_rejected_by_hard = hard_constraint_mode == STRICT_HARD_CONSTRAINT_MODE and (not is_hard_feasible)

    return {
        'protocol_version': eval_protocol,
        'comparable_to_drl': comparable,
        'comparability_notes': [] if comparable else [
            'Legacy baseline protocol: metrics are not strictly comparable to DRL hard-mask decode semantics.'
        ],
        'objective_cost_legacy': objective_legacy,
        'raw_total_cost_legacy': raw_legacy,
        'objective_cost_aligned': objective_legacy + hard_violation_penalty,
        'raw_total_cost_aligned': raw_legacy + hard_violation_penalty,
        'hard_violation_penalty_weight': applied_penalty_weight,
        'hard_violation_penalty': hard_violation_penalty,
        'hard_violation_count': hard_violation_count,
        'passenger_pickup_hard_violations': pickup_hard_violations,
        'passenger_total_ride_time_violations': total_ride_hard_violations,
        'passenger_excess_ride_time_violations': excess_ride_hard_violations,
        'is_hard_feasible': is_hard_feasible,
        'hard_constraint_mode': hard_constraint_mode,
        'strict_rejected_by_hard': strict_rejected_by_hard,
        'strict_infeasible_cost': float(strict_infeasible_cost),
    }


def evaluate_node_routes(
    sample,
    node_routes,
    eval_protocol=LEGACY_PROTOCOL,
    hard_violation_penalty_weight=0.0,
    hard_constraint_mode=PENALTY_HARD_CONSTRAINT_MODE,
    strict_infeasible_cost=1e12,
    return_eval_meta=False,
):
    n_orders = int(sample['n_orders'])
    batch = sample_to_batch(sample)
    pi = node_routes_to_pi(node_routes, n_orders)
    with torch.no_grad():
        objective_cost, details = MCVRPPDTW.get_costs(batch, pi, return_details=True)

    eval_meta = _build_eval_meta(
        details,
        objective_cost.item(),
        eval_protocol=eval_protocol,
        hard_violation_penalty_weight=hard_violation_penalty_weight,
        hard_constraint_mode=hard_constraint_mode,
        strict_infeasible_cost=strict_infeasible_cost,
    )

    score_cost = eval_meta['objective_cost_aligned'] if eval_meta['comparable_to_drl'] else eval_meta['objective_cost_legacy']
    if eval_meta['strict_rejected_by_hard']:
        score_cost = float(strict_infeasible_cost)
    if return_eval_meta:
        return score_cost, details, pi, eval_meta
    return score_cost, details, pi



def evaluate_order_routes(
    sample,
    order_routes,
    num_vehicles=None,
    fill_missing_orders=True,
    eval_protocol=LEGACY_PROTOCOL,
    hard_violation_penalty_weight=0.0,
    hard_constraint_mode=PENALTY_HARD_CONSTRAINT_MODE,
    strict_infeasible_cost=1e12,
):
    n_orders = int(sample['n_orders'])
    if hard_constraint_mode == STRICT_HARD_CONSTRAINT_MODE:
        repaired_routes, dropped_orders = prune_order_routes_for_strict(
            sample,
            order_routes,
            n_orders,
            num_vehicles=num_vehicles,
        )
    else:
        repaired_routes = repair_order_routes(
            order_routes,
            n_orders,
            num_vehicles=num_vehicles,
            fill_missing_orders=fill_missing_orders,
        )
        dropped_orders = []

    node_routes = order_routes_to_node_routes(repaired_routes, n_orders)
    objective_cost, details, pi, eval_meta = evaluate_node_routes(
        sample,
        node_routes,
        eval_protocol=eval_protocol,
        hard_violation_penalty_weight=hard_violation_penalty_weight,
        hard_constraint_mode=hard_constraint_mode,
        strict_infeasible_cost=strict_infeasible_cost,
        return_eval_meta=True,
    )

    eval_meta['strict_pruned_orders'] = float(len(dropped_orders))
    eval_meta['strict_pruned_order_ids'] = [int(order_id) for order_id in dropped_orders]

    if eval_protocol == ALIGNED_PROTOCOL and fill_missing_orders and hard_constraint_mode != STRICT_HARD_CONSTRAINT_MODE:
        eval_meta['comparable_to_drl'] = False
        eval_meta['comparability_notes'].append(
            'fill_missing_orders=True may bias service upward; use --no-fill-missing-orders for strict DRL comparability.'
        )

    return objective_cost, details, pi, repaired_routes, node_routes, eval_meta


def build_solution_info(objective_cost, details, node_routes, algorithm, extra=None, eval_meta=None):
    eval_meta = eval_meta or _build_eval_meta(details, objective_cost, LEGACY_PROTOCOL, 0.0)

    objective_effective = float(eval_meta['objective_cost_aligned'] if eval_meta['comparable_to_drl'] else eval_meta['objective_cost_legacy'])
    raw_effective = float(eval_meta['raw_total_cost_aligned'] if eval_meta['comparable_to_drl'] else eval_meta['raw_total_cost_legacy'])
    if eval_meta.get('strict_rejected_by_hard', False):
        objective_effective = float(eval_meta.get('strict_infeasible_cost', objective_effective))

    unfulfilled_orders = _detail_scalar(details, 'unfulfilled_orders', 0.0)
    pickup_only_orders = _detail_scalar(details, 'pickup_only_orders', 0.0)
    started_not_completed_orders = _detail_scalar(details, 'started_not_completed_orders', 0.0)
    untouched_unrejected_orders = _detail_scalar(details, 'untouched_unrejected_orders', 0.0)

    info = {
        'algorithm': algorithm,
        'objective_cost': objective_effective,
        'objective_cost_legacy': float(eval_meta['objective_cost_legacy']),
        'objective_cost_aligned': float(eval_meta['objective_cost_aligned']),
        'raw_total_cost': raw_effective,
        'raw_total_cost_legacy': float(eval_meta['raw_total_cost_legacy']),
        'raw_total_cost_aligned': float(eval_meta['raw_total_cost_aligned']),
        'energy_cost': _detail_scalar(details, 'energy_cost_raw', 0.0),
        'passenger_delivery_delay_cost': _detail_scalar(details, 'passenger_delivery_delay_cost_raw', 0.0),
        'cargo_delay': _detail_scalar(details, 'cargo_delay_cost_raw', 0.0),
        'trip_overtime_penalty': _detail_scalar(details, 'trip_overtime_penalty', 0.0),
        'reject_penalty': _detail_scalar(details, 'reject_penalty', 0.0),
        'unfulfilled_penalty': _detail_scalar(details, 'unfulfilled_penalty', 0.0),
        'completed_orders': _detail_scalar(details, 'completed_orders', 0.0),
        'rejected_orders': _detail_scalar(details, 'rejected_orders', 0.0),
        'unfulfilled_orders': unfulfilled_orders,
        'untouched_orders': _detail_scalar(details, 'untouched_orders', 0.0),
        'untouched_unrejected_orders': untouched_unrejected_orders,
        'pickup_only_orders': pickup_only_orders,
        'started_not_completed_orders': started_not_completed_orders,
        'vehicle_cost': _detail_scalar(details, 'vehicle_cost_raw', 0.0),
        'total_distance': _detail_scalar(details, 'total_distance', 0.0),
        'used_vehicles': int(round(_detail_scalar(details, 'used_vehicles', 0.0))),
        'passenger_pickup_hard_violations': float(eval_meta['passenger_pickup_hard_violations']),
        'passenger_total_ride_time_violations': float(eval_meta['passenger_total_ride_time_violations']),
        'passenger_excess_ride_time_violations': float(eval_meta['passenger_excess_ride_time_violations']),
        'hard_violation_count': float(eval_meta['hard_violation_count']),
        'hard_violation_penalty_weight': float(eval_meta['hard_violation_penalty_weight']),
        'hard_violation_penalty': float(eval_meta['hard_violation_penalty']),
        'is_hard_feasible': bool(eval_meta['is_hard_feasible']),
        'hard_constraint_mode': eval_meta.get('hard_constraint_mode', PENALTY_HARD_CONSTRAINT_MODE),
        'strict_rejected_by_hard': bool(eval_meta.get('strict_rejected_by_hard', False)),
        'strict_infeasible_cost': float(eval_meta.get('strict_infeasible_cost', 0.0)),
        'strict_pruned_orders': float(eval_meta.get('strict_pruned_orders', 0.0)),
        'protocol_version': eval_meta['protocol_version'],
        'comparable_to_drl': bool(eval_meta['comparable_to_drl']),
        'comparability_notes': list(eval_meta.get('comparability_notes', [])),
        'routes': node_routes,
    }

    info['service_rate'] = float(info['completed_orders'])
    info['business_clean'] = bool(
        info['is_hard_feasible']
        and abs(unfulfilled_orders) <= 1e-9
        and abs(pickup_only_orders) <= 1e-9
        and abs(started_not_completed_orders) <= 1e-9
        and abs(untouched_unrejected_orders) <= 1e-9
    )

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


def summarize_results(results, graph_size=None):
    if not results:
        return None

    summary = {
        'num_instances': len(results),
        'avg_objective_cost': float(np.mean([result['objective_cost'] for result in results])),
        'avg_objective_cost_legacy': float(np.mean([result.get('objective_cost_legacy', result['objective_cost']) for result in results])),
        'avg_objective_cost_aligned': float(np.mean([result.get('objective_cost_aligned', result['objective_cost']) for result in results])),
        'avg_raw_total_cost': float(np.mean([result['raw_total_cost'] for result in results])),
        'avg_raw_total_cost_legacy': float(np.mean([result.get('raw_total_cost_legacy', result['raw_total_cost']) for result in results])),
        'avg_raw_total_cost_aligned': float(np.mean([result.get('raw_total_cost_aligned', result['raw_total_cost']) for result in results])),
        'avg_energy_cost': float(np.mean([result['energy_cost'] for result in results])),
        'avg_passenger_delivery_delay_cost': float(np.mean([result['passenger_delivery_delay_cost'] for result in results])),
        'avg_cargo_delay': float(np.mean([result['cargo_delay'] for result in results])),
        'avg_trip_overtime_penalty': float(np.mean([result['trip_overtime_penalty'] for result in results])),
        'avg_reject_penalty': float(np.mean([result['reject_penalty'] for result in results])),
        'avg_unfulfilled_penalty': float(np.mean([result['unfulfilled_penalty'] for result in results])),
        'avg_completed_orders': float(np.mean([result['completed_orders'] for result in results])),
        'avg_rejected_orders': float(np.mean([result['rejected_orders'] for result in results])),
        'avg_unfulfilled_orders': float(np.mean([result['unfulfilled_orders'] for result in results])),
        'avg_untouched_orders': float(np.mean([result['untouched_orders'] for result in results])),
        'avg_untouched_unrejected_orders': float(np.mean([result['untouched_unrejected_orders'] for result in results])),
        'avg_pickup_only_orders': float(np.mean([result['pickup_only_orders'] for result in results])),
        'avg_started_not_completed_orders': float(np.mean([result['started_not_completed_orders'] for result in results])),
        'avg_vehicle_cost': float(np.mean([result['vehicle_cost'] for result in results])),
        'avg_total_distance': float(np.mean([result['total_distance'] for result in results])),
        'avg_used_vehicles': float(np.mean([result['used_vehicles'] for result in results])),
        'avg_passenger_pickup_hard_violations': float(np.mean([result.get('passenger_pickup_hard_violations', 0.0) for result in results])),
        'avg_passenger_total_ride_time_violations': float(np.mean([result.get('passenger_total_ride_time_violations', 0.0) for result in results])),
        'avg_passenger_excess_ride_time_violations': float(np.mean([result.get('passenger_excess_ride_time_violations', 0.0) for result in results])),
        'avg_hard_violation_count': float(np.mean([result.get('hard_violation_count', 0.0) for result in results])),
        'avg_hard_violation_penalty': float(np.mean([result.get('hard_violation_penalty', 0.0) for result in results])),
        'hard_feasible_rate': float(np.mean([1.0 if result.get('is_hard_feasible', False) else 0.0 for result in results])),
        'business_clean_rate': float(np.mean([1.0 if result.get('business_clean', False) else 0.0 for result in results])),
        'protocol_versions': sorted({result.get('protocol_version', LEGACY_PROTOCOL) for result in results}),
        'comparable_to_drl': bool(all(result.get('comparable_to_drl', False) for result in results)),
        'results': results,
    }

    if graph_size is not None and int(graph_size) > 0:
        g = float(graph_size)
        summary.update({
            'service_rate_mean': summary['avg_completed_orders'] / g,
            'rejected_rate_mean': summary['avg_rejected_orders'] / g,
            'unfulfilled_rate_mean': summary['avg_unfulfilled_orders'] / g,
            'untouched_unrejected_rate_mean': summary['avg_untouched_unrejected_orders'] / g,
            'served_plus_rejected_rate_mean': min(1.0, (summary['avg_completed_orders'] + summary['avg_rejected_orders']) / g),
            'business_clean': (
                summary['avg_hard_violation_count'] <= 1e-9
                and abs(summary['avg_unfulfilled_orders']) <= 1e-9
                and abs(summary['avg_pickup_only_orders']) <= 1e-9
                and abs(summary['avg_started_not_completed_orders']) <= 1e-9
                and abs(summary['avg_untouched_unrejected_orders']) <= 1e-9
            ),
        })

    return summary
