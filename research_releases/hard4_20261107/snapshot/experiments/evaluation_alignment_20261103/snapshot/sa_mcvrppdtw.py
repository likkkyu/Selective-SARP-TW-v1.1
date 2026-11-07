"""Simulated Annealing baseline for MCVRP-PDTW.

实现约束：
- Pickup / Delivery 始终绑定为同一订单、同一车辆
- 成本与惩罚统一复用 `MCVRPPDTW.get_costs`
"""

import argparse
import json
import math
import random
import time

from baseline_utils import (
    ALIGNED_PROTOCOL,
    DRL_ALIGNED_SEMANTIC_MODE,
    HARD_CONSTRAINT_MODES,
    LEGACY_PROTOCOL,
    LEGACY_SEMANTIC_MODE,
    PENALTY_HARD_CONSTRAINT_MODE,
    SEMANTIC_MODES,
    STRICT_HARD_CONSTRAINT_MODE,
    build_solution_info,
    clone_routes,
    default_num_vehicles,
    ensure_minimum_routes,
    evaluate_order_routes,
    random_initial_order_routes,
    summarize_results,
)
from problem_mcvrptw_v2 import Config, MCVRPPDTWDataset


class SimulatedAnnealingSolver:
    def __init__(
        self,
        iterations,
        initial_temperature,
        cooling_rate,
        num_vehicles=None,
        seed=1234,
        eval_protocol=ALIGNED_PROTOCOL,
        fill_missing_orders=False,
        hard_violation_penalty_weight=0.0,
        hard_constraint_mode=PENALTY_HARD_CONSTRAINT_MODE,
        strict_infeasible_cost=1e12,
        semantic_mode=DRL_ALIGNED_SEMANTIC_MODE,
        state_kwargs=None,
        allow_reject=True,
        scoring_mode='pax_lex',
    ):
        self.iterations = iterations
        self.initial_temperature = initial_temperature
        self.cooling_rate = cooling_rate
        self.num_vehicles = num_vehicles
        self.rng = random.Random(seed)
        self.eval_protocol = eval_protocol
        self.fill_missing_orders = fill_missing_orders
        self.hard_violation_penalty_weight = float(hard_violation_penalty_weight)
        self.hard_constraint_mode = hard_constraint_mode if hard_constraint_mode in HARD_CONSTRAINT_MODES else PENALTY_HARD_CONSTRAINT_MODE
        self.strict_infeasible_cost = float(strict_infeasible_cost)
        self.semantic_mode = semantic_mode if semantic_mode in SEMANTIC_MODES else LEGACY_SEMANTIC_MODE
        self.state_kwargs = dict(state_kwargs or {})
        self.allow_reject = bool(allow_reject)
        self.scoring_mode = str(scoring_mode)

    def repair_pd_order(self, order_routes, n_orders):
        from baseline_utils import repair_order_routes
        return repair_order_routes(
            order_routes,
            n_orders,
            num_vehicles=self.num_vehicles,
            fill_missing_orders=self.fill_missing_orders,
        )

    def _search_objective(self, info, objective_cost):
        if self.scoring_mode == 'objective':
            return float(objective_cost)

        hard_violation_count = float(info.get('hard_violation_count', 0.0))
        passenger_delay_minutes = float(info.get('passenger_delivery_delay_minutes', 0.0))
        raw_total_cost = float(info.get('raw_total_cost', objective_cost))
        # 词典序近似：先硬可行，再 passenger delay，再 raw total cost。
        return hard_violation_count * 1e9 + passenger_delay_minutes * 1e6 + raw_total_cost

    def evaluate_cost(self, sample, order_routes):
        objective_cost, details, _, repaired_routes, node_routes, eval_meta = evaluate_order_routes(
            sample,
            order_routes,
            num_vehicles=self.num_vehicles,
            fill_missing_orders=self.fill_missing_orders,
            eval_protocol=self.eval_protocol,
            hard_violation_penalty_weight=self.hard_violation_penalty_weight,
            hard_constraint_mode=self.hard_constraint_mode,
            strict_infeasible_cost=self.strict_infeasible_cost,
            semantic_mode=self.semantic_mode,
            state_kwargs=self.state_kwargs,
            allow_reject=self.allow_reject,
        )
        info = build_solution_info(
            objective_cost,
            details,
            node_routes,
            algorithm='SA',
            eval_meta=eval_meta,
        )
        search_cost = self._search_objective(info, objective_cost)
        return search_cost, info, repaired_routes

    def swap(self, order_routes):
        routes = [route for route in clone_routes(order_routes) if route]
        if not routes:
            return routes
        route_a = self.rng.randrange(len(routes))
        route_b = self.rng.randrange(len(routes))
        index_a = self.rng.randrange(len(routes[route_a]))
        index_b = self.rng.randrange(len(routes[route_b]))
        routes[route_a][index_a], routes[route_b][index_b] = routes[route_b][index_b], routes[route_a][index_a]
        return routes

    def relocate(self, order_routes):
        routes = ensure_minimum_routes(clone_routes(order_routes), self.num_vehicles)
        non_empty_indices = [index for index, route in enumerate(routes) if route]
        if not non_empty_indices:
            return routes
        source_index = self.rng.choice(non_empty_indices)
        target_index = self.rng.randrange(len(routes))
        source_route = routes[source_index]
        order_position = self.rng.randrange(len(source_route))
        order_id = source_route.pop(order_position)
        insert_position = self.rng.randrange(len(routes[target_index]) + 1)
        routes[target_index].insert(insert_position, order_id)
        return [route for route in routes if route]

    def two_opt(self, order_routes):
        routes = [route for route in clone_routes(order_routes) if len(route) >= 2]
        if not routes:
            return clone_routes(order_routes)
        route_index = self.rng.randrange(len(routes))
        route = routes[route_index]
        left, right = sorted(self.rng.sample(range(len(route)), 2))
        route[left:right + 1] = list(reversed(route[left:right + 1]))
        return routes

    def cross(self, order_routes):
        routes = [route for route in clone_routes(order_routes) if route]
        if len(routes) < 2:
            return routes
        route_a, route_b = self.rng.sample(range(len(routes)), 2)
        cut_a = self.rng.randrange(len(routes[route_a]) + 1)
        cut_b = self.rng.randrange(len(routes[route_b]) + 1)
        tail_a = routes[route_a][cut_a:]
        tail_b = routes[route_b][cut_b:]
        routes[route_a] = routes[route_a][:cut_a] + tail_b
        routes[route_b] = routes[route_b][:cut_b] + tail_a
        return routes

    def merge_small_route(self, order_routes):
        """Merge 算子：当存在很短的路线（<=2 单）时，尝试把它合并进其他路线并删除该路线。

        说明：
        - 这里的 route 是“订单级”列表（List[int]），pickup/delivery 会在评估阶段再展开。
        - 为了保持 SA 邻域生成足够轻量，这里不做昂贵的逐点可行性检验；
          合并后的可行性与好坏由 `evaluate_order_routes -> get_costs` 的软约束成本自然体现。
        """
        routes = [route for route in clone_routes(order_routes) if route]
        if len(routes) < 2:
            return routes

        small_indices = [index for index, route in enumerate(routes) if len(route) <= 2]
        if not small_indices:
            return routes

        source_index = self.rng.choice(small_indices)
        source_route = routes.pop(source_index)
        if not routes:
            # 理论上不会发生（len(routes) < 2 已 return），这里防御一下
            return [source_route]

        for order_id in source_route:
            target_index = self.rng.randrange(len(routes))
            insert_position = self.rng.randrange(len(routes[target_index]) + 1)
            routes[target_index].insert(insert_position, int(order_id))

        return [route for route in routes if route]

    def relocate_pair_block(self, order_routes):
        """v5 新增邻域算子：将一段连续 2-3 个订单的“小块”整体搬到另一条路线。

        说明：
        - 由于 SA 这里以「订单级」表示路径（pickup/delivery 由评估阶段展开），
          因此「成对搬迁」实际等价于把若干 order_id 作为不可拆的 block 整体移动，
          天然不会破坏 PD 同车约束。
        - 当 block 大小受限于 source_route 长度，最少 1 个、最多 3 个。
        """
        routes = ensure_minimum_routes(clone_routes(order_routes), self.num_vehicles)
        non_empty_indices = [index for index, route in enumerate(routes) if route]
        if not non_empty_indices:
            return [route for route in routes if route]

        source_index = self.rng.choice(non_empty_indices)
        source_route = routes[source_index]

        block_size = self.rng.randint(1, min(3, len(source_route)))
        start = self.rng.randrange(len(source_route) - block_size + 1)
        block = source_route[start:start + block_size]
        del source_route[start:start + block_size]

        target_index = self.rng.randrange(len(routes))
        insert_position = self.rng.randrange(len(routes[target_index]) + 1)
        routes[target_index][insert_position:insert_position] = block

        return [route for route in routes if route]

    def generate_neighbor(self, order_routes):
        operator = self.rng.choice([
            self.swap,
            self.relocate,
            self.two_opt,
            self.cross,
            self.merge_small_route,
            self.relocate_pair_block,
        ])
        return operator(order_routes)

    def solve(self, sample):
        n_orders = int(sample['n_orders'])
        if self.num_vehicles is None:
            self.num_vehicles = default_num_vehicles(n_orders)

        current_routes = self.repair_pd_order(
            random_initial_order_routes(n_orders, self.num_vehicles, self.rng),
            n_orders,
        )
        current_cost, current_info, current_routes = self.evaluate_cost(sample, current_routes)
        best_cost = current_cost
        best_info = current_info
        best_routes = current_routes

        temperature = self.initial_temperature
        strict_mode = self.hard_constraint_mode == STRICT_HARD_CONSTRAINT_MODE
        if strict_mode and current_info.get('strict_rejected_by_hard', False):
            max_init_retry = 64
            for _ in range(max_init_retry):
                trial_routes = self.repair_pd_order(
                    random_initial_order_routes(n_orders, self.num_vehicles, self.rng),
                    n_orders,
                )
                trial_cost, trial_info, trial_routes = self.evaluate_cost(sample, trial_routes)
                if not trial_info.get('strict_rejected_by_hard', False):
                    current_cost = trial_cost
                    current_info = trial_info
                    current_routes = trial_routes
                    best_cost = trial_cost
                    best_info = trial_info
                    best_routes = trial_routes
                    break

        for _ in range(self.iterations):
            candidate_routes = self.repair_pd_order(self.generate_neighbor(current_routes), n_orders)
            candidate_cost, candidate_info, candidate_routes = self.evaluate_cost(sample, candidate_routes)
            if strict_mode and candidate_info.get('strict_rejected_by_hard', False):
                temperature *= self.cooling_rate
                continue
            delta = candidate_cost - current_cost
            accept = delta < 0 or self.rng.random() < math.exp(-delta / max(temperature, 1e-6))
            if accept:
                current_cost = candidate_cost
                current_info = candidate_info
                current_routes = candidate_routes
            if candidate_cost < best_cost:
                best_cost = candidate_cost
                best_info = candidate_info
                best_routes = candidate_routes
            temperature *= self.cooling_rate

        best_info['order_routes'] = best_routes
        return best_info



def run_sa_benchmark(args):
    dataset = MCVRPPDTWDataset(num_samples=args.num_samples, graph_size=args.graph_size, seed=args.seed)
    if args.semantic_mode == DRL_ALIGNED_SEMANTIC_MODE:
        state_kwargs = {
            'max_concurrent_open_orders': max(int(args.max_concurrent_open_orders), 1),
            'min_orders_per_dispatch': max(int(args.min_orders_per_dispatch), 1),
            'enable_delivery_viability': bool(args.enable_delivery_viability),
            'enable_viability_fallback': bool(args.enable_viability_fallback),
            'relax_pickup_commitment_trip_time': bool(args.relax_pickup_commitment_trip_time),
        }
    else:
        state_kwargs = {}

    solver = SimulatedAnnealingSolver(
        iterations=args.iterations,
        initial_temperature=args.initial_temperature,
        cooling_rate=args.cooling_rate,
        num_vehicles=args.num_vehicles,
        seed=args.seed,
        eval_protocol=args.eval_protocol,
        fill_missing_orders=args.fill_missing_orders,
        hard_violation_penalty_weight=args.hard_violation_penalty_weight,
        hard_constraint_mode=args.hard_constraint_mode,
        strict_infeasible_cost=args.strict_infeasible_cost,
        semantic_mode=args.semantic_mode,
        state_kwargs=state_kwargs,
        allow_reject=args.allow_reject,
        scoring_mode=args.scoring_mode,
    )

    results = []
    for instance_id, sample in enumerate(dataset):
        start_time = time.time()
        info = solver.solve(sample)
        info['instance_id'] = instance_id
        info['solve_time'] = time.time() - start_time
        info['iterations'] = args.iterations
        results.append(info)
        print(
            f"[SA] instance={instance_id} objective={info['objective_cost']:.2f} "
            f"raw={info['raw_total_cost']:.2f} CNY vehicles={info['used_vehicles']}"
        )

    summary = summarize_results(results, graph_size=args.graph_size)
    if summary is None:
        return None
    summary.update({
        'graph_size': args.graph_size,
        'num_samples': args.num_samples,
        'iterations': args.iterations,
        'initial_temperature': args.initial_temperature,
        'cooling_rate': args.cooling_rate,
        'num_vehicles': solver.num_vehicles,
        'protocol_version': args.eval_protocol,
        'semantic_mode': args.semantic_mode,
        'state_kwargs': state_kwargs,
        'allow_reject': bool(args.allow_reject),
        'scoring_mode': args.scoring_mode,
        'fill_missing_orders': bool(args.fill_missing_orders),
        'hard_violation_penalty_weight': float(args.hard_violation_penalty_weight),
        'hard_constraint_mode': args.hard_constraint_mode,
        'strict_infeasible_cost': float(args.strict_infeasible_cost),
        'comparability_notes': [] if args.eval_protocol == ALIGNED_PROTOCOL else [
            'Legacy baseline protocol: results are not strictly comparable to DRL hard-mask decode semantics.'
        ],
    })

    with open(args.output_file, 'w', encoding='utf-8') as output_file:
        json.dump(summary, output_file, indent=2, ensure_ascii=False)

    print(f"\n[SA] 结果已保存到 {args.output_file}")
    return summary


def parse_args():
    parser = argparse.ArgumentParser(description='Simulated Annealing baseline for MCVRP-PDTW')
    parser.add_argument('--graph_size', type=int, choices=[25, 50, 100, 200], required=True)
    parser.add_argument('--num_samples', type=int, default=1)
    parser.add_argument('--iterations', type=int, default=1500)
    parser.add_argument('--initial_temperature', type=float, default=50.0)
    parser.add_argument('--cooling_rate', type=float, default=0.995)
    parser.add_argument('--num_vehicles', type=int, default=None)
    parser.add_argument('--seed', type=int, default=1234)

    parser.add_argument('--eval_protocol', choices=[LEGACY_PROTOCOL, ALIGNED_PROTOCOL], default=ALIGNED_PROTOCOL)
    parser.add_argument('--semantic_mode', choices=list(SEMANTIC_MODES), default=DRL_ALIGNED_SEMANTIC_MODE)
    parser.add_argument('--scoring_mode', choices=['objective', 'pax_lex'], default='pax_lex')

    parser.add_argument('--hard_constraint_mode', choices=list(HARD_CONSTRAINT_MODES), default=PENALTY_HARD_CONSTRAINT_MODE)
    parser.add_argument('--strict_infeasible_cost', type=float, default=1e12)
    parser.add_argument('--hard_violation_penalty_weight', type=float, default=float(Config.ALPHA_REJECT))

    parser.add_argument('--fill-missing-orders', dest='fill_missing_orders', action='store_true', default=False)
    parser.add_argument('--no-fill-missing-orders', dest='fill_missing_orders', action='store_false')

    parser.add_argument('--allow-reject', dest='allow_reject', action='store_true', default=True)
    parser.add_argument('--no-allow-reject', dest='allow_reject', action='store_false')

    parser.add_argument('--max_concurrent_open_orders', type=int, default=1)
    parser.add_argument('--min_orders_per_dispatch', type=int, default=4)
    parser.add_argument('--enable_delivery_viability', dest='enable_delivery_viability', action='store_true', default=True)
    parser.add_argument('--disable_delivery_viability', dest='enable_delivery_viability', action='store_false')
    parser.add_argument('--enable_viability_fallback', dest='enable_viability_fallback', action='store_true', default=False)
    parser.add_argument('--relax_pickup_commitment_trip_time', action='store_true', default=False)

    parser.add_argument('--output_file', default='sa_results.json')
    return parser.parse_args()


if __name__ == '__main__':
    run_sa_benchmark(parse_args())
