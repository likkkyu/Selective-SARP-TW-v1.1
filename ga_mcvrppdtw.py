"""Genetic Algorithm baseline for MCVRP-PDTW.

核心设计：
- 染色体编码为订单序列（order id permutation）
- `chromosome_to_routes` 在订单级别切分车辆路径，因此 pickup/delivery 始终在同一车辆上
- 所有成本评估统一复用 `MCVRPPDTW.get_costs`
"""

import argparse
import json
import random
import time

from baseline_utils import (
    build_solution_info,
    default_num_vehicles,
    evaluate_order_routes,
    summarize_results,
)
from problem_mcvrptw_v2 import MCVRPPDTWDataset, Config


class GeneticAlgorithmSolver:
    def __init__(self, population_size, generations, mutation_rate, elite_size=4, num_vehicles=None, seed=1234):
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.elite_size = elite_size
        self.num_vehicles = num_vehicles
        self.rng = random.Random(seed)

    def repair_pd_order(self, chromosome, n_orders):
        """去重+补全，保证 0..n_orders-1 恰好各出现一次。"""
        seen = set()
        repaired = []
        for gene in chromosome:
            gene = int(gene)
            if 0 <= gene < n_orders and gene not in seen:
                repaired.append(gene)
                seen.add(gene)
        for gene in range(n_orders):
            if gene not in seen:
                repaired.append(gene)
        return repaired

    def chromosome_to_routes(self, chromosome, n_orders, sample):
        """将染色体（一条订单序列）转成车辆级 routes。
        添加时间窗预估：如果当前车辆预估总行程超时 (MAX_TRIP_TIME) 或超过营业时间，也触发开新车。
        """
        if self.num_vehicles is None:
            self.num_vehicles = default_num_vehicles(n_orders)
        max_vehicles = max(int(self.num_vehicles), 1)

        demand_p_per_order = sample['demand_passenger'][:n_orders]
        demand_c_per_order = sample['demand_cargo'][:n_orders]

        max_cap_p = 1.0  
        max_cap_c = 1.0

        from problem_mcvrptw_v2 import Config
        AREA_SIZE = Config.AREA_SIZE
        VEHICLE_SPEED = Config.VEHICLE_SPEED
        SERVICE_TIME = Config.SERVICE_TIME
        MAX_TRIP_TIME = Config.MAX_TRIP_TIME
        OP_START = Config.OPERATION_START
        OP_END = Config.OPERATION_END

        loc = sample['loc']  # [2N, 2]
        depot = sample['depot'].squeeze() # [2]
        time_windows = sample['time_windows'] # [2N, 2]
        
        def get_loc(node_idx):
            if node_idx == 0:
                return depot
            return loc[node_idx - 1]

        
        routes = []
        cur_route = []
        cur_load_p = 0.0
        cur_load_c = 0.0
        
        # Track time and location for current vehicle
        cur_time = OP_START
        last_node = 0  # 0 is depot

        for gene in chromosome:
            order_id = int(gene)
            if not (0 <= order_id < n_orders):
                continue

            dp = float(demand_p_per_order[order_id].item())
            dc = float(demand_c_per_order[order_id].item())

            # 节点索引
            pickup_node = order_id + 1
            delivery_node = order_id + n_orders + 1
            
            # ====== 时间窗预估逻辑 ======
            # 1. 走到 pickup
            d_p = (get_loc(last_node) - get_loc(pickup_node)).norm(p=2).item() * AREA_SIZE
            t_p_arr = cur_time + d_p / VEHICLE_SPEED
            t_p_tw_start = float(time_windows[order_id, 0].item())
            t_p_start = max(t_p_arr, t_p_tw_start)
            t_p_finish = t_p_start + SERVICE_TIME
            
            # 2. 走到 delivery
            d_d = (get_loc(pickup_node) - get_loc(delivery_node)).norm(p=2).item() * AREA_SIZE
            t_d_arr = t_p_finish + d_d / VEHICLE_SPEED
            t_d_tw_start = float(time_windows[order_id + n_orders, 0].item())
            t_d_start = max(t_d_arr, t_d_tw_start)
            t_d_finish = t_d_start + SERVICE_TIME
            
            # 3. 如果此刻回 depot
            d_depot = (get_loc(delivery_node) - get_loc(0)).norm(p=2).item() * AREA_SIZE
            t_depot_arr = t_d_finish + d_depot / VEHICLE_SPEED
            
            # 检查：总行程耗时是否 > MAX_TRIP_TIME，或者服务完成时间 > OP_END
            trip_time = t_depot_arr - OP_START
            will_exceed_time = (trip_time > MAX_TRIP_TIME) or (t_p_start > OP_END) or (t_d_start > OP_END)

            will_exceed_p = (cur_load_p + max(dp, 0.0)) > max_cap_p + 1e-6
            will_exceed_c = (cur_load_c + max(dc, 0.0)) > max_cap_c + 1e-6
            
            if cur_route and (will_exceed_p or will_exceed_c or will_exceed_time) and len(routes) + 1 < max_vehicles:
                # 开启新车
                routes.append(cur_route)
                cur_route = []
                cur_load_p = 0.0
                cur_load_c = 0.0
                
                # 新车的首次访问时间重新计算
                cur_time = OP_START
                last_node = 0
                
                d_p = (get_loc(last_node) - get_loc(pickup_node)).norm(p=2).item() * AREA_SIZE
                t_p_arr = cur_time + d_p / VEHICLE_SPEED
                t_p_start = max(t_p_arr, t_p_tw_start)
                t_p_finish = t_p_start + SERVICE_TIME
                d_d = (get_loc(pickup_node) - get_loc(delivery_node)).norm(p=2).item() * AREA_SIZE
                t_d_arr = t_p_finish + d_d / VEHICLE_SPEED
                t_d_start = max(t_d_arr, t_d_tw_start)
                t_d_finish = t_d_start + SERVICE_TIME

            # 更新当前车状态
            cur_route.append(order_id)
            cur_load_p += max(dp, 0.0)
            cur_load_c += max(dc, 0.0)
            cur_time = t_d_finish
            last_node = delivery_node

        if cur_route:
            routes.append(cur_route)

        return routes[:max_vehicles]

    def evaluate_chromosome(self, sample, chromosome):
        n_orders = int(sample['n_orders'])
        repaired = self.repair_pd_order(chromosome, n_orders)
        order_routes = self.chromosome_to_routes(repaired, n_orders, sample)
        objective_cost, details, _, repaired_routes, node_routes = evaluate_order_routes(
            sample,
            order_routes,
            num_vehicles=self.num_vehicles,
        )
        info = build_solution_info(objective_cost, details, node_routes, algorithm='GA')
        info['chromosome'] = repaired
        info['order_routes'] = repaired_routes
        return objective_cost, info

    def initialize_population(self, n_orders):
        base = list(range(n_orders))
        population = []
        for _ in range(self.population_size):
            chromosome = base[:]
            self.rng.shuffle(chromosome)
            population.append(chromosome)
        return population

    def ordered_crossover(self, parent_a, parent_b):
        size = len(parent_a)
        left, right = sorted(self.rng.sample(range(size), 2))
        child = [None] * size
        child[left:right + 1] = parent_a[left:right + 1]

        fill_values = [gene for gene in parent_b if gene not in child]
        fill_index = 0
        for index in range(size):
            if child[index] is None:
                child[index] = fill_values[fill_index]
                fill_index += 1
        return child

    def mutate(self, chromosome):
        mutated = chromosome[:]
        if self.rng.random() < self.mutation_rate:
            left, right = self.rng.sample(range(len(mutated)), 2)
            mutated[left], mutated[right] = mutated[right], mutated[left]
        return mutated

    def solve(self, sample):
        n_orders = int(sample['n_orders'])
        # 若用户未显式给最大车辆数，则按配置给一个上界
        if self.num_vehicles is None:
            self.num_vehicles = default_num_vehicles(n_orders)

        population = self.initialize_population(n_orders)
        best_info = None
        best_cost = float('inf')

        for _ in range(self.generations):
            scored_population = []
            for chromosome in population:
                cost, info = self.evaluate_chromosome(sample, chromosome)
                scored_population.append((cost, chromosome, info))
                if cost < best_cost:
                    best_cost = cost
                    best_info = info

            scored_population.sort(key=lambda item: item[0])
            elites = [chromosome for _, chromosome, _ in scored_population[:self.elite_size]]
            next_population = elites[:]

            while len(next_population) < self.population_size:
                parent_a = self.rng.choice(scored_population[: max(self.population_size // 2, 2)])[1]
                parent_b = self.rng.choice(scored_population[: max(self.population_size // 2, 2)])[1]
                child = self.ordered_crossover(parent_a, parent_b)
                next_population.append(self.mutate(child))

            population = next_population

        return best_info


def run_ga_benchmark(args):
    dataset = MCVRPPDTWDataset(num_samples=args.num_samples, graph_size=args.graph_size, seed=args.seed)
    solver = GeneticAlgorithmSolver(
        population_size=args.population_size,
        generations=args.generations,
        mutation_rate=args.mutation_rate,
        elite_size=args.elite_size,
        num_vehicles=args.num_vehicles,
        seed=args.seed,
    )

    results = []
    for instance_id, sample in enumerate(dataset):
        start_time = time.time()
        info = solver.solve(sample)
        info['instance_id'] = instance_id
        info['solve_time'] = time.time() - start_time
        info['population_size'] = args.population_size
        info['generations'] = args.generations
        results.append(info)
        print(
            f"[GA] instance={instance_id} objective={info['objective_cost']:.2f} "
            f"raw={info['raw_total_cost']:.2f} CNY vehicles={info['used_vehicles']}"
        )

    summary = summarize_results(results)
    if summary is None:
        return None
    summary.update({
        'graph_size': args.graph_size,
        'num_samples': args.num_samples,
        'population_size': args.population_size,
        'generations': args.generations,
        'mutation_rate': args.mutation_rate,
        'num_vehicles': solver.num_vehicles,
    })

    with open(args.output_file, 'w', encoding='utf-8') as output_file:
        json.dump(summary, output_file, indent=2, ensure_ascii=False)

    print(f"\n[GA] 结果已保存到 {args.output_file}")
    return summary


def parse_args():
    parser = argparse.ArgumentParser(description='Genetic Algorithm baseline for MCVRP-PDTW')
    parser.add_argument('--graph_size', type=int, choices=[25, 50, 100], required=True)
    parser.add_argument('--num_samples', type=int, default=1)
    parser.add_argument('--population_size', type=int, default=48)
    parser.add_argument('--generations', type=int, default=120)
    parser.add_argument('--mutation_rate', type=float, default=0.2)
    parser.add_argument('--elite_size', type=int, default=4)
    parser.add_argument('--num_vehicles', type=int, default=None)
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--output_file', default='ga_results.json')
    return parser.parse_args()


if __name__ == '__main__':
    run_ga_benchmark(parse_args())
