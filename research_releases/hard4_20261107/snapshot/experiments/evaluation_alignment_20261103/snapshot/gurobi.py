"""
Gurobi-based Baseline Solver for MCVRPPDTW (v5)

v5 关键修复（评审项 1 → 致命）：
- 使用三维决策变量 x[i,j,k]，恢复"Pickup 与 Delivery 必须由同一辆车 k 完成"的硬约束
- 双舱容量都用 MTZ 累计载量约束（独立 u_p[i,k] / u_c[i,k]）
- 目标函数除距离外，可选打开"加权能耗 + 延迟"项，与 DRL 训练目标对齐
- 时间连续性 / 时间窗 / Pickup-先于-Delivery 全部按硬约束写入

代价：
- x[i,j,k] 是 K 倍变量量，对 N=100、K=25 来说约 250000 个 binary，需要充足求解时间
- 推荐用法：N=25 / N=50 上跑严格版作为参考下界；N=100 用 K=ceil(N/4) 与较宽 time_limit
"""

import numpy as np
import time
from gurobipy import Model, GRB, quicksum
import torch
from typing import List, Tuple, Dict, Optional
import json

from problem_mcvrptw_v2 import MCVRPPDTW, MCVRPPDTWDataset, Config


class GurobiMCVRPPDTWSolver:
    """求解严格 MCVRP-PDTW（含 PD 同车硬约束）的 Gurobi 求解器。"""

    def __init__(
        self,
        time_limit: Optional[int] = 3600,
        mip_gap: float = 0.01,
        threads: int = 4,
        verbose: bool = True,
        relax_constraints: bool = False,    # v5 默认不放宽
        objective_mode: str = 'distance',   # 'distance' | 'cost'   v5 新增
    ):
        # time_limit=None 表示不设置 TimeLimit（无限时）
        self.time_limit = None if time_limit is None else int(time_limit)
        self.mip_gap = mip_gap
        self.threads = threads
        self.verbose = verbose
        self.relax_constraints = relax_constraints
        self.objective_mode = objective_mode

    def solve(self, instance: dict) -> Tuple[float, List, Dict]:
        start_time = time.time()

        coords = instance['coords']
        demands = instance['demands']
        demand_p = instance['demand_passenger']
        demand_c = instance['demand_cargo']
        time_windows = instance['time_windows'].copy()
        service_times = instance['service_times']
        capacity_p = instance['passenger_capacity']
        capacity_c = instance['cargo_capacity']
        num_vehicles = instance['num_vehicles']
        depot = instance.get('depot_idx', 0)
        node_type = instance.get('node_type')  # v5: 用于成本目标 (1=passenger, 0=cargo)

        n = len(coords)
        K = num_vehicles

        dist_matrix = self._compute_distance_matrix(coords)
        pd_pairs = self._identify_pickup_delivery_pairs(demands)

        # 可选轻度放宽（默认关闭）
        if self.relax_constraints:
            print("  🔧 应用约束放松策略 (v5: 仅放宽时间窗 ±20%, 不再放宽 PD 同车)")
            for i in range(n):
                if i != depot:
                    width = max(time_windows[i][1] - time_windows[i][0], 1e-3)
                    time_windows[i][0] = max(0, time_windows[i][0] - width * 0.2)
                    time_windows[i][1] = min(48, time_windows[i][1] + width * 0.2)

        # ==================== 模型 ====================
        model = Model("MCVRPPDTW_v5")
        if self.time_limit is not None:
            model.setParam('TimeLimit', self.time_limit)
        model.setParam('MIPGap', self.mip_gap)
        model.setParam('Threads', self.threads)
        model.setParam('MIPFocus', 1)
        model.setParam('NoRelHeurTime', 60)
        model.setParam('PumpPasses', 50)
        model.setParam('PreSparsify', 1)
        if not self.verbose:
            model.setParam('OutputFlag', 0)

        # 决策变量：x[i,j,k] 三维
        x = {}
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                for k in range(K):
                    x[i, j, k] = model.addVar(vtype=GRB.BINARY, name=f'x_{i}_{j}_{k}')

        # 时间到达变量
        t = {}
        for i in range(n):
            t[i] = model.addVar(
                vtype=GRB.CONTINUOUS,
                lb=time_windows[i][0],
                ub=time_windows[i][1],
                name=f't_{i}'
            )

        # 容量累计变量（每辆车独立）
        u_p, u_c = {}, {}
        for i in range(n):
            for k in range(K):
                u_p[i, k] = model.addVar(vtype=GRB.CONTINUOUS, lb=0, ub=capacity_p, name=f'up_{i}_{k}')
                u_c[i, k] = model.addVar(vtype=GRB.CONTINUOUS, lb=0, ub=capacity_c, name=f'uc_{i}_{k}')

        # 车辆是否启用
        y = {k: model.addVar(vtype=GRB.BINARY, name=f'y_{k}') for k in range(K)}

        model.update()

        # ==================== 约束 ====================
        # C1: 每个客户节点恰被一辆车访问一次
        for i in range(1, n):
            model.addConstr(
                quicksum(x[j, i, k] for j in range(n) if j != i for k in range(K)) == 1,
                name=f"visit_{i}"
            )

        # C2: 每辆车每个节点入出对偶（流平衡）
        for k in range(K):
            for h in range(n):
                model.addConstr(
                    quicksum(x[i, h, k] for i in range(n) if i != h)
                    == quicksum(x[h, j, k] for j in range(n) if j != h),
                    name=f"flow_{h}_{k}"
                )

        # C3: 每辆车从 depot 出发至多一次（启用即出发）
        for k in range(K):
            model.addConstr(quicksum(x[depot, j, k] for j in range(1, n)) <= y[k], name=f"depot_out_{k}")
            model.addConstr(quicksum(x[i, depot, k] for i in range(1, n)) <= y[k], name=f"depot_in_{k}")

        # C4: 时间连续性 (Big-M)
        # v5: 速度从 Config 取，与 problem 一致
        speed = Config.VEHICLE_SPEED
        M_time = 48.0
        for k in range(K):
            for i in range(n):
                for j in range(1, n):
                    if i == j:
                        continue
                    travel_time = dist_matrix[i, j] / speed
                    model.addConstr(
                        t[j] >= t[i] + service_times[i] + travel_time - M_time * (1 - x[i, j, k]),
                        name=f"time_{i}_{j}_{k}"
                    )

        # C5: 双舱容量 MTZ (每辆车)
        M_p = max(capacity_p * 2, 1.0)
        M_c = max(capacity_c * 2, 1.0)
        for k in range(K):
            for i in range(n):
                for j in range(1, n):
                    if i == j:
                        continue
                    dp = float(demand_p[j])
                    dc = float(demand_c[j])
                    # pickup demand 才累加（dp>0），delivery 时累计载量减少
                    model.addConstr(
                        u_p[j, k] >= u_p[i, k] + dp - M_p * (1 - x[i, j, k]),
                        name=f"cap_p_{i}_{j}_{k}"
                    )
                    model.addConstr(
                        u_c[j, k] >= u_c[i, k] + dc - M_c * (1 - x[i, j, k]),
                        name=f"cap_c_{i}_{j}_{k}"
                    )

        # C6: ★★ Pickup-Delivery 同车硬约束（v5 关键修复）
        for p_idx, d_idx in pd_pairs:
            # 6a 同车
            for k in range(K):
                model.addConstr(
                    quicksum(x[j, p_idx, k] for j in range(n) if j != p_idx)
                    == quicksum(x[j, d_idx, k] for j in range(n) if j != d_idx),
                    name=f"same_vehicle_{p_idx}_{d_idx}_{k}"
                )
            # 6b 时间先序
            model.addConstr(
                t[d_idx] >= t[p_idx] + service_times[p_idx],
                name=f"pd_time_{p_idx}_{d_idx}"
            )

        # C7: 单趟时长上限（与 problem 一致，硬约束）
        # 这里以 t[depot] 为参考无意义，简化为 max{t[i]} 上限通过 time_window 隐含；
        # 严格版本需引入 trip_start[k]，此处用 OPERATION_END 作宏观上界即可。

        # ==================== 目标函数 ====================
        if self.objective_mode == 'cost' and node_type is not None:
            # v6: 距离 × 平均能耗 × 电价 + 车辆固定成本
            # 平均能耗按 MODELING.md §6 公式 η(W) = 0.18·(1+W/10000) 估算：
            # 在容量上限 W_max = 20·65 + 200·1 = 1500 kg 时 η_full ≈ 0.207 kWh/km
            # 取 W=W_max/2 的近似值作为线性化平均：η_avg ≈ 0.18·(1 + 750/10000) = 0.1935 kWh/km
            W_max_kg = (Config.PASSENGER_CAPACITY * Config.PASSENGER_WEIGHT_KG
                        + Config.CARGO_CAPACITY * Config.CARGO_UNIT_WEIGHT_KG)
            avg_energy = Config.ENERGY_BASE * (1.0 + 0.5 * W_max_kg / Config.ENERGY_WEIGHT_REF)
            energy_term = quicksum(
                dist_matrix[i, j] * avg_energy * Config.ELECTRICITY_PRICE * x[i, j, k]
                for i in range(n) for j in range(n) if i != j for k in range(K)
            )
            # delay 项需要辅助变量；为保持模型轻量，这里只对距离 + 车辆固定成本建模
            vehicle_fixed = quicksum(Config.VEHICLE_COST * y[k] for k in range(K))
            model.setObjective(energy_term + vehicle_fixed, GRB.MINIMIZE)
        else:
            obj_dist = quicksum(
                dist_matrix[i, j] * x[i, j, k]
                for i in range(n) for j in range(n) if i != j for k in range(K)
            )
            model.setObjective(obj_dist, GRB.MINIMIZE)

        # ==================== 求解 ====================
        print(f"  🚀 开始求解 (规模: {n} 节点, K={K}, 严格 PD-同车 模型)")
        print(f"    预估二元变量数: ~{n*n*K}")
        model.optimize()
        solve_time = time.time() - start_time

        status_map = {GRB.OPTIMAL: 'optimal', GRB.TIME_LIMIT: 'time_limit', GRB.INFEASIBLE: 'infeasible'}
        status_str = status_map.get(model.status, 'other')

        if model.SolCount > 0:
            print(f"  ✅ 找到解! Obj: {model.objVal:.2f}")
            routes = self._extract_routes_per_vehicle(x, n, K, depot)
            info = {
                'status': status_str,
                'obj_value': model.objVal,
                'solve_time': solve_time,
                'num_vehicles_used': len([r for r in routes if len(r) > 2]),
                'total_distance': sum(
                    dist_matrix[r[i], r[i + 1]] for r in routes for i in range(len(r) - 1)
                ),
                'routes': routes,
            }
            return model.objVal, routes, info
        else:
            print(f"  ❌ 未找到可行解")
            return float('inf'), [], {'status': status_str, 'solve_time': solve_time}

    # -------------------- helpers --------------------
    def _compute_distance_matrix(self, coords: np.ndarray) -> np.ndarray:
        """计算两两距离 (km).

        v6.2 修复 (评审项 #1):
        - DRL 侧 `problem_mcvrptw_v2._compute_distance_energy` 计算欧氏距离后
          会乘以 `Config.AREA_SIZE` 把归一化坐标 [0,1] 转换成 km。
        - 这里 Gurobi 端必须采用同一单位, 否则会导致 distance / cost 与 DRL
          差出一个 AREA_SIZE 倍 (默认 10×), 论文对比直接失真。
        """
        n = len(coords)
        dist_matrix = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                if i != j:
                    dist_matrix[i, j] = np.linalg.norm(coords[i] - coords[j]) * Config.AREA_SIZE
        return dist_matrix

    def _identify_pickup_delivery_pairs(self, demands: np.ndarray) -> List[Tuple[int, int]]:
        n = len(demands)
        n_orders = (n - 1) // 2
        return [(i + 1, i + 1 + n_orders) for i in range(n_orders)]

    def _extract_routes_per_vehicle(self, x: dict, n: int, K: int, depot: int) -> List[List[int]]:
        """v5: 按车辆 k 分别还原路径。"""
        routes = []
        for k in range(K):
            adj = {i: [] for i in range(n)}
            used = False
            for i in range(n):
                for j in range(n):
                    if i != j and (i, j, k) in x and x[i, j, k].X > 0.5:
                        adj[i].append(j)
                        used = True
            if not used:
                continue
            # 从 depot 出发追路径
            current = depot
            route = [depot]
            visited_in_route = set()
            while True:
                if current in visited_in_route:
                    print(f"    警告: 车辆 {k} 路径中检测到环")
                    break
                visited_in_route.add(current)
                next_options = adj.get(current, [])
                if not next_options:
                    break
                nxt = next_options[0]
                route.append(nxt)
                if nxt == depot:
                    break
                current = nxt
            if len(route) > 2:
                routes.append(route)
        return routes if routes else [[depot]]


# ==================== 工具函数 ====================
def convert_torch_to_gurobi_format(data):
    depot = data['depot'].numpy()
    locs = data['loc'].numpy()
    coords = np.vstack([depot, locs])

    demand_p = data['demand_passenger'].numpy()
    demand_c = data['demand_cargo'].numpy()

    if abs(demand_p).max() <= 1.0 + 1e-5:
        demand_p = demand_p * Config.PASSENGER_CAPACITY
        demand_c = demand_c * Config.CARGO_CAPACITY

    demand_p = np.concatenate([[0.0], demand_p])
    demand_c = np.concatenate([[0.0], demand_c])

    n_orders = data['n_orders']
    demands = np.zeros_like(demand_p)
    for i in range(1, n_orders + 1):
        demands[i] = 1
        demands[i + n_orders] = -1

    time_windows = data['time_windows'].numpy()
    depot_tw = np.array([[Config.OPERATION_START, Config.OPERATION_END]])
    time_windows = np.vstack([depot_tw, time_windows])

    service_times = np.full(len(coords), Config.SERVICE_TIME)
    service_times[0] = 0.0

    node_type = np.zeros(len(coords))
    node_type[1:] = data['node_type'].numpy() if hasattr(data['node_type'], 'numpy') else np.array(data['node_type'])

    # 当前代码口径：K_max = ceil(n_orders / 6)，与 Config.DEFAULT_NUM_VEHICLE_RATIO=1/6 对齐
    import math as _math
    num_vehicles = max(1, _math.ceil(n_orders / 6))

    return {
        'coords': coords,
        'demands': demands,
        'demand_passenger': demand_p,
        'demand_cargo': demand_c,
        'time_windows': time_windows,
        'service_times': service_times,
        'node_type': node_type,
        'passenger_capacity': Config.PASSENGER_CAPACITY,
        'cargo_capacity': Config.CARGO_CAPACITY,
        'num_vehicles': num_vehicles,
        'depot_idx': 0,
    }


def evaluate_gurobi_baseline(dataset, time_limit=600, output_file='gurobi_result.json',
                             objective_mode='distance', relax_constraints=False, budget_seconds=None):
    solver = GurobiMCVRPPDTWSolver(
        time_limit=time_limit,
        verbose=True,
        objective_mode=objective_mode,
        relax_constraints=relax_constraints,
    )
    results = []

    print(f"\n>> v5 严格模式: PD 同车硬约束已恢复, objective={objective_mode}")
    limit_text = 'unlimited' if time_limit is None else f'{int(time_limit)}s'
    print(f">> 开始处理数据集 (共 {len(dataset)} 个样本), time_limit={limit_text}")

    for i, data in enumerate(dataset):
        print(f"\nProcessing instance {i + 1}/{len(dataset)}...")
        instance = convert_torch_to_gurobi_format(data)
        obj, routes, info = solver.solve(instance)
        # v6.2 修复 (评审项 #2): 始终写 instance_id, 让重评估按 id 对齐, 避免
        # 跳过 infeasible 后下游再用 enumerate 索引就错配。
        info['instance_id'] = int(i)
        info.setdefault('status', 'unknown')
        if info['status'] in ['optimal', 'time_limit'] and obj < 1e10:
            info['feasible'] = True
            info['time_limit_seconds'] = int(time_limit) if time_limit is not None else None
            if budget_seconds is not None:
                info['budget_seconds'] = int(budget_seconds)
            results.append(info)
        else:
            print("  ⚠️ 该实例未找到解，跳过统计 (instance_id 已记录)。")
            # 记录到 infeasible 列表, 仍然写入 summary 便于追踪
            results.append({
                'instance_id': int(i),
                'status': info.get('status', 'infeasible'),
                'obj_value': float('inf'),
                'solve_time': info.get('solve_time', 0.0),
                'routes': [],
                'feasible': False,
                'time_limit_seconds': int(time_limit) if time_limit is not None else None,
                'budget_seconds': int(budget_seconds) if budget_seconds is not None else None,
            })

    if not results:
        print("\n❌ 所有实例均未找到可行解。")
        return None

    feasible_results = [r for r in results if r.get('feasible', False)]
    status_counts = {}
    for record in results:
        status = str(record.get('status', 'unknown'))
        status_counts[status] = status_counts.get(status, 0) + 1

    if not feasible_results:
        print("\n❌ 所有实例均未找到可行解 (但已写入 instance_id 用于对齐)。")
        avg_cost = float('inf')
        avg_time = float(np.mean([r['solve_time'] for r in results]))
    else:
        avg_cost = float(np.mean([r['obj_value'] for r in feasible_results]))
        avg_time = float(np.mean([r['solve_time'] for r in feasible_results]))

    num_total = len(results)
    num_feasible = len(feasible_results)
    feasible_rate = float(num_feasible / num_total) if num_total > 0 else 0.0

    summary = {
        'metrics_schema_version': 'fair_v1',
        'avg_cost': avg_cost,
        'avg_cost_feasible_only': avg_cost,
        'avg_solve_time': avg_time,
        'objective_mode': objective_mode,
        'num_total': num_total,
        'num_feasible': num_feasible,
        'feasible_rate': feasible_rate,
        'coverage_solver_feasible': feasible_rate,
        'status_counts': status_counts,
        'time_limit_seconds': int(time_limit) if time_limit is not None else None,
        'budget_seconds': int(budget_seconds) if budget_seconds is not None else None,
        'results': results,
    }
    with open(output_file, 'w') as f:
        json.dump(summary, f, indent=2, default=str)

    print(f"\n✅ 完成! 平均成本: {avg_cost:.2f}, 平均时间: {avg_time:.2f}s")
    return summary


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='MCVRP-PDTW Gurobi 基线 (v6)')
    parser.add_argument('--graph-sizes', type=int, nargs='+', default=[25, 50],
                        help='求解的订单规模列表 (default: 25 50)')
    parser.add_argument('--num-samples', type=int, default=2,
                        help='每种规模的测试实例数 (default: 2)')
    parser.add_argument('--seed', type=int, default=42,
                        help='测试集随机种子 (default: 42)')
    parser.add_argument('--objective-mode', type=str, default='distance',
                        choices=['distance', 'cost'],
                        help="目标函数: 'distance' 线性快速 (默认), 'cost' 含能耗+车辆固定 (5-10× 慢)")
    parser.add_argument('--time-limit-25', type=int, default=600)
    parser.add_argument('--time-limit-50', type=int, default=1800)
    parser.add_argument('--time-limit-100', type=int, default=3600)
    parser.add_argument('--time-limit-200', type=int, default=1800)
    parser.add_argument('--time-limit-default', type=int, default=1800,
                        help='未显式指定规模的默认 time limit (s)')
    parser.add_argument('--no-time-limit', action='store_true',
                        help='不设置 TimeLimit（无限时，直到求解器自然结束）')
    parser.add_argument('--time-budgets', type=int, nargs='*', default=None,
                        help='可选: 预算 sweep (秒)。提供后会按 budget 生成多组结果文件')
    parser.add_argument('--relax', action='store_true',
                        help='放宽时间窗 ±20% (default: False)')
    args = parser.parse_args()

    print("=" * 80)
    print(f"MCVRP-PDTW 完整基准测试 v6 (Gurobi, objective_mode={args.objective_mode})")
    print("=" * 80)

    time_limit_map = {
        25: args.time_limit_25,
        50: args.time_limit_50,
        100: args.time_limit_100,
        200: args.time_limit_200,
    }

    datasets = {}
    for size in args.graph_sizes:
        print(f"生成 {size} 节点测试数据集...")
        datasets[size] = MCVRPPDTWDataset(num_samples=args.num_samples, graph_size=size, seed=args.seed)

    budgets = [int(value) for value in args.time_budgets] if args.time_budgets else None

    for size in args.graph_sizes:
        print(f"\n求解 {size} 节点问题...")
        default_limit = int(time_limit_map.get(size, args.time_limit_default))

        if args.no_time_limit:
            run_budgets = [None]
        else:
            run_budgets = budgets if budgets else [default_limit]

        for budget in run_budgets:
            # 默认保持历史文件名；开启预算 sweep 时附加 budget 后缀
            if budget is None:
                output_file = f'gurobi_results_{size}_{args.objective_mode}.json'
            elif budgets:
                output_file = f'gurobi_results_{size}_{args.objective_mode}_budget{int(budget)}.json'
            else:
                output_file = f'gurobi_results_{size}_{args.objective_mode}.json'

            budget_label = 'unlimited' if budget is None else str(int(budget))
            try:
                evaluate_gurobi_baseline(
                    dataset=datasets[size],
                    time_limit=None if budget is None else int(budget),
                    output_file=output_file,
                    objective_mode=args.objective_mode,
                    relax_constraints=args.relax,
                    budget_seconds=None if budget is None else int(budget),
                )
            except Exception as e:
                print(f"✗ {size} 节点求解失败 (budget={budget_label}): {e}")
