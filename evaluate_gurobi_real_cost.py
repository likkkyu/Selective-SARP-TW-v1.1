"""Gurobi结果重评估工具。

说明：
- 统一使用 `MCVRPPDTW.get_costs(return_details=True)` 的人民币口径字段
- 兼容 `gurobi.py` / `gurobi_v2.py` 导出的结果文件
"""

import argparse
import json
import os

import numpy as np

from baseline_utils import build_solution_info, evaluate_node_routes
from problem_mcvrptw_v2 import MCVRPPDTWDataset



def reevaluate_gurobi_results(graph_size, gurobi_json_path, num_samples=None, seed=42):
    if not os.path.exists(gurobi_json_path):
        raise FileNotFoundError(f'文件不存在: {gurobi_json_path}')

    with open(gurobi_json_path, 'r', encoding='utf-8') as input_file:
        gurobi_data = json.load(input_file)

    if 'results' in gurobi_data:
        gurobi_results = gurobi_data['results']
    elif 'detailed_results' in gurobi_data:
        gurobi_results = gurobi_data['detailed_results']
    else:
        raise ValueError('无法识别的 Gurobi JSON 结构')

    # v6.2 (评审项 #2): 过滤掉 infeasible 项, 按 instance_id 对齐数据集
    feasible_entries = [
        r for r in gurobi_results
        if r.get('feasible', True) and r.get('routes') and r.get('obj_value', float('inf')) < 1e10
    ]
    if not feasible_entries:
        print('Gurobi 结果中无可重评估的可行解。')
        return None

    max_id = max(r.get('instance_id', idx) for idx, r in enumerate(feasible_entries))
    if num_samples is None:
        num_samples = max_id + 1

    dataset = MCVRPPDTWDataset(num_samples=num_samples, graph_size=graph_size, seed=seed)
    reevaluated_results = []
    for index, result in enumerate(feasible_entries):
        instance_id = result.get('instance_id', index)
        if instance_id >= len(dataset):
            print(f'  ⚠️ instance_id={instance_id} 超出数据集范围 ({len(dataset)}), 跳过')
            continue
        routes = result.get('routes', [])
        if not routes:
            continue

        sample = dataset[instance_id]
        objective_cost, details, _ = evaluate_node_routes(sample, routes)
        info = build_solution_info(objective_cost, details, routes, algorithm='Gurobi-Reeval')
        info.update({
            'instance_id': instance_id,
            'gurobi_cost': result.get('cost', result.get('obj_value', result.get('distance_objective', float('inf')))),
        })
        reevaluated_results.append(info)

    if not reevaluated_results:
        return None

    summary = {
        'graph_size': graph_size,
        'num_samples': len(reevaluated_results),
        'gurobi_avg_cost': float(np.mean([entry['gurobi_cost'] for entry in reevaluated_results])),
        'real_avg_cost': float(np.mean([entry['raw_total_cost'] for entry in reevaluated_results])),
        'real_std_cost': float(np.std([entry['raw_total_cost'] for entry in reevaluated_results])),
        'avg_energy_cost': float(np.mean([entry['energy_cost'] for entry in reevaluated_results])),
        'avg_passenger_delivery_delay_cost': float(np.mean([entry['passenger_delivery_delay_cost'] for entry in reevaluated_results])),
        'avg_cargo_delay_cost': float(np.mean([entry['cargo_delay'] for entry in reevaluated_results])),
        'avg_trip_overtime_penalty': float(np.mean([entry.get('trip_overtime_penalty', 0.0) for entry in reevaluated_results])),
        'avg_vehicle_cost': float(np.mean([entry['vehicle_cost'] for entry in reevaluated_results])),
        'avg_distance': float(np.mean([entry['total_distance'] for entry in reevaluated_results])),
        'detailed_results': reevaluated_results,
    }
    return summary



def main():
    parser = argparse.ArgumentParser(description='Re-evaluate Gurobi outputs with DRL costs')
    parser.add_argument('--graph_size', type=int, choices=[25, 50, 100, 200], required=True)
    parser.add_argument('--input', required=True, help='Gurobi 结果 JSON 路径')
    parser.add_argument('--output', default=None, help='重评估结果输出路径')
    parser.add_argument('--seed', type=int, default=99999, help='重评估数据集 seed（应与 DRL 正式评估保持一致）')
    args = parser.parse_args()

    summary = reevaluate_gurobi_results(args.graph_size, args.input, seed=args.seed)
    if summary is None:
        print('没有可重评估的结果。')
        return

    output_path = args.output or os.path.join(
        'outputs',
        'gurobi_reevaluated',
        f'gurobi_reevaluated_{args.graph_size}.json',
    )
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as output_file:
        json.dump(summary, output_file, indent=2, ensure_ascii=False)

    print(f"重评估完成，结果已保存到 {output_path}")
    print(f"真实总成本均值: {summary['real_avg_cost']:.2f} RMB")


if __name__ == '__main__':
    main()
