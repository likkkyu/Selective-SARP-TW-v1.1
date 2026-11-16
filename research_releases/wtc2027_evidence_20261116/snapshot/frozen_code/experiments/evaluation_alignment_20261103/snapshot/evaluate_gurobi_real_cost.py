"""Gurobi结果重评估工具。

说明：
- 统一使用 `MCVRPPDTW.get_costs(return_details=True)` 的人民币口径字段
- 兼容 `gurobi.py` / `gurobi_v2.py` 导出的结果文件
- 新增公平评估输出：覆盖率 + feasible_only + all_samples(惩罚口径)
"""

import argparse
import json
import os

import numpy as np

from baseline_utils import (
    ALIGNED_PROTOCOL,
    LEGACY_PROTOCOL,
    PENALTY_HARD_CONSTRAINT_MODE,
    STRICT_HARD_CONSTRAINT_MODE,
    build_solution_info,
    evaluate_node_routes,
    summarize_results,
)
from problem_mcvrptw_v2 import MCVRPPDTWDataset



def _source_is_feasible(entry):
    return (
        entry.get('feasible', True)
        and bool(entry.get('routes'))
        and entry.get('obj_value', float('inf')) < 1e10
    )



def reevaluate_gurobi_results(
    graph_size,
    gurobi_json_path,
    num_samples=None,
    seed=42,
    coverage_mode='feasible_only',
    eval_protocol=ALIGNED_PROTOCOL,
    hard_constraint_mode=PENALTY_HARD_CONSTRAINT_MODE,
    hard_violation_penalty_weight=0.0,
    strict_infeasible_cost=1e12,
):
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

    if not gurobi_results:
        return {
            'metrics_schema_version': 'fair_v1',
            'graph_size': graph_size,
            'num_samples': 0,
            'coverage': {
                'num_total_input': 0,
                'num_feasible_input': 0,
                'feasible_rate_input': 0.0,
                'num_reevaluated': 0,
                'reevaluated_rate_over_total': 0.0,
            },
            'feasible_only': {'summary': None},
            'all_samples': {'summary': None, 'penalized_objective_mean': None, 'penalized_objective_std': None},
            'detailed_results': [],
        }

    max_id = max(int(record.get('instance_id', index)) for index, record in enumerate(gurobi_results))
    if num_samples is None:
        num_samples = max_id + 1

    dataset = MCVRPPDTWDataset(num_samples=num_samples, graph_size=graph_size, seed=seed)

    feasible_entries = [record for record in gurobi_results if _source_is_feasible(record)]
    if coverage_mode == 'all_samples':
        target_entries = gurobi_results
    else:
        target_entries = feasible_entries

    reevaluated_results = []
    penalized_objectives_all = []

    for index, result in enumerate(target_entries):
        instance_id = int(result.get('instance_id', index))
        source_feasible = _source_is_feasible(result)

        if instance_id >= len(dataset):
            print(f'  ⚠️ instance_id={instance_id} 超出数据集范围 ({len(dataset)}), 跳过')
            continue

        routes = result.get('routes', [])
        if source_feasible:
            sample = dataset[instance_id]
            objective_cost, details, _ = evaluate_node_routes(
                sample,
                routes,
                eval_protocol=eval_protocol,
                hard_violation_penalty_weight=hard_violation_penalty_weight,
                hard_constraint_mode=hard_constraint_mode,
                strict_infeasible_cost=strict_infeasible_cost,
            )
            info = build_solution_info(
                objective_cost,
                details,
                routes,
                algorithm='Gurobi-Reeval',
                extra={'graph_size': graph_size},
            )
            info.update({
                'instance_id': instance_id,
                'source_status': result.get('status', 'unknown'),
                'source_feasible': bool(result.get('feasible', source_feasible)),
                'source_solve_time': float(result.get('solve_time', 0.0)),
                'source_obj_value': float(result.get('obj_value', float('inf'))),
                'source_time_limit_seconds': result.get('time_limit_seconds'),
                'source_budget_seconds': result.get('budget_seconds'),
                'gurobi_cost': result.get('cost', result.get('obj_value', result.get('distance_objective', float('inf')))),
            })
            reevaluated_results.append(info)
            penalized_objectives_all.append(float(info['objective_cost']))
        elif coverage_mode == 'all_samples':
            penalized_objectives_all.append(float(strict_infeasible_cost))

    feasible_only_summary = summarize_results(reevaluated_results, graph_size=graph_size) if reevaluated_results else None

    num_total_input = len(gurobi_results)
    num_feasible_input = len(feasible_entries)
    num_reevaluated = len(reevaluated_results)

    output = {
        'metrics_schema_version': 'fair_v1',
        'graph_size': graph_size,
        'seed': seed,
        'num_samples': num_samples,
        'eval_config': {
            'coverage_mode': coverage_mode,
            'eval_protocol': eval_protocol,
            'hard_constraint_mode': hard_constraint_mode,
            'hard_violation_penalty_weight': float(hard_violation_penalty_weight),
            'strict_infeasible_cost': float(strict_infeasible_cost),
        },
        'coverage': {
            'num_total_input': num_total_input,
            'num_feasible_input': num_feasible_input,
            'feasible_rate_input': float(num_feasible_input / num_total_input) if num_total_input > 0 else 0.0,
            'num_reevaluated': num_reevaluated,
            'reevaluated_rate_over_total': float(num_reevaluated / num_total_input) if num_total_input > 0 else 0.0,
            'reevaluated_rate_over_feasible': float(num_reevaluated / num_feasible_input) if num_feasible_input > 0 else 0.0,
        },
        'feasible_only': {
            'num_samples': num_reevaluated,
            'summary': feasible_only_summary,
        },
        'all_samples': {
            'num_samples': num_total_input,
            'summary': feasible_only_summary,
            'penalized_objective_mean': float(np.mean(penalized_objectives_all)) if penalized_objectives_all else None,
            'penalized_objective_std': float(np.std(penalized_objectives_all)) if penalized_objectives_all else None,
        },
        'detailed_results': reevaluated_results,
    }

    # 兼容旧字段（历史脚本仍会读取）
    if reevaluated_results:
        output.update({
            'gurobi_avg_cost': float(np.mean([entry['gurobi_cost'] for entry in reevaluated_results])),
            'real_avg_cost': float(np.mean([entry['raw_total_cost'] for entry in reevaluated_results])),
            'real_std_cost': float(np.std([entry['raw_total_cost'] for entry in reevaluated_results])),
            'avg_energy_cost': float(np.mean([entry['energy_cost'] for entry in reevaluated_results])),
            'avg_passenger_delivery_delay_cost': float(np.mean([entry['passenger_delivery_delay_cost'] for entry in reevaluated_results])),
            'avg_cargo_delay_cost': float(np.mean([entry['cargo_delay'] for entry in reevaluated_results])),
            'avg_trip_overtime_penalty': float(np.mean([entry.get('trip_overtime_penalty', 0.0) for entry in reevaluated_results])),
            'avg_vehicle_cost': float(np.mean([entry['vehicle_cost'] for entry in reevaluated_results])),
            'avg_distance': float(np.mean([entry['total_distance'] for entry in reevaluated_results])),
        })
    else:
        output.update({
            'gurobi_avg_cost': None,
            'real_avg_cost': None,
            'real_std_cost': None,
            'avg_energy_cost': None,
            'avg_passenger_delivery_delay_cost': None,
            'avg_cargo_delay_cost': None,
            'avg_trip_overtime_penalty': None,
            'avg_vehicle_cost': None,
            'avg_distance': None,
        })

    return output



def main():
    parser = argparse.ArgumentParser(description='Re-evaluate Gurobi outputs with DRL costs')
    parser.add_argument('--graph_size', type=int, choices=[25, 50, 100, 200], required=True)
    parser.add_argument('--input', required=True, help='Gurobi 结果 JSON 路径')
    parser.add_argument('--output', default=None, help='重评估结果输出路径')
    parser.add_argument('--seed', type=int, default=99999, help='重评估数据集 seed（应与 DRL 正式评估保持一致）')
    parser.add_argument('--num-samples', type=int, default=None, help='可选：显式数据集样本数，默认按输入 instance_id 自动推断')
    parser.add_argument('--coverage-mode', choices=['feasible_only', 'all_samples'], default='feasible_only',
                        help='重评估条目范围；推荐 all_samples 用于论文透明报告')
    parser.add_argument('--eval-protocol', choices=[LEGACY_PROTOCOL, ALIGNED_PROTOCOL], default=ALIGNED_PROTOCOL)
    parser.add_argument('--hard-constraint-mode', choices=[PENALTY_HARD_CONSTRAINT_MODE, STRICT_HARD_CONSTRAINT_MODE],
                        default=PENALTY_HARD_CONSTRAINT_MODE)
    parser.add_argument('--hard-violation-penalty-weight', type=float, default=0.0)
    parser.add_argument('--strict-infeasible-cost', type=float, default=1e12)
    args = parser.parse_args()

    summary = reevaluate_gurobi_results(
        graph_size=args.graph_size,
        gurobi_json_path=args.input,
        num_samples=args.num_samples,
        seed=args.seed,
        coverage_mode=args.coverage_mode,
        eval_protocol=args.eval_protocol,
        hard_constraint_mode=args.hard_constraint_mode,
        hard_violation_penalty_weight=args.hard_violation_penalty_weight,
        strict_infeasible_cost=args.strict_infeasible_cost,
    )

    output_path = args.output or os.path.join(
        'outputs',
        'gurobi_reevaluated',
        f'gurobi_reevaluated_{args.graph_size}.json',
    )
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as output_file:
        json.dump(summary, output_file, indent=2, ensure_ascii=False)

    print(f"重评估完成，结果已保存到 {output_path}")
    coverage = summary.get('coverage', {})
    print(
        f"覆盖率: feasible_input={coverage.get('num_feasible_input', 0)}/{coverage.get('num_total_input', 0)}"
        f" ({coverage.get('feasible_rate_input', 0.0):.2%})"
    )
    if summary.get('real_avg_cost') is not None:
        print(f"真实总成本均值(feasible_only): {summary['real_avg_cost']:.2f} RMB")


if __name__ == '__main__':
    main()
