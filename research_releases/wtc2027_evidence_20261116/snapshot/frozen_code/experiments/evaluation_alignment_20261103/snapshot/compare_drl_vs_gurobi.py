"""
Selective SARP-TW：DRL vs Gurobi 公平对比脚本
Fair Comparison between DRL and Gurobi

【双轨口径】
- Track-1【距离口径】DRL.total_distance vs Gurobi(objective_mode='distance')
- Track-2【真实成本口径】DRL.total_cost_raw vs Gurobi 重评估 raw_total_cost

依赖:
- gurobi_results_{N}_distance.json
- gurobi_results_{N}_cost.json
- outputs/gurobi_reevaluated/gurobi_reevaluated_{N}.json
"""

import argparse
import json
import os
import time

import numpy as np
import torch

from problem_mcvrptw_v2 import MCVRPPDTW, MCVRPPDTWDataset
from nets.attention_model import AttentionModel, set_decode_type


def build_model_from_checkpoint(checkpoint, device, problem):
    args_dict = checkpoint.get('args', {}) or {}
    if not isinstance(args_dict, dict):
        try:
            args_dict = vars(args_dict)
        except Exception:
            args_dict = {}

    checkpoint_shrink_size = args_dict.get('shrink_size', None)
    decode_pickup_urgency_bias = args_dict.get('decode_pickup_urgency_bias', 0.0)
    decode_pickup_urgency_horizon_hours = args_dict.get('decode_pickup_urgency_horizon_hours', 1.0)

    return AttentionModel(
        embedding_dim=args_dict.get('embedding_dim', 256),
        hidden_dim=args_dict.get('hidden_dim', 256),
        n_encode_layers=args_dict.get('n_encode_layers', 6),
        tanh_clipping=args_dict.get('tanh_clipping', 10.),
        normalization=args_dict.get('normalization', 'batch'),
        n_heads=args_dict.get('n_heads', 8),
        shrink_size=checkpoint_shrink_size,
        decode_pickup_urgency_bias=decode_pickup_urgency_bias,
        decode_pickup_urgency_horizon_hours=decode_pickup_urgency_horizon_hours,
        problem=problem,
    ).to(device)


def _parse_checkpoint_overrides(values):
    mapping = {}
    for item in values or []:
        text = str(item)
        if '=' not in text:
            raise ValueError(f"Invalid --checkpoint-overrides entry: {text}. Expected N=path")
        size_text, path = text.split('=', 1)
        mapping[int(size_text)] = path
    return mapping


def _resolve_checkpoint_path(graph_size, checkpoint_overrides):
    if graph_size in checkpoint_overrides:
        return checkpoint_overrides[graph_size]

    candidates = [
        f'outputs/pomo_n{graph_size}_optimized/model_best.pt',
        f'outputs/pomo_n{graph_size}_optimized/model_best_service.pt',
        f'outputs/pomo_n{graph_size}_optimized/model_best_objective.pt',
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    return candidates[0]


def evaluate_drl(graph_size, num_samples=5, seed=42, checkpoint_path=None, device=None, state_kwargs=None):
    """评估 DRL 模型，返回距离 + raw_cost 双轨指标。"""
    device = device or torch.device('cpu')

    if not os.path.exists(checkpoint_path):
        print(f"  ✗ 模型不存在: {checkpoint_path}")
        return None

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = build_model_from_checkpoint(checkpoint, device, MCVRPPDTW)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    set_decode_type(model, 'greedy')

    dataset = MCVRPPDTWDataset(num_samples=num_samples, graph_size=graph_size, seed=seed)

    distances, raw_costs, energy_costs, times = [], [], [], []
    for sample in dataset.data:
        batch = {
            key: (value.unsqueeze(0).to(device) if torch.is_tensor(value) else value)
            for key, value in sample.items()
        }
        start = time.time()
        with torch.no_grad():
            _, _, pi = model(batch, return_pi=True, state_kwargs=state_kwargs)
            _, details = MCVRPPDTW.get_costs(batch, pi, return_details=True)
        times.append(time.time() - start)
        distances.append(details['total_distance'].item())
        raw_costs.append(details['total_cost_raw'].item())
        energy_costs.append(details['energy_cost_raw'].item())

    return {
        'checkpoint': checkpoint_path,
        'avg_distance': float(np.mean(distances)),
        'std_distance': float(np.std(distances)),
        'avg_total_cost_raw': float(np.mean(raw_costs)),
        'std_total_cost_raw': float(np.std(raw_costs)),
        'avg_energy_cost': float(np.mean(energy_costs)),
        'avg_time': float(np.mean(times)),
    }


def _load_json_if_exists(path):
    if not os.path.exists(path):
        return None
    with open(path, 'r', encoding='utf-8') as input_file:
        return json.load(input_file)


def _extract_coverage_fields(data):
    if not isinstance(data, dict):
        return {
            'num_total': None,
            'num_feasible': None,
            'feasible_rate': None,
        }
    num_total = data.get('num_total')
    num_feasible = data.get('num_feasible')
    feasible_rate = data.get('feasible_rate')
    if feasible_rate is None and isinstance(num_total, int) and num_total > 0 and isinstance(num_feasible, int):
        feasible_rate = float(num_feasible / num_total)
    return {
        'num_total': num_total,
        'num_feasible': num_feasible,
        'feasible_rate': feasible_rate,
    }


def load_gurobi_distance_track(size, results_dir='.', budget_seconds=None):
    """Track-1 距离口径：读 gurobi_results_{N}_distance(.json / budget).json。"""
    candidates = []
    if budget_seconds is not None:
        candidates.append(os.path.join(results_dir, f'gurobi_results_{size}_distance_budget{int(budget_seconds)}.json'))
    candidates.extend([
        os.path.join(results_dir, f'gurobi_results_{size}_distance.json'),
        os.path.join(results_dir, f'gurobi_results_{size}.json'),  # 兼容旧文件名
    ])

    for path in candidates:
        data = _load_json_if_exists(path)
        if data is None:
            continue
        coverage = _extract_coverage_fields(data)
        return {
            'avg_distance': data.get('avg_cost', data.get('avg_cost_feasible_only', 0)),
            'avg_time': data.get('avg_solve_time', 0),
            'source': path,
            'coverage': coverage,
            'time_limit_seconds': data.get('time_limit_seconds'),
            'budget_seconds': data.get('budget_seconds', budget_seconds),
        }
    return None


def load_gurobi_cost_track(size, results_dir='.', reeval_dir='outputs/gurobi_reevaluated', budget_seconds=None):
    """Track-2 真实成本口径：优先读重评估文件，其次读 cost 模式结果。"""
    reeval_path = os.path.join(reeval_dir, f'gurobi_reevaluated_{size}.json')
    data = _load_json_if_exists(reeval_path)
    if data is not None:
        feasible_summary = data.get('feasible_only', {}).get('summary') if isinstance(data.get('feasible_only'), dict) else None
        real_avg_cost = data.get('real_avg_cost')
        if real_avg_cost is None and isinstance(feasible_summary, dict):
            real_avg_cost = feasible_summary.get('avg_raw_total_cost')
        coverage_data = data.get('coverage', {}) if isinstance(data.get('coverage'), dict) else {}
        coverage = {
            'num_total': coverage_data.get('num_total_input'),
            'num_feasible': coverage_data.get('num_feasible_input'),
            'feasible_rate': coverage_data.get('feasible_rate_input'),
        }
        return {
            'avg_total_cost_raw': real_avg_cost,
            'avg_energy_cost': data.get('avg_energy_cost'),
            'avg_distance': data.get('avg_distance'),
            'avg_time': None,
            'source': reeval_path,
            'coverage': coverage,
            'all_samples_penalized_objective_mean': data.get('all_samples', {}).get('penalized_objective_mean')
            if isinstance(data.get('all_samples'), dict) else None,
        }

    candidates = []
    if budget_seconds is not None:
        candidates.append(os.path.join(results_dir, f'gurobi_results_{size}_cost_budget{int(budget_seconds)}.json'))
    candidates.append(os.path.join(results_dir, f'gurobi_results_{size}_cost.json'))

    for cost_path in candidates:
        raw = _load_json_if_exists(cost_path)
        if raw is None:
            continue
        coverage = _extract_coverage_fields(raw)
        return {
            'avg_total_cost_raw': raw.get('avg_cost', raw.get('avg_cost_feasible_only', 0)),
            'avg_energy_cost': None,
            'avg_distance': None,
            'avg_time': raw.get('avg_solve_time', 0),
            'source': cost_path,
            'coverage': coverage,
            'all_samples_penalized_objective_mean': None,
        }
    return None


def load_gurobi_anytime_curve(size, objective_mode, results_dir, budgets):
    points = []
    if not budgets:
        return points
    for budget in budgets:
        path = os.path.join(results_dir, f'gurobi_results_{size}_{objective_mode}_budget{int(budget)}.json')
        data = _load_json_if_exists(path)
        if data is None:
            continue
        points.append({
            'budget_seconds': int(budget),
            'objective': data.get('avg_cost', data.get('avg_cost_feasible_only')),
            'feasible_rate': data.get('feasible_rate'),
            'num_total': data.get('num_total'),
            'num_feasible': data.get('num_feasible'),
            'source': path,
        })
    points.sort(key=lambda item: item['budget_seconds'])
    return points


def parse_args():
    parser = argparse.ArgumentParser(description='DRL vs Gurobi 双轨对比 (v8 fair-aware)')
    parser.add_argument('--graph-sizes', type=int, nargs='+', default=[25, 50, 100])
    parser.add_argument('--num-samples', type=int, default=5)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--no-cuda', action='store_true')
    parser.add_argument('--max-concurrent-open-orders', type=int, default=6)
    parser.add_argument('--min-orders-per-dispatch', type=int, default=4)
    parser.add_argument('--enable-delivery-viability', action='store_true', default=True)
    parser.add_argument('--disable-delivery-viability', action='store_false', dest='enable_delivery_viability')
    parser.add_argument('--enable-viability-fallback', action='store_true', default=False)
    parser.add_argument('--relax-pickup-commitment-trip-time', action='store_true', default=False)
    parser.add_argument('--results-dir', type=str, default='.')
    parser.add_argument('--reeval-dir', type=str, default=os.path.join('outputs', 'gurobi_reevaluated'))
    parser.add_argument('--checkpoint-overrides', nargs='*', default=None,
                        help='可选：为特定规模指定 checkpoint 路径，如 100=outputs/x/model_best_service.pt')
    parser.add_argument('--comparison-mode', choices=['legacy', 'fair'], default='fair',
                        help='legacy 保持历史输出；fair 增加 coverage 与结论守卫')
    parser.add_argument('--min-gurobi-coverage', type=float, default=0.5,
                        help='fair 模式下低覆盖阈值')
    parser.add_argument('--budget-seconds', type=int, default=None,
                        help='可选：读取某个 budget 后缀文件进行对比')
    parser.add_argument('--anytime-budgets-sec', type=int, nargs='*', default=None,
                        help='可选：读取 budget sweep 文件并输出 anytime 曲线数据')
    parser.add_argument('--output', type=str, default=os.path.join('outputs', 'drl_vs_gurobi_comparison.json'))
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device('cuda' if (torch.cuda.is_available() and not args.no_cuda) else 'cpu')
    checkpoint_overrides = _parse_checkpoint_overrides(args.checkpoint_overrides)
    state_kwargs = {
        'max_concurrent_open_orders': int(args.max_concurrent_open_orders),
        'min_orders_per_dispatch': int(args.min_orders_per_dispatch),
        'enable_delivery_viability': bool(args.enable_delivery_viability),
        'enable_viability_fallback': bool(args.enable_viability_fallback),
        'relax_pickup_commitment_trip_time': bool(args.relax_pickup_commitment_trip_time),
    }

    print('=' * 80)
    print('DRL vs Gurobi 双轨对比 (v8 fair-aware)')
    print('=' * 80)
    print(f"state_kwargs: {state_kwargs}")

    print('\n[1] 评估 DRL 模型...')
    drl_results = {}
    for size in args.graph_sizes:
        checkpoint_path = _resolve_checkpoint_path(size, checkpoint_overrides)
        print(f"\n  评估 N={size} ... checkpoint={checkpoint_path}")
        result = evaluate_drl(
            size,
            num_samples=args.num_samples,
            seed=args.seed,
            checkpoint_path=checkpoint_path,
            device=device,
            state_kwargs=state_kwargs,
        )
        if result:
            drl_results[size] = result
            print(f"    距离          : {result['avg_distance']:.3f} ± {result['std_distance']:.3f} km")
            print(f"    总成本(raw)   : {result['avg_total_cost_raw']:.3f} ± {result['std_total_cost_raw']:.3f} RMB")
            print(f"    能耗成本(raw) : {result['avg_energy_cost']:.3f} RMB")
            print(f"    DRL 推理时间  : {result['avg_time'] * 1000:.1f} ms")

    print('\n[2] 加载 Gurobi 双轨结果...')
    gurobi_dist = {
        size: load_gurobi_distance_track(size, results_dir=args.results_dir, budget_seconds=args.budget_seconds)
        for size in args.graph_sizes
    }
    gurobi_cost = {
        size: load_gurobi_cost_track(size, results_dir=args.results_dir, reeval_dir=args.reeval_dir, budget_seconds=args.budget_seconds)
        for size in args.graph_sizes
    }

    if not any(gurobi_dist.values()) and not any(gurobi_cost.values()):
        print('  ✗ 未找到任何 Gurobi 结果文件')
        print('  请先运行: python gurobi.py --objective-mode distance')
        print('        或: python gurobi.py --objective-mode cost')
        return

    print('\n' + '=' * 80)
    print('【Track 1】距离口径 (km)')
    print('=' * 80)
    print(f"\n{'规模':<8} | {'指标':<14} | {'DRL':<14} | {'Gurobi':<14} | {'Gap':>10}")
    print('-' * 80)
    for size in args.graph_sizes:
        if size in drl_results and gurobi_dist.get(size):
            drl = drl_results[size]
            gur = gurobi_dist[size]
            d_drl = drl['avg_distance']
            d_gur = gur['avg_distance']
            gap = (d_drl - d_gur) / d_gur * 100 if d_gur and d_gur > 0 else 0
            print(f"{size}订单  | {'距离(km)':<14} | {d_drl:<14.3f} | {d_gur:<14.3f} | {gap:>+9.1f}%")
            t_drl = drl['avg_time']
            t_gur = gur['avg_time']
            speedup = t_gur / t_drl if t_drl > 0 and t_gur is not None else 0
            print(f"{'':8} | {'时间(s)':<14} | {t_drl:<14.4f} | {t_gur:<14.2f} | {speedup:>9.0f}x")
            print('-' * 80)

    print('\n' + '=' * 80)
    print('【Track 2】真实成本口径 (RMB)')
    print('=' * 80)
    print(f"\n{'规模':<8} | {'指标':<14} | {'DRL':<14} | {'Gurobi':<14} | {'Gap':>10}")
    print('-' * 80)
    for size in args.graph_sizes:
        if size in drl_results and gurobi_cost.get(size):
            drl = drl_results[size]
            gur = gurobi_cost[size]
            c_drl = drl['avg_total_cost_raw']
            c_gur = gur['avg_total_cost_raw']
            if c_gur is None or c_gur <= 0:
                gap = None
            else:
                gap = (c_drl - c_gur) / c_gur * 100
            gap_text = f"{gap:+9.1f}%" if gap is not None else '   n/a   '
            print(f"{size}订单  | {'总成本(RMB)':<14} | {c_drl:<14.3f} | {str(round(c_gur, 3)) if c_gur is not None else 'n/a':<14} | {gap_text}")
            print(f"{'':8} | {'数据来源':<14} | DRL eval     | {os.path.basename(gur['source']):<14}")
            print('-' * 80)

    fair_summary = None
    if args.comparison_mode == 'fair':
        fair_by_size = {}
        low_coverage_sizes = []
        print('\n' + '=' * 80)
        print('【Fair 模式】覆盖率与结论守卫')
        print('=' * 80)
        print(f"\n{'规模':<8} | {'num_feasible/num_total':<24} | {'feasible_rate':<14} | {'状态':<16}")
        print('-' * 80)
        for size in args.graph_sizes:
            coverage = None
            source = None
            if gurobi_dist.get(size) and gurobi_dist[size].get('coverage'):
                coverage = gurobi_dist[size]['coverage']
                source = gurobi_dist[size].get('source')
            elif gurobi_cost.get(size) and gurobi_cost[size].get('coverage'):
                coverage = gurobi_cost[size]['coverage']
                source = gurobi_cost[size].get('source')

            num_total = coverage.get('num_total') if coverage else None
            num_feasible = coverage.get('num_feasible') if coverage else None
            feasible_rate = coverage.get('feasible_rate') if coverage else None
            if feasible_rate is None and isinstance(num_total, int) and num_total > 0 and isinstance(num_feasible, int):
                feasible_rate = float(num_feasible / num_total)

            low_coverage = bool(feasible_rate is not None and feasible_rate < float(args.min_gurobi_coverage))
            if low_coverage:
                low_coverage_sizes.append(size)
            status_text = 'LOW_COVERAGE' if low_coverage else 'OK'

            ratio_text = 'n/a'
            if isinstance(num_feasible, int) and isinstance(num_total, int):
                ratio_text = f'{num_feasible}/{num_total}'
            rate_text = f'{feasible_rate:.2%}' if feasible_rate is not None else 'n/a'
            print(f"{size:<8} | {ratio_text:<24} | {rate_text:<14} | {status_text:<16}")

            fair_by_size[str(size)] = {
                'coverage': {
                    'num_total': num_total,
                    'num_feasible': num_feasible,
                    'feasible_rate': feasible_rate,
                },
                'low_coverage': low_coverage,
                'coverage_source': source,
            }
        print('-' * 80)

        anytime_curves = {
            'gurobi_distance': {
                str(size): load_gurobi_anytime_curve(size, 'distance', args.results_dir, args.anytime_budgets_sec)
                for size in args.graph_sizes
            },
            'gurobi_cost': {
                str(size): load_gurobi_anytime_curve(size, 'cost', args.results_dir, args.anytime_budgets_sec)
                for size in args.graph_sizes
            },
            'drl_constant': {
                str(size): {
                    'avg_distance': drl_results[size]['avg_distance'],
                    'avg_total_cost_raw': drl_results[size]['avg_total_cost_raw'],
                    'avg_time_seconds': drl_results[size]['avg_time'],
                }
                for size in args.graph_sizes if size in drl_results
            },
        }

        fair_summary = {
            'mode': 'fair',
            'min_gurobi_coverage_threshold': float(args.min_gurobi_coverage),
            'by_size': fair_by_size,
            'low_coverage_sizes': low_coverage_sizes,
            'conclusion_guard': {
                'allow_mainline_head_to_head': len(low_coverage_sizes) == 0,
                'reason': (
                    'all sizes coverage above threshold'
                    if len(low_coverage_sizes) == 0
                    else f'low coverage at sizes: {low_coverage_sizes}'
                ),
            },
            'anytime_curves': anytime_curves,
        }

    out_path = args.output
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    comparison = {
        'run': {
            'graph_sizes': args.graph_sizes,
            'num_samples': args.num_samples,
            'seed': args.seed,
            'device': str(device),
            'state_kwargs': state_kwargs,
            'results_dir': args.results_dir,
            'reeval_dir': args.reeval_dir,
            'checkpoint_overrides': checkpoint_overrides,
            'comparison_mode': args.comparison_mode,
            'budget_seconds': args.budget_seconds,
            'anytime_budgets_sec': args.anytime_budgets_sec,
        },
        'drl_results': {str(k): v for k, v in drl_results.items()},
        'gurobi_distance_track': {str(k): v for k, v in gurobi_dist.items() if v},
        'gurobi_cost_track': {str(k): v for k, v in gurobi_cost.items() if v},
    }
    if fair_summary is not None:
        comparison['fair_comparison'] = fair_summary

    with open(out_path, 'w', encoding='utf-8') as output_file:
        json.dump(comparison, output_file, indent=2, ensure_ascii=False)
    print(f"\n✓ 对比结果已保存: {out_path}")


if __name__ == '__main__':
    main()
