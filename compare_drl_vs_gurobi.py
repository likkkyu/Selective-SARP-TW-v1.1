"""
Selective SARP-TW：DRL vs Gurobi 公平对比脚本
Fair Comparison between DRL and Gurobi

【当前双轨口径】
- Track-1【距离口径】DRL.total_distance vs Gurobi(objective_mode='distance').avg_cost
- Track-2【真实成本口径】DRL.total_cost_raw vs Gurobi 重评估 raw_total_cost

依赖:
- gurobi_results_{N}_distance.json    (gurobi.py --objective-mode distance)
- gurobi_results_{N}_cost.json        (gurobi.py --objective-mode cost, 可选)
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


def load_gurobi_distance_track(size, results_dir='.'):
    """Track-1 距离口径：直接读 gurobi_results_{N}_distance.json。"""
    candidates = [
        os.path.join(results_dir, f'gurobi_results_{size}_distance.json'),
        os.path.join(results_dir, f'gurobi_results_{size}.json'),  # 兼容旧文件名
    ]
    for path in candidates:
        if os.path.exists(path):
            with open(path, 'r', encoding='utf-8') as input_file:
                data = json.load(input_file)
            return {
                'avg_distance': data.get('avg_cost', 0),
                'avg_time': data.get('avg_solve_time', 0),
                'source': path,
            }
    return None


def load_gurobi_cost_track(size, results_dir='.', reeval_dir='outputs/gurobi_reevaluated'):
    """Track-2 真实成本口径：优先读重评估文件，其次读 cost 模式结果。"""
    reeval_path = os.path.join(reeval_dir, f'gurobi_reevaluated_{size}.json')
    if os.path.exists(reeval_path):
        with open(reeval_path, 'r', encoding='utf-8') as input_file:
            data = json.load(input_file)
        return {
            'avg_total_cost_raw': data.get('real_avg_cost', 0),
            'avg_energy_cost': data.get('avg_energy_cost', 0),
            'avg_distance': data.get('avg_distance', 0),
            'avg_time': None,
            'source': reeval_path,
        }

    cost_path = os.path.join(results_dir, f'gurobi_results_{size}_cost.json')
    if os.path.exists(cost_path):
        with open(cost_path, 'r', encoding='utf-8') as input_file:
            data = json.load(input_file)
        return {
            'avg_total_cost_raw': data.get('avg_cost', 0),
            'avg_energy_cost': None,
            'avg_distance': None,
            'avg_time': data.get('avg_solve_time', 0),
            'source': cost_path,
        }
    return None


def parse_args():
    parser = argparse.ArgumentParser(description='DRL vs Gurobi 双轨对比 (v7)')
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
    print('DRL vs Gurobi 双轨对比 (v7)')
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
    gurobi_dist = {size: load_gurobi_distance_track(size, results_dir=args.results_dir) for size in args.graph_sizes}
    gurobi_cost = {
        size: load_gurobi_cost_track(size, results_dir=args.results_dir, reeval_dir=args.reeval_dir)
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
            gap = (d_drl - d_gur) / d_gur * 100 if d_gur > 0 else 0
            print(f"{size}订单  | {'距离(km)':<14} | {d_drl:<14.3f} | {d_gur:<14.3f} | {gap:>+9.1f}%")
            t_drl = drl['avg_time']
            t_gur = gur['avg_time']
            speedup = t_gur / t_drl if t_drl > 0 else 0
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
            gap = (c_drl - c_gur) / c_gur * 100 if c_gur > 0 else 0
            print(f"{size}订单  | {'总成本(RMB)':<14} | {c_drl:<14.3f} | {c_gur:<14.3f} | {gap:>+9.1f}%")
            print(f"{'':8} | {'数据来源':<14} | DRL eval     | {os.path.basename(gur['source']):<14}")
            print('-' * 80)

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
        },
        'drl_results': {str(k): v for k, v in drl_results.items()},
        'gurobi_distance_track': {str(k): v for k, v in gurobi_dist.items() if v},
        'gurobi_cost_track': {str(k): v for k, v in gurobi_cost.items() if v},
    }
    with open(out_path, 'w', encoding='utf-8') as output_file:
        json.dump(comparison, output_file, indent=2, ensure_ascii=False)
    print(f"\n✓ 对比结果已保存: {out_path}")


if __name__ == '__main__':
    main()
