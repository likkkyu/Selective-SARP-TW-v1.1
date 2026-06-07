"""
Selective SARP-TW：DRL vs Gurobi 公平对比脚本
Fair Comparison between DRL and Gurobi

【当前双轨口径】
- Track-1【距离口径】DRL.total_distance vs Gurobi(objective_mode='distance').avg_cost
  对齐 VRP/TSP 文献主流报告口径，便于和经典基线横向比较。
- Track-2【真实成本口径】DRL.total_cost_raw vs Gurobi(objective_mode='cost') 重评估的 raw_total_cost
  反映 MCVRP-PDTW 真实经济成本，用于消融与论文 Section 5。

依赖:
- gurobi_results_{N}_distance.json    (gurobi.py --objective-mode distance)
- gurobi_results_{N}_cost.json        (gurobi.py --objective-mode cost, 可选)
- outputs/gurobi_reevaluated/gurobi_reevaluated_{N}.json
        (evaluate_gurobi_real_cost.py --input gurobi_results_{N}_distance.json)
"""

import argparse
import json
import os
import time

import numpy as np
import torch

from problem_mcvrptw_v2 import MCVRPPDTW, MCVRPPDTWDataset, Config
from nets.attention_model import AttentionModel, set_decode_type


def build_model_from_checkpoint(checkpoint, device, problem):
    args_dict = checkpoint.get('args', {}) or {}
    if not isinstance(args_dict, dict):
        try:
            args_dict = vars(args_dict)
        except Exception:
            args_dict = {}
    return AttentionModel(
        embedding_dim=args_dict.get('embedding_dim', 256),
        hidden_dim=args_dict.get('hidden_dim', 256),
        n_encode_layers=args_dict.get('n_encode_layers', 6),
        tanh_clipping=args_dict.get('tanh_clipping', 10.),
        normalization=args_dict.get('normalization', 'batch'),
        n_heads=args_dict.get('n_heads', 8),
        problem=problem,
    ).to(device)


def evaluate_drl(graph_size, num_samples=5, seed=42, checkpoint_path=None, device=None):
    """评估 DRL 模型，返回距离 + raw_cost 双轨指标。"""
    device = device or torch.device('cpu')
    if checkpoint_path is None:
        checkpoint_path = f'outputs/pomo_n{graph_size}_optimized/model_best.pt'

    if not os.path.exists(checkpoint_path):
        print(f"  ✗ 模型不存在: {checkpoint_path}")
        return None

    # 安全建议：仅加载本仓库训练或可信来源的 checkpoint。
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = build_model_from_checkpoint(checkpoint, device, MCVRPPDTW)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    set_decode_type(model, 'greedy')

    dataset = MCVRPPDTWDataset(num_samples=num_samples, graph_size=graph_size, seed=seed)

    distances, raw_costs, energy_costs, times = [], [], [], []
    for sample in dataset.data:
        # 当前实现修复：batch 必须搬到 model 所在 device, 否则 CUDA 下会报
        # device mismatch (RuntimeError: Expected all tensors to be on the same device).
        batch = {
            k: (v.unsqueeze(0).to(device) if torch.is_tensor(v) else v)
            for k, v in sample.items()
        }
        start = time.time()
        with torch.no_grad():
            cost_train, _, pi = model(batch, return_pi=True)
            _, details = MCVRPPDTW.get_costs(batch, pi, return_details=True)
        times.append(time.time() - start)
        distances.append(details['total_distance'].item())
        raw_costs.append(details['total_cost_raw'].item())
        energy_costs.append(details['energy_cost_raw'].item())

    return {
        'avg_distance': float(np.mean(distances)),
        'std_distance': float(np.std(distances)),
        'avg_total_cost_raw': float(np.mean(raw_costs)),
        'std_total_cost_raw': float(np.std(raw_costs)),
        'avg_energy_cost': float(np.mean(energy_costs)),
        'avg_time': float(np.mean(times)),
    }


def load_gurobi_distance_track(size):
    """Track-1 距离口径：直接读 gurobi_results_{N}_distance.json"""
    candidates = [
        f'gurobi_results_{size}_distance.json',
        f'gurobi_results_{size}.json',  # 兼容旧文件名
    ]
    for path in candidates:
        if os.path.exists(path):
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            return {
                'avg_distance': data.get('avg_cost', 0),  # distance 模式下 avg_cost 即为距离
                'avg_time': data.get('avg_solve_time', 0),
                'source': path,
            }
    return None


def load_gurobi_cost_track(size):
    """Track-2 真实成本口径：优先读重评估文件，其次读 cost 模式直接结果。"""
    reeval_path = os.path.join('outputs', 'gurobi_reevaluated', f'gurobi_reevaluated_{size}.json')
    if os.path.exists(reeval_path):
        with open(reeval_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return {
            'avg_total_cost_raw': data.get('real_avg_cost', 0),
            'avg_energy_cost': data.get('avg_energy_cost', 0),
            'avg_distance': data.get('avg_distance', 0),
            'avg_time': None,  # 重评估没有求解时间
            'source': reeval_path,
        }
    cost_path = f'gurobi_results_{size}_cost.json'
    if os.path.exists(cost_path):
        with open(cost_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return {
            'avg_total_cost_raw': data.get('avg_cost', 0),  # cost 模式下 avg_cost ≈ 部分真实成本
            'avg_energy_cost': None,
            'avg_distance': None,
            'avg_time': data.get('avg_solve_time', 0),
            'source': cost_path,
        }
    return None


def main():
    parser = argparse.ArgumentParser(description='DRL vs Gurobi 双轨对比 (v6)')
    parser.add_argument('--graph-sizes', type=int, nargs='+', default=[25, 50, 100])
    parser.add_argument('--num-samples', type=int, default=5)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--no-cuda', action='store_true')
    args = parser.parse_args()

    device = torch.device('cuda' if (torch.cuda.is_available() and not args.no_cuda) else 'cpu')

    print("=" * 80)
    print("DRL vs Gurobi 双轨对比 (v6)")
    print("=" * 80)

    print("\n[1] 评估 DRL 模型...")
    drl_results = {}
    for size in args.graph_sizes:
        print(f"\n  评估 N={size} ...")
        result = evaluate_drl(size, num_samples=args.num_samples, seed=args.seed, device=device)
        if result:
            drl_results[size] = result
            print(f"    距离          : {result['avg_distance']:.3f} ± {result['std_distance']:.3f} km")
            print(f"    总成本(raw)   : {result['avg_total_cost_raw']:.3f} ± {result['std_total_cost_raw']:.3f} RMB")
            print(f"    能耗成本(raw) : {result['avg_energy_cost']:.3f} RMB")
            print(f"    DRL 推理时间  : {result['avg_time'] * 1000:.1f} ms")

    print("\n[2] 加载 Gurobi 双轨结果...")
    gurobi_dist = {s: load_gurobi_distance_track(s) for s in args.graph_sizes}
    gurobi_cost = {s: load_gurobi_cost_track(s) for s in args.graph_sizes}

    if not any(gurobi_dist.values()) and not any(gurobi_cost.values()):
        print("  ✗ 未找到任何 Gurobi 结果文件")
        print("  请先运行: python gurobi.py --objective-mode distance")
        print("        或: python gurobi.py --objective-mode cost")
        return

    # ===== Track 1 距离口径 =====
    print("\n" + "=" * 80)
    print("【Track 1】距离口径 (km)  —  与 VRP/TSP 文献主流口径一致")
    print("=" * 80)
    print(f"\n{'规模':<8} | {'指标':<14} | {'DRL':<14} | {'Gurobi':<14} | {'Gap':>10}")
    print("-" * 80)
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
            print("-" * 80)

    # ===== Track 2 真实成本口径 =====
    print("\n" + "=" * 80)
    print("【Track 2】真实成本口径 (RMB)  —  MCVRP-PDTW 经济成本")
    print("=" * 80)
    print(f"\n{'规模':<8} | {'指标':<14} | {'DRL':<14} | {'Gurobi':<14} | {'Gap':>10}")
    print("-" * 80)
    for size in args.graph_sizes:
        if size in drl_results and gurobi_cost.get(size):
            drl = drl_results[size]
            gur = gurobi_cost[size]
            c_drl = drl['avg_total_cost_raw']
            c_gur = gur['avg_total_cost_raw']
            gap = (c_drl - c_gur) / c_gur * 100 if c_gur > 0 else 0
            print(f"{size}订单  | {'总成本(RMB)':<14} | {c_drl:<14.3f} | {c_gur:<14.3f} | {gap:>+9.1f}%")
            print(f"{'':8} | {'数据来源':<14} | DRL eval     | {os.path.basename(gur['source']):<14}")
            print("-" * 80)

    # ===== 总结 =====
    print("\n" + "=" * 80)
    print("分析总结")
    print("=" * 80)
    print("""
1. 双轨口径设计：
   - Track 1 (distance) ：Gurobi 纯距离最小化目标（线性、快），与 VRP 主流文献对齐。
   - Track 2 (real cost)：含能耗 + 延误 + 车辆固定 + 拒单 + 单趟超时的真实经济成本。

2. Gurobi 在大规模 (N=100) 通常无法收敛到最优，仅作为可行解参考。

3. DRL 优势：
   - 推理 ≈ 毫秒级，Gurobi 通常分钟级以上 → 1000x+ 加速；
   - DRL 在真实成本口径下能直接优化目标，路径质量更接近实际运营需求。
""")

    out_path = os.path.join('outputs', 'drl_vs_gurobi_comparison.json')
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    comparison = {
        'drl_results': {str(k): v for k, v in drl_results.items()},
        'gurobi_distance_track': {str(k): v for k, v in gurobi_dist.items() if v},
        'gurobi_cost_track': {str(k): v for k, v in gurobi_cost.items() if v},
    }
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(comparison, f, indent=2, ensure_ascii=False)
    print(f"\n✓ 对比结果已保存: {out_path}")


if __name__ == '__main__':
    main()
