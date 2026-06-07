"""
计算DRL模型的准时率 (On-Time Delivery Ratio)
用于与Gurobi进行服务质量对比
"""

import torch
import numpy as np
from tqdm import tqdm
import json

from problem_mcvrptw_v2 import MCVRPPDTW, MCVRPPDTWDataset, Config
from nets.attention_model import AttentionModel, set_decode_type


def collate_fn(batch):
    keys = batch[0].keys()
    return {
        key: torch.stack([sample[key] for sample in batch], dim=0)
        if torch.is_tensor(batch[0][key]) else batch[0][key]
        for key in keys
    }


def load_model(graph_size, device):
    """加载训练好的模型"""
    model_path = f'outputs/pomo_n{graph_size}_optimized/model_best.pt'
    checkpoint = torch.load(model_path, map_location=device)
    
    args_dict = checkpoint['args']
    model = AttentionModel(
        embedding_dim=args_dict.get('embedding_dim', 256),
        hidden_dim=args_dict.get('hidden_dim', 256),
        n_encode_layers=args_dict.get('n_encode_layers', 6),
        tanh_clipping=args_dict.get('tanh_clipping', 10.),
        normalization=args_dict.get('normalization', 'batch'),
        n_heads=args_dict.get('n_heads', 8),
        problem=MCVRPPDTW
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    return model


def calculate_ontime_ratio_single(sample, pi, tolerance_minutes=5):
    """按统一口径计算 delivery 准时率与 passenger 服务质量。"""
    batch = {k: v.unsqueeze(0) if torch.is_tensor(v) else v for k, v in sample.items()}
    with torch.no_grad():
        _, details = MCVRPPDTW.get_costs(batch, pi.unsqueeze(0), return_details=True)

    passenger_delivery_total = max(float(details['passenger_delivery_total'][0].item()), 1.0)
    cargo_delivery_total = max(float(details['cargo_delivery_total'][0].item()), 1.0)
    total_deliveries = passenger_delivery_total + cargo_delivery_total

    passenger_ontime = float(details['passenger_delivery_on_time'][0].item())
    cargo_ontime = float(details['cargo_delivery_on_time'][0].item())
    total_ontime = passenger_ontime + cargo_ontime
    total_late = total_deliveries - total_ontime

    avg_delay = (
        float(details['passenger_delivery_delay_minutes'][0].item())
        + float(details['cargo_delay_minutes'][0].item())
    ) / max(total_late, 1.0)

    return {
        'ontime_count': total_ontime,
        'late_count': total_late,
        'early_count': 0,
        'total_nodes': total_deliveries,
        'ontime_ratio': total_ontime / max(total_deliveries, 1.0),
        'late_ratio': total_late / max(total_deliveries, 1.0),
        'avg_delay': avg_delay if total_late > 0 else 0,
        'max_delay': max(
            float(details['passenger_delivery_delay_minutes'][0].item()) / passenger_delivery_total,
            float(details['cargo_delay_minutes'][0].item()) / cargo_delivery_total,
        ),
        'passenger_pickup_hard_violations': float(details['passenger_pickup_hard_violations'][0].item()),
        'passenger_ride_time_violations': float(details['passenger_ride_time_violations'][0].item()),
        'completed_orders': float(details['completed_orders'][0].item()),
        'rejected_orders': float(details['rejected_orders'][0].item()),
    }


def evaluate_ontime_ratio(graph_size, num_samples=100):
    """评估指定规模的准时率"""
    
    print(f"\n{'='*60}")
    print(f"评估 {graph_size} 订单的准时率")
    print("="*60)
    
    device = torch.device('cpu')
    
    # 加载模型
    print("加载模型...")
    model = load_model(graph_size, device)
    set_decode_type(model, 'greedy')
    
    # 生成测试数据
    print(f"生成 {num_samples} 个测试样本...")
    dataset = MCVRPPDTWDataset(num_samples=num_samples, graph_size=graph_size, seed=42)
    
    # 统计结果
    all_results = []
    
    print("开始推理...")
    for i, sample in enumerate(tqdm(dataset.data)):
        # 构造batch
        batch = {k: v.unsqueeze(0) if torch.is_tensor(v) else v 
                 for k, v in sample.items()}
        
        # 推理
        with torch.no_grad():
            cost, _, pi = model(batch, return_pi=True)
        
        # 计算准时率
        result = calculate_ontime_ratio_single(sample, pi[0], tolerance_minutes=5)
        all_results.append(result)
    
    # 汇总统计
    total_ontime = sum(r['ontime_count'] for r in all_results)
    total_late = sum(r['late_count'] for r in all_results)
    total_early = sum(r['early_count'] for r in all_results)
    total_nodes = sum(r['total_nodes'] for r in all_results)
    
    avg_ontime_ratio = np.mean([r['ontime_ratio'] for r in all_results])
    std_ontime_ratio = np.std([r['ontime_ratio'] for r in all_results])
    
    avg_delay = np.mean([r['avg_delay'] for r in all_results]) if all_results else 0
    max_delay = max([r['max_delay'] for r in all_results]) if all_results else 0
    
    # 打印结果
    avg_pickup_hard_violations = np.mean([r['passenger_pickup_hard_violations'] for r in all_results])
    avg_ride_time_violations = np.mean([r['passenger_ride_time_violations'] for r in all_results])

    print(f"\nDelivery 准时率统计:")
    print(f"  准时 delivery: {total_ontime:.0f} / {total_nodes:.0f} ({total_ontime/total_nodes*100:.1f}%)")
    print(f"  迟到 delivery: {total_late:.0f} / {total_nodes:.0f} ({total_late/total_nodes*100:.1f}%)")
    print(f"\n平均 delivery 准时率: {avg_ontime_ratio*100:.2f}% ± {std_ontime_ratio*100:.2f}%")
    print(f"平均 delivery 延误时间: {avg_delay:.2f} 分钟")
    print(f"最大平均延误时间: {max_delay:.2f} 分钟")
    print(f"平均 passenger pickup 硬违约: {avg_pickup_hard_violations:.2f}")
    print(f"平均 passenger ride-time 违约: {avg_ride_time_violations:.2f}")
    
    return {
        'graph_size': graph_size,
        'num_samples': num_samples,
        'avg_ontime_ratio': avg_ontime_ratio,
        'std_ontime_ratio': std_ontime_ratio,
        'total_ontime': total_ontime,
        'total_late': total_late,
        'total_early': total_early,
        'total_nodes': total_nodes,
        'avg_delay': avg_delay,
        'max_delay': max_delay,
        'avg_passenger_pickup_hard_violations': avg_pickup_hard_violations,
        'avg_passenger_ride_time_violations': avg_ride_time_violations,
    }


def main():
    """主函数"""
    
    print("="*70)
    print("DRL模型准时率评估工具")
    print("On-Time Delivery Ratio Evaluation for DRL Model")
    print("="*70)
    
    results = {}
    
    for size in [25, 50, 100]:
        result = evaluate_ontime_ratio(size, num_samples=100)
        results[size] = result
    
    # 生成对比表格
    print(f"\n{'='*70}")
    print("综合对比表")
    print("="*70)
    print(f"{'规模':<10} {'准时率':<15} {'迟到率':<15} {'平均延误':<15}")
    print("-"*70)
    
    for size in [25, 50, 100]:
        r = results[size]
        print(f"{size}订单    {r['avg_ontime_ratio']*100:>6.2f}%        "
              f"{r['total_late']/r['total_nodes']*100:>6.2f}%        "
              f"{r['avg_delay']:>6.2f} 分钟")
    
    # 保存结果
    output_file = 'outputs/ontime_ratio_results.json'
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n结果已保存至: {output_file}")
    print("="*70)


if __name__ == '__main__':
    main()

