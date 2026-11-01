"""
导出分类型（乘客/货物）服务率与延误率汇总表。

目标：输出 n25/n50/n100/n200 四个规模的论文可贴表 CSV，包含：
- 乘客服务率
- 货物服务率
- 乘客延误率
- 货物延误率

说明：
- 分类型指标依赖逐样本回放（model + get_costs(details)）。
- 若某规模 checkpoint 缺失（当前常见是 n200），脚本会回退读取 summary JSON，
  保留 overall 指标并将分类型指标置空，同时给出备注。
"""

import argparse
import csv
import json
import os
from types import SimpleNamespace

import torch
from torch.utils.data import DataLoader

from evaluate_model import build_model_from_checkpoint, _resolve_state_kwargs
from nets.attention_model import set_decode_type
from problem_mcvrptw_v2 import MCVRPPDTW, MCVRPPDTWDataset


DEFAULT_CHECKPOINTS = {
    25: 'outputs_cmp/n25_ours_rw1_fixed/pomo_n25_optimized/model_best_service.pt',
    50: 'outputs_cmp/n50_ours_rw1/pomo_n50_optimized/model_best_service.pt',
    100: 'outputs_cmp/n100_ours_rw1_fixed/pomo_n100_optimized/model_best_service.pt',
    200: 'outputs_n200_phaseC_rw1_e10/pomo_n200_optimized/model_best_service.pt',
}

# 当 checkpoint 不在本地时，允许从已有 eval500 summary 回填 overall 信息
DEFAULT_SUMMARY_FALLBACKS = {
    200: 'outputs_cmp/n200_phaseC_rw1_e10_eval500.json',
}


def parse_args():
    parser = argparse.ArgumentParser(description='Export passenger/cargo service-delay metrics table')
    parser.add_argument('--graph-sizes', type=str, default='25,50,100,200',
                        help='Comma-separated graph sizes, e.g. 25,50,100,200')
    parser.add_argument('--num-samples', type=int, default=500)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--seed', type=int, default=99999)
    parser.add_argument('--decode', type=str, choices=['greedy', 'sampling'], default='greedy')
    parser.add_argument('--no-cuda', action='store_true', help='Force CPU evaluation')
    parser.add_argument('--output-csv', type=str,
                        default='outputs_cmp/service_delay_table_by_type.csv')
    return parser.parse_args()


def collate_fn(batch):
    keys = batch[0].keys()
    return {
        key: torch.stack([sample[key] for sample in batch], dim=0)
        if torch.is_tensor(batch[0][key]) else batch[0][key]
        for key in keys
    }


def safe_div(numer, denom):
    if denom is None or float(denom) <= 1e-12:
        return None
    return float(numer) / float(denom)


def parse_graph_sizes(spec):
    out = []
    for chunk in str(spec).split(','):
        chunk = chunk.strip()
        if not chunk:
            continue
        out.append(int(chunk))
    return out


def build_state_args_stub():
    """构造 _resolve_state_kwargs 所需最小参数集合。"""
    return SimpleNamespace(
        max_concurrent_open_orders=1,
        min_orders_per_dispatch=None,
        enable_delivery_viability=False,
        enable_viability_fallback=False,
        relax_pickup_commitment_trip_time=False,
        hard_cargo_pickup_timewindow=None,
        cargo_delay_tier1_min=None,
        cargo_delay_tier2_min=None,
        cargo_delay_cost_0_30=None,
        cargo_delay_cost_30_60=None,
        cargo_delay_cost_60_plus=None,
    )


def evaluate_type_metrics(graph_size, checkpoint_path, num_samples, batch_size, seed, decode, device):
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    model = build_model_from_checkpoint(checkpoint, device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    set_decode_type(model, decode)

    state_args = build_state_args_stub()
    state_kwargs = _resolve_state_kwargs(state_args, checkpoint)

    dataset = MCVRPPDTWDataset(num_samples=num_samples, graph_size=graph_size, seed=seed)
    loader = DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn)

    sum_passenger_orders = 0.0
    sum_cargo_orders = 0.0

    sum_passenger_served = 0.0
    sum_cargo_served = 0.0

    sum_passenger_on_time = 0.0
    sum_cargo_on_time = 0.0

    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}

            _, _, pi = model(batch, return_pi=True, state_kwargs=state_kwargs)
            _, details = MCVRPPDTW.get_costs(batch, pi, return_details=True)

            n_orders = int(graph_size)
            node_type_pickups = batch['node_type'][:, :n_orders]

            passenger_orders = (node_type_pickups == 1).sum(dim=1).float()
            cargo_orders = float(n_orders) - passenger_orders

            passenger_served = details['passenger_delivery_total'].float()
            cargo_served = details['cargo_delivery_total'].float()

            passenger_on_time = details['passenger_delivery_on_time'].float()
            cargo_on_time = details['cargo_delivery_on_time'].float()

            sum_passenger_orders += float(passenger_orders.sum().item())
            sum_cargo_orders += float(cargo_orders.sum().item())

            sum_passenger_served += float(passenger_served.sum().item())
            sum_cargo_served += float(cargo_served.sum().item())

            sum_passenger_on_time += float(passenger_on_time.sum().item())
            sum_cargo_on_time += float(cargo_on_time.sum().item())

    sum_passenger_late = max(0.0, sum_passenger_served - sum_passenger_on_time)
    sum_cargo_late = max(0.0, sum_cargo_served - sum_cargo_on_time)

    total_orders = sum_passenger_orders + sum_cargo_orders
    total_served = sum_passenger_served + sum_cargo_served
    total_late = sum_passenger_late + sum_cargo_late

    return {
        'graph_size': int(graph_size),
        'source': 'model_eval',
        'checkpoint': checkpoint_path,
        'num_samples': int(num_samples),
        'seed': int(seed),

        'passenger_orders_total': sum_passenger_orders,
        'cargo_orders_total': sum_cargo_orders,
        'passenger_served_total': sum_passenger_served,
        'cargo_served_total': sum_cargo_served,
        'passenger_on_time_total': sum_passenger_on_time,
        'cargo_on_time_total': sum_cargo_on_time,
        'passenger_late_total': sum_passenger_late,
        'cargo_late_total': sum_cargo_late,

        # 你论文里最常用的 4 个指标
        'passenger_service_rate': safe_div(sum_passenger_served, sum_passenger_orders),
        'cargo_service_rate': safe_div(sum_cargo_served, sum_cargo_orders),
        'passenger_delay_rate': safe_div(sum_passenger_late, sum_passenger_served),
        'cargo_delay_rate': safe_div(sum_cargo_late, sum_cargo_served),

        # 附加口径（便于核对）
        'passenger_delay_rate_over_total': safe_div(sum_passenger_late, sum_passenger_orders),
        'cargo_delay_rate_over_total': safe_div(sum_cargo_late, sum_cargo_orders),
        'passenger_on_time_rate': safe_div(sum_passenger_on_time, sum_passenger_served),
        'cargo_on_time_rate': safe_div(sum_cargo_on_time, sum_cargo_served),
        'overall_service_rate': safe_div(total_served, total_orders),
        'overall_delay_rate': safe_div(total_late, total_served),

        'note': '',
    }


def fallback_from_summary(graph_size, summary_path):
    with open(summary_path, 'r', encoding='utf-8') as f:
        payload = json.load(f)

    run = payload.get('run', {})
    agg = payload.get('aggregate', {})

    return {
        'graph_size': int(graph_size),
        'source': 'summary_fallback',
        'checkpoint': run.get('checkpoint', ''),
        'num_samples': run.get('num_samples', ''),
        'seed': run.get('seed', ''),

        'passenger_orders_total': None,
        'cargo_orders_total': None,
        'passenger_served_total': None,
        'cargo_served_total': None,
        'passenger_on_time_total': None,
        'cargo_on_time_total': None,
        'passenger_late_total': None,
        'cargo_late_total': None,

        'passenger_service_rate': None,
        'cargo_service_rate': None,
        'passenger_delay_rate': None,
        'cargo_delay_rate': None,

        'passenger_delay_rate_over_total': None,
        'cargo_delay_rate_over_total': None,
        'passenger_on_time_rate': None,
        'cargo_on_time_rate': None,
        'overall_service_rate': agg.get('service_rate_mean', None),
        'overall_delay_rate': None,

        'note': (
            'checkpoint_missing_locally; type-level rates unavailable from summary-only. '
            f"summary_service_rate={agg.get('service_rate_mean', None)}, "
            f"passenger_delay_cost_mean={agg.get('passenger_delivery_delay_cost_mean', None)}, "
            f"cargo_delay_cost_mean={agg.get('cargo_delay_cost_mean', None)}"
        ),
    }


def main():
    args = parse_args()
    graph_sizes = parse_graph_sizes(args.graph_sizes)

    device = torch.device('cuda' if (torch.cuda.is_available() and not args.no_cuda) else 'cpu')
    print(f'[info] device={device}')

    rows = []

    for graph_size in graph_sizes:
        checkpoint_path = DEFAULT_CHECKPOINTS.get(graph_size)
        if checkpoint_path and os.path.exists(checkpoint_path):
            print(f'[run] n{graph_size} -> eval from checkpoint: {checkpoint_path}')
            row = evaluate_type_metrics(
                graph_size=graph_size,
                checkpoint_path=checkpoint_path,
                num_samples=args.num_samples,
                batch_size=args.batch_size,
                seed=args.seed,
                decode=args.decode,
                device=device,
            )
            rows.append(row)
            continue

        fallback_path = DEFAULT_SUMMARY_FALLBACKS.get(graph_size)
        if fallback_path and os.path.exists(fallback_path):
            print(f'[warn] n{graph_size} checkpoint missing, fallback summary: {fallback_path}')
            row = fallback_from_summary(graph_size, fallback_path)
            rows.append(row)
            continue

        print(f'[skip] n{graph_size}: checkpoint and fallback summary both missing')
        rows.append({
            'graph_size': int(graph_size),
            'source': 'missing',
            'checkpoint': checkpoint_path or '',
            'num_samples': '',
            'seed': '',
            'passenger_orders_total': None,
            'cargo_orders_total': None,
            'passenger_served_total': None,
            'cargo_served_total': None,
            'passenger_on_time_total': None,
            'cargo_on_time_total': None,
            'passenger_late_total': None,
            'cargo_late_total': None,
            'passenger_service_rate': None,
            'cargo_service_rate': None,
            'passenger_delay_rate': None,
            'cargo_delay_rate': None,
            'passenger_delay_rate_over_total': None,
            'cargo_delay_rate_over_total': None,
            'passenger_on_time_rate': None,
            'cargo_on_time_rate': None,
            'overall_service_rate': None,
            'overall_delay_rate': None,
            'note': 'checkpoint and summary missing',
        })

    os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)

    fieldnames = [
        'graph_size', 'source', 'checkpoint', 'num_samples', 'seed',
        'passenger_orders_total', 'cargo_orders_total',
        'passenger_served_total', 'cargo_served_total',
        'passenger_on_time_total', 'cargo_on_time_total',
        'passenger_late_total', 'cargo_late_total',
        'passenger_service_rate', 'cargo_service_rate',
        'passenger_delay_rate', 'cargo_delay_rate',
        'passenger_delay_rate_over_total', 'cargo_delay_rate_over_total',
        'passenger_on_time_rate', 'cargo_on_time_rate',
        'overall_service_rate', 'overall_delay_rate',
        'note',
    ]

    with open(args.output_csv, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    print(f'\n[done] wrote: {args.output_csv}')
    for row in rows:
        print(
            f"  n{row['graph_size']}: "
            f"p_service={row['passenger_service_rate']}, c_service={row['cargo_service_rate']}, "
            f"p_delay={row['passenger_delay_rate']}, c_delay={row['cargo_delay_rate']}, "
            f"source={row['source']}"
        )


if __name__ == '__main__':
    main()
