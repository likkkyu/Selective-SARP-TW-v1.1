"""
Selective SARP-TW 模型评估脚本（通用 CLI）

通用版评估脚本：
- 通过 CLI 参数指定 graph_size / checkpoint / num_samples 等
- 自动从 checkpoint['args'] 恢复模型超参数（不再硬编码 128/3）
- 主指标：total_cost_raw（论文比较口径，纯 RMB）
- 完整 breakdown：能耗、乘客 delivery 延误、货物 delivery 延误、车辆固定、拒单、单趟超时

用法示例：
    python evaluate_model.py --graph-size 25
    python evaluate_model.py --graph-size 50 --checkpoint outputs/pomo_n50_optimized/model_best.pt
    python evaluate_model.py --graph-size 100 --num-samples 200 --batch-size 16
"""
import argparse
import os
import statistics

import torch
from torch.utils.data import DataLoader

from problem_mcvrptw_v2 import MCVRPPDTWDataset, MCVRPPDTW, Config
from nets.attention_model import AttentionModel, set_decode_type


def parse_args():
    parser = argparse.ArgumentParser(description='Selective SARP-TW 通用评估脚本')
    parser.add_argument('--graph-size', type=int, default=25,
                        help='订单数 (default: 25)')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='模型 checkpoint 路径 (default: outputs/pomo_n{N}_optimized/model_best.pt)')
    parser.add_argument('--num-samples', type=int, default=100,
                        help='测试样本数 (default: 100)')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size (default: 32)')
    parser.add_argument('--seed', type=int, default=99999,
                        help='测试集随机种子 (default: 99999)')
    parser.add_argument('--decode', type=str, default='greedy',
                        choices=['greedy', 'sampling'],
                        help='解码方式 (default: greedy)')
    parser.add_argument('--no-cuda', action='store_true',
                        help='强制使用 CPU')
    parser.add_argument('--diagnostics', action='store_true',
                        help='输出 mask 与动作可行性诊断指标')
    parser.add_argument('--max-concurrent-open-orders', type=int, default=1,
                        help='共享实验：允许的最大并发 open 单数量 (default: 1)')
    parser.add_argument('--enable-delivery-viability', action='store_true',
                        help='共享实验：对 delivery 也启用 viability 过滤')
    parser.add_argument('--enable-viability-fallback', action='store_true',
                        help='共享实验：若 delivery viability 全挡死则启用安全回退')
    parser.add_argument('--relax-pickup-commitment-trip-time', action='store_true',
                        help='仅对 pickup_commitment 的 completion proof 放松 trip_time gate')
    parser.add_argument('--passenger-tw-period-weights', nargs=3, type=float, default=None,
                        metavar=('MORNING', 'MIDDAY', 'EVENING'),
                        help='覆盖默认 passenger 三时段 TW 权重')
    parser.add_argument('--cargo-tw-period-weights', nargs=3, type=float, default=None,
                        metavar=('MORNING', 'MIDDAY', 'EVENING'),
                        help='覆盖默认 cargo 三时段 TW 权重')
    parser.add_argument('--passenger-tw-period-bounds', nargs=6, type=float, default=None,
                        metavar=('MORNING_START', 'MORNING_END', 'MIDDAY_START', 'MIDDAY_END', 'EVENING_START', 'EVENING_END'),
                        help='覆盖默认 passenger 三时段 pickup TW 区间')
    parser.add_argument('--cargo-tw-period-bounds', nargs=6, type=float, default=None,
                        metavar=('MORNING_START', 'MORNING_END', 'MIDDAY_START', 'MIDDAY_END', 'EVENING_START', 'EVENING_END'),
                        help='覆盖默认 cargo 三时段 pickup TW 区间')
    return parser.parse_args()


def collate_fn(batch):
    keys = batch[0].keys()
    return {
        key: torch.stack([sample[key] for sample in batch], dim=0)
        if torch.is_tensor(batch[0][key]) else batch[0][key]
        for key in keys
    }


def _normalize_ratio_triplet(values):
    total = sum(max(float(v), 0.0) for v in values)
    if total <= 0:
        return tuple(1.0 / len(values) for _ in values)
    return tuple(max(float(v), 0.0) / total for v in values)


def _parse_period_bounds(values):
    if values is None:
        return None
    if len(values) != 6:
        raise ValueError('TW period bounds must provide exactly 6 numbers: s1 e1 s2 e2 s3 e3')
    bounds = []
    for idx in range(0, 6, 2):
        start = float(values[idx])
        end = float(values[idx + 1])
        if end <= start:
            raise ValueError(f'Invalid TW bounds pair #{idx // 2 + 1}: end must be greater than start')
        bounds.append((start, end))
    return tuple(bounds)


def build_model_from_checkpoint(checkpoint, device):
    """从 checkpoint['args'] 恢复模型超参数；缺省时回退到当前默认值 (256/256/6/8)。"""
    args_dict = checkpoint.get('args', {}) or {}
    if not isinstance(args_dict, dict):
        try:
            args_dict = vars(args_dict)
        except Exception:
            args_dict = {}

    embedding_dim = args_dict.get('embedding_dim', 256)
    hidden_dim = args_dict.get('hidden_dim', 256)
    n_encode_layers = args_dict.get('n_encode_layers', 6)
    n_heads = args_dict.get('n_heads', 8)
    tanh_clipping = args_dict.get('tanh_clipping', 10.0)
    normalization = args_dict.get('normalization', 'batch')

    model = AttentionModel(
        embedding_dim=embedding_dim,
        hidden_dim=hidden_dim,
        problem=MCVRPPDTW,
        n_encode_layers=n_encode_layers,
        n_heads=n_heads,
        tanh_clipping=tanh_clipping,
        normalization=normalization,
    ).to(device)

    print(f'  embedding_dim={embedding_dim}, hidden_dim={hidden_dim}, '
          f'n_encode_layers={n_encode_layers}, n_heads={n_heads}, '
          f'tanh_clipping={tanh_clipping}, normalization={normalization}')

    return model


def _resolve_state_kwargs(args, checkpoint):
    ckpt_args = checkpoint.get('args', {}) or {}
    if not isinstance(ckpt_args, dict):
        try:
            ckpt_args = vars(ckpt_args)
        except Exception:
            ckpt_args = {}

    max_open = args.max_concurrent_open_orders
    if max_open == 1 and 'max_concurrent_open_orders' in ckpt_args:
        max_open = ckpt_args['max_concurrent_open_orders']

    enable_delivery_viability = args.enable_delivery_viability
    if (not enable_delivery_viability) and ('enable_delivery_viability' in ckpt_args):
        enable_delivery_viability = bool(ckpt_args['enable_delivery_viability'])

    enable_viability_fallback = args.enable_viability_fallback
    if (not enable_viability_fallback) and ('enable_viability_fallback' in ckpt_args):
        enable_viability_fallback = bool(ckpt_args['enable_viability_fallback'])

    return {
        'max_concurrent_open_orders': max_open,
        'enable_delivery_viability': enable_delivery_viability,
        'enable_viability_fallback': enable_viability_fallback,
        'relax_pickup_commitment_trip_time': bool(getattr(args, 'relax_pickup_commitment_trip_time', False)),
    }


def evaluate():
    args = parse_args()

    device = torch.device('cuda' if (torch.cuda.is_available() and not args.no_cuda) else 'cpu')

    if args.checkpoint is None:
        args.checkpoint = f'outputs/pomo_n{args.graph_size}_optimized/model_best.pt'

    if not os.path.exists(args.checkpoint):
        print(f'✗ 模型不存在: {args.checkpoint}')
        return

    print('=' * 64)
    print('Selective SARP-TW 模型评估')
    print('=' * 64)
    print(f'  device       : {device}')
    print(f'  checkpoint   : {args.checkpoint}')
    print(f'  graph_size   : {args.graph_size}')
    print(f'  num_samples  : {args.num_samples}')
    print(f'  batch_size   : {args.batch_size}')
    print(f'  seed         : {args.seed}')
    print(f'  decode       : {args.decode}')
    print('-' * 64)

    # 安全建议：仅加载可信 checkpoint。
    # 注: weights_only=True 在 PyTorch 2.x 才支持; 训练 checkpoint 中保存了
    # vars(args) 字典 / normalization_profile 等非 tensor 对象, 这些在
    # weights_only 严格模式下需要通过 add_safe_globals 显式放行。这里默认
    # 保持 weights_only=False, 但在文档与下方的安全提示中明确"仅加载本仓库
    # 自训练或可信来源的 .pt 文件"。
    try:
        checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    except TypeError:
        # 旧版 PyTorch 不支持 weights_only 参数
        checkpoint = torch.load(args.checkpoint, map_location=device)
    print(f'  loaded epoch : {checkpoint.get("epoch", "?")}')
    state_kwargs = _resolve_state_kwargs(args, checkpoint)
    print(f"  shared env   : max_open={state_kwargs['max_concurrent_open_orders']}, "
          f"delivery_viability={state_kwargs['enable_delivery_viability']}, "
          f"viability_fallback={state_kwargs['enable_viability_fallback']}, "
          f"relax_commitment_trip_time={state_kwargs['relax_pickup_commitment_trip_time']}")

    model = build_model_from_checkpoint(checkpoint, device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    set_decode_type(model, args.decode)

    dataset_kwargs = {}
    if args.passenger_tw_period_weights is not None:
        dataset_kwargs['passenger_tw_period_weights_override'] = _normalize_ratio_triplet(args.passenger_tw_period_weights)
    if args.cargo_tw_period_weights is not None:
        dataset_kwargs['cargo_tw_period_weights_override'] = _normalize_ratio_triplet(args.cargo_tw_period_weights)
    if args.passenger_tw_period_bounds is not None:
        dataset_kwargs['passenger_tw_period_bounds_override'] = _parse_period_bounds(args.passenger_tw_period_bounds)
    if args.cargo_tw_period_bounds is not None:
        dataset_kwargs['cargo_tw_period_bounds_override'] = _parse_period_bounds(args.cargo_tw_period_bounds)

    test_dataset = MCVRPPDTWDataset(
        num_samples=args.num_samples,
        graph_size=args.graph_size,
        seed=args.seed,
        **dataset_kwargs,
    )
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size, collate_fn=collate_fn,
    )

    all_cost_train = []
    all_cost_raw = []
    all_energy_raw = []
    all_pax_delay = []
    all_cargo_delay = []
    all_vehicle_cost = []
    all_reject_penalty = []
    all_unfulfilled_penalty = []
    all_trip_overtime = []
    all_distance = []
    all_num_vehicles = []
    all_num_rejected = []
    all_num_unfulfilled = []
    all_num_completed = []
    all_num_untouched = []
    all_num_untouched_unrejected = []
    all_num_pickup_only = []
    all_num_started_not_completed = []
    all_pickup_hard_violations = []
    all_total_ride_time_violations = []
    all_excess_ride_time_violations = []
    all_diagnostics = {}

    with torch.no_grad():
        for batch in test_loader:
            batch = {k: v.to(device) if torch.is_tensor(v) else v
                     for k, v in batch.items()}
            if args.diagnostics:
                cost, _, pi, debug = model(batch, return_pi=True, return_debug=True, state_kwargs=state_kwargs)
                for key, value in debug.items():
                    if torch.is_tensor(value):
                        all_diagnostics.setdefault(key, []).extend(value.tolist())
            else:
                cost, _, pi = model(batch, return_pi=True, state_kwargs=state_kwargs)
            _, details = MCVRPPDTW.get_costs(batch, pi, return_details=True)

            all_cost_train.extend(cost.tolist())

            def _maybe(name, store):
                if name in details and torch.is_tensor(details[name]):
                    store.extend(details[name].tolist())

            _maybe('total_cost_raw', all_cost_raw)
            _maybe('energy_cost_raw', all_energy_raw)
            _maybe('passenger_delivery_delay_cost_raw', all_pax_delay)
            _maybe('cargo_delay_cost', all_cargo_delay)
            _maybe('vehicle_cost', all_vehicle_cost)
            _maybe('reject_penalty', all_reject_penalty)
            _maybe('unfulfilled_penalty', all_unfulfilled_penalty)
            _maybe('trip_overtime_penalty', all_trip_overtime)
            _maybe('total_distance', all_distance)
            _maybe('passenger_pickup_hard_violations', all_pickup_hard_violations)
            _maybe('passenger_total_ride_time_violations', all_total_ride_time_violations)
            _maybe('passenger_excess_ride_time_violations', all_excess_ride_time_violations)
            # 当前 get_costs 中 rejected_orders 表示主动 reject 数；unfulfilled_orders 单独统计静默未完成
            _maybe('used_vehicles', all_num_vehicles)
            _maybe('rejected_orders', all_num_rejected)
            _maybe('unfulfilled_orders', all_num_unfulfilled)
            _maybe('completed_orders', all_num_completed)
            _maybe('untouched_orders', all_num_untouched)
            _maybe('untouched_unrejected_orders', all_num_untouched_unrejected)
            _maybe('pickup_only_orders', all_num_pickup_only)
            _maybe('started_not_completed_orders', all_num_started_not_completed)

    def _stats(label, data, unit='RMB', fmt='{:.3f}'):
        if not data:
            print(f'  {label:<28}: (n/a)')
            return
        if len(data) >= 2:
            mu, sd = statistics.mean(data), statistics.stdev(data)
            print(f'  {label:<28}: ' + fmt.format(mu) + ' ± ' + fmt.format(sd) + f' {unit}')
        else:
            print(f'  {label:<28}: ' + fmt.format(data[0]) + f' {unit}')

    print('-' * 64)
    print(f'测试结果 ({args.num_samples} 个实例, graph_size={args.graph_size})')
    print('-' * 64)
    print('【主指标 - 论文比较口径】')
    _stats('Total Cost (raw, RMB)', all_cost_raw)
    print()
    print('【成本分项 (RMB)】')
    _stats('  Energy Cost', all_energy_raw)
    _stats('  Passenger Delivery Delay Cost', all_pax_delay)
    _stats('  Cargo Delay Cost', all_cargo_delay)
    _stats('  Vehicle Fixed Cost', all_vehicle_cost)
    _stats('  Reject Penalty', all_reject_penalty)
    _stats('  Unfulfilled Penalty', all_unfulfilled_penalty)
    _stats('  Trip Overtime Penalty', all_trip_overtime)
    print()
    print('【运营指标】')
    service_rates = [value / args.graph_size for value in all_num_completed]
    rejected_rates = [value / args.graph_size for value in all_num_rejected]
    unfulfilled_rates = [value / args.graph_size for value in all_num_unfulfilled]
    untouched_unrejected_rates = [value / args.graph_size for value in all_num_untouched_unrejected]
    served_plus_rejected_rates = [min(1.0, s + r) for s, r in zip(service_rates, rejected_rates)]
    _stats('Total Distance', all_distance, unit='km', fmt='{:.3f}')
    _stats('Passenger Pickup Hard Viol.', all_pickup_hard_violations, unit='', fmt='{:.2f}')
    _stats('Passenger Total Ride Viol.', all_total_ride_time_violations, unit='', fmt='{:.2f}')
    _stats('Passenger Excess Ride Viol.', all_excess_ride_time_violations, unit='', fmt='{:.2f}')
    _stats('# Vehicles Used', all_num_vehicles, unit='', fmt='{:.2f}')
    _stats('# Completed Orders', all_num_completed, unit='', fmt='{:.2f}')
    _stats('# Rejected Orders', all_num_rejected, unit='', fmt='{:.2f}')
    _stats('# Unfulfilled Orders', all_num_unfulfilled, unit='', fmt='{:.2f}')
    _stats('# Untouched Orders', all_num_untouched, unit='', fmt='{:.2f}')
    _stats('# Untouched-Unrejected', all_num_untouched_unrejected, unit='', fmt='{:.2f}')
    _stats('Service Rate', service_rates, unit='', fmt='{:.3f}')
    _stats('Rejected Rate', rejected_rates, unit='', fmt='{:.3f}')
    _stats('Unfulfilled Rate', unfulfilled_rates, unit='', fmt='{:.3f}')
    _stats('Served+Rejected Rate', served_plus_rejected_rates, unit='', fmt='{:.3f}')
    _stats('Untouched-Unrejected Rate', untouched_unrejected_rates, unit='', fmt='{:.3f}')
    _stats('# Pickup-only Orders', all_num_pickup_only, unit='', fmt='{:.2f}')
    _stats('# Started-not-completed', all_num_started_not_completed, unit='', fmt='{:.2f}')
    if args.diagnostics and all_diagnostics:
        print()
        print('【Mask / 可行性诊断】')
        for label, key, fmt in [
            ('Feasible pickups/step', 'diag_feasible_pickups', '{:.2f}'),
            ('Feasible deliveries/step', 'diag_feasible_deliveries', '{:.2f}'),
            ('Any service feasible rate', 'diag_any_service_feasible', '{:.2f}'),
            ('Depot-only rate', 'diag_depot_only', '{:.2f}'),
            ('Reject available rate', 'diag_reject_available_rate', '{:.2f}'),
            ('Feasible->depot rate', 'diag_service_feasible_but_selected_depot', '{:.2f}'),
            ('Feasible->reject rate', 'diag_service_feasible_but_selected_reject', '{:.2f}'),
            ('Mask by pickup TW', 'diag_mask_pickup_tw', '{:.2f}'),
            ('Mask by ride time', 'diag_mask_ride_time', '{:.2f}'),
            ('Mask by trip time', 'diag_mask_trip_time', '{:.2f}'),
            ('Mask by ops end', 'diag_mask_ops_end', '{:.2f}'),
            ('Mask by pickup commitment', 'diag_mask_pickup_commitment', '{:.2f}'),
            ('Open started count', 'diag_open_started_count', '{:.2f}'),
            ('Open started eq2 rate', 'diag_open_started_eq2', '{:.2f}'),
            ('Second pickup feasible', 'diag_second_pickup_feasible', '{:.2f}'),
            ('Second pickup blocked', 'diag_second_pickup_blocked_by_commitment', '{:.2f}'),
            ('Commitment block by K', 'diag_pickup_commitment_block_by_k', '{:.2f}'),
            ('Commitment block by completion', 'diag_pickup_commitment_block_by_completion', '{:.2f}'),
            ('  - completion by ride time', 'diag_pickup_commitment_block_by_completion_ride_time', '{:.2f}'),
            ('  - completion by trip time', 'diag_pickup_commitment_block_by_completion_trip_time', '{:.2f}'),
            ('  - completion by ops end', 'diag_pickup_commitment_block_by_completion_ops_end', '{:.2f}'),
            ('  - completion by open>6', 'diag_pickup_commitment_block_by_completion_open_over_6', '{:.2f}'),
            ('  - completion by other', 'diag_pickup_commitment_block_by_completion_other', '{:.2f}'),
            ('Commitment block by next state', 'diag_pickup_commitment_block_by_next_state', '{:.2f}'),
            ('  - next state by precedence', 'diag_pickup_commitment_block_by_next_state_precedence', '{:.2f}'),
            ('  - next state by ride time', 'diag_pickup_commitment_block_by_next_state_ride_time', '{:.2f}'),
            ('  - next state by trip time', 'diag_pickup_commitment_block_by_next_state_trip_time', '{:.2f}'),
            ('  - next state by ops end', 'diag_pickup_commitment_block_by_next_state_ops_end', '{:.2f}'),
            ('  - next state by delivery viability', 'diag_pickup_commitment_block_by_next_state_delivery_viability', '{:.2f}'),
            ('  - next state by vehicle limit', 'diag_pickup_commitment_block_by_next_state_vehicle_limit', '{:.2f}'),
            ('  - next state by mixed', 'diag_pickup_commitment_block_by_next_state_mixed', '{:.2f}'),
            ('  - next state by other', 'diag_pickup_commitment_block_by_next_state_other', '{:.2f}'),
            ('Commitment block by fallback', 'diag_pickup_commitment_block_by_fallback', '{:.2f}'),
            ('Delivery viability masked', 'diag_delivery_viability_masked', '{:.2f}'),
            ('Delivery viability fallback', 'diag_delivery_viability_fallback', '{:.2f}'),
            ('Mask by vehicle limit', 'diag_mask_vehicle_limit', '{:.2f}'),
            ('Reject predeparture rate', 'diag_reject_predeparture_available', '{:.2f}'),
            ('Reject in-route rate', 'diag_reject_inroute_available', '{:.2f}'),
        ]:
            _stats(label, all_diagnostics.get(key, []), unit='', fmt=fmt)
    print()
    print('【训练目标 (归一化, 仅供参考)】')
    _stats('Total Cost (train)', all_cost_train, unit='', fmt='{:.4f}')
    print()
    print('【问题配置 (Config)】')
    print(f'  Area Size            : {Config.AREA_SIZE} km × {Config.AREA_SIZE} km')
    print(f'  Operation Window     : {Config.OPERATION_START:.1f}:00 - {Config.OPERATION_END:.1f}:00')
    print(f'  Passenger Capacity   : {Config.PASSENGER_CAPACITY} 人')
    print(f'  Cargo Capacity       : {Config.CARGO_CAPACITY} 单位')
    print(f'  Passenger Pickup TW  : {Config.PASSENGER_TW_WIDTH:.1f} h')
    print(f'  Max Trip Time        : {Config.MAX_TRIP_TIME} h')
    print(f'  Passenger Total Ride : {Config.PASSENGER_MAX_RIDE_TIME_MINUTES:.0f} min')
    print(f'  Passenger Excess Ride: {Config.PASSENGER_MAX_EXCESS_RIDE_TIME_MINUTES:.0f} min')
    print(f'  Electricity Price    : {Config.ELECTRICITY_PRICE} RMB/kWh')
    print(f'  Passenger Delivery Delay Cost : {Config.PASSENGER_DELAY_COST} RMB/min')
    print(f'  Cargo Delay Cost     : {Config.CARGO_DELAY_COST} RMB/min')
    print(f'  Vehicle Fixed Cost   : {Config.VEHICLE_COST} RMB/车')
    print(f'  Reject Penalty (α)   : {Config.ALPHA_REJECT} RMB/主动 reject 单')
    print(f'  Unfulfilled (α)      : {Config.ALPHA_UNFULFILLED} RMB/未履约单')
    print(f'  Trip Overtime (α)    : {Config.ALPHA_TRIP_OVERTIME} RMB/h')
    print('=' * 64)


if __name__ == '__main__':
    evaluate()
