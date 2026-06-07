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
    return parser.parse_args()


def collate_fn(batch):
    keys = batch[0].keys()
    return {
        key: torch.stack([sample[key] for sample in batch], dim=0)
        if torch.is_tensor(batch[0][key]) else batch[0][key]
        for key in keys
    }


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

    model = build_model_from_checkpoint(checkpoint, device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    set_decode_type(model, args.decode)

    test_dataset = MCVRPPDTWDataset(
        num_samples=args.num_samples,
        graph_size=args.graph_size,
        seed=args.seed,
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
    all_pickup_hard_violations = []
    all_total_ride_time_violations = []
    all_excess_ride_time_violations = []

    with torch.no_grad():
        for batch in test_loader:
            batch = {k: v.to(device) if torch.is_tensor(v) else v
                     for k, v in batch.items()}
            cost, _, pi = model(batch, return_pi=True)
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
    _stats('Total Distance', all_distance, unit='km', fmt='{:.3f}')
    _stats('Passenger Pickup Hard Viol.', all_pickup_hard_violations, unit='', fmt='{:.2f}')
    _stats('Passenger Total Ride Viol.', all_total_ride_time_violations, unit='', fmt='{:.2f}')
    _stats('Passenger Excess Ride Viol.', all_excess_ride_time_violations, unit='', fmt='{:.2f}')
    _stats('# Vehicles Used', all_num_vehicles, unit='', fmt='{:.2f}')
    _stats('# Completed Orders', all_num_completed, unit='', fmt='{:.2f}')
    _stats('# Rejected Orders', all_num_rejected, unit='', fmt='{:.2f}')
    _stats('# Unfulfilled Orders', all_num_unfulfilled, unit='', fmt='{:.2f}')
    print()
    print('【训练目标 (归一化, 仅供参考)】')
    _stats('Total Cost (train)', all_cost_train, unit='', fmt='{:.4f}')
    print()
    print('【问题配置 (Config)】')
    print(f'  Area Size            : {Config.AREA_SIZE} km × {Config.AREA_SIZE} km')
    print(f'  Operation Window     : {Config.OPERATION_START:.1f}:00 - {Config.OPERATION_END:.1f}:00')
    print(f'  Passenger Capacity   : {Config.PASSENGER_CAPACITY} 人')
    print(f'  Cargo Capacity       : {Config.CARGO_CAPACITY} 单位')
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
