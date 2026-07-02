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
import json
import os
import statistics

import torch
from torch.utils.data import DataLoader

from problem_mcvrptw_v2 import MCVRPPDTWDataset, MCVRPPDTW, Config
from nets.attention_model import AttentionModel, set_decode_type
from state_mcvrptw_v2 import StateMCVRPPDTW


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
    parser.add_argument('--min-orders-per-dispatch', type=int, default=None,
                        help='硬约束：车辆一旦发车，至少完成该数量订单后才可回 depot；默认沿用 checkpoint')
    parser.add_argument('--hard-cargo-pickup-timewindow', dest='hard_cargo_pickup_timewindow', action='store_true',
                        help='启用 cargo pickup 硬时间窗（覆盖 checkpoint）')
    parser.add_argument('--soft-cargo-pickup-timewindow', dest='hard_cargo_pickup_timewindow', action='store_false',
                        help='关闭 cargo pickup 硬时间窗（覆盖 checkpoint）')
    parser.set_defaults(hard_cargo_pickup_timewindow=None)
    parser.add_argument('--cargo-delay-tier1-min', type=float, default=None,
                        help='cargo delivery 分段迟到阈值1（分钟）；默认沿用 checkpoint')
    parser.add_argument('--cargo-delay-tier2-min', type=float, default=None,
                        help='cargo delivery 分段迟到阈值2（分钟）；默认沿用 checkpoint')
    parser.add_argument('--cargo-delay-cost-0-30', type=float, default=None,
                        help='cargo delivery 0~tier1 每分钟成本；默认沿用 checkpoint')
    parser.add_argument('--cargo-delay-cost-30-60', type=float, default=None,
                        help='cargo delivery tier1~tier2 每分钟成本；默认沿用 checkpoint')
    parser.add_argument('--cargo-delay-cost-60-plus', type=float, default=None,
                        help='cargo delivery >tier2 每分钟成本；默认沿用 checkpoint')
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
    parser.add_argument('--trace-first-untouched-unrejected', action='store_true',
                        help='导出第一个 untouched_unrejected>0 的 rollout 样本末尾轨迹')
    parser.add_argument('--trace-output', type=str, default=None,
                        help='轨迹 JSON 输出路径 (default: <checkpoint_dir>/residual_trace.json)')
    parser.add_argument('--shrink-size', type=int, default=None,
                        help='覆盖模型 shrink_size；0 表示禁用，默认沿用 checkpoint / AttentionModel 默认值')
    parser.add_argument('--decode-pickup-urgency-bias', type=float, default=None,
                        help='解码打分：pickup 紧迫度加分系数 beta（默认沿用 checkpoint）')
    parser.add_argument('--decode-pickup-urgency-horizon-hours', type=float, default=None,
                        help='解码打分：pickup 紧迫度窗口 horizon（小时，默认沿用 checkpoint）')
    parser.add_argument('--json-output', type=str, default=None,
                        help='将评估摘要写入 JSON 文件，便于多 checkpoint 正式对比')
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


def build_model_from_checkpoint(checkpoint, device, shrink_size_override=None, decode_pickup_urgency_bias_override=None,
                                decode_pickup_urgency_horizon_hours_override=None):
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

    checkpoint_shrink_size = args_dict.get('shrink_size', None)
    if shrink_size_override is None:
        shrink_size = checkpoint_shrink_size
    elif shrink_size_override <= 0:
        shrink_size = None
    else:
        shrink_size = shrink_size_override

    checkpoint_pickup_urgency_bias = args_dict.get('decode_pickup_urgency_bias', 0.0)
    checkpoint_pickup_urgency_horizon_hours = args_dict.get('decode_pickup_urgency_horizon_hours', 1.0)
    decode_pickup_urgency_bias = (
        checkpoint_pickup_urgency_bias
        if decode_pickup_urgency_bias_override is None
        else decode_pickup_urgency_bias_override
    )
    decode_pickup_urgency_horizon_hours = (
        checkpoint_pickup_urgency_horizon_hours
        if decode_pickup_urgency_horizon_hours_override is None
        else decode_pickup_urgency_horizon_hours_override
    )

    model = AttentionModel(
        embedding_dim=embedding_dim,
        hidden_dim=hidden_dim,
        problem=MCVRPPDTW,
        n_encode_layers=n_encode_layers,
        n_heads=n_heads,
        tanh_clipping=tanh_clipping,
        normalization=normalization,
        shrink_size=shrink_size,
        decode_pickup_urgency_bias=decode_pickup_urgency_bias,
        decode_pickup_urgency_horizon_hours=decode_pickup_urgency_horizon_hours,
    ).to(device)

    print(f'  embedding_dim={embedding_dim}, hidden_dim={hidden_dim}, '
          f'n_encode_layers={n_encode_layers}, n_heads={n_heads}, '
          f'tanh_clipping={tanh_clipping}, normalization={normalization}, '
          f'shrink_size={shrink_size}')
    print(f'  decode_pickup_urgency_bias={float(decode_pickup_urgency_bias):.6f}, '
          f'decode_pickup_urgency_horizon_hours={float(decode_pickup_urgency_horizon_hours):.6f}')

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

    min_orders_per_dispatch = getattr(args, 'min_orders_per_dispatch', None)
    if min_orders_per_dispatch is None:
        min_orders_per_dispatch = int(ckpt_args.get('min_orders_per_dispatch', 4))

    enable_delivery_viability = getattr(args, 'enable_delivery_viability', False)
    if enable_delivery_viability is None:
        enable_delivery_viability = bool(ckpt_args.get('enable_delivery_viability', True))
    elif (not enable_delivery_viability) and ('enable_delivery_viability' in ckpt_args):
        enable_delivery_viability = bool(ckpt_args['enable_delivery_viability'])

    enable_viability_fallback = getattr(args, 'enable_viability_fallback', False)
    if enable_viability_fallback is None:
        enable_viability_fallback = bool(ckpt_args.get('enable_viability_fallback', False))
    elif (not enable_viability_fallback) and ('enable_viability_fallback' in ckpt_args):
        enable_viability_fallback = bool(ckpt_args['enable_viability_fallback'])

    if getattr(args, 'hard_cargo_pickup_timewindow', None) is None:
        Config.HARD_CARGO_PICKUP_TIMEWINDOW = bool(ckpt_args.get('hard_cargo_pickup_timewindow', Config.HARD_CARGO_PICKUP_TIMEWINDOW))
    else:
        Config.HARD_CARGO_PICKUP_TIMEWINDOW = bool(args.hard_cargo_pickup_timewindow)

    arg_tier1 = getattr(args, 'cargo_delay_tier1_min', None)
    arg_tier2 = getattr(args, 'cargo_delay_tier2_min', None)
    tier1 = arg_tier1 if arg_tier1 is not None else ckpt_args.get('cargo_delay_tier1_min', Config.CARGO_DELAY_TIER1_MIN)
    tier2 = arg_tier2 if arg_tier2 is not None else ckpt_args.get('cargo_delay_tier2_min', Config.CARGO_DELAY_TIER2_MIN)
    Config.CARGO_DELAY_TIER1_MIN = max(float(tier1), 0.0)
    Config.CARGO_DELAY_TIER2_MIN = max(float(tier2), Config.CARGO_DELAY_TIER1_MIN)
    arg_cost0 = getattr(args, 'cargo_delay_cost_0_30', None)
    arg_cost1 = getattr(args, 'cargo_delay_cost_30_60', None)
    arg_cost2 = getattr(args, 'cargo_delay_cost_60_plus', None)
    cost0 = arg_cost0 if arg_cost0 is not None else ckpt_args.get('cargo_delay_cost_0_30', Config.CARGO_DELAY_COST)
    cost1 = arg_cost1 if arg_cost1 is not None else ckpt_args.get('cargo_delay_cost_30_60', Config.CARGO_DELAY_COST_30_60)
    cost2 = arg_cost2 if arg_cost2 is not None else ckpt_args.get('cargo_delay_cost_60_plus', Config.CARGO_DELAY_COST_60_PLUS)
    Config.CARGO_DELAY_COST = max(float(cost0), 0.0)
    Config.CARGO_DELAY_COST_30_60 = max(float(cost1), 0.0)
    Config.CARGO_DELAY_COST_60_PLUS = max(float(cost2), 0.0)

    return {
        'max_concurrent_open_orders': max_open,
        'min_orders_per_dispatch': int(max(min_orders_per_dispatch, 1)),
        'enable_delivery_viability': enable_delivery_viability,
        'enable_viability_fallback': enable_viability_fallback,
        'relax_pickup_commitment_trip_time': bool(getattr(args, 'relax_pickup_commitment_trip_time', False)),
    }


def _summarize_order_set(order_indices):
    return [int(idx) + 1 for idx in order_indices]


def _trace_snapshot(state, mask, debug, step, selected=None, note=''):
    def _debug_scalar(key, default=0.0):
        value = debug.get(key)
        if torch.is_tensor(value):
            return float(value[0].item())
        if value is None:
            return float(default)
        return float(value)

    n_orders = state.n_orders
    open_orders = torch.nonzero(state.get_open_started_mask()[0], as_tuple=False).squeeze(-1).tolist()
    pickup_mask = mask[0, 0, 1:1 + n_orders]
    delivery_mask = mask[0, 0, 1 + n_orders:1 + 2 * n_orders]
    feasible_pickups = torch.nonzero(~pickup_mask, as_tuple=False).squeeze(-1).tolist()
    feasible_deliveries = torch.nonzero(~delivery_mask, as_tuple=False).squeeze(-1).tolist()
    untouched_mask = (~state.visited[0, 0, 1:n_orders + 1].bool()) & (~state.visited[0, 0, n_orders + 1:2 * n_orders + 1].bool())
    untouched_unrejected_mask = untouched_mask & (~state.rejected_[0, 0].bool())
    any_service_feasible = float((~mask[0, 0, 1:1 + 2 * n_orders]).any().item())
    depot_only = float((~mask[0, 0, 0]).item() and bool(mask[0, 0, 1:1 + 2 * n_orders].all().item()) and bool(mask[0, 0, state.reject_index].item()))
    snapshot = {
        'step': int(step),
        'note': note,
        'selected': None if selected is None else int(selected),
        'prev_node': int(state.prev_a[0, 0].item()),
        'current_time': float(state.current_time[0, 0].item()),
        'trip_start_time': float(state.trip_start_time[0, 0].item()),
        'used_vehicles': float(state.used_vehicles[0, 0].item()),
        'deadlock_count': int(state.deadlock_count[0, 0].item()),
        'deadlock_limit': int(state.deadlock_limit[0, 0].item()),
        'terminal': bool(state.terminal_[0, 0].item()),
        'finished': bool(state.get_finished()[0].item()),
        'depot_available': bool((~mask[0, 0, 0]).item()),
        'reject_available': bool((~mask[0, 0, state.reject_index]).item()),
        'open_orders': _summarize_order_set(open_orders),
        'feasible_pickups': _summarize_order_set(feasible_pickups),
        'feasible_deliveries': _summarize_order_set(feasible_deliveries),
        'untouched_unrejected_orders': _summarize_order_set(torch.nonzero(untouched_unrejected_mask, as_tuple=False).squeeze(-1).tolist()),
        'diag_open_started_count': _debug_scalar('diag_open_started_count', len(open_orders)),
        'diag_any_service_feasible': _debug_scalar('diag_any_service_feasible', any_service_feasible),
        'diag_depot_only': _debug_scalar('diag_depot_only', depot_only),
        'diag_reject_allowed': _debug_scalar('diag_reject_allowed', float((~mask[0, 0, state.reject_index]).item())),
        'diag_reject_predeparture_available': _debug_scalar('diag_reject_predeparture_available', 0.0),
        'diag_reject_inroute_available': _debug_scalar('diag_reject_inroute_available', 0.0),
        'diag_depot_fallback_used': _debug_scalar('diag_depot_fallback_used', 0.0),
        'diag_mask_precedence': _debug_scalar('diag_mask_precedence', 0.0),
        'diag_mask_pickup_tw': _debug_scalar('diag_mask_pickup_tw', 0.0),
        'diag_mask_trip_time': _debug_scalar('diag_mask_trip_time', 0.0),
        'diag_mask_pickup_commitment': _debug_scalar('diag_mask_pickup_commitment', 0.0),
        'diag_delivery_viability_masked': _debug_scalar('diag_delivery_viability_masked', 0.0),
    }
    return snapshot


def _order_audit_from_state(state):
    batch_size = state.ids.size(0)
    n_orders = state.n_orders
    device = state.coords.device

    if n_orders == 0:
        empty_mask = torch.zeros(batch_size, 0, dtype=torch.bool, device=device)
        zero = torch.zeros(batch_size, device=device)
        return {
            'completed_mask': empty_mask,
            'rejected_mask': empty_mask,
            'untouched_mask': empty_mask,
            'pickup_only_mask': empty_mask,
            'delivery_without_pickup_mask': empty_mask,
            'started_not_completed_mask': empty_mask,
            'untouched_unrejected_mask': empty_mask,
            'unfulfilled_mask': empty_mask,
            'completed_orders': zero,
            'rejected_orders': zero,
            'untouched_orders': zero,
            'pickup_only_orders': zero,
            'delivery_without_pickup_orders': zero,
            'started_not_completed_orders': zero,
            'untouched_unrejected_orders': zero,
            'unfulfilled_orders': zero,
            'partition_ok': torch.ones(batch_size, dtype=torch.bool, device=device),
        }

    visited = state.visited[:, 0, :].bool()
    pickup_visited = visited[:, 1:n_orders + 1]
    delivery_visited = visited[:, n_orders + 1:2 * n_orders + 1]
    rejected_mask = state.rejected_.squeeze(1).bool()

    completed_mask = pickup_visited & delivery_visited & (~rejected_mask)
    untouched_mask = (~pickup_visited) & (~delivery_visited)
    pickup_only_mask = pickup_visited & (~delivery_visited)
    delivery_without_pickup_mask = delivery_visited & (~pickup_visited)
    started_not_completed_mask = pickup_only_mask | delivery_without_pickup_mask
    untouched_unrejected_mask = untouched_mask & (~rejected_mask)
    unfulfilled_mask = started_not_completed_mask | untouched_unrejected_mask

    partition_sum = (
        completed_mask.to(torch.int64)
        + rejected_mask.to(torch.int64)
        + started_not_completed_mask.to(torch.int64)
        + untouched_unrejected_mask.to(torch.int64)
    )

    return {
        'completed_mask': completed_mask,
        'rejected_mask': rejected_mask,
        'untouched_mask': untouched_mask,
        'pickup_only_mask': pickup_only_mask,
        'delivery_without_pickup_mask': delivery_without_pickup_mask,
        'started_not_completed_mask': started_not_completed_mask,
        'untouched_unrejected_mask': untouched_unrejected_mask,
        'unfulfilled_mask': unfulfilled_mask,
        'completed_orders': completed_mask.float().sum(1),
        'rejected_orders': rejected_mask.float().sum(1),
        'untouched_orders': untouched_mask.float().sum(1),
        'pickup_only_orders': pickup_only_mask.float().sum(1),
        'delivery_without_pickup_orders': delivery_without_pickup_mask.float().sum(1),
        'started_not_completed_orders': started_not_completed_mask.float().sum(1),
        'untouched_unrejected_orders': untouched_unrejected_mask.float().sum(1),
        'unfulfilled_orders': unfulfilled_mask.float().sum(1),
        'partition_ok': partition_sum.eq(1).all(dim=1),
    }


def _mask_to_order_ids(mask_row):
    return _summarize_order_set(torch.nonzero(mask_row, as_tuple=False).squeeze(-1).tolist())


def _audit_sample_payload(audit, sample_idx):
    return {
        'completed_orders': float(audit['completed_orders'][sample_idx].item()),
        'rejected_orders': float(audit['rejected_orders'][sample_idx].item()),
        'untouched_orders': float(audit['untouched_orders'][sample_idx].item()),
        'pickup_only_orders': float(audit['pickup_only_orders'][sample_idx].item()),
        'delivery_without_pickup_orders': float(audit['delivery_without_pickup_orders'][sample_idx].item()),
        'started_not_completed_orders': float(audit['started_not_completed_orders'][sample_idx].item()),
        'untouched_unrejected_orders': float(audit['untouched_unrejected_orders'][sample_idx].item()),
        'unfulfilled_orders': float(audit['unfulfilled_orders'][sample_idx].item()),
        'partition_ok': bool(audit['partition_ok'][sample_idx].item()),
        'completed_order_ids': _mask_to_order_ids(audit['completed_mask'][sample_idx]),
        'rejected_order_ids': _mask_to_order_ids(audit['rejected_mask'][sample_idx]),
        'untouched_order_ids': _mask_to_order_ids(audit['untouched_mask'][sample_idx]),
        'pickup_only_order_ids': _mask_to_order_ids(audit['pickup_only_mask'][sample_idx]),
        'delivery_without_pickup_order_ids': _mask_to_order_ids(audit['delivery_without_pickup_mask'][sample_idx]),
        'started_not_completed_order_ids': _mask_to_order_ids(audit['started_not_completed_mask'][sample_idx]),
        'untouched_unrejected_order_ids': _mask_to_order_ids(audit['untouched_unrejected_mask'][sample_idx]),
        'unfulfilled_order_ids': _mask_to_order_ids(audit['unfulfilled_mask'][sample_idx]),
    }


def _replay_order_audit(batch, pi, state_kwargs):
    state = StateMCVRPPDTW.initialize(batch, **state_kwargs)

    if pi.dim() == 1:
        pi = pi[:, None]

    for step in range(pi.size(1)):
        if state.all_finished():
            break
        selected = pi[:, step]
        mask = state.get_mask()
        state = state.update(selected, current_mask=mask)

    audit = _order_audit_from_state(state)
    audit['final_state'] = state
    return audit


def _trace_rollout_case(model, sample, state_kwargs, device):
    batch = {k: (v.unsqueeze(0).to(device) if torch.is_tensor(v) else v) for k, v in sample.items()}
    embeddings, _ = model.embedder(model._init_embed(batch), pd_pair_mask=model._build_pd_pair_mask(batch))
    fixed = model._precompute(embeddings)
    state = StateMCVRPPDTW.initialize(batch, **state_kwargs)
    sequences = []
    trace = []
    consecutive_depot = torch.zeros(1, dtype=torch.long, device=device)
    max_steps = model.max_decode_steps or max(embeddings.size(1) * 3, 8)

    for step in range(max_steps):
        if state.all_finished():
            break
        log_p, mask, debug = model._get_log_p(
            fixed,
            state,
            consecutive_depot=consecutive_depot,
            return_debug=True,
        )
        selected = model._select_node(log_p.exp()[:, 0, :], mask[:, 0, :])
        trace.append(_trace_snapshot(state, mask, debug, step, selected=int(selected[0].item()), note='before_update'))
        sequences.append(selected)
        consecutive_depot = torch.where(selected == 0, consecutive_depot + 1, torch.zeros_like(consecutive_depot))
        state = state.update(selected, current_mask=mask)

    if sequences:
        pi = torch.stack(sequences, 1)
    else:
        pi = torch.zeros((1, 1), dtype=torch.long, device=device)
    _, details = MCVRPPDTW.get_costs(batch, pi, return_details=True)
    audit = _order_audit_from_state(state)
    final_mask, final_debug = state.get_mask(return_debug=True)
    trace.append(_trace_snapshot(state, final_mask, final_debug, len(trace), note='final_state'))
    return {
        'pi': pi[0].tolist(),
        'trace': trace,
        'final_details': {
            'completed_orders': float(details['completed_orders'][0].item()),
            'rejected_orders': float(details['rejected_orders'][0].item()),
            'unfulfilled_orders': float(details['unfulfilled_orders'][0].item()),
            'untouched_orders': float(details['untouched_orders'][0].item()),
            'untouched_unrejected_orders': float(details['untouched_unrejected_orders'][0].item()),
            'pickup_only_orders': float(details['pickup_only_orders'][0].item()),
            'started_not_completed_orders': float(details['started_not_completed_orders'][0].item()),
            'audited_completed_orders': float(audit['completed_orders'][0].item()),
            'audited_rejected_orders': float(audit['rejected_orders'][0].item()),
            'audited_unfulfilled_orders': float(audit['unfulfilled_orders'][0].item()),
            'audited_untouched_orders': float(audit['untouched_orders'][0].item()),
            'audited_untouched_unrejected_orders': float(audit['untouched_unrejected_orders'][0].item()),
            'audited_pickup_only_orders': float(audit['pickup_only_orders'][0].item()),
            'audited_delivery_without_pickup_orders': float(audit['delivery_without_pickup_orders'][0].item()),
            'audited_started_not_completed_orders': float(audit['started_not_completed_orders'][0].item()),
            'partition_ok': bool(audit['partition_ok'][0].item()),
            'completed_order_ids': _mask_to_order_ids(audit['completed_mask'][0]),
            'rejected_order_ids': _mask_to_order_ids(audit['rejected_mask'][0]),
            'untouched_order_ids': _mask_to_order_ids(audit['untouched_mask'][0]),
            'untouched_unrejected_order_ids': _mask_to_order_ids(audit['untouched_unrejected_mask'][0]),
            'pickup_only_order_ids': _mask_to_order_ids(audit['pickup_only_mask'][0]),
            'delivery_without_pickup_order_ids': _mask_to_order_ids(audit['delivery_without_pickup_mask'][0]),
            'started_not_completed_order_ids': _mask_to_order_ids(audit['started_not_completed_mask'][0]),
            'unfulfilled_order_ids': _mask_to_order_ids(audit['unfulfilled_mask'][0]),
        },
    }


def _mean_or_none(values):
    return None if not values else float(statistics.mean(values))



def _is_effectively_zero(value, tol=1e-9):
    return value is not None and abs(float(value)) <= tol



def _stdev_or_none(values):
    if not values:
        return None
    if len(values) == 1:
        return 0.0
    return float(statistics.stdev(values))



def _build_eval_summary(args, checkpoint, device, state_kwargs,
                        all_cost_train, all_cost_raw, all_energy_raw, all_pax_delay, all_cargo_delay,
                        all_vehicle_cost, all_reject_penalty, all_unfulfilled_penalty, all_trip_overtime,
                        all_distance, all_num_vehicles, all_num_rejected, all_num_unfulfilled,
                        all_num_completed, all_num_untouched, all_num_untouched_unrejected,
                        all_num_pickup_only, all_num_started_not_completed,
                        all_pickup_hard_violations, all_total_ride_time_violations, all_excess_ride_time_violations,
                        all_audited_completed, all_audited_rejected, all_audited_untouched,
                        all_audited_untouched_unrejected, all_audited_pickup_only,
                        all_audited_delivery_without_pickup, all_audited_started_not_completed,
                        all_audited_unfulfilled, partition_ok_count, aggregate_match_count,
                        legacy_untouched_match_count, first_audit_mismatch, first_legacy_untouched_mismatch,
                        all_diagnostics=None):
    service_rates = [value / args.graph_size for value in all_num_completed]
    rejected_rates = [value / args.graph_size for value in all_num_rejected]
    unfulfilled_rates = [value / args.graph_size for value in all_num_unfulfilled]
    untouched_unrejected_rates = [value / args.graph_size for value in all_num_untouched_unrejected]
    served_plus_rejected_rates = [min(1.0, s + r) for s, r in zip(service_rates, rejected_rates)]

    audited_service_rates = [value / args.graph_size for value in all_audited_completed]
    audited_rejected_rates = [value / args.graph_size for value in all_audited_rejected]
    audited_unfulfilled_rates = [value / args.graph_size for value in all_audited_unfulfilled]
    audited_untouched_unrejected_rates = [value / args.graph_size for value in all_audited_untouched_unrejected]

    train_cost_mean = _mean_or_none(all_cost_train)
    total_cost_raw_mean = _mean_or_none(all_cost_raw)
    completed_orders_mean = _mean_or_none(all_num_completed)
    rejected_orders_mean = _mean_or_none(all_num_rejected)
    unfulfilled_orders_mean = _mean_or_none(all_num_unfulfilled)
    untouched_orders_mean = _mean_or_none(all_num_untouched)
    untouched_unrejected_orders_mean = _mean_or_none(all_num_untouched_unrejected)
    pickup_only_orders_mean = _mean_or_none(all_num_pickup_only)
    started_not_completed_orders_mean = _mean_or_none(all_num_started_not_completed)

    audited_completed_orders_mean = _mean_or_none(all_audited_completed)
    audited_rejected_orders_mean = _mean_or_none(all_audited_rejected)
    audited_unfulfilled_orders_mean = _mean_or_none(all_audited_unfulfilled)
    audited_untouched_orders_mean = _mean_or_none(all_audited_untouched)
    audited_untouched_unrejected_orders_mean = _mean_or_none(all_audited_untouched_unrejected)
    audited_pickup_only_orders_mean = _mean_or_none(all_audited_pickup_only)
    audited_delivery_without_pickup_orders_mean = _mean_or_none(all_audited_delivery_without_pickup)
    audited_started_not_completed_orders_mean = _mean_or_none(all_audited_started_not_completed)

    service_rate_mean = _mean_or_none(service_rates)
    rejected_rate_mean = _mean_or_none(rejected_rates)
    unfulfilled_rate_mean = _mean_or_none(unfulfilled_rates)
    untouched_unrejected_rate_mean = _mean_or_none(untouched_unrejected_rates)
    served_plus_rejected_rate_mean = _mean_or_none(served_plus_rejected_rates)

    audited_service_rate_mean = _mean_or_none(audited_service_rates)
    audited_rejected_rate_mean = _mean_or_none(audited_rejected_rates)
    audited_unfulfilled_rate_mean = _mean_or_none(audited_unfulfilled_rates)
    audited_untouched_unrejected_rate_mean = _mean_or_none(audited_untouched_unrejected_rates)

    business_acceptance = {
        'gate_unfulfilled_zero': _is_effectively_zero(audited_unfulfilled_orders_mean),
        'gate_pickup_only_zero': _is_effectively_zero(audited_pickup_only_orders_mean),
        'gate_started_not_completed_zero': _is_effectively_zero(audited_started_not_completed_orders_mean),
        'gate_untouched_unrejected_zero': _is_effectively_zero(audited_untouched_unrejected_orders_mean),
    }
    business_acceptance['clean'] = all(business_acceptance.values())
    business_acceptance['ranking_key'] = [
        1 if business_acceptance['clean'] else 0,
        service_rate_mean or 0.0,
        -((train_cost_mean if train_cost_mean is not None else float('inf'))),
    ]

    aggregate = {
        'train_cost_mean': train_cost_mean,
        'train_cost_std': _stdev_or_none(all_cost_train),
        'total_cost_raw_mean': total_cost_raw_mean,
        'total_cost_raw_std': _stdev_or_none(all_cost_raw),
        'energy_cost_mean': _mean_or_none(all_energy_raw),
        'energy_cost_std': _stdev_or_none(all_energy_raw),
        'passenger_delivery_delay_cost_mean': _mean_or_none(all_pax_delay),
        'passenger_delivery_delay_cost_std': _stdev_or_none(all_pax_delay),
        'cargo_delay_cost_mean': _mean_or_none(all_cargo_delay),
        'cargo_delay_cost_std': _stdev_or_none(all_cargo_delay),
        'vehicle_cost_mean': _mean_or_none(all_vehicle_cost),
        'vehicle_cost_std': _stdev_or_none(all_vehicle_cost),
        'reject_penalty_mean': _mean_or_none(all_reject_penalty),
        'reject_penalty_std': _stdev_or_none(all_reject_penalty),
        'unfulfilled_penalty_mean': _mean_or_none(all_unfulfilled_penalty),
        'unfulfilled_penalty_std': _stdev_or_none(all_unfulfilled_penalty),
        'trip_overtime_penalty_mean': _mean_or_none(all_trip_overtime),
        'trip_overtime_penalty_std': _stdev_or_none(all_trip_overtime),
        'distance_mean': _mean_or_none(all_distance),
        'distance_std': _stdev_or_none(all_distance),
        'used_vehicles_mean': _mean_or_none(all_num_vehicles),
        'used_vehicles_std': _stdev_or_none(all_num_vehicles),
        'completed_orders_mean': completed_orders_mean,
        'completed_orders_std': _stdev_or_none(all_num_completed),
        'rejected_orders_mean': rejected_orders_mean,
        'rejected_orders_std': _stdev_or_none(all_num_rejected),
        'unfulfilled_orders_mean': unfulfilled_orders_mean,
        'unfulfilled_orders_std': _stdev_or_none(all_num_unfulfilled),
        'untouched_orders_mean': untouched_orders_mean,
        'untouched_orders_std': _stdev_or_none(all_num_untouched),
        'untouched_unrejected_orders_mean': untouched_unrejected_orders_mean,
        'untouched_unrejected_orders_std': _stdev_or_none(all_num_untouched_unrejected),
        'pickup_only_orders_mean': pickup_only_orders_mean,
        'pickup_only_orders_std': _stdev_or_none(all_num_pickup_only),
        'started_not_completed_orders_mean': started_not_completed_orders_mean,
        'started_not_completed_orders_std': _stdev_or_none(all_num_started_not_completed),
        'service_rate_mean': service_rate_mean,
        'service_rate_std': _stdev_or_none(service_rates),
        'rejected_rate_mean': rejected_rate_mean,
        'rejected_rate_std': _stdev_or_none(rejected_rates),
        'unfulfilled_rate_mean': unfulfilled_rate_mean,
        'unfulfilled_rate_std': _stdev_or_none(unfulfilled_rates),
        'untouched_unrejected_rate_mean': untouched_unrejected_rate_mean,
        'untouched_unrejected_rate_std': _stdev_or_none(untouched_unrejected_rates),
        'served_plus_rejected_rate_mean': served_plus_rejected_rate_mean,
        'served_plus_rejected_rate_std': _stdev_or_none(served_plus_rejected_rates),
        'passenger_pickup_hard_violations_mean': _mean_or_none(all_pickup_hard_violations),
        'passenger_pickup_hard_violations_std': _stdev_or_none(all_pickup_hard_violations),
        'passenger_total_ride_time_violations_mean': _mean_or_none(all_total_ride_time_violations),
        'passenger_total_ride_time_violations_std': _stdev_or_none(all_total_ride_time_violations),
        'passenger_excess_ride_time_violations_mean': _mean_or_none(all_excess_ride_time_violations),
        'passenger_excess_ride_time_violations_std': _stdev_or_none(all_excess_ride_time_violations),
    }

    audit = {
        'completed_orders_mean': audited_completed_orders_mean,
        'completed_orders_std': _stdev_or_none(all_audited_completed),
        'rejected_orders_mean': audited_rejected_orders_mean,
        'rejected_orders_std': _stdev_or_none(all_audited_rejected),
        'untouched_orders_mean': audited_untouched_orders_mean,
        'untouched_orders_std': _stdev_or_none(all_audited_untouched),
        'untouched_unrejected_orders_mean': audited_untouched_unrejected_orders_mean,
        'untouched_unrejected_orders_std': _stdev_or_none(all_audited_untouched_unrejected),
        'pickup_only_orders_mean': audited_pickup_only_orders_mean,
        'pickup_only_orders_std': _stdev_or_none(all_audited_pickup_only),
        'delivery_without_pickup_orders_mean': audited_delivery_without_pickup_orders_mean,
        'delivery_without_pickup_orders_std': _stdev_or_none(all_audited_delivery_without_pickup),
        'started_not_completed_orders_mean': audited_started_not_completed_orders_mean,
        'started_not_completed_orders_std': _stdev_or_none(all_audited_started_not_completed),
        'unfulfilled_orders_mean': audited_unfulfilled_orders_mean,
        'unfulfilled_orders_std': _stdev_or_none(all_audited_unfulfilled),
        'service_rate_mean': audited_service_rate_mean,
        'service_rate_std': _stdev_or_none(audited_service_rates),
        'rejected_rate_mean': audited_rejected_rate_mean,
        'rejected_rate_std': _stdev_or_none(audited_rejected_rates),
        'unfulfilled_rate_mean': audited_unfulfilled_rate_mean,
        'unfulfilled_rate_std': _stdev_or_none(audited_unfulfilled_rates),
        'untouched_unrejected_rate_mean': audited_untouched_unrejected_rate_mean,
        'untouched_unrejected_rate_std': _stdev_or_none(audited_untouched_unrejected_rates),
        'partition_consistent_samples': partition_ok_count,
        'core_aggregate_match_samples': aggregate_match_count,
        'legacy_untouched_match_samples': legacy_untouched_match_count,
        'total_samples': args.num_samples,
        'first_audit_mismatch': first_audit_mismatch,
        'first_legacy_untouched_mismatch': first_legacy_untouched_mismatch,
    }

    summary = {
        'run': {
            'checkpoint': args.checkpoint,
            'loaded_epoch': checkpoint.get('epoch'),
            'graph_size': args.graph_size,
            'num_samples': args.num_samples,
            'batch_size': args.batch_size,
            'seed': args.seed,
            'decode': args.decode,
            'device': str(device),
            'state_kwargs': state_kwargs,
            'decode_pickup_urgency_bias': getattr(args, 'resolved_decode_pickup_urgency_bias', None),
            'decode_pickup_urgency_horizon_hours': getattr(args, 'resolved_decode_pickup_urgency_horizon_hours', None),
            'checkpoint_role': checkpoint.get('checkpoint_role'),
            'selection_rule': checkpoint.get('selection_rule'),
            'business_clean': checkpoint.get('business_clean'),
        },
        'aggregate': aggregate,
        'audit': audit,
        'business_acceptance': business_acceptance,
    }
    if all_diagnostics:
        summary['diagnostics'] = {
            key: {
                'mean': _mean_or_none(values),
                'std': _stdev_or_none(values),
            }
            for key, values in all_diagnostics.items()
        }
    return summary



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
          f"min_orders_per_dispatch={state_kwargs['min_orders_per_dispatch']}, "
          f"delivery_viability={state_kwargs['enable_delivery_viability']}, "
          f"viability_fallback={state_kwargs['enable_viability_fallback']}, "
          f"relax_commitment_trip_time={state_kwargs['relax_pickup_commitment_trip_time']}")

    shrink_size_override = args.shrink_size
    model = build_model_from_checkpoint(
        checkpoint,
        device,
        shrink_size_override=shrink_size_override,
        decode_pickup_urgency_bias_override=args.decode_pickup_urgency_bias,
        decode_pickup_urgency_horizon_hours_override=args.decode_pickup_urgency_horizon_hours,
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    set_decode_type(model, args.decode)
    args.resolved_decode_pickup_urgency_bias = float(getattr(model, 'decode_pickup_urgency_bias', 0.0))
    args.resolved_decode_pickup_urgency_horizon_hours = float(getattr(model, 'decode_pickup_urgency_horizon_hours', 1.0))

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
    all_audited_completed = []
    all_audited_rejected = []
    all_audited_untouched = []
    all_audited_untouched_unrejected = []
    all_audited_pickup_only = []
    all_audited_delivery_without_pickup = []
    all_audited_started_not_completed = []
    all_audited_unfulfilled = []
    partition_ok_count = 0
    aggregate_match_count = 0
    legacy_untouched_match_count = 0
    first_audit_mismatch = None
    first_legacy_untouched_mismatch = None
    residual_trace_payload = None
    sample_offset = 0

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
            audit = _replay_order_audit(batch, pi, state_kwargs)

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

            all_audited_completed.extend(audit['completed_orders'].tolist())
            all_audited_rejected.extend(audit['rejected_orders'].tolist())
            all_audited_untouched.extend(audit['untouched_orders'].tolist())
            all_audited_untouched_unrejected.extend(audit['untouched_unrejected_orders'].tolist())
            all_audited_pickup_only.extend(audit['pickup_only_orders'].tolist())
            all_audited_delivery_without_pickup.extend(audit['delivery_without_pickup_orders'].tolist())
            all_audited_started_not_completed.extend(audit['started_not_completed_orders'].tolist())
            all_audited_unfulfilled.extend(audit['unfulfilled_orders'].tolist())
            partition_ok_count += int(audit['partition_ok'].sum().item())

            batch_size = pi.size(0)
            sample_match_mask = audit['partition_ok'].clone()
            for detail_key, audit_key in [
                ('completed_orders', 'completed_orders'),
                ('rejected_orders', 'rejected_orders'),
                ('unfulfilled_orders', 'unfulfilled_orders'),
                ('untouched_unrejected_orders', 'untouched_unrejected_orders'),
                ('pickup_only_orders', 'pickup_only_orders'),
                ('started_not_completed_orders', 'started_not_completed_orders'),
            ]:
                sample_match_mask &= torch.isclose(details[detail_key].float(), audit[audit_key].float())
            aggregate_match_count += int(sample_match_mask.sum().item())
            legacy_untouched_match_mask = torch.isclose(details['untouched_orders'].float(), audit['untouched_orders'].float())
            legacy_untouched_match_count += int(legacy_untouched_match_mask.sum().item())

            if first_audit_mismatch is None:
                mismatch_indices = torch.nonzero(~sample_match_mask, as_tuple=False).squeeze(-1)
                if mismatch_indices.numel() > 0:
                    local_idx = int(mismatch_indices[0].item())
                    first_audit_mismatch = {
                        'sample_index': sample_offset + local_idx,
                        'pi': pi[local_idx].detach().cpu().tolist(),
                        'aggregate_details': {
                            'completed_orders': float(details['completed_orders'][local_idx].item()),
                            'rejected_orders': float(details['rejected_orders'][local_idx].item()),
                            'unfulfilled_orders': float(details['unfulfilled_orders'][local_idx].item()),
                            'untouched_orders': float(details['untouched_orders'][local_idx].item()),
                            'untouched_unrejected_orders': float(details['untouched_unrejected_orders'][local_idx].item()),
                            'pickup_only_orders': float(details['pickup_only_orders'][local_idx].item()),
                            'started_not_completed_orders': float(details['started_not_completed_orders'][local_idx].item()),
                        },
                        'audited_details': _audit_sample_payload(audit, local_idx),
                    }

            if first_legacy_untouched_mismatch is None:
                mismatch_indices = torch.nonzero(~legacy_untouched_match_mask, as_tuple=False).squeeze(-1)
                if mismatch_indices.numel() > 0:
                    local_idx = int(mismatch_indices[0].item())
                    first_legacy_untouched_mismatch = {
                        'sample_index': sample_offset + local_idx,
                        'pi': pi[local_idx].detach().cpu().tolist(),
                        'legacy_untouched_orders': float(details['untouched_orders'][local_idx].item()),
                        'audited_untouched_orders': float(audit['untouched_orders'][local_idx].item()),
                        'audited_details': _audit_sample_payload(audit, local_idx),
                    }

            if args.trace_first_untouched_unrejected and residual_trace_payload is None:
                untouched_unrejected = audit['untouched_unrejected_orders']
                if torch.is_tensor(untouched_unrejected):
                    hit_indices = torch.nonzero(untouched_unrejected > 0, as_tuple=False).squeeze(-1)
                    if hit_indices.numel() > 0:
                        local_idx = int(hit_indices[0].item())
                        sample = {
                            key: (value[local_idx].detach().cpu() if torch.is_tensor(value) else value)
                            for key, value in batch.items()
                        }
                        trace_result = _trace_rollout_case(model, sample, state_kwargs, device)
                        residual_trace_payload = {
                            'sample_index': sample_offset + local_idx,
                            'checkpoint': args.checkpoint,
                            'graph_size': args.graph_size,
                            'seed': args.seed,
                            'state_kwargs': state_kwargs,
                            **trace_result,
                        }
            sample_offset += batch['loc'].size(0)

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
    _stats('# Rejected Before Service', all_num_rejected, unit='', fmt='{:.2f}')
    _stats('# Residual Unfulfilled Orders', all_num_unfulfilled, unit='', fmt='{:.2f}')
    _stats('# Untouched Orders (legacy accounting; includes explicit rejects)', all_num_untouched, unit='', fmt='{:.2f}')
    _stats('# Untouched, Not Explicitly Rejected', all_num_untouched_unrejected, unit='', fmt='{:.2f}')
    _stats('Service Rate', service_rates, unit='', fmt='{:.3f}')
    _stats('Rejected Before Service Rate', rejected_rates, unit='', fmt='{:.3f}')
    _stats('Residual Unfulfilled Rate', unfulfilled_rates, unit='', fmt='{:.3f}')
    _stats('Served+Rejected Rate', served_plus_rejected_rates, unit='', fmt='{:.3f}')
    _stats('Untouched, Not Explicitly Rejected Rate', untouched_unrejected_rates, unit='', fmt='{:.3f}')
    _stats('# Pickup-only Orders', all_num_pickup_only, unit='', fmt='{:.2f}')
    _stats('# Started but Not Completed', all_num_started_not_completed, unit='', fmt='{:.2f}')
    print()
    print('【Order-level replay audit】')
    audited_service_rates = [value / args.graph_size for value in all_audited_completed]
    audited_rejected_rates = [value / args.graph_size for value in all_audited_rejected]
    audited_unfulfilled_rates = [value / args.graph_size for value in all_audited_unfulfilled]
    audited_untouched_unrejected_rates = [value / args.graph_size for value in all_audited_untouched_unrejected]
    _stats('Audited Completed Orders', all_audited_completed, unit='', fmt='{:.2f}')
    _stats('Audited Rejected Before Service', all_audited_rejected, unit='', fmt='{:.2f}')
    _stats('Audited Untouched Orders (legacy accounting view)', all_audited_untouched, unit='', fmt='{:.2f}')
    _stats('Audited Untouched, Not Explicitly Rejected', all_audited_untouched_unrejected, unit='', fmt='{:.2f}')
    _stats('Audited Pickup-only Orders', all_audited_pickup_only, unit='', fmt='{:.2f}')
    _stats('Audited Delivery Without Pickup', all_audited_delivery_without_pickup, unit='', fmt='{:.2f}')
    _stats('Audited Started but Not Completed', all_audited_started_not_completed, unit='', fmt='{:.2f}')
    _stats('Audited Residual Unfulfilled Orders', all_audited_unfulfilled, unit='', fmt='{:.2f}')
    _stats('Audited Service Rate', audited_service_rates, unit='', fmt='{:.3f}')
    _stats('Audited Rejected Before Service Rate', audited_rejected_rates, unit='', fmt='{:.3f}')
    _stats('Audited Residual Unfulfilled Rate', audited_unfulfilled_rates, unit='', fmt='{:.3f}')
    _stats('Audited Untouched, Not Explicitly Rejected Rate', audited_untouched_unrejected_rates, unit='', fmt='{:.3f}')
    print(f'  Partition-consistent samples   : {partition_ok_count}/{args.num_samples}')
    print(f'  Core aggregate-match samples  : {aggregate_match_count}/{args.num_samples}')
    print(f'  Legacy untouched-match samples: {legacy_untouched_match_count}/{args.num_samples}')
    if first_audit_mismatch is not None:
        print('  First core aggregate mismatch : sample #{sample_index}'.format(**first_audit_mismatch))
        print('    aggregate details           : ' + json.dumps(first_audit_mismatch['aggregate_details'], ensure_ascii=False))
        print('    audited details             : ' + json.dumps(first_audit_mismatch['audited_details'], ensure_ascii=False))
    if first_legacy_untouched_mismatch is not None:
        print('  First legacy untouched gap    : sample #{sample_index}'.format(**first_legacy_untouched_mismatch))
        print('    legacy untouched_orders     : {legacy_untouched_orders} (legacy accounting count)'.format(**first_legacy_untouched_mismatch))
        print('    audited untouched_orders    : {audited_untouched_orders} (true untouched-after-replay count)'.format(**first_legacy_untouched_mismatch))
        print('    audited details             : ' + json.dumps(first_legacy_untouched_mismatch['audited_details'], ensure_ascii=False))
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
    if args.trace_first_untouched_unrejected:
        print()
        print('【Residual trace】')
        if residual_trace_payload is None:
            print('  No untouched-unrejected sample found.')
        else:
            trace_output = args.trace_output or os.path.join(os.path.dirname(args.checkpoint), 'residual_trace.json')
            with open(trace_output, 'w', encoding='utf-8') as trace_file:
                json.dump(residual_trace_payload, trace_file, ensure_ascii=False, indent=2)
            print(f'  Saved trace to: {trace_output}')
            print(f"  Sample index : {residual_trace_payload['sample_index']}")
            print(f"  Final untouched-unrejected: {residual_trace_payload['final_details']['untouched_unrejected_orders']:.0f}")
    print()
    summary_payload = _build_eval_summary(
        args,
        checkpoint,
        device,
        state_kwargs,
        all_cost_train,
        all_cost_raw,
        all_energy_raw,
        all_pax_delay,
        all_cargo_delay,
        all_vehicle_cost,
        all_reject_penalty,
        all_unfulfilled_penalty,
        all_trip_overtime,
        all_distance,
        all_num_vehicles,
        all_num_rejected,
        all_num_unfulfilled,
        all_num_completed,
        all_num_untouched,
        all_num_untouched_unrejected,
        all_num_pickup_only,
        all_num_started_not_completed,
        all_pickup_hard_violations,
        all_total_ride_time_violations,
        all_excess_ride_time_violations,
        all_audited_completed,
        all_audited_rejected,
        all_audited_untouched,
        all_audited_untouched_unrejected,
        all_audited_pickup_only,
        all_audited_delivery_without_pickup,
        all_audited_started_not_completed,
        all_audited_unfulfilled,
        partition_ok_count,
        aggregate_match_count,
        legacy_untouched_match_count,
        first_audit_mismatch,
        first_legacy_untouched_mismatch,
        all_diagnostics,
    )
    if args.json_output:
        with open(args.json_output, 'w', encoding='utf-8') as json_file:
            json.dump(summary_payload, json_file, ensure_ascii=False, indent=2)
        print(f'JSON summary saved to: {args.json_output}')
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
    print(f'  Cargo Pickup TW Hard : {Config.HARD_CARGO_PICKUP_TIMEWINDOW}')
    print(f'  Cargo Delay Piecewise: [0,{Config.CARGO_DELAY_TIER1_MIN:.0f}]={Config.CARGO_DELAY_COST} RMB/min, '
          f'({Config.CARGO_DELAY_TIER1_MIN:.0f},{Config.CARGO_DELAY_TIER2_MIN:.0f}]={Config.CARGO_DELAY_COST_30_60} RMB/min, '
          f'>{Config.CARGO_DELAY_TIER2_MIN:.0f}={Config.CARGO_DELAY_COST_60_PLUS} RMB/min')
    print(f'  Vehicle Fixed Cost   : {Config.VEHICLE_COST} RMB/车')
    print(f'  Reject Penalty (α)   : {Config.ALPHA_REJECT} RMB/主动 reject 单')
    print(f'  Unfulfilled (α)      : {Config.ALPHA_UNFULFILLED} RMB/未履约单')
    print(f'  Trip Overtime (α)    : {Config.ALPHA_TRIP_OVERTIME} RMB/h')
    print('=' * 64)


if __name__ == '__main__':
    evaluate()
