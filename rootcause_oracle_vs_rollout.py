import argparse
import json
import os
import random
import statistics
import sys
from datetime import datetime

import torch
from torch.utils.data import DataLoader

from evaluate_model import build_model_from_checkpoint, _resolve_state_kwargs
from nets.attention_model import set_decode_type
from problem_mcvrptw_v2 import MCVRPPDTWDataset, MCVRPPDTW
from state_mcvrptw_v2 import StateMCVRPPDTW


HEURISTIC_POLICIES = ('oracle_style', 'random', 'bad')
ALL_POLICIES = ('rollout',) + HEURISTIC_POLICIES
RATE_DEBUG_KEYS = {
    'diag_feasible_pickups',
    'diag_feasible_deliveries',
    'diag_feasible_service',
    'diag_any_service_feasible',
    'diag_depot_only',
    'diag_reject_available_rate',
    'diag_service_feasible_but_selected_depot',
    'diag_service_feasible_but_selected_reject',
    'diag_selected_pickup',
    'diag_selected_delivery',
    'diag_selected_depot',
    'diag_selected_reject',
    'diag_mask_visited',
    'diag_mask_precedence',
    'diag_mask_cap_passenger',
    'diag_mask_cap_cargo',
    'diag_mask_pickup_tw',
    'diag_mask_ride_time',
    'diag_mask_trip_time',
    'diag_mask_ops_end',
    'diag_mask_pickup_commitment',
    'diag_open_started_count',
    'diag_open_started_eq2',
    'diag_second_pickup_feasible',
    'diag_second_pickup_blocked_by_commitment',
    'diag_pickup_commitment_block_by_k',
    'diag_pickup_commitment_block_by_completion',
    'diag_pickup_commitment_block_by_completion_ride_time',
    'diag_pickup_commitment_block_by_completion_trip_time',
    'diag_pickup_commitment_block_by_completion_ops_end',
    'diag_pickup_commitment_block_by_completion_open_over_6',
    'diag_pickup_commitment_block_by_completion_other',
    'diag_pickup_commitment_block_by_next_state',
    'diag_pickup_commitment_block_by_fallback',
    'diag_delivery_viability_masked',
    'diag_delivery_viability_fallback',
    'diag_mask_vehicle_limit',
    'diag_depot_carry_block',
    'diag_depot_no_work_block',
    'diag_depot_fallback_used',
    'diag_reject_candidate_available',
    'diag_reject_allowed',
    'diag_reject_predeparture_available',
    'diag_reject_inroute_available',
}
DETAIL_KEYS = [
    'objective_total',
    'total_cost_raw',
    'energy_cost_raw',
    'passenger_delivery_delay_cost_raw',
    'cargo_delay_cost_raw',
    'vehicle_cost_raw',
    'reject_penalty',
    'unfulfilled_penalty',
    'trip_overtime_penalty',
    'used_vehicles',
    'rejected_orders',
    'unfulfilled_orders',
    'completed_orders',
    'pickup_only_orders',
    'started_not_completed_orders',
    'passenger_pickup_hard_violations',
    'passenger_total_ride_time_violations',
    'passenger_excess_ride_time_violations',
    'total_distance',
]
MASK_SHARE_KEYS = [
    'diag_mask_visited',
    'diag_mask_precedence',
    'diag_mask_cap_passenger',
    'diag_mask_cap_cargo',
    'diag_mask_pickup_tw',
    'diag_mask_ride_time',
    'diag_mask_trip_time',
    'diag_mask_ops_end',
    'diag_mask_pickup_commitment',
    'diag_delivery_viability_masked',
    'diag_mask_vehicle_limit',
]


def parse_args():
    parser = argparse.ArgumentParser(
        description='Compare rollout against oracle-style / heuristic baselines on identical Selective SARP-TW instances.'
    )
    parser.add_argument('--graph-size', type=int, default=25)
    parser.add_argument('--num-samples', type=int, default=16)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--seeds', nargs='+', type=int, required=True,
                        help='Dataset seeds to evaluate. The same instances are used for all policies per seed.')
    parser.add_argument('--policies', nargs='+', default=['rollout', 'oracle_style', 'random'],
                        choices=ALL_POLICIES)
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Required when rollout is included. Path to model checkpoint.')
    parser.add_argument('--decode', type=str, default='greedy', choices=['greedy', 'sampling'])
    parser.add_argument('--no-cuda', action='store_true')
    parser.add_argument('--max-concurrent-open-orders', type=int, default=1,
                        help='If left at 1, rollout state kwargs will inherit the checkpoint value when available.')
    parser.add_argument('--enable-delivery-viability', dest='enable_delivery_viability', action='store_true')
    parser.add_argument('--disable-delivery-viability', dest='enable_delivery_viability', action='store_false')
    parser.set_defaults(enable_delivery_viability=None)
    parser.add_argument('--enable-viability-fallback', dest='enable_viability_fallback', action='store_true')
    parser.add_argument('--disable-viability-fallback', dest='enable_viability_fallback', action='store_false')
    parser.set_defaults(enable_viability_fallback=None)
    parser.add_argument('--deadlock-limit', type=int, default=2)
    parser.add_argument('--passenger-tw-period-weights', nargs=3, type=float, default=None,
                        metavar=('MORNING', 'MIDDAY', 'EVENING'))
    parser.add_argument('--cargo-tw-period-weights', nargs=3, type=float, default=None,
                        metavar=('MORNING', 'MIDDAY', 'EVENING'))
    parser.add_argument('--output-dir', type=str, required=True)
    parser.add_argument('--trace-first-failure', action='store_true',
                        help='For heuristic policies, save the first failure trace in the per-seed JSON.')
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


def _build_dataset_kwargs(args):
    kwargs = {}
    if args.passenger_tw_period_weights is not None:
        kwargs['passenger_tw_period_weights_override'] = _normalize_ratio_triplet(args.passenger_tw_period_weights)
    if args.cargo_tw_period_weights is not None:
        kwargs['cargo_tw_period_weights_override'] = _normalize_ratio_triplet(args.cargo_tw_period_weights)
    return kwargs


def _node_coord(batch, node_idx):
    if node_idx == 0:
        return batch['depot'][0]
    return batch['loc'][0, node_idx - 1]


def _order_deadline(batch, order_idx):
    return float(batch['time_windows'][0, order_idx, 1].item())


def _service_coord(batch, state, action):
    kind, order_idx = action
    if kind == 'P':
        node_idx = order_idx + 1
    else:
        node_idx = state.n_orders + order_idx + 1
    return _node_coord(batch, node_idx)


def choose_heuristic_action(state, policy, rng, batch, pickups, deliveries):
    service_actions = [('P', idx) for idx in pickups] + [('D', idx) for idx in deliveries]
    if not service_actions:
        return None

    cur_coord = state.cur_coord[0, 0]
    if policy == 'bad':
        pickup_actions = [action for action in service_actions if action[0] == 'P']
        delivery_actions = [action for action in service_actions if action[0] == 'D']
        candidates = pickup_actions if pickup_actions else delivery_actions
        return max(candidates, key=lambda action: torch.norm(_service_coord(batch, state, action) - cur_coord).item())

    if policy == 'oracle_style':
        if deliveries:
            return ('D', min(deliveries, key=lambda idx: _order_deadline(batch, idx)))
        return ('P', min(pickups, key=lambda idx: _order_deadline(batch, idx)))

    return rng.choice(service_actions)


def _service_candidates(mask, state):
    feasible = (~mask[0, 0]).nonzero(as_tuple=False).squeeze(-1).tolist()
    pickups = [idx - 1 for idx in feasible if 1 <= idx <= state.n_orders]
    deliveries = [idx - (state.n_orders + 1) for idx in feasible if state.n_orders + 1 <= idx <= 2 * state.n_orders]
    return pickups, deliveries


def _summarize_order_set(order_indices):
    return [int(idx) + 1 for idx in order_indices]


def _collect_trace_snapshot(state, mask, debug, pickups, deliveries, step, selected_action=None, note=''):
    open_mask = state.get_open_started_mask()[0]
    open_orders = torch.nonzero(open_mask, as_tuple=False).squeeze(-1).tolist()
    snapshot = {
        'step': int(step),
        'note': note,
        'selected_action': selected_action,
        'open_orders': _summarize_order_set(open_orders),
        'pickups': _summarize_order_set(pickups),
        'deliveries': _summarize_order_set(deliveries),
        'current_time': float(state.current_time[0, 0].item()),
        'trip_start_time': float(state.trip_start_time[0, 0].item()),
        'prev_node': int(state.prev_a[0, 0].item()),
        'used_vehicles': float(state.used_vehicles[0, 0].item()),
        'diag_open_started_count': float(debug['diag_open_started_count'][0].item()),
        'diag_second_pickup_feasible': float(debug['diag_second_pickup_feasible'][0].item()),
        'diag_second_pickup_blocked_by_commitment': float(debug['diag_second_pickup_blocked_by_commitment'][0].item()),
        'diag_delivery_viability_masked': float(debug['diag_delivery_viability_masked'][0].item()),
        'diag_delivery_viability_fallback': float(debug['diag_delivery_viability_fallback'][0].item()),
        'diag_mask_precedence': float(debug['diag_mask_precedence'][0].item()),
        'diag_mask_pickup_tw': float(debug['diag_mask_pickup_tw'][0].item()),
        'diag_mask_ride_time': float(debug['diag_mask_ride_time'][0].item()),
        'diag_mask_trip_time': float(debug['diag_mask_trip_time'][0].item()),
        'diag_mask_ops_end': float(debug['diag_mask_ops_end'][0].item()),
        'diag_mask_pickup_commitment': float(debug['diag_mask_pickup_commitment'][0].item()),
        'diag_mask_vehicle_limit': float(debug['diag_mask_vehicle_limit'][0].item()),
        'diag_depot_fallback_used': float(debug['diag_depot_fallback_used'][0].item()),
    }
    return snapshot


def _base_step_debug(mask, state, debug):
    service_mask = mask[0, 0, 1:1 + 2 * state.n_orders]
    pickup_mask = mask[0, 0, 1:1 + state.n_orders]
    delivery_mask = mask[0, 0, 1 + state.n_orders:1 + 2 * state.n_orders]
    step_debug = {key: float(debug[key][0].item()) for key in debug}
    step_debug['diag_steps'] = 1.0
    step_debug['diag_feasible_service'] = float((~service_mask).sum().item())
    step_debug['diag_feasible_pickups'] = float((~pickup_mask).sum().item())
    step_debug['diag_feasible_deliveries'] = float((~delivery_mask).sum().item())
    step_debug['diag_any_service_feasible'] = float((~service_mask).any().item())
    step_debug['diag_depot_only'] = float((not bool(mask[0, 0, 0].item())) and bool(service_mask.all().item()) and bool(mask[0, 0, -1].item()))
    step_debug['diag_reject_available_rate'] = float((~mask[0, 0, -1]).item())
    step_debug['diag_selected_pickup'] = 0.0
    step_debug['diag_selected_delivery'] = 0.0
    step_debug['diag_selected_depot'] = 0.0
    step_debug['diag_selected_reject'] = 0.0
    step_debug['diag_service_feasible_but_selected_depot'] = 0.0
    step_debug['diag_service_feasible_but_selected_reject'] = 0.0
    return step_debug


def _update_selected_debug(step_debug, state, selected):
    reject_index = state.reject_index
    service_feasible = step_debug['diag_any_service_feasible'] > 0
    if selected == 0:
        step_debug['diag_selected_depot'] = 1.0
        if service_feasible:
            step_debug['diag_service_feasible_but_selected_depot'] = 1.0
    elif selected == reject_index:
        step_debug['diag_selected_reject'] = 1.0
        if service_feasible:
            step_debug['diag_service_feasible_but_selected_reject'] = 1.0
    elif 1 <= selected <= state.n_orders:
        step_debug['diag_selected_pickup'] = 1.0
    elif state.n_orders + 1 <= selected <= 2 * state.n_orders:
        step_debug['diag_selected_delivery'] = 1.0


def _normalize_debug(debug_totals):
    steps = max(float(debug_totals.get('diag_steps', 0.0)), 1.0)
    normalized = {}
    for key, value in debug_totals.items():
        if key == 'diag_steps':
            normalized[key] = value
        elif key in RATE_DEBUG_KEYS:
            normalized[key] = value / steps
        else:
            normalized[key] = value
    return normalized


def _summarize_details(details, graph_size):
    summary = {}
    for key in DETAIL_KEYS:
        values = details.get(key, [])
        summary[key] = float(statistics.fmean(values)) if values else 0.0
    summary['service_rate'] = summary['completed_orders'] / float(graph_size)
    summary['rejected_rate'] = summary['rejected_orders'] / float(graph_size)
    summary['unfulfilled_rate'] = summary['unfulfilled_orders'] / float(graph_size)
    summary['pickup_only_rate'] = summary['pickup_only_orders'] / float(graph_size)
    summary['started_not_completed_rate'] = summary['started_not_completed_orders'] / float(graph_size)
    summary['served_plus_rejected_orders'] = summary['completed_orders'] + summary['rejected_orders']
    summary['served_plus_rejected_rate'] = summary['served_plus_rejected_orders'] / float(graph_size)
    return summary


def _postprocess_debug(debug_summary, graph_size):
    processed = dict(debug_summary)
    denom = max(float(2 * graph_size), 1.0)
    for key in MASK_SHARE_KEYS:
        processed[f'{key}_pct_of_service_slots'] = 100.0 * float(debug_summary.get(key, 0.0)) / denom
    return processed


def _aggregate_records(records):
    if not records:
        return {}
    keys = sorted({key for record in records for key in record})
    result = {}
    for key in keys:
        values = [float(record[key]) for record in records if key in record]
        if not values:
            continue
        result[key] = {
            'mean': float(statistics.fmean(values)),
            'std': float(statistics.pstdev(values)) if len(values) > 1 else 0.0,
        }
    return result


def _run_heuristic_episode(sample, policy, state_kwargs, deadlock_limit, trace_first_failure, rng):
    batch = {k: v.unsqueeze(0) if torch.is_tensor(v) else v for k, v in sample.items()}
    state = StateMCVRPPDTW.initialize(
        batch,
        deadlock_limit=deadlock_limit,
        max_concurrent_open_orders=state_kwargs['max_concurrent_open_orders'],
        enable_delivery_viability=state_kwargs['enable_delivery_viability'],
        enable_viability_fallback=state_kwargs['enable_viability_fallback'],
    )
    actions = []
    trace = []
    debug_totals = {'diag_steps': 0.0}
    no_move_count = 0.0
    fallback_count = 0.0

    for step in range(max(16, state.n_orders * 8)):
        mask, debug = state.get_mask(return_debug=True)
        pickups, deliveries = _service_candidates(mask, state)
        step_debug = _base_step_debug(mask, state, debug)
        open_count = int(debug['diag_open_started_count'][0].item())
        fallback_count += float(debug['diag_delivery_viability_fallback'][0].item())

        if trace_first_failure:
            trace.append(_collect_trace_snapshot(state, mask, debug, pickups, deliveries, step, note='before_action'))

        if not pickups and not deliveries:
            if open_count > 0:
                no_move_count += 1.0
                for key, value in step_debug.items():
                    debug_totals[key] = debug_totals.get(key, 0.0) + float(value)
                if trace_first_failure:
                    trace.append(_collect_trace_snapshot(state, mask, debug, pickups, deliveries, step, note='no_move_break'))
                break
            if state.get_finished().all():
                for key, value in step_debug.items():
                    debug_totals[key] = debug_totals.get(key, 0.0) + float(value)
                if trace_first_failure:
                    trace.append(_collect_trace_snapshot(state, mask, debug, pickups, deliveries, step, note='finished_break'))
                break
            if not bool(mask[0, 0, 0].item()):
                selected = 0
                actions.append(selected)
                _update_selected_debug(step_debug, state, selected)
                for key, value in step_debug.items():
                    debug_totals[key] = debug_totals.get(key, 0.0) + float(value)
                if trace_first_failure:
                    trace.append(_collect_trace_snapshot(state, mask, debug, pickups, deliveries, step, selected_action=('DEPOT', -1), note='depot_return'))
                state = state.update(torch.tensor([0], dtype=torch.long, device=state.coords.device))
                continue
            for key, value in step_debug.items():
                debug_totals[key] = debug_totals.get(key, 0.0) + float(value)
            if trace_first_failure:
                trace.append(_collect_trace_snapshot(state, mask, debug, pickups, deliveries, step, note='blocked_break'))
            break

        chosen = choose_heuristic_action(state, policy, rng, batch, pickups, deliveries)
        if chosen is None:
            if open_count > 0:
                no_move_count += 1.0
            for key, value in step_debug.items():
                debug_totals[key] = debug_totals.get(key, 0.0) + float(value)
            if trace_first_failure:
                trace.append(_collect_trace_snapshot(state, mask, debug, pickups, deliveries, step, note='selected_none_break'))
            break

        kind, order_idx = chosen
        selected = order_idx + 1 if kind == 'P' else state.n_orders + order_idx + 1
        actions.append(selected)
        _update_selected_debug(step_debug, state, selected)
        for key, value in step_debug.items():
            debug_totals[key] = debug_totals.get(key, 0.0) + float(value)
        if trace_first_failure:
            trace.append(_collect_trace_snapshot(state, mask, debug, pickups, deliveries, step, selected_action=chosen, note='apply_action'))
        state = state.update(torch.tensor([selected], dtype=torch.long, device=state.coords.device))

    pi = torch.tensor([actions if actions else [0]], dtype=torch.long)
    _, details = MCVRPPDTW.get_costs(batch, pi, return_details=True)
    detail_record = {key: float(details[key][0].item()) for key in DETAIL_KEYS}
    debug_record = _normalize_debug(debug_totals)
    debug_record['no_move_count'] = no_move_count
    debug_record['delivery_viability_fallback_count'] = fallback_count
    if no_move_count <= 0 and detail_record['started_not_completed_orders'] <= 0:
        trace = []
    return detail_record, debug_record, trace


def evaluate_heuristic_policy(policy, dataset, state_kwargs, args, seed):
    detail_records = []
    debug_records = []
    traces = []
    rng = random.Random(seed)
    for sample_idx, sample in enumerate(dataset):
        detail_record, debug_record, trace = _run_heuristic_episode(
            sample,
            policy,
            state_kwargs,
            args.deadlock_limit,
            args.trace_first_failure and not traces,
            rng,
        )
        detail_records.append(detail_record)
        debug_records.append(debug_record)
        if trace:
            traces.append({'sample_index': sample_idx, 'trace': trace})

    detail_summary = _summarize_details({key: [record[key] for record in detail_records] for key in DETAIL_KEYS}, args.graph_size)
    debug_summary = _postprocess_debug(_aggregate_mean_dicts(debug_records), args.graph_size)
    return {
        'policy': policy,
        'detail_summary': detail_summary,
        'debug_summary': debug_summary,
        'sample_count': len(detail_records),
        'trace': traces[0] if traces else None,
    }


def _aggregate_mean_dicts(records):
    if not records:
        return {}
    keys = sorted({key for record in records for key in record})
    return {
        key: float(statistics.fmean([float(record[key]) for record in records if key in record]))
        for key in keys
    }


def evaluate_rollout_policy(model, dataset, checkpoint, device, args):
    test_loader = DataLoader(dataset, batch_size=args.batch_size, collate_fn=collate_fn)
    detail_buffers = {key: [] for key in DETAIL_KEYS}
    debug_buffers = {}
    state_kwargs = _resolve_state_kwargs(args, checkpoint)
    with torch.no_grad():
        for batch in test_loader:
            batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
            cost, _, pi, debug = model(batch, return_pi=True, return_debug=True, state_kwargs=state_kwargs)
            _, details = MCVRPPDTW.get_costs(batch, pi, return_details=True)
            for key in DETAIL_KEYS:
                detail_buffers[key].extend(details[key].tolist())
            debug_buffers.setdefault('diag_steps', []).extend(debug['diag_steps'].tolist())
            for key in debug:
                debug_buffers.setdefault(key, []).extend(debug[key].tolist())
    detail_summary = _summarize_details(detail_buffers, args.graph_size)
    debug_summary = _postprocess_debug(
        {key: float(statistics.fmean(values)) for key, values in debug_buffers.items() if values},
        args.graph_size,
    )
    return {
        'policy': 'rollout',
        'detail_summary': detail_summary,
        'debug_summary': debug_summary,
        'sample_count': sum(len(v) for v in detail_buffers.values()) // max(len(DETAIL_KEYS), 1),
        'trace': None,
        'state_kwargs': state_kwargs,
    }


def _load_rollout_model(args, device):
    if 'rollout' not in args.policies:
        return None, None
    if args.checkpoint is None:
        raise ValueError('--checkpoint is required when rollout is included in --policies')
    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f'Checkpoint not found: {args.checkpoint}')
    try:
        checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(args.checkpoint, map_location=device)
    model = build_model_from_checkpoint(checkpoint, device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    set_decode_type(model, args.decode)
    return model, checkpoint


def _resolve_shared_state_kwargs(args, checkpoint):
    if checkpoint is not None:
        return _resolve_state_kwargs(args, checkpoint)
    max_open = args.max_concurrent_open_orders
    if max_open < 1:
        max_open = 1
    enable_delivery_viability = bool(args.enable_delivery_viability) if args.enable_delivery_viability is not None else True
    enable_viability_fallback = bool(args.enable_viability_fallback) if args.enable_viability_fallback is not None else False
    return {
        'max_concurrent_open_orders': max_open,
        'enable_delivery_viability': enable_delivery_viability,
        'enable_viability_fallback': enable_viability_fallback,
    }


def _compact_policy_result(result):
    payload = {
        'policy': result['policy'],
        'sample_count': result['sample_count'],
        'detail_summary': result['detail_summary'],
        'debug_summary': result['debug_summary'],
    }
    if result.get('trace'):
        payload['trace'] = result['trace']
    if result.get('state_kwargs'):
        payload['state_kwargs'] = result['state_kwargs']
    return payload


def _write_json(path, payload):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def _write_summary_markdown(path, config_payload, per_seed_payload, overall_summary):
    lines = []
    lines.append('# Root-cause oracle-vs-rollout audit')
    lines.append('')
    lines.append('This report compares the trained rollout policy against heuristic baselines on identical instances.')
    lines.append('')
    lines.append('**Important:** `oracle_style` is a heuristic comparator, not a proof of global optimality.')
    lines.append('')
    lines.append('## Config')
    lines.append('')
    lines.append(f"- graph_size: {config_payload['graph_size']}")
    lines.append(f"- num_samples_per_seed: {config_payload['num_samples']}")
    lines.append(f"- seeds: {config_payload['seeds']}")
    lines.append(f"- policies: {config_payload['policies']}")
    lines.append(f"- checkpoint: {config_payload.get('checkpoint')}")
    lines.append('')
    lines.append('## Overall summary (mean ± std across seeds)')
    lines.append('')
    for policy, metrics in overall_summary.items():
        lines.append(f'### {policy}')
        lines.append('')
        for key in [
            'completed_orders',
            'service_rate',
            'served_plus_rejected_orders',
            'served_plus_rejected_rate',
            'rejected_orders',
            'rejected_rate',
            'unfulfilled_orders',
            'unfulfilled_rate',
            'pickup_only_orders',
            'started_not_completed_orders',
            'passenger_pickup_hard_violations',
            'passenger_total_ride_time_violations',
            'passenger_excess_ride_time_violations',
            'trip_overtime_penalty',
            'diag_mask_precedence_pct_of_service_slots',
            'diag_mask_vehicle_limit_pct_of_service_slots',
            'diag_mask_pickup_tw_pct_of_service_slots',
            'diag_mask_trip_time_pct_of_service_slots',
        ]:
            if key in metrics:
                mean = metrics[key]['mean']
                std = metrics[key]['std']
                lines.append(f"- {key}: {mean:.4f} ± {std:.4f}")
        lines.append('')
    lines.append('## Per-seed artifact files')
    lines.append('')
    for seed_payload in per_seed_payload:
        lines.append(f"- seed {seed_payload['seed']}: `seed_{seed_payload['seed']}.json`")
    lines.append('')
    with open(path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')


def main():
    args = parse_args()
    device = torch.device('cuda' if (torch.cuda.is_available() and not args.no_cuda) else 'cpu')
    os.makedirs(args.output_dir, exist_ok=True)

    model, checkpoint = _load_rollout_model(args, device)
    shared_state_kwargs = _resolve_shared_state_kwargs(args, checkpoint)
    dataset_kwargs = _build_dataset_kwargs(args)

    per_seed_payload = []
    by_policy_records = {policy: [] for policy in args.policies}

    for seed in args.seeds:
        dataset = MCVRPPDTWDataset(
            num_samples=args.num_samples,
            graph_size=args.graph_size,
            seed=seed,
            **dataset_kwargs,
        )
        seed_result = {
            'seed': seed,
            'policies': {},
        }
        for policy in args.policies:
            if policy == 'rollout':
                result = evaluate_rollout_policy(model, dataset, checkpoint, device, args)
            else:
                result = evaluate_heuristic_policy(policy, dataset, shared_state_kwargs, args, seed)
            compact = _compact_policy_result(result)
            seed_result['policies'][policy] = compact
            merged = dict(compact['detail_summary'])
            merged.update(compact['debug_summary'])
            by_policy_records[policy].append(merged)
        per_seed_payload.append(seed_result)
        _write_json(os.path.join(args.output_dir, f'seed_{seed}.json'), seed_result)

    overall_summary = {
        policy: _aggregate_records(records)
        for policy, records in by_policy_records.items()
    }

    config_payload = {
        'generated_at': datetime.utcnow().isoformat() + 'Z',
        'command': ' '.join(sys.argv),
        'graph_size': args.graph_size,
        'num_samples': args.num_samples,
        'batch_size': args.batch_size,
        'seeds': args.seeds,
        'policies': args.policies,
        'checkpoint': args.checkpoint,
        'decode': args.decode,
        'device': str(device),
        'state_kwargs': shared_state_kwargs,
        'deadlock_limit': args.deadlock_limit,
        'dataset_kwargs': dataset_kwargs,
        'note': 'oracle_style is a heuristic comparator, not a globally optimal oracle.',
    }

    _write_json(os.path.join(args.output_dir, 'oracle_vs_rollout_config.json'), config_payload)
    _write_json(os.path.join(args.output_dir, 'oracle_vs_rollout_by_seed.json'), per_seed_payload)
    _write_json(os.path.join(args.output_dir, 'oracle_vs_rollout_summary.json'), overall_summary)
    _write_summary_markdown(
        os.path.join(args.output_dir, 'oracle_vs_rollout_summary.md'),
        config_payload,
        per_seed_payload,
        overall_summary,
    )

    print('=' * 72)
    print('Root-cause oracle-vs-rollout audit complete')
    print('=' * 72)
    print(f'Output dir: {args.output_dir}')
    print(f'Policies  : {args.policies}')
    print(f'Seeds     : {args.seeds}')
    print('Important : oracle_style is heuristic, not globally optimal.')
    print('-' * 72)
    for policy, metrics in overall_summary.items():
        service = metrics.get('service_rate', {'mean': 0.0, 'std': 0.0})
        completed = metrics.get('completed_orders', {'mean': 0.0, 'std': 0.0})
        rejected = metrics.get('rejected_orders', {'mean': 0.0, 'std': 0.0})
        unfulfilled = metrics.get('unfulfilled_orders', {'mean': 0.0, 'std': 0.0})
        precedence = metrics.get('diag_mask_precedence_pct_of_service_slots', {'mean': 0.0, 'std': 0.0})
        vehicle_limit = metrics.get('diag_mask_vehicle_limit_pct_of_service_slots', {'mean': 0.0, 'std': 0.0})
        print(f'{policy:12s} service={service["mean"]:.4f}±{service["std"]:.4f} '
              f'completed={completed["mean"]:.2f}±{completed["std"]:.2f} '
              f'rejected={rejected["mean"]:.2f}±{rejected["std"]:.2f} '
              f'unfulfilled={unfulfilled["mean"]:.2f}±{unfulfilled["std"]:.2f} '
              f'precedence={precedence["mean"]:.2f}%±{precedence["std"]:.2f}% '
              f'vehicle_limit={vehicle_limit["mean"]:.2f}%±{vehicle_limit["std"]:.2f}%')
    print('=' * 72)


if __name__ == '__main__':
    main()
