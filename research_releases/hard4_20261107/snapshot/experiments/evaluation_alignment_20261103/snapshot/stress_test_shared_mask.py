import argparse
import random

import torch

from problem_mcvrptw_v2 import MCVRPPDTWDataset, MCVRPPDTW
from state_mcvrptw_v2 import StateMCVRPPDTW


def parse_args():
    parser = argparse.ArgumentParser(description='共享 mask 坏策略压力测试')
    parser.add_argument('--graph-size', type=int, default=25)
    parser.add_argument('--num-samples', type=int, default=24)
    parser.add_argument('--seed', type=int, default=12345)
    parser.add_argument('--policy', type=str, default='bad', choices=['bad', 'random', 'oracle'])
    parser.add_argument('--max-concurrent-open-orders', type=int, default=6)
    parser.add_argument('--min-orders-per-dispatch', type=int, default=4)
    parser.add_argument('--enable-delivery-viability', action='store_true', default=True)
    parser.add_argument('--disable-delivery-viability', action='store_false', dest='enable_delivery_viability')
    parser.add_argument('--enable-viability-fallback', action='store_true')
    parser.add_argument('--trace-first-failure', action='store_true',
                        help='打印首个 no_move / orphan 样本的逐步轨迹')
    return parser.parse_args()


def _node_coord(batch, node_idx):
    if node_idx == 0:
        return batch['depot'][0]
    return batch['loc'][0, node_idx - 1]


def _order_deadline(batch, state, order_idx):
    return float(batch['time_windows'][0, order_idx, 1].item())


def _service_coord(batch, state, action):
    kind, order_idx = action
    if kind == 'P':
        node_idx = order_idx + 1
    else:
        node_idx = state.n_orders + order_idx + 1
    return _node_coord(batch, node_idx)


def choose_action(state, policy, rng, batch, pickups, deliveries):
    service_actions = [('P', idx) for idx in pickups] + [('D', idx) for idx in deliveries]
    if not service_actions:
        return None

    cur_coord = state.cur_coord[0, 0]
    if policy == 'bad':
        pickup_actions = [action for action in service_actions if action[0] == 'P']
        delivery_actions = [action for action in service_actions if action[0] == 'D']
        candidates = pickup_actions if pickup_actions else delivery_actions
        return max(candidates, key=lambda action: torch.norm(_service_coord(batch, state, action) - cur_coord).item())

    if policy == 'oracle':
        if deliveries:
            return ('D', min(deliveries, key=lambda idx: _order_deadline(batch, state, idx)))
        return ('P', min(pickups, key=lambda idx: _order_deadline(batch, state, idx)))

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
        'diag_mask_pickup_tw': float(debug['diag_mask_pickup_tw'][0].item()),
        'diag_mask_ride_time': float(debug['diag_mask_ride_time'][0].item()),
        'diag_mask_trip_time': float(debug['diag_mask_trip_time'][0].item()),
        'diag_mask_ops_end': float(debug['diag_mask_ops_end'][0].item()),
        'diag_mask_pickup_commitment': float(debug['diag_mask_pickup_commitment'][0].item()),
        'diag_depot_fallback_used': float(debug['diag_depot_fallback_used'][0].item()),
    }
    return snapshot


def _print_trace(sample_idx, trace, result):
    print('-' * 64)
    print(f'TRACE sample={sample_idx}')
    for snapshot in trace:
        selected = snapshot['selected_action']
        if selected is None:
            selected_text = 'None'
        else:
            kind, order_idx = selected
            selected_text = f'{kind}{int(order_idx) + 1}'
        print(
            f"step={snapshot['step']:02d} note={snapshot['note']} selected={selected_text} "
            f"open={snapshot['open_orders']} pickups={snapshot['pickups']} deliveries={snapshot['deliveries']} "
            f"t={snapshot['current_time']:.2f} trip_start={snapshot['trip_start_time']:.2f} prev={snapshot['prev_node']}"
        )
        print(
            '  '
            f"diag_open={snapshot['diag_open_started_count']:.2f} second_ok={snapshot['diag_second_pickup_feasible']:.2f} "
            f"second_block={snapshot['diag_second_pickup_blocked_by_commitment']:.2f} delivery_masked={snapshot['diag_delivery_viability_masked']:.2f} "
            f"fallback={snapshot['diag_delivery_viability_fallback']:.2f} pickup_tw={snapshot['diag_mask_pickup_tw']:.2f} "
            f"ride={snapshot['diag_mask_ride_time']:.2f} trip={snapshot['diag_mask_trip_time']:.2f} "
            f"ops={snapshot['diag_mask_ops_end']:.2f} commitment={snapshot['diag_mask_pickup_commitment']:.2f} "
            f"depot_fallback={snapshot['diag_depot_fallback_used']:.2f}"
        )
    print(
        f"result completed={result['completed']:.2f} picked={result['picked']:.2f} "
        f"orphans={result['orphans']:.2f} dead_end={result['dead_end']:.2f} "
        f"fallback={result['fallback']:.2f} no_move={result['no_move']:.2f}"
    )
    print('-' * 64)


def run_episode(sample, policy, args, rng):
    batch = {k: v.unsqueeze(0) if torch.is_tensor(v) else v for k, v in sample.items()}
    state = StateMCVRPPDTW.initialize(
        batch,
        max_concurrent_open_orders=args.max_concurrent_open_orders,
        min_orders_per_dispatch=args.min_orders_per_dispatch,
        enable_delivery_viability=args.enable_delivery_viability,
        enable_viability_fallback=args.enable_viability_fallback,
    )
    actions = []
    trace = []
    max_depth = 0
    fallback_count = 0.0
    dead_end_count = 0.0
    no_move_count = 0.0

    for step in range(max(16, state.n_orders * 8)):
        mask, debug = state.get_mask(return_debug=True)
        max_depth = max(max_depth, int(debug['diag_open_started_count'][0].item()))
        pickups, deliveries = _service_candidates(mask, state)
        open_count = int(debug['diag_open_started_count'][0].item())
        fallback_flag = float(debug['diag_delivery_viability_fallback'][0].item())
        fallback_count += fallback_flag
        if args.enable_viability_fallback:
            dead_end_count += fallback_flag

        if args.trace_first_failure:
            trace.append(_collect_trace_snapshot(state, mask, debug, pickups, deliveries, step, note='before_action'))

        if not pickups and not deliveries:
            if open_count > 0:
                no_move_count += 1.0
                if args.trace_first_failure:
                    trace.append(_collect_trace_snapshot(state, mask, debug, pickups, deliveries, step, note='no_move_break'))
                break
            if state.get_finished().all():
                if args.trace_first_failure:
                    trace.append(_collect_trace_snapshot(state, mask, debug, pickups, deliveries, step, note='finished_break'))
                break
            if not bool(mask[0, 0, 0].item()):
                actions.append(0)
                if args.trace_first_failure:
                    trace.append(_collect_trace_snapshot(state, mask, debug, pickups, deliveries, step, selected_action=('DEPOT', -1), note='depot_return'))
                state = state.update(torch.tensor([0], dtype=torch.long, device=state.coords.device))
                continue
            if args.trace_first_failure:
                trace.append(_collect_trace_snapshot(state, mask, debug, pickups, deliveries, step, note='blocked_break'))
            break

        selected_action = choose_action(state, policy, rng, batch, pickups, deliveries)
        if selected_action is None:
            if open_count > 0:
                no_move_count += 1.0
            if args.trace_first_failure:
                trace.append(_collect_trace_snapshot(state, mask, debug, pickups, deliveries, step, note='selected_none_break'))
            break

        kind, order_idx = selected_action
        selected = order_idx + 1 if kind == 'P' else state.n_orders + order_idx + 1
        actions.append(selected)
        if args.trace_first_failure:
            trace.append(_collect_trace_snapshot(state, mask, debug, pickups, deliveries, step, selected_action=selected_action, note='apply_action'))
        state = state.update(torch.tensor([selected], dtype=torch.long, device=state.coords.device))

    pi = torch.tensor([actions if actions else [0]], dtype=torch.long)
    _, details = MCVRPPDTW.get_costs(batch, pi, return_details=True)
    picked = float(details['completed_orders'][0].item() + details['pickup_only_orders'][0].item())
    return {
        'completed': float(details['completed_orders'][0].item()),
        'picked': picked,
        'orphans': float(details['started_not_completed_orders'][0].item()),
        'max_depth': float(max_depth),
        'dead_end': float(dead_end_count),
        'fallback': float(fallback_count),
        'no_move': float(no_move_count),
        'trace': trace,
    }


def main():
    args = parse_args()
    dataset = MCVRPPDTWDataset(num_samples=args.num_samples, graph_size=args.graph_size, seed=args.seed)
    rng = random.Random(args.seed)
    totals = {'completed': 0.0, 'picked': 0.0, 'orphans': 0.0, 'max_depth': 0.0, 'dead_end': 0.0, 'fallback': 0.0, 'no_move': 0.0}
    traced_failure = False
    for sample_idx, sample in enumerate(dataset):
        result = run_episode(sample, args.policy, args, rng)
        for key in totals:
            totals[key] += result[key]
        if args.trace_first_failure and (not traced_failure):
            if result['no_move'] > 0 or result['orphans'] > 0:
                _print_trace(sample_idx, result['trace'], result)
                traced_failure = True
    scale = float(args.num_samples)
    print('=' * 64)
    print('共享 mask 压力测试结果')
    print('=' * 64)
    print(f'policy    : {args.policy}')
    print(f'samples   : {args.num_samples}')
    print(f'max_open  : {args.max_concurrent_open_orders}')
    print(f'min_orders_per_dispatch : {args.min_orders_per_dispatch}')
    print(f'delivery_viability : {args.enable_delivery_viability}')
    print(f'fallback  : {args.enable_viability_fallback}')
    print('-' * 64)
    print(f"completed : {totals['completed'] / scale:.2f}")
    print(f"picked    : {totals['picked'] / scale:.2f}")
    print(f"orphans   : {totals['orphans'] / scale:.2f}")
    print(f"max_depth : {totals['max_depth'] / scale:.2f}")
    print(f"dead_end  : {totals['dead_end'] / scale:.2f}")
    print(f"fallback  : {totals['fallback'] / scale:.2f}")
    print(f"no_move   : {totals['no_move'] / scale:.2f}")
    print('=' * 64)


if __name__ == '__main__':
    main()
