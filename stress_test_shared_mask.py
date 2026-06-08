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
    parser.add_argument('--policy', type=str, default='bad', choices=['bad', 'random'])
    parser.add_argument('--max-concurrent-open-orders', type=int, default=6)
    parser.add_argument('--enable-delivery-viability', action='store_true')
    parser.add_argument('--enable-viability-fallback', action='store_true')
    return parser.parse_args()


def choose_action(mask, state, policy, rng):
    feasible = (~mask[0, 0]).nonzero(as_tuple=False).squeeze(-1).tolist()
    pickups = [idx for idx in feasible if 1 <= idx <= state.n_orders]
    deliveries = [idx for idx in feasible if state.n_orders + 1 <= idx <= 2 * state.n_orders]
    if policy == 'bad':
        if pickups:
            return max(pickups)
        if deliveries:
            return max(deliveries)
        if 0 in feasible:
            return 0
        if state.reject_index in feasible:
            return state.reject_index
        return None
    if feasible:
        return rng.choice(feasible)
    return None


def run_episode(sample, policy, args, rng):
    batch = {k: v.unsqueeze(0) if torch.is_tensor(v) else v for k, v in sample.items()}
    state = StateMCVRPPDTW.initialize(
        batch,
        max_concurrent_open_orders=args.max_concurrent_open_orders,
        enable_delivery_viability=args.enable_delivery_viability,
        enable_viability_fallback=args.enable_viability_fallback,
    )
    actions = []
    max_depth = 0
    fallback_count = 0.0
    dead_end_count = 0.0
    no_move_count = 0.0
    for _ in range(max(8, state.n_orders * 6)):
        mask, debug = state.get_mask(return_debug=True)
        max_depth = max(max_depth, int(debug['diag_open_started_count'][0].item()))
        fallback_count += float(debug['diag_delivery_viability_fallback'][0].item())
        open_count = int(debug['diag_open_started_count'][0].item())
        feasible_delivery_count = int((~mask[0, 0, state.n_orders + 1:2 * state.n_orders + 1]).sum().item())
        feasible_pickup_count = int((~mask[0, 0, 1:state.n_orders + 1]).sum().item())
        if open_count > 0 and feasible_delivery_count == 0:
            dead_end_count += 1.0
        if state.get_finished().all():
            break
        selected = choose_action(mask, state, policy, rng)
        if selected is None:
            no_move_count += 1.0
            break
        actions.append(selected)
        state = state.update(torch.tensor([selected]))
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
    }


def main():
    args = parse_args()
    dataset = MCVRPPDTWDataset(num_samples=args.num_samples, graph_size=args.graph_size, seed=args.seed)
    rng = random.Random(args.seed)
    totals = {'completed': 0.0, 'picked': 0.0, 'orphans': 0.0, 'max_depth': 0.0, 'dead_end': 0.0, 'fallback': 0.0, 'no_move': 0.0}
    for sample in dataset:
        result = run_episode(sample, args.policy, args, rng)
        for key in totals:
            totals[key] += result[key]
    scale = float(args.num_samples)
    print('=' * 64)
    print('共享 mask 压力测试结果')
    print('=' * 64)
    print(f'policy    : {args.policy}')
    print(f'samples   : {args.num_samples}')
    print(f'max_open  : {args.max_concurrent_open_orders}')
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
