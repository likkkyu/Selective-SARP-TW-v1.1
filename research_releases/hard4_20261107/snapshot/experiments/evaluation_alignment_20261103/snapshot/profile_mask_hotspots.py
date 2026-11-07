import argparse
import time
from collections import defaultdict
from functools import wraps

import torch

from problem_mcvrptw_v2 import MCVRPPDTWDataset
from state_mcvrptw_v2 import StateMCVRPPDTW


class MethodProfiler:
    def __init__(self):
        self.timings = defaultdict(float)
        self.calls = defaultdict(int)
        self.get_mask_depth = 0

    def add(self, key, duration):
        self.timings[key] += duration
        self.calls[key] += 1


def parse_args():
    parser = argparse.ArgumentParser(description='Profile get_mask hotspots on sampled states')
    parser.add_argument('--graph-sizes', nargs='+', type=int, default=[25, 50, 100, 200])
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--repeats', type=int, default=5)
    parser.add_argument('--steps', type=int, default=20)
    parser.add_argument('--max-concurrent-open-orders', type=int, default=6)
    parser.add_argument('--min-orders-per-dispatch', type=int, default=4)
    parser.add_argument('--enable-delivery-viability', action='store_true', default=True)
    parser.add_argument('--disable-delivery-viability', action='store_false', dest='enable_delivery_viability')
    parser.add_argument('--enable-viability-fallback', action='store_true', default=False)
    return parser.parse_args()


def _collate_dataset(dataset, batch_size):
    sample0 = dataset[0]
    return {
        key: torch.stack([dataset[i][key] for i in range(batch_size)], dim=0)
        if torch.is_tensor(sample0[key]) else sample0[key]
        for key in sample0.keys()
    }


def _wrap_method(name, original, profiler):
    @wraps(original)
    def wrapped(self, *args, **kwargs):
        if name == 'get_mask':
            nested = profiler.get_mask_depth > 0
            skip_pickup_commitment = bool(kwargs.get('skip_pickup_commitment', False))
            if nested and skip_pickup_commitment:
                key = 'get_mask_nested_skip'
            elif nested:
                key = 'get_mask_nested_full'
            else:
                key = 'get_mask_top'
            profiler.get_mask_depth += 1
            start = time.perf_counter()
            try:
                return original(self, *args, **kwargs)
            finally:
                profiler.get_mask_depth -= 1
                profiler.add(key, time.perf_counter() - start)

        if name == 'update':
            key = 'update_nested_in_get_mask' if profiler.get_mask_depth > 0 else 'update_rollout'
        else:
            key = name

        start = time.perf_counter()
        try:
            return original(self, *args, **kwargs)
        finally:
            profiler.add(key, time.perf_counter() - start)

    return wrapped


def _profile_graph_size(args, graph_size):
    profiler = MethodProfiler()
    methods = [
        'get_mask',
        'update',
        '_has_feasible_open_completion',
        '_get_legal_delivery_orders',
        '_has_legal_delivery_path',
        '_has_any_physical_delivery_step',
        '_delivery_step_feasible',
        '_build_completion_scalar_caches',
    ]
    originals = {name: getattr(StateMCVRPPDTW, name) for name in methods}

    for name, original in originals.items():
        setattr(StateMCVRPPDTW, name, _wrap_method(name, original, profiler))

    try:
        dataset = MCVRPPDTWDataset(num_samples=args.batch_size, graph_size=graph_size, seed=args.seed)
        batch = _collate_dataset(dataset, args.batch_size)
        initial_state = StateMCVRPPDTW.initialize(
            batch,
            max_concurrent_open_orders=args.max_concurrent_open_orders,
            min_orders_per_dispatch=args.min_orders_per_dispatch,
            enable_delivery_viability=args.enable_delivery_viability,
            enable_viability_fallback=args.enable_viability_fallback,
        )

        rollout_steps = 0
        for _ in range(args.repeats):
            state = initial_state
            for _ in range(args.steps):
                if state.all_finished():
                    break
                mask = state.get_mask()
                selected = []
                for batch_idx in range(state.ids.size(0)):
                    feasible = torch.nonzero(~mask[batch_idx, 0], as_tuple=False).squeeze(-1)
                    selected.append(int(feasible[0].item()) if feasible.numel() > 0 else 0)
                state = state.update(torch.tensor(selected, dtype=torch.long), current_mask=mask)
                rollout_steps += 1
    finally:
        for name, original in originals.items():
            setattr(StateMCVRPPDTW, name, original)

    top_calls = max(1, profiler.calls.get('get_mask_top', 0))
    summary = {
        'graph_size': graph_size,
        'rollout_steps': rollout_steps,
        'get_mask_top_ms': profiler.timings.get('get_mask_top', 0.0) * 1000.0 / top_calls,
        'update_rollout_ms': profiler.timings.get('update_rollout', 0.0) * 1000.0 / max(1, profiler.calls.get('update_rollout', 0)),
    }

    ordered_keys = [
        'get_mask_top',
        'get_mask_nested_skip',
        'get_mask_nested_full',
        'update_rollout',
        'update_nested_in_get_mask',
        '_has_feasible_open_completion',
        '_get_legal_delivery_orders',
        '_has_legal_delivery_path',
        '_has_any_physical_delivery_step',
        '_delivery_step_feasible',
        '_build_completion_scalar_caches',
    ]

    print('=' * 72)
    print(f'graph_size={graph_size} steps={rollout_steps}')
    print('=' * 72)
    print(f"avg top-level get_mask : {summary['get_mask_top_ms']:.2f} ms/call")
    print(f"avg rollout update     : {summary['update_rollout_ms']:.2f} ms/call")
    print('-' * 72)
    print(f"{'metric':32s} {'calls':>10s} {'total_ms':>12s} {'per_top_ms':>12s}")
    for key in ordered_keys:
        calls = profiler.calls.get(key, 0)
        total_ms = profiler.timings.get(key, 0.0) * 1000.0
        per_top_ms = total_ms / top_calls
        print(f"{key:32s} {calls:10d} {total_ms:12.2f} {per_top_ms:12.2f}")
    print()


def main():
    args = parse_args()
    for graph_size in args.graph_sizes:
        _profile_graph_size(args, graph_size)


if __name__ == '__main__':
    main()
