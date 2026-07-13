"""
POMO Training Script for MCVRP-PDTW (Optimized Version)
Multi-Compartment VRP with Pickup-Delivery and Time Windows

支持特性：
1. 分别训练 25 / 50 / 100 订单
2. 按 graph_size 自动校准/加载归一化常数
3. 验证阶段同时输出训练目标（objective）与真实人民币成本（CNY）
"""

import argparse
import json
import math
import os
import random
import time
from contextlib import nullcontext

import numpy as np
import torch
import torch.distributed as dist
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    def tqdm(iterable, **_kwargs):
        return iterable

from nets.attention_model import AttentionModel, set_decode_type
from problem_mcvrptw_v2 import (
    Config,
    MCVRPPDTW,
    MCVRPPDTWDataset,
    calibrate_normalization,
)


PHASE_CONFIGS = {
    25: {
        'n_epochs': 50,
        'batch_size': 8,
        'pomo_size': 2,
        'n_encode_layers': 6,
        'hidden_dim': 256,
        'lr': 5e-5,
        'epoch_size': 8000,
    },
    50: {
        'n_epochs': 60,
        'batch_size': 8,
        'pomo_size': 2,
        'n_encode_layers': 6,
        'hidden_dim': 256,
        'lr': 5e-5,
        'epoch_size': 8000,
    },
    100: {
        'n_epochs': 80,
        'batch_size': 4,
        'pomo_size': 1,
        'n_encode_layers': 6,
        'hidden_dim': 256,
        'lr': 3e-5,
        'epoch_size': 10000,
    },
    200: {
        'n_epochs': 100,
        'batch_size': 2,
        'pomo_size': 1,
        'n_encode_layers': 6,
        'hidden_dim': 256,
        'lr': 2e-5,
        'epoch_size': 12000,
    },
}


def set_global_seed(seed):
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, 'cudnn'):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def collate_fn(batch):
    keys = batch[0].keys()
    return {
        key: torch.stack([sample[key] for sample in batch], dim=0)
        if torch.is_tensor(batch[0][key]) else batch[0][key]
        for key in keys
    }


class BenchmarkAccumulator:
    def __init__(self):
        self.seconds = {}
        self.calls = {}

    def add(self, name, seconds, calls=1):
        self.seconds[name] = self.seconds.get(name, 0.0) + float(seconds)
        self.calls[name] = self.calls.get(name, 0) + int(calls)

    def merge(self, other):
        if other is None:
            return
        for name, seconds in other.seconds.items():
            self.seconds[name] = self.seconds.get(name, 0.0) + float(seconds)
        for name, calls in other.calls.items():
            self.calls[name] = self.calls.get(name, 0) + int(calls)

    def total(self, name):
        return float(self.seconds.get(name, 0.0))

    def count(self, name):
        return int(self.calls.get(name, 0))

    def mean_ms(self, name):
        calls = self.count(name)
        if calls <= 0:
            return 0.0
        return self.total(name) * 1000.0 / calls

    def snapshot(self):
        return {
            'seconds': dict(self.seconds),
            'calls': dict(self.calls),
        }


def _normalize_state_dict_keys(state_dict):
    if not isinstance(state_dict, dict):
        return state_dict
    normalized = {}
    for key, value in state_dict.items():
        if isinstance(key, str) and key.startswith('module.'):
            normalized[key[len('module.'):]] = value
        else:
            normalized[key] = value
    return normalized


def _str_to_bool(value):
    if isinstance(value, bool):
        return value
    value = str(value).strip().lower()
    if value in {'1', 'true', 'yes', 'y', 'on'}:
        return True
    if value in {'0', 'false', 'no', 'n', 'off'}:
        return False
    raise ValueError(f'Cannot parse boolean value: {value}')


def _infer_distributed_from_env():
    world_size = int(os.environ.get('WORLD_SIZE', '1'))
    return world_size > 1


def _resolve_amp_dtype(name):
    amp_name = str(name).strip().lower()
    if amp_name == 'fp16':
        return torch.float16
    if amp_name == 'bf16':
        return torch.bfloat16
    raise ValueError(f'Unsupported amp dtype: {name}')


def _build_grad_scaler(enabled):
    if hasattr(torch, 'amp') and hasattr(torch.amp, 'GradScaler'):
        try:
            return torch.amp.GradScaler('cuda', enabled=enabled)
        except TypeError:
            return torch.amp.GradScaler(enabled=enabled)
    return torch.cuda.amp.GradScaler(enabled=enabled)


class POMOTrainerOptimized:
    """POMO Trainer with graph-size-aware normalization calibration."""

    def __init__(self, args):
        self.args = args
        self.distributed = bool(getattr(self.args, 'distributed', False))
        self.rank = int(getattr(self.args, 'rank', 0))
        self.world_size = int(getattr(self.args, 'world_size', 1))
        self.local_rank = int(getattr(self.args, 'local_rank', 0))
        self.is_main_process = (not self.distributed) or self.rank == 0
        set_global_seed(self.args.seed + self.rank)
        self.device = torch.device('cuda', self.local_rank) if self.args.use_cuda else torch.device('cpu')
        if self.device.type == 'cuda':
            torch.cuda.set_device(self.device)
        self.problem = MCVRPPDTW
        self.normalization_profile = self._prepare_normalization_profile()
        base_model = self._create_model().to(self.device)
        self.model = base_model
        if self.distributed:
            self.model = DistributedDataParallel(
                base_model,
                device_ids=[self.local_rank] if self.device.type == 'cuda' else None,
                output_device=self.local_rank if self.device.type == 'cuda' else None,
                find_unused_parameters=self.args.ddp_find_unused_parameters,
            )
        self.raw_model = self.model.module if isinstance(self.model, DistributedDataParallel) else self.model

        self.optimizer = optim.AdamW(
            self.raw_model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay,
        )
        self.lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=args.n_epochs,
            eta_min=args.lr * 0.01,
        )

        self.train_log = []
        self.val_log = []
        self.start_epoch = 1
        self.best_val_objective = float('inf')
        self.best_business_key = None
        self.best_business_rate = float('-inf')
        self.best_service_key = None
        self.best_service_rate = float('-inf')
        self.benchmark = BenchmarkAccumulator() if self.args.benchmark_mode else None
        self.benchmark_epoch_summaries = []
        self.amp_enabled = bool(getattr(self.args, 'amp', False)) and self.device.type == 'cuda'
        self.amp_dtype = _resolve_amp_dtype(self.args.amp_dtype) if self.amp_enabled else None
        self.grad_accum_steps = max(1, int(getattr(self.args, 'grad_accum_steps', 1)))
        self.scaler = _build_grad_scaler(enabled=self.amp_enabled and self.amp_dtype == torch.float16)

        if args.resume_path:
            self._load_checkpoint(args.resume_path)

        use_cuda = self.device.type == 'cuda'
        self.log_interval = max(1, int(self.args.log_interval))
        self.loader_kwargs = {
            'num_workers': max(0, self.args.num_workers),
            'pin_memory': use_cuda and not self.args.no_pin_memory,
        }
        if self.loader_kwargs['num_workers'] > 0:
            self.loader_kwargs['persistent_workers'] = not self.args.disable_persistent_workers
            self.loader_kwargs['prefetch_factor'] = max(1, self.args.prefetch_factor)

        if self.is_main_process:
            print(f"Device: {self.device}")
            print(f"Graph size: {self.args.graph_size} orders")
            print(f"Normalization profile: {json.dumps(self.normalization_profile, ensure_ascii=False)}")
            print(f"Shared env defaults: max_open={self.args.max_concurrent_open_orders}, "
                  f"min_orders_per_dispatch={self.args.min_orders_per_dispatch}, "
                  f"delivery_viability={self.args.enable_delivery_viability}, "
                  f"viability_fallback={self.args.enable_viability_fallback}")
            print(f"Reward profile: energy={self.args.alpha_energy}, delay={self.args.alpha_delay}, "
                  f"vehicle={self.args.alpha_vehicle}, reject={self.args.alpha_reject}, "
                  f"unfulfilled={self.args.alpha_unfulfilled}, overtime={self.args.alpha_trip_overtime}")
            print(f"Cargo pickup hard TW: {Config.HARD_CARGO_PICKUP_TIMEWINDOW}")
            print(f"Cargo delay piecewise: [0,{Config.CARGO_DELAY_TIER1_MIN:.0f}]={Config.CARGO_DELAY_COST:.3f}, "
                  f"({Config.CARGO_DELAY_TIER1_MIN:.0f},{Config.CARGO_DELAY_TIER2_MIN:.0f}]={Config.CARGO_DELAY_COST_30_60:.3f}, "
                  f">{Config.CARGO_DELAY_TIER2_MIN:.0f}={Config.CARGO_DELAY_COST_60_PLUS:.3f}")
            print(f"Passenger pickup TW width: {Config.PASSENGER_TW_WIDTH:.1f} h")
            print(f"Global seed: {self.args.seed}")
            print(f"Model params: {sum(p.numel() for p in self.raw_model.parameters()):,}")
            print(f"Distributed: {self.distributed} (rank={self.rank}, world_size={self.world_size})")
            print(f"AMP: {self.amp_enabled} ({self.args.amp_dtype if self.amp_enabled else 'fp32'})")
            print(f"Grad accumulation steps: {self.grad_accum_steps}")
            if self.args.benchmark_mode:
                print(f"Benchmark mode: warmup_epochs={self.args.benchmark_warmup_epochs}, "
                      f"skip_validation={self.args.benchmark_skip_validation}, "
                      f"disable_checkpoint={self.args.benchmark_disable_checkpoint}, "
                      f"disable_log_save={self.args.benchmark_disable_log_save}, "
                      f"batch_timing={self.args.benchmark_batch_timing}")

    def _prepare_normalization_profile(self):
        if self.args.calibrate_before_train:
            if self.is_main_process:
                print(
                    f"[Normalization] 开始校准 graph_size={self.args.graph_size}, "
                    f"samples={self.args.normalization_samples}, seed={self.args.normalization_seed}"
                )
                profile = calibrate_normalization(
                    graph_size=self.args.graph_size,
                    num_samples=self.args.normalization_samples,
                    seed=self.args.normalization_seed,
                    num_vehicles=self.args.num_vehicles,
                    output_path=Config.get_normalization_output_path(
                        self.args.graph_size,
                        base_dir=self.args.normalization_dir,
                    ),
                )
                if self.distributed:
                    dist.barrier()
                return profile
            if self.distributed:
                dist.barrier()

        profile = Config.load_normalization_profile(
            self.args.graph_size,
            base_dir=self.args.normalization_dir,
        )
        return Config.set_normalization_profile(self.args.graph_size, profile)

    def _create_model(self):
        return AttentionModel(
            embedding_dim=self.args.embedding_dim,
            hidden_dim=self.args.hidden_dim,
            problem=self.problem,
            n_encode_layers=self.args.n_encode_layers,
            tanh_clipping=self.args.tanh_clipping,
            mask_inner=True,
            mask_logits=True,
            normalization=self.args.normalization,
            n_heads=self.args.n_heads,
            checkpoint_encoder=self.args.checkpoint_encoder,
            shrink_size=self.args.shrink_size,
            max_decode_steps=self.args.max_decode_steps,
            max_consecutive_depot=self.args.max_consecutive_depot,
            reject_init_bias=self.args.reject_init_bias,
            decode_pickup_urgency_bias=self.args.decode_pickup_urgency_bias,
            decode_pickup_urgency_horizon_hours=self.args.decode_pickup_urgency_horizon_hours,
        )

    def _to_device(self, batch):
        return {
            key: value.to(self.device, non_blocking=True) if torch.is_tensor(value) else value
            for key, value in batch.items()
        }

    def _build_state_kwargs(self, allow_reject, benchmark_timing=False):
        return {
            'allow_reject': allow_reject,
            'deadlock_limit': self.args.deadlock_limit,
            'max_concurrent_open_orders': self.args.max_concurrent_open_orders,
            'min_orders_per_dispatch': self.args.min_orders_per_dispatch,
            'enable_delivery_viability': self.args.enable_delivery_viability,
            'enable_viability_fallback': self.args.enable_viability_fallback,
            'relax_pickup_commitment_trip_time': self.args.relax_pickup_commitment_trip_time,
            'benchmark_timing': bool(benchmark_timing),
        }

    def _build_default_dataset_kwargs(self):
        kwargs = {}
        if self.args.passenger_tw_period_weights_override is not None:
            kwargs['passenger_tw_period_weights_override'] = self.args.passenger_tw_period_weights_override
        if self.args.cargo_tw_period_weights_override is not None:
            kwargs['cargo_tw_period_weights_override'] = self.args.cargo_tw_period_weights_override
        if self.args.passenger_tw_period_bounds_override is not None:
            kwargs['passenger_tw_period_bounds_override'] = self.args.passenger_tw_period_bounds_override
        if self.args.cargo_tw_period_bounds_override is not None:
            kwargs['cargo_tw_period_bounds_override'] = self.args.cargo_tw_period_bounds_override
        return kwargs

    def _validation_dataset_seed(self):
        return int(self.args.seed)

    def _training_dataset_seed(self, epoch):
        return int(self.args.seed + epoch * 1000)

    def _sync_if_needed(self):
        if self.args.benchmark_mode and self.device.type == 'cuda':
            torch.cuda.synchronize(self.device)

    def _amp_context(self):
        if not self.amp_enabled:
            return nullcontext()
        return torch.autocast(device_type='cuda', dtype=self.amp_dtype)

    def _distributed_average(self, value):
        if not self.distributed:
            return value
        if isinstance(value, torch.Tensor):
            tensor = value.detach().clone().to(self.device)
        else:
            tensor = torch.tensor(float(value), device=self.device)
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        tensor /= float(self.world_size)
        if isinstance(value, torch.Tensor):
            return tensor
        return float(tensor.item())

    def _should_run_validation(self):
        if self.args.benchmark_skip_validation:
            return False
        if not self.distributed:
            return True
        return bool(self.args.dist_eval) or self.is_main_process

    def _emit_benchmark_summary(self, payload):
        if not self.args.benchmark_mode:
            return
        print("  [Benchmark]")
        print(f"    epoch_total_s           : {payload['epoch_total_s']:.3f}")
        print(f"    train_s                : {payload['train_s']:.3f}")
        print(f"    validate_s             : {payload['validate_s']:.3f}")
        print(f"    checkpoint_s           : {payload['checkpoint_s']:.3f}")
        print(f"    dataset_build_s        : {payload['dataset_build_s']:.3f}")
        print(f"    dataloader_build_s     : {payload['dataloader_build_s']:.3f}")
        print(f"    mean_batch_total_ms    : {payload['mean_batch_total_ms']:.2f}")
        print(f"    mean_batch_forward_ms  : {payload['mean_batch_forward_ms']:.2f}")
        print(f"    mean_batch_backward_ms : {payload['mean_batch_backward_ms']:.2f}")
        print(f"    mean_batch_optim_ms    : {payload['mean_batch_optimizer_ms']:.2f}")
        print(f"    mean_model_init_ms     : {payload['mean_model_init_embed_ms']:.2f}")
        print(f"    mean_model_encoder_ms  : {payload['mean_model_encoder_ms']:.2f}")
        print(f"    mean_decode_inner_ms   : {payload['mean_model_decode_inner_ms']:.2f}")
        print(f"    mean_decode_state_ms   : {payload['mean_decode_make_state_ms']:.2f}")
        print(f"    mean_decode_fixed_ms   : {payload['mean_decode_precompute_fixed_ms']:.2f}")
        print(f"    mean_decode_logp_ms    : {payload['mean_decode_get_log_p_total_ms']:.2f}")
        print(f"    mean_decode_mask_ms    : {payload['mean_decode_get_mask_ms']:.2f}")
        print(f"    mean_mask_pickup_tw_ms : {payload['mean_mask_pickup_tw_ms']:.2f}")
        print(f"    mean_mask_ride_ms      : {payload['mean_mask_ride_time_ms']:.2f}")
        print(f"    mean_mask_trip_ms      : {payload['mean_mask_trip_time_ms']:.2f}")
        print(f"    mean_mask_ops_end_ms   : {payload['mean_mask_ops_end_ms']:.2f}")
        print(f"    mean_mask_commit_ms    : {payload['mean_mask_pickup_commitment_ms']:.2f}")
        print(f"    mean_pc_complete_ms    : {payload['mean_pc_has_feasible_open_completion_ms']:.2f}")
        print(f"    mean_pc_step_ms        : {payload['mean_pc_delivery_step_feasible_ms']:.2f}")
        print(f"    mean_pc_eval_open_ms   : {payload['mean_pc_evaluate_post_pickup_open_delivery_ms']:.2f}")
        print(f"    mean_pc_legal_ms       : {payload['mean_pc_get_legal_delivery_orders_ms']:.2f}")
        print(f"    mean_pc_path_ms        : {payload['mean_pc_has_legal_delivery_path_ms']:.2f}")
        print(f"    mean_pc_phys_ms        : {payload['mean_pc_has_any_physical_delivery_step_ms']:.2f}")
        print(f"    mean_pc_comp_calls     : {payload['mean_pc_has_feasible_open_completion_calls']:.2f}")
        print(f"    mean_pc_comp_hit_rate  : {payload['mean_pc_completion_memo_hit_rate']:.3f}")
        print(f"    mean_pc_comp_branches  : {payload['mean_pc_completion_branch_attempts']:.2f}")
        print(f"    mean_pc_comp_open_bits : {payload['mean_pc_completion_open_bits']:.2f}")
        print(f"    mean_pc_comp_avg_depth : {payload['mean_pc_completion_avg_recursion_depth']:.2f}")
        print(f"    mean_pc_comp_max_depth : {payload['mean_pc_completion_max_recursion_depth']:.2f}")
        print(f"    mean_pc_step_calls     : {payload['mean_pc_delivery_step_feasible_calls']:.2f}")
        print(f"    mean_pc_step_hit_rate  : {payload['mean_pc_step_memo_hit_rate']:.3f}")
        print(f"    mean_pc_step_ok_calls  : {payload['mean_pc_step_reason_ok']:.2f}")
        print(f"    mean_pc_step_trip_calls: {payload['mean_pc_step_reason_trip_time']:.2f}")
        print(f"    mean_pc_step_ride_calls: {payload['mean_pc_step_reason_ride_time']:.2f}")
        print(f"    mean_pc_step_ops_calls : {payload['mean_pc_step_reason_ops_end']:.2f}")
        print(f"    mean_mask_viability_ms : {payload['mean_mask_delivery_viability_ms']:.2f}")
        print(f"    mean_mask_final_ms     : {payload['mean_mask_finalize_ms']:.2f}")
        print(f"    mean_decode_update_ms  : {payload['mean_decode_state_update_ms']:.2f}")
        print(f"    mean_decode_logits_ms  : {payload['mean_decode_logits_ms']:.2f}")
        print(f"    mean_decode_softmax_ms : {payload['mean_decode_log_softmax_ms']:.2f}")
        print(f"    mean_decode_select_ms  : {payload['mean_decode_select_node_ms']:.2f}")
        print(f"    mean_get_costs_ms      : {payload['mean_model_get_costs_ms']:.2f}")
        print(f"    mean_val_forward_ms    : {payload['mean_validate_model_forward_ms']:.2f}")
        print(f"    mean_val_costs_ms      : {payload['mean_validate_get_costs_ms']:.2f}")
        print(f"    batches_per_s          : {payload['batches_per_s']:.3f}")
        print(f"    samples_per_s          : {payload['samples_per_s']:.3f}")
        if self.args.benchmark_json:
            print(json.dumps({'type': 'benchmark_epoch', **payload}, ensure_ascii=False))

    def _emit_benchmark_final_summary(self):
        if not self.args.benchmark_mode:
            return
        warmup_epochs = max(0, int(self.args.benchmark_warmup_epochs))
        measured = [item for item in self.benchmark_epoch_summaries if item['epoch'] > warmup_epochs]
        if not measured:
            measured = list(self.benchmark_epoch_summaries)
        if not measured:
            return

        def _avg(key):
            return sum(item[key] for item in measured) / len(measured)

        summary = {
            'measured_epochs': len(measured),
            'warmup_epochs': warmup_epochs,
            'avg_epoch_total_s': _avg('epoch_total_s'),
            'avg_train_s': _avg('train_s'),
            'avg_validate_s': _avg('validate_s'),
            'avg_checkpoint_s': _avg('checkpoint_s'),
            'avg_dataset_build_s': _avg('dataset_build_s'),
            'avg_dataloader_build_s': _avg('dataloader_build_s'),
            'avg_mean_batch_total_ms': _avg('mean_batch_total_ms'),
            'avg_mean_batch_forward_ms': _avg('mean_batch_forward_ms'),
            'avg_mean_batch_backward_ms': _avg('mean_batch_backward_ms'),
            'avg_mean_batch_optimizer_ms': _avg('mean_batch_optimizer_ms'),
            'avg_mean_model_init_embed_ms': _avg('mean_model_init_embed_ms'),
            'avg_mean_model_encoder_ms': _avg('mean_model_encoder_ms'),
            'avg_mean_model_decode_inner_ms': _avg('mean_model_decode_inner_ms'),
            'avg_mean_decode_make_state_ms': _avg('mean_decode_make_state_ms'),
            'avg_mean_decode_precompute_fixed_ms': _avg('mean_decode_precompute_fixed_ms'),
            'avg_mean_decode_get_log_p_total_ms': _avg('mean_decode_get_log_p_total_ms'),
            'avg_mean_decode_get_mask_ms': _avg('mean_decode_get_mask_ms'),
            'avg_mean_mask_pickup_tw_ms': _avg('mean_mask_pickup_tw_ms'),
            'avg_mean_mask_ride_time_ms': _avg('mean_mask_ride_time_ms'),
            'avg_mean_mask_trip_time_ms': _avg('mean_mask_trip_time_ms'),
            'avg_mean_mask_ops_end_ms': _avg('mean_mask_ops_end_ms'),
            'avg_mean_mask_pickup_commitment_ms': _avg('mean_mask_pickup_commitment_ms'),
            'avg_mean_pc_has_feasible_open_completion_ms': _avg('mean_pc_has_feasible_open_completion_ms'),
            'avg_mean_pc_delivery_step_feasible_ms': _avg('mean_pc_delivery_step_feasible_ms'),
            'avg_mean_pc_evaluate_post_pickup_open_delivery_ms': _avg('mean_pc_evaluate_post_pickup_open_delivery_ms'),
            'avg_mean_pc_get_legal_delivery_orders_ms': _avg('mean_pc_get_legal_delivery_orders_ms'),
            'avg_mean_pc_has_legal_delivery_path_ms': _avg('mean_pc_has_legal_delivery_path_ms'),
            'avg_mean_pc_has_any_physical_delivery_step_ms': _avg('mean_pc_has_any_physical_delivery_step_ms'),
            'avg_mean_pc_has_feasible_open_completion_calls': _avg('mean_pc_has_feasible_open_completion_calls'),
            'avg_mean_pc_completion_memo_hit_rate': _avg('mean_pc_completion_memo_hit_rate'),
            'avg_mean_pc_completion_branch_attempts': _avg('mean_pc_completion_branch_attempts'),
            'avg_mean_pc_completion_open_bits': _avg('mean_pc_completion_open_bits'),
            'avg_mean_pc_completion_avg_recursion_depth': _avg('mean_pc_completion_avg_recursion_depth'),
            'avg_mean_pc_completion_max_recursion_depth': _avg('mean_pc_completion_max_recursion_depth'),
            'avg_mean_pc_delivery_step_feasible_calls': _avg('mean_pc_delivery_step_feasible_calls'),
            'avg_mean_pc_step_memo_hit_rate': _avg('mean_pc_step_memo_hit_rate'),
            'avg_mean_pc_step_reason_ok': _avg('mean_pc_step_reason_ok'),
            'avg_mean_pc_step_reason_trip_time': _avg('mean_pc_step_reason_trip_time'),
            'avg_mean_pc_step_reason_ride_time': _avg('mean_pc_step_reason_ride_time'),
            'avg_mean_pc_step_reason_ops_end': _avg('mean_pc_step_reason_ops_end'),
            'avg_mean_mask_delivery_viability_ms': _avg('mean_mask_delivery_viability_ms'),
            'avg_mean_mask_finalize_ms': _avg('mean_mask_finalize_ms'),
            'avg_mean_decode_state_update_ms': _avg('mean_decode_state_update_ms'),
            'avg_mean_decode_logits_ms': _avg('mean_decode_logits_ms'),
            'avg_mean_model_get_costs_ms': _avg('mean_model_get_costs_ms'),
            'avg_mean_validate_model_forward_ms': _avg('mean_validate_model_forward_ms'),
            'avg_mean_validate_get_costs_ms': _avg('mean_validate_get_costs_ms'),
            'avg_batches_per_s': _avg('batches_per_s'),
            'avg_samples_per_s': _avg('samples_per_s'),
        }
        print('\n' + '=' * 70)
        print('Benchmark Summary')
        print('=' * 70)
        for key, value in summary.items():
            if isinstance(value, float):
                print(f"{key}: {value:.3f}")
            else:
                print(f"{key}: {value}")
        if self.args.benchmark_json:
            print(json.dumps({'type': 'benchmark_final', **summary}, ensure_ascii=False))

    def _build_curriculum_dataset_kwargs(self, epoch):
        kwargs = self._build_default_dataset_kwargs()
        if not self.args.enable_rideshare_curriculum:
            return kwargs

        phase1_epochs = max(1, self.args.curriculum_warmup_epochs)
        phase2_epochs = max(1, self.args.curriculum_mix_epochs)
        if epoch <= phase1_epochs:
            kwargs.update({
                'passenger_ratio_override': self.args.curriculum_phase1_passenger_ratio,
                'passenger_distance_mix_override': (
                    self.args.curriculum_phase1_short_ratio,
                    self.args.curriculum_phase1_mid_ratio,
                    self.args.curriculum_phase1_long_ratio,
                ),
                'passenger_tw_period_weights_override': (
                    self.args.curriculum_phase1_passenger_tw_morning,
                    self.args.curriculum_phase1_passenger_tw_midday,
                    self.args.curriculum_phase1_passenger_tw_evening,
                ),
                'cargo_tw_period_weights_override': (
                    self.args.curriculum_phase1_cargo_tw_morning,
                    self.args.curriculum_phase1_cargo_tw_midday,
                    self.args.curriculum_phase1_cargo_tw_evening,
                ),
            })
            return kwargs

        if epoch <= phase1_epochs + phase2_epochs:
            kwargs.update({
                'passenger_ratio_override': self.args.curriculum_phase2_passenger_ratio,
                'passenger_distance_mix_override': (
                    self.args.curriculum_phase2_short_ratio,
                    self.args.curriculum_phase2_mid_ratio,
                    self.args.curriculum_phase2_long_ratio,
                ),
                'passenger_tw_period_weights_override': (
                    self.args.curriculum_phase2_passenger_tw_morning,
                    self.args.curriculum_phase2_passenger_tw_midday,
                    self.args.curriculum_phase2_passenger_tw_evening,
                ),
                'cargo_tw_period_weights_override': (
                    self.args.curriculum_phase2_cargo_tw_morning,
                    self.args.curriculum_phase2_cargo_tw_midday,
                    self.args.curriculum_phase2_cargo_tw_evening,
                ),
            })
            return kwargs

        return kwargs

    def _describe_curriculum(self, epoch):
        kwargs = self._build_curriculum_dataset_kwargs(epoch)
        if not kwargs:
            return 'default-distribution'

        if not self.args.enable_rideshare_curriculum:
            parts = []
            tw_mix = kwargs.get('passenger_tw_period_weights_override')
            cargo_tw_mix = kwargs.get('cargo_tw_period_weights_override')
            if tw_mix is not None:
                parts.append(f"passenger_tw={tuple(round(v, 2) for v in tw_mix)}")
            if cargo_tw_mix is not None:
                parts.append(f"cargo_tw={tuple(round(v, 2) for v in cargo_tw_mix)}")
            passenger_tw_bounds = kwargs.get('passenger_tw_period_bounds_override')
            cargo_tw_bounds = kwargs.get('cargo_tw_period_bounds_override')
            if passenger_tw_bounds is not None:
                parts.append(f"passenger_bounds={tuple((round(s, 2), round(e, 2)) for s, e in passenger_tw_bounds)}")
            if cargo_tw_bounds is not None:
                parts.append(f"cargo_bounds={tuple((round(s, 2), round(e, 2)) for s, e in cargo_tw_bounds)}")
            return 'dataset-override(' + ', '.join(parts) + ')'

        passenger_ratio = kwargs.get('passenger_ratio_override')
        distance_mix = kwargs.get('passenger_distance_mix_override')
        tw_mix = kwargs.get('passenger_tw_period_weights_override')
        return (
            f"curriculum(passenger_ratio={passenger_ratio:.2f}, "
            f"distance_mix={tuple(round(v, 2) for v in distance_mix)}, "
            f"passenger_tw={tuple(round(v, 2) for v in tw_mix)})"
        )

    def _repeat_for_pomo(self, batch, pomo_size):
        return {
            key: value.repeat_interleave(pomo_size, dim=0) if torch.is_tensor(value) else value
            for key, value in batch.items()
        }

    def _add_benchmark_payload(self, timing, payload):
        if timing is None or payload is None:
            return
        seconds_map = payload.get('seconds', {})
        calls_map = payload.get('calls', {})
        for name, seconds in seconds_map.items():
            calls = calls_map.get(name, 1)
            timing.add(name, seconds, calls=calls)
        for name, calls in calls_map.items():
            if name not in seconds_map:
                timing.add(name, 0.0, calls=calls)

    @staticmethod
    def _per_batch_ms(timing, name, batch_count):
        if timing is None or batch_count <= 0:
            return 0.0
        return timing.total(name) * 1000.0 / batch_count

    @staticmethod
    def _per_batch_count(timing, name, batch_count):
        if timing is None or batch_count <= 0:
            return 0.0
        return timing.count(name) / batch_count

    @staticmethod
    def _safe_ratio(numerator, denominator):
        if denominator <= 0:
            return 0.0
        return float(numerator) / float(denominator)

    def _pomo_forward(self, batch, state_kwargs=None, timing=None):
        return_benchmark = timing is not None
        if self.args.pomo_size <= 1:
            if return_benchmark:
                cost, log_likelihood, benchmark_payload = self.model(
                    batch,
                    state_kwargs=state_kwargs,
                    return_benchmark=True,
                )
                self._add_benchmark_payload(timing, benchmark_payload)
            else:
                cost, log_likelihood = self.model(batch, state_kwargs=state_kwargs)
            return cost.unsqueeze(1), log_likelihood.unsqueeze(1)

        batch_size = batch['loc'].size(0)
        if return_benchmark:
            cost, log_likelihood, benchmark_payload = self.model(
                batch,
                state_kwargs=state_kwargs,
                return_benchmark=True,
                logical_pomo_size=self.args.pomo_size,
            )
            self._add_benchmark_payload(timing, benchmark_payload)
        else:
            cost, log_likelihood = self.model(
                batch,
                state_kwargs=state_kwargs,
                logical_pomo_size=self.args.pomo_size,
            )
        return cost.reshape(batch_size, self.args.pomo_size), log_likelihood.reshape(batch_size, self.args.pomo_size)

    @staticmethod
    def _pomo_loss(costs, log_probs, baseline_mode='auto'):
        if baseline_mode == 'auto':
            if costs.size(1) <= 1:
                baseline = costs.mean().detach()
            else:
                baseline = costs.mean(dim=1, keepdim=True)
        elif baseline_mode == 'batch_mean':
            baseline = costs.mean().detach()
        elif baseline_mode == 'instance_mean':
            if costs.size(1) <= 1:
                raise ValueError('baseline_mode=instance_mean requires pomo_size > 1')
            baseline = costs.mean(dim=1, keepdim=True).detach()
        else:
            raise ValueError(f'Unsupported baseline_mode: {baseline_mode}')
        advantage = costs - baseline
        loss = (advantage * log_probs).mean()
        min_cost = costs.min(dim=1)[0].mean()
        return loss, min_cost

    @staticmethod
    def _service_priority_key(results):
        return (
            float(results.get('service_rate', 0.0)),
            -float(results.get('unfulfilled_rate', 1.0)),
            -float(results.get('rejected_rate', 1.0)),
            -float(results.get('avg_objective', float('inf'))),
        )

    @staticmethod
    def _is_zero_metric(value, tol=1e-9):
        return abs(float(value)) <= tol

    @classmethod
    def _is_business_clean(cls, results):
        return (
            cls._is_zero_metric(results.get('avg_unfulfilled_orders', 0.0))
            and cls._is_zero_metric(results.get('avg_pickup_only_orders', 0.0))
            and cls._is_zero_metric(results.get('avg_started_not_completed_orders', 0.0))
            and cls._is_zero_metric(results.get('avg_untouched_unrejected_orders', 0.0))
        )

    @classmethod
    def _business_priority_key(cls, results):
        return (
            1.0 if cls._is_business_clean(results) else 0.0,
            float(results.get('service_rate', 0.0)),
            -float(results.get('avg_objective', float('inf'))),
        )

    def _augment_service_metrics(self, results):
        graph_size = max(int(self.args.graph_size), 1)
        service_rate = float(results.get('avg_completed_orders', 0.0)) / graph_size
        rejected_rate = float(results.get('avg_rejected_orders', 0.0)) / graph_size
        unfulfilled_rate = float(results.get('avg_unfulfilled_orders', 0.0)) / graph_size
        untouched_unrejected_rate = float(results.get('avg_untouched_unrejected_orders', 0.0)) / graph_size
        pickup_only_rate = float(results.get('avg_pickup_only_orders', 0.0)) / graph_size
        started_not_completed_rate = float(results.get('avg_started_not_completed_orders', 0.0)) / graph_size
        served_plus_rejected_rate = min(1.0, service_rate + rejected_rate)
        results['service_rate'] = service_rate
        results['completed_rate'] = service_rate
        results['rejected_rate'] = rejected_rate
        results['unfulfilled_rate'] = unfulfilled_rate
        results['untouched_unrejected_rate'] = untouched_unrejected_rate
        results['pickup_only_rate'] = pickup_only_rate
        results['started_not_completed_rate'] = started_not_completed_rate
        results['served_plus_rejected_rate'] = served_plus_rejected_rate
        results['non_service_rate'] = min(1.0, rejected_rate + unfulfilled_rate)
        results['business_clean'] = self._is_business_clean(results)
        return results

    def train_epoch(self, epoch, train_loader):
        self.model.train()
        set_decode_type(self.model, 'sampling')
        allow_reject = epoch > self.args.reject_warmup_epochs
        state_kwargs = self._build_state_kwargs(
            allow_reject=allow_reject,
            benchmark_timing=self.args.benchmark_batch_timing,
        )

        epoch_loss = 0.0
        epoch_objective = 0.0
        n_batches = 0
        timing = BenchmarkAccumulator() if self.args.benchmark_mode else None

        current_lr = self.optimizer.param_groups[0]['lr']
        progress = tqdm(train_loader, desc=f"Epoch {epoch} (lr={current_lr:.2e})", disable=not self.is_main_process)
        epoch_train_start = time.perf_counter() if self.args.benchmark_mode else None
        self.optimizer.zero_grad(set_to_none=True)
        for batch_idx, batch in enumerate(progress, start=1):
            batch_start = time.perf_counter() if self.args.benchmark_mode else None

            to_device_start = time.perf_counter() if self.args.benchmark_batch_timing else None
            batch = self._to_device(batch)
            if self.args.benchmark_batch_timing:
                timing.add('batch_to_device', time.perf_counter() - to_device_start)

            forward_start = time.perf_counter() if self.args.benchmark_batch_timing else None
            with self._amp_context():
                costs, log_probs = self._pomo_forward(
                    batch,
                    state_kwargs=state_kwargs,
                    timing=timing if self.args.benchmark_batch_timing else None,
                )
                loss, mean_objective = self._pomo_loss(
                    costs,
                    log_probs,
                    baseline_mode=self.args.baseline_mode,
                )
                scaled_loss = loss / self.grad_accum_steps
            if self.args.benchmark_batch_timing:
                self._sync_if_needed()
                timing.add('batch_forward', time.perf_counter() - forward_start)

            loss_start = time.perf_counter() if self.args.benchmark_batch_timing else None
            if self.args.benchmark_batch_timing:
                timing.add('batch_loss', time.perf_counter() - loss_start)

            backward_start = time.perf_counter() if self.args.benchmark_batch_timing else None
            if self.scaler.is_enabled():
                self.scaler.scale(scaled_loss).backward()
            else:
                scaled_loss.backward()
            if self.args.benchmark_batch_timing:
                self._sync_if_needed()
                timing.add('batch_backward', time.perf_counter() - backward_start)

            should_step = (batch_idx % self.grad_accum_steps == 0) or (batch_idx == len(train_loader))
            if should_step:
                clip_start = time.perf_counter() if self.args.benchmark_batch_timing else None
                if self.scaler.is_enabled():
                    self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.raw_model.parameters(), self.args.max_grad_norm)
                if self.args.benchmark_batch_timing:
                    timing.add('batch_grad_clip', time.perf_counter() - clip_start)

                step_start = time.perf_counter() if self.args.benchmark_batch_timing else None
                if self.scaler.is_enabled():
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()
                self.optimizer.zero_grad(set_to_none=True)
                if self.args.benchmark_batch_timing:
                    self._sync_if_needed()
                    timing.add('batch_optimizer_step', time.perf_counter() - step_start)
            if self.args.benchmark_mode:
                timing.add('batch_total', time.perf_counter() - batch_start)

            epoch_loss += loss.item()
            epoch_objective += mean_objective.item()
            n_batches += 1

            if self.is_main_process and ((batch_idx % self.log_interval == 0) or (batch_idx == len(train_loader))):
                progress.set_postfix({'loss': f'{loss.item():.4f}', 'objective': f'{mean_objective.item():.2f}'})

        self.lr_scheduler.step()
        avg_loss = epoch_loss / max(n_batches, 1)
        avg_objective = epoch_objective / max(n_batches, 1)
        avg_loss = self._distributed_average(avg_loss)
        avg_objective = self._distributed_average(avg_objective)
        if self.args.benchmark_mode:
            timing.add('epoch_train', time.perf_counter() - epoch_train_start)
        return avg_loss, avg_objective, timing

    def validate(self, val_loader):
        self.model.eval()
        set_decode_type(self.model, 'greedy')

        timing = BenchmarkAccumulator() if self.args.benchmark_mode else None
        validate_start = time.perf_counter() if self.args.benchmark_mode else None
        all_objectives = []
        raw_total_costs = []
        detail_buffers = {
            'energy_cost_raw': [],
            'passenger_delivery_delay_cost_raw': [],
            'cargo_delay_cost_raw': [],
            'trip_overtime_penalty': [],
            'reject_penalty': [],
            'unfulfilled_penalty': [],
            'rejected_orders': [],
            'unfulfilled_orders': [],
            'completed_orders': [],
            'untouched_orders': [],
            'untouched_unrejected_orders': [],
            'pickup_only_orders': [],
            'started_not_completed_orders': [],
            'vehicle_cost_raw': [],
            'total_distance': [],
            'passenger_pickup_hard_violations': [],
            'passenger_total_ride_time_violations': [],
            'passenger_excess_ride_time_violations': [],
        }
        debug_buffers = {} if self.args.collect_mask_diagnostics else None

        val_state_kwargs = self._build_state_kwargs(
            allow_reject=True,
            benchmark_timing=self.args.benchmark_batch_timing,
        )
        with torch.no_grad():
            for batch in tqdm(val_loader, desc='Validating', disable=not self.is_main_process):
                batch = self._to_device(batch)
                model_start = time.perf_counter() if self.args.benchmark_mode else None
                with self._amp_context():
                    if self.args.collect_mask_diagnostics:
                        objective_cost, _, pi, debug = self.model(
                            batch,
                            return_pi=True,
                            state_kwargs=val_state_kwargs,
                            return_debug=True,
                        )
                        benchmark_payload = debug.pop('benchmark_timing', None)
                        self._add_benchmark_payload(timing, benchmark_payload)
                        for key, value in debug.items():
                            debug_buffers.setdefault(key, []).append(value)
                    else:
                        if self.args.benchmark_batch_timing:
                            objective_cost, _, pi, benchmark_payload = self.model(
                                batch,
                                return_pi=True,
                                state_kwargs=val_state_kwargs,
                                return_benchmark=True,
                            )
                            self._add_benchmark_payload(timing, benchmark_payload)
                        else:
                            objective_cost, _, pi = self.model(
                                batch,
                                return_pi=True,
                                state_kwargs=val_state_kwargs
                            )
                if self.args.benchmark_mode:
                    self._sync_if_needed()
                    timing.add('validate_model_forward', time.perf_counter() - model_start)
                costs_start = time.perf_counter() if self.args.benchmark_mode else None
                _, details = self.problem.get_costs(batch, pi, return_details=True)
                if self.args.benchmark_mode:
                    self._sync_if_needed()
                    timing.add('validate_get_costs', time.perf_counter() - costs_start)

                all_objectives.append(objective_cost)
                raw_total_costs.append(details['total_cost_raw'])
                for key in detail_buffers:
                    detail_buffers[key].append(details[key])

        all_objectives = torch.cat(all_objectives, dim=0)
        raw_total_costs = torch.cat(raw_total_costs, dim=0)
        for key in detail_buffers:
            detail_buffers[key] = torch.cat(detail_buffers[key], dim=0)
        if debug_buffers is not None:
            for key in debug_buffers:
                debug_buffers[key] = torch.cat(debug_buffers[key], dim=0)

        results = {
            'avg_objective': all_objectives.mean().item(),
            'std_objective': all_objectives.std().item(),
            'avg_cost': raw_total_costs.mean().item(),
            'std_cost': raw_total_costs.std().item(),
            'min_cost': raw_total_costs.min().item(),
            'max_cost': raw_total_costs.max().item(),
            'avg_energy_cost': detail_buffers['energy_cost_raw'].mean().item(),
            'avg_passenger_delivery_delay_cost': detail_buffers['passenger_delivery_delay_cost_raw'].mean().item(),
            'avg_cargo_delay_cost': detail_buffers['cargo_delay_cost_raw'].mean().item(),
            'avg_trip_overtime_penalty': detail_buffers['trip_overtime_penalty'].mean().item(),
            'avg_reject_penalty': detail_buffers['reject_penalty'].mean().item(),
            'avg_unfulfilled_penalty': detail_buffers['unfulfilled_penalty'].mean().item(),
            'avg_rejected_orders': detail_buffers['rejected_orders'].mean().item(),
            'avg_unfulfilled_orders': detail_buffers['unfulfilled_orders'].mean().item(),
            'avg_completed_orders': detail_buffers['completed_orders'].mean().item() if 'completed_orders' in detail_buffers else 0.0,
            'avg_untouched_orders': detail_buffers['untouched_orders'].mean().item() if 'untouched_orders' in detail_buffers else 0.0,
            'avg_untouched_unrejected_orders': detail_buffers['untouched_unrejected_orders'].mean().item() if 'untouched_unrejected_orders' in detail_buffers else 0.0,
            'avg_pickup_only_orders': detail_buffers['pickup_only_orders'].mean().item() if 'pickup_only_orders' in detail_buffers else 0.0,
            'avg_started_not_completed_orders': detail_buffers['started_not_completed_orders'].mean().item() if 'started_not_completed_orders' in detail_buffers else 0.0,
            'avg_vehicle_cost': detail_buffers['vehicle_cost_raw'].mean().item(),
            'avg_distance': detail_buffers['total_distance'].mean().item(),
            'avg_passenger_pickup_hard_violations': detail_buffers['passenger_pickup_hard_violations'].mean().item(),
            'avg_passenger_total_ride_time_violations': detail_buffers['passenger_total_ride_time_violations'].mean().item(),
            'avg_passenger_excess_ride_time_violations': detail_buffers['passenger_excess_ride_time_violations'].mean().item(),
            'avg_passenger_ride_time_violations': detail_buffers['passenger_total_ride_time_violations'].mean().item(),
            'avg_passenger_delay_cost': detail_buffers['passenger_delivery_delay_cost_raw'].mean().item(),
        }
        results = self._augment_service_metrics(results)
        if debug_buffers is not None:
            for key, value in debug_buffers.items():
                results[key] = value.mean().item()
        if self.distributed and self.args.dist_eval:
            reduce_keys = [key for key, value in results.items() if isinstance(value, (int, float, bool))]
            for key in reduce_keys:
                results[key] = self._distributed_average(float(results[key]))
            results = self._augment_service_metrics(results)
        if self.args.benchmark_mode:
            timing.add('epoch_validate', time.perf_counter() - validate_start)
        return results, timing

    def train(self):
        if self.is_main_process:
            print('\n' + '=' * 70)
            print('Starting Optimized POMO Training')
            print('=' * 70)
            print(f"Graph size: {self.args.graph_size} orders ({self.args.graph_size * 2} nodes)")
            print(f"Batch size: {self.args.batch_size}")
            print(f"POMO size: {self.args.pomo_size}")
            print(f"Epochs: {self.args.n_epochs}")
            print(f"Seed: {self.args.seed}")
            print(f"Reject warmup epochs: {self.args.reject_warmup_epochs}")
            print(f"Reject init bias: {self.args.reject_init_bias}")
            print(f"Decode pickup urgency bias: {self.args.decode_pickup_urgency_bias}")
            print(f"Decode pickup urgency horizon (h): {self.args.decode_pickup_urgency_horizon_hours}")
            print(f"Baseline mode: {self.args.baseline_mode}")
            print(f"Shared env: {self._build_state_kwargs(allow_reject=True)}")
            print(f"Curriculum enabled: {self.args.enable_rideshare_curriculum}")
            if self.args.n_epochs <= self.args.reject_warmup_epochs:
                print("[Warning] n_epochs <= reject_warmup_epochs: 训练期间 reject 始终被屏蔽，")
                print("          train/val 行为可能出现偏差（短训 smoke 建议显式设置 --reject-warmup-epochs 0）。")
            print('=' * 70)

        os.makedirs(self.args.save_dir, exist_ok=True)
        val_seed = self._validation_dataset_seed()
        if self.is_main_process:
            print(f"Validation dataset seed: {val_seed}")
        val_dataset = MCVRPPDTWDataset(
            num_samples=self.args.val_size,
            graph_size=self.args.graph_size,
            seed=val_seed,
            **self._build_default_dataset_kwargs(),
        )
        val_sampler = None
        if self.distributed and self.args.dist_eval:
            val_sampler = DistributedSampler(val_dataset, num_replicas=self.world_size, rank=self.rank, shuffle=False)
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.args.batch_size,
            collate_fn=collate_fn,
            sampler=val_sampler,
            shuffle=False,
            **self.loader_kwargs,
        )

        best_val_objective = self.best_val_objective
        best_business_key = self.best_business_key
        best_business_rate = self.best_business_rate
        best_service_key = self.best_service_key
        best_service_rate = self.best_service_rate
        for epoch in range(self.start_epoch, self.args.n_epochs + 1):
            epoch_bench = BenchmarkAccumulator() if self.args.benchmark_mode else None
            epoch_total_start = time.perf_counter() if self.args.benchmark_mode else None
            curriculum_kwargs = self._build_curriculum_dataset_kwargs(epoch)
            train_seed = self._training_dataset_seed(epoch)
            if self.is_main_process:
                print(f"Epoch {epoch} training dataset seed: {train_seed}")
            dataset_build_start = time.perf_counter() if self.args.benchmark_mode else None
            train_dataset = MCVRPPDTWDataset(
                num_samples=self.args.epoch_size,
                graph_size=self.args.graph_size,
                seed=train_seed,
                **curriculum_kwargs,
            )
            if self.args.benchmark_mode:
                epoch_bench.add('epoch_dataset_build_train', time.perf_counter() - dataset_build_start)
            dataloader_build_start = time.perf_counter() if self.args.benchmark_mode else None
            train_sampler = None
            if self.distributed:
                train_sampler = DistributedSampler(train_dataset, num_replicas=self.world_size, rank=self.rank, shuffle=True)
                train_sampler.set_epoch(epoch)
            train_loader = DataLoader(
                train_dataset,
                batch_size=self.args.batch_size,
                shuffle=train_sampler is None,
                sampler=train_sampler,
                collate_fn=collate_fn,
                **self.loader_kwargs,
            )
            if self.args.benchmark_mode:
                epoch_bench.add('epoch_dataloader_build_train', time.perf_counter() - dataloader_build_start)

            train_loss, train_objective, train_timing = self.train_epoch(epoch, train_loader)
            if self.args.benchmark_mode:
                epoch_bench.merge(train_timing)
            if self.args.benchmark_skip_validation or (not self._should_run_validation()):
                val_results = {
                    'avg_objective': float('nan'),
                    'std_objective': float('nan'),
                    'avg_cost': float('nan'),
                    'std_cost': float('nan'),
                    'min_cost': float('nan'),
                    'max_cost': float('nan'),
                    'avg_energy_cost': float('nan'),
                    'avg_passenger_delivery_delay_cost': float('nan'),
                    'avg_cargo_delay_cost': float('nan'),
                    'avg_trip_overtime_penalty': float('nan'),
                    'avg_reject_penalty': float('nan'),
                    'avg_unfulfilled_penalty': float('nan'),
                    'avg_rejected_orders': float('nan'),
                    'avg_unfulfilled_orders': float('nan'),
                    'avg_completed_orders': float('nan'),
                    'avg_untouched_orders': float('nan'),
                    'avg_untouched_unrejected_orders': float('nan'),
                    'avg_pickup_only_orders': float('nan'),
                    'avg_started_not_completed_orders': float('nan'),
                    'avg_vehicle_cost': float('nan'),
                    'avg_distance': float('nan'),
                    'avg_passenger_pickup_hard_violations': float('nan'),
                    'avg_passenger_total_ride_time_violations': float('nan'),
                    'avg_passenger_excess_ride_time_violations': float('nan'),
                    'avg_passenger_ride_time_violations': float('nan'),
                    'avg_passenger_delay_cost': float('nan'),
                    'service_rate': float('nan'),
                    'rejected_rate': float('nan'),
                    'unfulfilled_rate': float('nan'),
                    'served_plus_rejected_rate': float('nan'),
                    'untouched_unrejected_rate': float('nan'),
                    'pickup_only_rate': float('nan'),
                    'started_not_completed_rate': float('nan'),
                    'non_service_rate': float('nan'),
                    'business_clean': False,
                }
                val_timing = BenchmarkAccumulator() if self.args.benchmark_mode else None
            else:
                val_results, val_timing = self.validate(val_loader)
                if self.args.benchmark_mode:
                    epoch_bench.merge(val_timing)

            if self.is_main_process:
                self.train_log.append({'epoch': epoch, 'loss': train_loss, 'objective': train_objective})
                self.val_log.append({'epoch': epoch, **val_results})

                print(f"\nEpoch {epoch}/{self.args.n_epochs}:")
                print(f"  Train Loss: {train_loss:.4f}")
                print(f"  Train Objective: {train_objective:.2f}")
                print(f"  Data Profile: {self._describe_curriculum(epoch)}")
                if self.args.benchmark_skip_validation:
                    print("  Validation: skipped (benchmark mode)")
                else:
                    print(f"  Val Objective: {val_results['avg_objective']:.2f} +/- {val_results['std_objective']:.2f}")
                    print(f"  Val Cost (CNY): {val_results['avg_cost']:.2f} +/- {val_results['std_cost']:.2f}")
                    print(f"  Energy: {val_results['avg_energy_cost']:.2f} RMB")
                    print(f"  Passenger Delivery Delay: {val_results['avg_passenger_delivery_delay_cost']:.2f} RMB")
                    print(f"  Cargo Delay: {val_results['avg_cargo_delay_cost']:.2f} RMB")
                    print(f"  Passenger Pickup Hard Violations: {val_results['avg_passenger_pickup_hard_violations']:.2f}")
                    print(f"  Passenger Total Ride-Time Violations: {val_results['avg_passenger_total_ride_time_violations']:.2f}")
                    print(f"  Passenger Excess Ride-Time Violations: {val_results['avg_passenger_excess_ride_time_violations']:.2f}")
                    print(f"  Trip Overtime: {val_results['avg_trip_overtime_penalty']:.2f} RMB")
                    print(f"  Reject Penalty: {val_results['avg_reject_penalty']:.2f} RMB")
                    print(f"  Unfulfilled Penalty: {val_results['avg_unfulfilled_penalty']:.2f} RMB")
                    print(f"  Avg Completed Orders: {val_results['avg_completed_orders']:.2f}")
                    print(f"  Avg Rejected Orders: {val_results['avg_rejected_orders']:.2f}")
                    print(f"  Avg Unfulfilled Orders: {val_results['avg_unfulfilled_orders']:.2f}")
                    print(f"  Avg Untouched Orders: {val_results['avg_untouched_orders']:.2f}")
                    print(f"  Avg Untouched-Unrejected Orders: {val_results['avg_untouched_unrejected_orders']:.2f}")
                    print(f"  Service Rate: {val_results['service_rate']:.3f}")
                    print(f"  Rejected Rate: {val_results['rejected_rate']:.3f}")
                    print(f"  Unfulfilled Rate: {val_results['unfulfilled_rate']:.3f}")
                    print(f"  Served+Rejected Rate: {val_results['served_plus_rejected_rate']:.3f}")
                    print(f"  Untouched-Unrejected Rate: {val_results['untouched_unrejected_rate']:.3f}")
                    print(f"  Avg Pickup-only Orders: {val_results['avg_pickup_only_orders']:.2f}")
                    print(f"  Avg Started-not-completed Orders: {val_results['avg_started_not_completed_orders']:.2f}")
                    print(f"  Vehicle Cost: {val_results['avg_vehicle_cost']:.2f} RMB")
                    print(f"  Distance: {val_results['avg_distance']:.2f} km")
                    if self.args.collect_mask_diagnostics:
                        print("  [Mask Diagnostics]")
                        print(f"    Feasible pickups/step     : {val_results.get('diag_feasible_pickups', 0.0):.2f}")
                        print(f"    Feasible deliveries/step  : {val_results.get('diag_feasible_deliveries', 0.0):.2f}")
                        print(f"    Any service feasible rate : {val_results.get('diag_any_service_feasible', 0.0):.2f}")
                        print(f"    Depot-only rate           : {val_results.get('diag_depot_only', 0.0):.2f}")
                        print(f"    Reject available rate     : {val_results.get('diag_reject_available_rate', 0.0):.2f}")
                        print(f"    Feasible->depot rate      : {val_results.get('diag_service_feasible_but_selected_depot', 0.0):.2f}")
                        print(f"    Feasible->reject rate     : {val_results.get('diag_service_feasible_but_selected_reject', 0.0):.2f}")
                        print(f"    Mask by pickup TW         : {val_results.get('diag_mask_pickup_tw', 0.0):.2f}")
                        print(f"    Mask by ride time         : {val_results.get('diag_mask_ride_time', 0.0):.2f}")
                        print(f"    Mask by trip time         : {val_results.get('diag_mask_trip_time', 0.0):.2f}")
                        print(f"    Mask by ops end           : {val_results.get('diag_mask_ops_end', 0.0):.2f}")
                        print(f"    Mask by pickup commitment : {val_results.get('diag_mask_pickup_commitment', 0.0):.2f}")
                        print(f"    Open started count        : {val_results.get('diag_open_started_count', 0.0):.2f}")
                        print(f"    Second pickup feasible    : {val_results.get('diag_second_pickup_feasible', 0.0):.2f}")
                        print(f"    Delivery viability masked : {val_results.get('diag_delivery_viability_masked', 0.0):.2f}")
                        print(f"    Delivery fallback rate    : {val_results.get('diag_delivery_viability_fallback', 0.0):.2f}")
                        print(f"    Mask by vehicle limit     : {val_results.get('diag_mask_vehicle_limit', 0.0):.2f}")
                        print(f"    Reject predeparture rate  : {val_results.get('diag_reject_predeparture_available', 0.0):.2f}")
                        print(f"    Reject in-route rate      : {val_results.get('diag_reject_inroute_available', 0.0):.2f}")

            if self.is_main_process and (not self.args.benchmark_skip_validation):
                business_key = self._business_priority_key(val_results)
                if best_business_key is None or business_key > best_business_key:
                    best_business_key = business_key
                    best_business_rate = float(val_results['service_rate'])
                    self.best_business_key = best_business_key
                    self.best_business_rate = best_business_rate
                    if not self.args.benchmark_disable_checkpoint:
                        checkpoint_start = time.perf_counter() if self.args.benchmark_mode else None
                        self._save_model(epoch, val_results, 'best', selection_rule='business_first')
                        if self.args.benchmark_mode:
                            epoch_bench.add('epoch_checkpoint', time.perf_counter() - checkpoint_start)
                        print('  [Saved best model by business priority]')

                service_key = self._service_priority_key(val_results)
                if best_service_key is None or service_key > best_service_key:
                    best_service_key = service_key
                    best_service_rate = float(val_results['service_rate'])
                    self.best_service_key = best_service_key
                    self.best_service_rate = best_service_rate
                    if not self.args.benchmark_disable_checkpoint:
                        checkpoint_start = time.perf_counter() if self.args.benchmark_mode else None
                        self._save_model(epoch, val_results, 'best_service', selection_rule='service_priority')
                        if self.args.benchmark_mode:
                            epoch_bench.add('epoch_checkpoint', time.perf_counter() - checkpoint_start)
                        print('  [Saved best model by service priority]')

                if val_results['avg_objective'] < best_val_objective:
                    best_val_objective = val_results['avg_objective']
                    self.best_val_objective = best_val_objective
                    if not self.args.benchmark_disable_checkpoint:
                        checkpoint_start = time.perf_counter() if self.args.benchmark_mode else None
                        self._save_model(epoch, val_results, 'best_objective', selection_rule='objective_min')
                        if self.args.benchmark_mode:
                            epoch_bench.add('epoch_checkpoint', time.perf_counter() - checkpoint_start)
                        print('  [Saved best model by objective]')

                if self.args.save_interval > 0 and epoch % self.args.save_interval == 0 and epoch != self.args.n_epochs and not self.args.benchmark_disable_checkpoint:
                    checkpoint_start = time.perf_counter() if self.args.benchmark_mode else None
                    self._save_model(epoch, val_results, f'epoch_{epoch}', selection_rule='periodic')
                    if self.args.benchmark_mode:
                        epoch_bench.add('epoch_checkpoint', time.perf_counter() - checkpoint_start)

            if self.args.benchmark_mode:
                epoch_bench.add('epoch_total', time.perf_counter() - epoch_total_start)
                batch_count = max(train_timing.count('batch_total'), 1)
                samples_seen = batch_count * self.args.batch_size * self.world_size
                epoch_payload = {
                    'epoch': epoch,
                    'epoch_total_s': epoch_bench.total('epoch_total'),
                    'train_s': epoch_bench.total('epoch_train'),
                    'validate_s': epoch_bench.total('epoch_validate'),
                    'checkpoint_s': epoch_bench.total('epoch_checkpoint'),
                    'dataset_build_s': epoch_bench.total('epoch_dataset_build_train'),
                    'dataloader_build_s': epoch_bench.total('epoch_dataloader_build_train'),
                    'mean_batch_total_ms': train_timing.mean_ms('batch_total'),
                    'mean_batch_forward_ms': train_timing.mean_ms('batch_forward'),
                    'mean_batch_backward_ms': train_timing.mean_ms('batch_backward'),
                    'mean_batch_optimizer_ms': train_timing.mean_ms('batch_optimizer_step'),
                    'mean_model_init_embed_ms': self._per_batch_ms(train_timing, 'model_init_embed', batch_count),
                    'mean_model_encoder_ms': self._per_batch_ms(train_timing, 'model_encoder', batch_count),
                    'mean_model_decode_inner_ms': self._per_batch_ms(train_timing, 'model_decode_inner', batch_count),
                    'mean_decode_make_state_ms': self._per_batch_ms(train_timing, 'decode_make_state', batch_count),
                    'mean_decode_precompute_fixed_ms': self._per_batch_ms(train_timing, 'decode_precompute_fixed', batch_count),
                    'mean_decode_get_log_p_total_ms': self._per_batch_ms(train_timing, 'decode_get_log_p_total', batch_count),
                    'mean_decode_step_context_ms': self._per_batch_ms(train_timing, 'decode_step_context', batch_count),
                    'mean_decode_attention_node_data_ms': self._per_batch_ms(train_timing, 'decode_attention_node_data', batch_count),
                    'mean_decode_get_mask_ms': self._per_batch_ms(train_timing, 'decode_get_mask', batch_count),
                    'mean_mask_active_views_ms': self._per_batch_ms(train_timing, 'mask_active_views', batch_count),
                    'mean_mask_init_visited_ms': self._per_batch_ms(train_timing, 'mask_init_visited', batch_count),
                    'mean_mask_precedence_ms': self._per_batch_ms(train_timing, 'mask_precedence', batch_count),
                    'mean_mask_capacity_ms': self._per_batch_ms(train_timing, 'mask_capacity', batch_count),
                    'mean_mask_pickup_tw_ms': self._per_batch_ms(train_timing, 'mask_pickup_tw', batch_count),
                    'mean_mask_open_started_ms': self._per_batch_ms(train_timing, 'mask_open_started', batch_count),
                    'mean_mask_ride_time_ms': self._per_batch_ms(train_timing, 'mask_ride_time', batch_count),
                    'mean_mask_trip_time_ms': self._per_batch_ms(train_timing, 'mask_trip_time', batch_count),
                    'mean_mask_ops_end_ms': self._per_batch_ms(train_timing, 'mask_ops_end', batch_count),
                    'mean_mask_pickup_commitment_ms': self._per_batch_ms(train_timing, 'mask_pickup_commitment', batch_count),
                    'mean_pc_has_feasible_open_completion_ms': self._per_batch_ms(train_timing, 'pc_has_feasible_open_completion', batch_count),
                    'mean_pc_delivery_step_feasible_ms': self._per_batch_ms(train_timing, 'pc_delivery_step_feasible', batch_count),
                    'mean_pc_evaluate_post_pickup_open_delivery_ms': self._per_batch_ms(train_timing, 'pc_evaluate_post_pickup_open_delivery', batch_count),
                    'mean_pc_get_legal_delivery_orders_ms': self._per_batch_ms(train_timing, 'pc_get_legal_delivery_orders', batch_count),
                    'mean_pc_has_legal_delivery_path_ms': self._per_batch_ms(train_timing, 'pc_has_legal_delivery_path', batch_count),
                    'mean_pc_has_any_physical_delivery_step_ms': self._per_batch_ms(train_timing, 'pc_has_any_physical_delivery_step', batch_count),
                    'mean_pc_has_feasible_open_completion_calls': self._per_batch_count(train_timing, 'pc_has_feasible_open_completion_calls', batch_count),
                    'mean_pc_completion_memo_lookups': self._per_batch_count(train_timing, 'pc_completion_memo_lookups', batch_count),
                    'mean_pc_completion_memo_hits': self._per_batch_count(train_timing, 'pc_completion_memo_hits', batch_count),
                    'mean_pc_completion_memo_misses': self._per_batch_count(train_timing, 'pc_completion_memo_misses', batch_count),
                    'mean_pc_completion_memo_stores': self._per_batch_count(train_timing, 'pc_completion_memo_stores', batch_count),
                    'mean_pc_completion_memo_hit_rate': self._safe_ratio(
                        train_timing.count('pc_completion_memo_hits'),
                        train_timing.count('pc_completion_memo_lookups'),
                    ),
                    'mean_pc_completion_branch_attempts': self._per_batch_count(train_timing, 'pc_completion_branch_attempts', batch_count),
                    'mean_pc_completion_open_bits': self._safe_ratio(
                        train_timing.count('pc_completion_open_bits_sum'),
                        train_timing.count('pc_has_feasible_open_completion_calls'),
                    ),
                    'mean_pc_completion_avg_recursion_depth': self._safe_ratio(
                        train_timing.count('pc_completion_recursion_depth_sum'),
                        train_timing.count('pc_has_feasible_open_completion_calls'),
                    ),
                    'mean_pc_completion_max_recursion_depth': self._per_batch_count(train_timing, 'pc_completion_max_recursion_depth', batch_count),
                    'mean_pc_delivery_step_feasible_calls': self._per_batch_count(train_timing, 'pc_delivery_step_feasible_calls', batch_count),
                    'mean_pc_step_memo_lookups': self._per_batch_count(train_timing, 'pc_step_memo_lookups', batch_count),
                    'mean_pc_step_memo_hits': self._per_batch_count(train_timing, 'pc_step_memo_hits', batch_count),
                    'mean_pc_step_memo_misses': self._per_batch_count(train_timing, 'pc_step_memo_misses', batch_count),
                    'mean_pc_step_memo_stores': self._per_batch_count(train_timing, 'pc_step_memo_stores', batch_count),
                    'mean_pc_step_memo_hit_rate': self._safe_ratio(
                        train_timing.count('pc_step_memo_hits'),
                        train_timing.count('pc_step_memo_lookups'),
                    ),
                    'mean_pc_step_reason_ok': self._per_batch_count(train_timing, 'pc_step_reason_ok', batch_count),
                    'mean_pc_step_reason_trip_time': self._per_batch_count(train_timing, 'pc_step_reason_trip_time', batch_count),
                    'mean_pc_step_reason_ride_time': self._per_batch_count(train_timing, 'pc_step_reason_ride_time', batch_count),
                    'mean_pc_step_reason_ops_end': self._per_batch_count(train_timing, 'pc_step_reason_ops_end', batch_count),
                    'mean_pc_step_reason_missing_pickup_time': self._per_batch_count(train_timing, 'pc_step_reason_missing_pickup_time', batch_count),
                    'mean_mask_delivery_viability_ms': self._per_batch_ms(train_timing, 'mask_delivery_viability', batch_count),
                    'mean_mask_finalize_ms': self._per_batch_ms(train_timing, 'mask_finalize', batch_count),
                    'mean_decode_logits_ms': self._per_batch_ms(train_timing, 'decode_logits', batch_count),
                    'mean_decode_log_softmax_ms': self._per_batch_ms(train_timing, 'decode_log_softmax', batch_count),
                    'mean_decode_select_node_ms': self._per_batch_ms(train_timing, 'decode_select_node', batch_count),
                    'mean_decode_state_update_ms': self._per_batch_ms(train_timing, 'decode_state_update', batch_count),
                    'mean_model_get_costs_ms': self._per_batch_ms(train_timing, 'model_get_costs', batch_count),
                    'mean_model_log_likelihood_ms': self._per_batch_ms(train_timing, 'model_log_likelihood', batch_count),
                    'mean_validate_model_forward_ms': val_timing.mean_ms('validate_model_forward') if val_timing is not None else 0.0,
                    'mean_validate_get_costs_ms': val_timing.mean_ms('validate_get_costs') if val_timing is not None else 0.0,
                    'batches_per_s': batch_count / max(epoch_bench.total('epoch_train'), 1e-9),
                    'samples_per_s': samples_seen / max(epoch_bench.total('epoch_train'), 1e-9),
                }
                if self.is_main_process:
                    self.benchmark_epoch_summaries.append(epoch_payload)
                    self.benchmark.merge(epoch_bench)
                    self._emit_benchmark_summary(epoch_payload)

        if self.is_main_process and not self.args.benchmark_disable_checkpoint:
            final_checkpoint_start = time.perf_counter() if self.args.benchmark_mode else None
            self._save_model(self.args.n_epochs, val_results, 'final', selection_rule='final_epoch')
            if self.args.benchmark_mode:
                self.benchmark.add('final_checkpoint', time.perf_counter() - final_checkpoint_start)
        if self.is_main_process and not self.args.benchmark_disable_log_save:
            final_log_start = time.perf_counter() if self.args.benchmark_mode else None
            self._save_logs()
            if self.args.benchmark_mode:
                self.benchmark.add('final_log_save', time.perf_counter() - final_log_start)

        if self.is_main_process:
            self._emit_benchmark_final_summary()

        self.best_business_key = best_business_key
        self.best_business_rate = best_business_rate
        self.best_service_key = best_service_key
        self.best_service_rate = best_service_rate
        self.best_val_objective = best_val_objective

        if self.is_main_process:
            print('\n' + '=' * 70)
            print('Training Complete!')
            if self.args.benchmark_skip_validation:
                print('Best validation business-clean service rate: skipped')
                print('Best validation service-priority rate: skipped')
                print('Best validation objective: skipped')
            else:
                print(f"Best validation business-clean service rate: {best_business_rate:.3f}")
                print(f"Best validation service-priority rate: {best_service_rate:.3f}")
                print(f"Best validation objective: {best_val_objective:.2f}")
            print('=' * 70)
        return {
            'best_business_rate': best_business_rate,
            'best_service_rate': best_service_rate,
            'best_objective': best_val_objective,
        }

    def _save_model(self, epoch, results, name, selection_rule=None):
        path = os.path.join(self.args.save_dir, f'model_{name}.pt')
        checkpoint_args = dict(vars(self.args))
        checkpoint_args.pop('dist_backend', None)
        checkpoint_args.pop('local_rank', None)
        checkpoint_args.pop('rank', None)
        checkpoint_args.pop('world_size', None)
        checkpoint_args.pop('distributed', None)
        checkpoint_args['amp'] = bool(self.amp_enabled)
        checkpoint_args['amp_dtype'] = self.args.amp_dtype
        checkpoint_args['grad_accum_steps'] = self.grad_accum_steps
        checkpoint_args['dist_eval'] = bool(self.args.dist_eval)
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.raw_model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'results': results,
            'args': checkpoint_args,
            'normalization_profile': self.normalization_profile,
            'checkpoint_role': name,
            'selection_rule': selection_rule,
            'business_clean': bool(results.get('business_clean', False)),
            'business_priority_key': list(self._business_priority_key(results)),
            'service_priority_key': list(self._service_priority_key(results)),
            'training_architecture_version': 2,
            'distributed_training': bool(self.distributed),
            'amp_enabled': bool(self.amp_enabled),
            'amp_dtype': self.args.amp_dtype if self.amp_enabled else 'fp32',
        }, path)

    def _save_logs(self):
        log_path = os.path.join(self.args.save_dir, 'training_log.json')
        with open(log_path, 'w', encoding='utf-8') as log_file:
            json.dump({
                'train': self.train_log,
                'val': self.val_log,
                'training_architecture': {
                    'distributed': bool(self.distributed),
                    'world_size': self.world_size,
                    'amp_enabled': bool(self.amp_enabled),
                    'amp_dtype': self.args.amp_dtype if self.amp_enabled else 'fp32',
                    'grad_accum_steps': self.grad_accum_steps,
                },
                'config': {
                    'AREA_SIZE': Config.AREA_SIZE,
                    'PASSENGER_CAPACITY': Config.PASSENGER_CAPACITY,
                    'CARGO_CAPACITY': Config.CARGO_CAPACITY,
                    'PASSENGER_TW_WIDTH': Config.PASSENGER_TW_WIDTH,
                    'MAX_TRIP_TIME': Config.MAX_TRIP_TIME,
                    'ELECTRICITY_PRICE': Config.ELECTRICITY_PRICE,
                    'PASSENGER_DELAY_COST': Config.PASSENGER_DELAY_COST,
                    'CARGO_DELAY_COST': Config.CARGO_DELAY_COST,
                    'CARGO_DELAY_TIER1_MIN': Config.CARGO_DELAY_TIER1_MIN,
                    'CARGO_DELAY_TIER2_MIN': Config.CARGO_DELAY_TIER2_MIN,
                    'CARGO_DELAY_COST_30_60': Config.CARGO_DELAY_COST_30_60,
                    'CARGO_DELAY_COST_60_PLUS': Config.CARGO_DELAY_COST_60_PLUS,
                    'HARD_CARGO_PICKUP_TIMEWINDOW': Config.HARD_CARGO_PICKUP_TIMEWINDOW,
                    'ALPHA_ENERGY': self.args.alpha_energy,
                    'ALPHA_DELAY': self.args.alpha_delay,
                    'ALPHA_VEHICLE': self.args.alpha_vehicle,
                    'ALPHA_REJECT': self.args.alpha_reject,
                    'ALPHA_UNFULFILLED': self.args.alpha_unfulfilled,
                    'ALPHA_TRIP_OVERTIME': self.args.alpha_trip_overtime,
                    'VEHICLE_SPEED': Config.VEHICLE_SPEED,
                    'seed': self.args.seed,
                    'validation_dataset_seed': self._validation_dataset_seed(),
                    'training_dataset_seed_formula': 'seed + epoch * 1000',
                    'OPERATION_START': Config.OPERATION_START,
                    'reject_warmup_epochs': self.args.reject_warmup_epochs,
                    'reject_init_bias': self.args.reject_init_bias,
                    'decode_pickup_urgency_bias': self.args.decode_pickup_urgency_bias,
                    'decode_pickup_urgency_horizon_hours': self.args.decode_pickup_urgency_horizon_hours,
                    'baseline_mode': self.args.baseline_mode,
                    'best_checkpoint_metric': 'business_first(clean -> service_rate -> avg_objective)',
                    'service_checkpoint_metric': 'service_priority(service_rate -> unfulfilled_rate -> rejected_rate -> avg_objective)',
                    'OPERATION_END': Config.OPERATION_END,
                    'max_concurrent_open_orders': self.args.max_concurrent_open_orders,
                    'min_orders_per_dispatch': self.args.min_orders_per_dispatch,
                    'enable_delivery_viability': self.args.enable_delivery_viability,
                    'enable_viability_fallback': self.args.enable_viability_fallback,
                    'enable_rideshare_curriculum': self.args.enable_rideshare_curriculum,
                },
                'normalization_profile': self.normalization_profile,
            }, log_file, indent=2, ensure_ascii=False)

    def _load_checkpoint(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        model_state = _normalize_state_dict_keys(checkpoint['model_state_dict'])
        self.raw_model.load_state_dict(model_state)

        if self.args.resume_weights_only:
            if self.is_main_process:
                print(f"[Resume] Loaded weights only: {checkpoint_path}")
                print(f"[Resume] Start epoch: {self.start_epoch}")
            return

        if 'optimizer_state_dict' in checkpoint:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        saved_profile = checkpoint.get('normalization_profile')
        if saved_profile:
            self.normalization_profile = Config.set_normalization_profile(self.args.graph_size, saved_profile)
        self.start_epoch = int(checkpoint.get('epoch', 0)) + 1
        saved_results = checkpoint.get('results', {}) or {}
        self.best_val_objective = float(saved_results.get('avg_objective', float('inf')))
        if saved_results:
            saved_results = self._augment_service_metrics(dict(saved_results))
            self.best_business_key = tuple(checkpoint.get('business_priority_key', self._business_priority_key(saved_results)))
            self.best_business_rate = float(saved_results.get('service_rate', float('-inf')))
            self.best_service_key = tuple(checkpoint.get('service_priority_key', self._service_priority_key(saved_results)))
            self.best_service_rate = float(saved_results.get('service_rate', float('-inf')))
        if self.is_main_process:
            print(f"[Resume] Loaded checkpoint: {checkpoint_path}")
            print(f"[Resume] Start epoch: {self.start_epoch}")
            print(f"[Resume] Best objective so far: {self.best_val_objective:.2f}")
            if self.best_business_key is not None:
                print(f"[Resume] Best business-clean service rate so far: {self.best_business_rate:.3f}")
            if self.best_service_key is not None:
                print(f"[Resume] Best service-priority rate so far: {self.best_service_rate:.3f}")


def parse_args():
    parser = argparse.ArgumentParser(description='Train POMO models for MCVRP-PDTW')
    parser.add_argument('--graph_sizes', nargs='+', type=int, default=[25, 50, 100], help='待执行的订单规模')
    parser.add_argument('--calibrate', action='store_true', help='仅执行归一化校准，不启动训练')
    parser.add_argument('--calibrate-before-train', action='store_true', help='训练前按 graph_size 自动校准归一化')
    parser.add_argument('--normalization-samples', type=int, default=256, help='校准样本数')
    parser.add_argument('--normalization-seed', type=int, default=1234, help='校准随机种子')
    parser.add_argument('--normalization-dir', default=Config.NORMALIZATION_OUTPUT_DIR, help='归一化配置输出目录')
    parser.add_argument('--num-vehicles', type=int, default=None, help='可选：覆盖默认车辆数 K')
    parser.add_argument('--epoch-size', type=int, default=None, help='覆盖每个 epoch 的训练样本数')
    parser.add_argument('--val-size', type=int, default=500, help='验证样本数')
    parser.add_argument('--batch-size', type=int, default=None, help='覆盖默认 batch size')
    parser.add_argument('--pomo-size', type=int, default=None, help='覆盖默认 POMO size')
    parser.add_argument('--epochs', type=int, default=None, help='覆盖默认训练轮数')
    parser.add_argument('--output-root', default='outputs', help='输出目录根路径')
    parser.add_argument('--embedding-dim', type=int, default=256)
    parser.add_argument('--hidden-dim', type=int, default=None)
    parser.add_argument('--n-encode-layers', type=int, default=None)
    parser.add_argument('--n-heads', type=int, default=8)
    parser.add_argument('--tanh-clipping', type=float, default=10.0)
    parser.add_argument('--normalization', default='batch')
    parser.add_argument('--weight-decay', type=float, default=1e-5)
    parser.add_argument('--max-grad-norm', type=float, default=1.0)
    parser.add_argument('--save-interval', type=int, default=20)
    parser.add_argument('--log-interval', type=int, default=20, help='tqdm 刷新间隔（按 batch）')
    parser.add_argument('--seed', type=int, default=1234, help='全局随机种子（模型初始化、DataLoader shuffle、采样 rollout）')
    parser.add_argument('--num-workers', type=int, default=4, help='DataLoader worker 数')
    parser.add_argument('--prefetch-factor', type=int, default=2, help='DataLoader prefetch 因子 (num_workers>0 时生效)')
    parser.add_argument('--disable-persistent-workers', action='store_true', help='禁用 DataLoader persistent_workers')
    parser.add_argument('--no-pin-memory', action='store_true', help='禁用 DataLoader pin_memory')
    parser.add_argument('--checkpoint-encoder', action='store_true', help='启用 encoder gradient checkpoint 以降低显存')
    parser.add_argument('--shrink-size', type=int, default=16, help='decoder shrink_size，0 表示禁用')
    parser.add_argument('--max-decode-steps', type=int, default=None, help='decoder 最大步数上限，默认按节点数自动推断')
    parser.add_argument('--max-consecutive-depot', type=int, default=8, help='连续 depot 选择上限，超过后强制终止当前 rollout')
    parser.add_argument('--reject-warmup-epochs', type=int, default=3, help='训练前若干 epoch 屏蔽 reject 动作，先学习服务')
    parser.add_argument('--reject-init-bias', type=float, default=-2.5, help='reject head 的初始 bias，负值用于抑制早期 reject')
    parser.add_argument('--decode-pickup-urgency-bias', type=float, default=0.0,
                        help='解码打分：pickup 紧迫度加分系数 beta（0 表示关闭）')
    parser.add_argument('--decode-pickup-urgency-horizon-hours', type=float, default=1.0,
                        help='解码打分：pickup 紧迫度窗口 horizon（小时）')
    parser.add_argument('--baseline-mode', choices=['auto', 'batch_mean', 'instance_mean'], default='auto',
                        help='训练 advantage 的 baseline 模式；auto 保持当前默认行为，batch_mean / instance_mean 用于解耦 POMO 对比')
    parser.add_argument('--collect-mask-diagnostics', action='store_true', help='在验证/评估中收集 mask 与动作可行性诊断指标')
    parser.add_argument('--benchmark-mode', action='store_true', help='启用训练 wall-clock benchmark 分解输出')
    parser.add_argument('--benchmark-skip-validation', action='store_true', help='benchmark 模式下跳过 validation')
    parser.add_argument('--benchmark-disable-checkpoint', action='store_true', help='benchmark 模式下禁用 checkpoint 保存')
    parser.add_argument('--benchmark-disable-log-save', action='store_true', help='benchmark 模式下禁用 training_log.json 写入')
    parser.add_argument('--benchmark-warmup-epochs', type=int, default=1, help='benchmark 汇总时忽略前若干 warmup epoch')
    parser.add_argument('--benchmark-batch-timing', action='store_true', help='benchmark 模式下输出 batch 子阶段计时')
    parser.add_argument('--benchmark-json', action='store_true', help='benchmark 模式下额外输出 JSON 摘要')
    parser.add_argument('--deadlock-limit', type=int, default=2, help='连续回 depot 且无可服务节点时的终止阈值')
    parser.add_argument('--max-concurrent-open-orders', type=int, default=6, help='共享主线：允许的最大并发 open 单数量')
    parser.add_argument('--enable-delivery-viability', action='store_true', default=True, help='共享主线：启用 delivery viability')
    parser.add_argument('--disable-delivery-viability', action='store_false', dest='enable_delivery_viability', help='关闭 delivery viability（仅对照实验）')
    parser.add_argument('--enable-viability-fallback', action='store_true', default=False, help='启用 delivery viability fallback（仅对照实验）')
    parser.add_argument('--relax-pickup-commitment-trip-time', action='store_true', default=False, help='仅对 pickup_commitment 的 completion proof 放松 trip_time gate（实验开关）')
    parser.add_argument('--min-orders-per-dispatch', type=int, default=4, help='硬约束：车辆一旦发车，至少完成该数量订单后才可回 depot')
    parser.add_argument('--hard-cargo-pickup-timewindow', dest='hard_cargo_pickup_timewindow', action='store_true', default=True, help='启用 cargo pickup 硬时间窗')
    parser.add_argument('--soft-cargo-pickup-timewindow', dest='hard_cargo_pickup_timewindow', action='store_false', help='关闭 cargo pickup 硬时间窗（用于消融）')
    parser.add_argument('--cargo-delay-tier1-min', type=float, default=Config.CARGO_DELAY_TIER1_MIN, help='cargo delivery 分段迟到阈值1（分钟）')
    parser.add_argument('--cargo-delay-tier2-min', type=float, default=Config.CARGO_DELAY_TIER2_MIN, help='cargo delivery 分段迟到阈值2（分钟）')
    parser.add_argument('--cargo-delay-cost-0-30', type=float, default=Config.CARGO_DELAY_COST, help='cargo delivery 0~tier1 每分钟成本')
    parser.add_argument('--cargo-delay-cost-30-60', type=float, default=Config.CARGO_DELAY_COST_30_60, help='cargo delivery tier1~tier2 每分钟成本')
    parser.add_argument('--cargo-delay-cost-60-plus', type=float, default=Config.CARGO_DELAY_COST_60_PLUS, help='cargo delivery >tier2 每分钟成本')
    parser.add_argument('--alpha-energy', type=float, default=Config.ALPHA_ENERGY, help='训练 objective 中 energy 项的权重')
    parser.add_argument('--alpha-delay', type=float, default=Config.ALPHA_DELAY, help='训练 objective 中 delay 项的权重')
    parser.add_argument('--alpha-vehicle', type=float, default=Config.ALPHA_VEHICLE, help='训练 objective 中 vehicle 项的权重')
    parser.add_argument('--alpha-reject', type=float, default=Config.ALPHA_REJECT, help='训练 objective 中 reject 惩罚')
    parser.add_argument('--alpha-unfulfilled', type=float, default=Config.ALPHA_UNFULFILLED, help='训练 objective 中 unfulfilled 惩罚')
    parser.add_argument('--alpha-trip-overtime', type=float, default=Config.ALPHA_TRIP_OVERTIME, help='训练 objective 中 trip overtime 惩罚')
    parser.add_argument('--passenger-tw-period-weights', nargs=3, type=float, default=None, metavar=('MORNING', 'MIDDAY', 'EVENING'), help='覆盖默认 passenger 三时段 TW 权重')
    parser.add_argument('--cargo-tw-period-weights', nargs=3, type=float, default=None, metavar=('MORNING', 'MIDDAY', 'EVENING'), help='覆盖默认 cargo 三时段 TW 权重')
    parser.add_argument('--passenger-tw-period-bounds', nargs=6, type=float, default=None,
                        metavar=('MORNING_START', 'MORNING_END', 'MIDDAY_START', 'MIDDAY_END', 'EVENING_START', 'EVENING_END'),
                        help='覆盖默认 passenger 三时段 pickup TW 区间')
    parser.add_argument('--cargo-tw-period-bounds', nargs=6, type=float, default=None,
                        metavar=('MORNING_START', 'MORNING_END', 'MIDDAY_START', 'MIDDAY_END', 'EVENING_START', 'EVENING_END'),
                        help='覆盖默认 cargo 三时段 pickup TW 区间')
    parser.add_argument('--enable-rideshare-curriculum', action='store_true', help='按 epoch 使用轻量共享导向 curriculum')
    parser.add_argument('--curriculum-warmup-epochs', type=int, default=5, help='curriculum 第 1 阶段持续 epoch 数')
    parser.add_argument('--curriculum-mix-epochs', type=int, default=10, help='curriculum 第 2 阶段持续 epoch 数')
    parser.add_argument('--curriculum-phase1-passenger-ratio', type=float, default=0.8, help='curriculum phase1 passenger ratio override')
    parser.add_argument('--curriculum-phase1-short-ratio', type=float, default=0.75, help='curriculum phase1 passenger short-trip ratio')
    parser.add_argument('--curriculum-phase1-mid-ratio', type=float, default=0.2, help='curriculum phase1 passenger mid-trip ratio')
    parser.add_argument('--curriculum-phase1-long-ratio', type=float, default=0.05, help='curriculum phase1 passenger long-trip ratio')
    parser.add_argument('--curriculum-phase1-passenger-tw-morning', type=float, default=0.7, help='curriculum phase1 passenger morning TW weight')
    parser.add_argument('--curriculum-phase1-passenger-tw-midday', type=float, default=0.2, help='curriculum phase1 passenger midday TW weight')
    parser.add_argument('--curriculum-phase1-passenger-tw-evening', type=float, default=0.1, help='curriculum phase1 passenger evening TW weight')
    parser.add_argument('--curriculum-phase1-cargo-tw-morning', type=float, default=0.6, help='curriculum phase1 cargo morning TW weight')
    parser.add_argument('--curriculum-phase1-cargo-tw-midday', type=float, default=0.25, help='curriculum phase1 cargo midday TW weight')
    parser.add_argument('--curriculum-phase1-cargo-tw-evening', type=float, default=0.15, help='curriculum phase1 cargo evening TW weight')
    parser.add_argument('--curriculum-phase2-passenger-ratio', type=float, default=0.7, help='curriculum phase2 passenger ratio override')
    parser.add_argument('--curriculum-phase2-short-ratio', type=float, default=0.6, help='curriculum phase2 passenger short-trip ratio')
    parser.add_argument('--curriculum-phase2-mid-ratio', type=float, default=0.3, help='curriculum phase2 passenger mid-trip ratio')
    parser.add_argument('--curriculum-phase2-long-ratio', type=float, default=0.1, help='curriculum phase2 passenger long-trip ratio')
    parser.add_argument('--curriculum-phase2-passenger-tw-morning', type=float, default=0.58, help='curriculum phase2 passenger morning TW weight')
    parser.add_argument('--curriculum-phase2-passenger-tw-midday', type=float, default=0.27, help='curriculum phase2 passenger midday TW weight')
    parser.add_argument('--curriculum-phase2-passenger-tw-evening', type=float, default=0.15, help='curriculum phase2 passenger evening TW weight')
    parser.add_argument('--curriculum-phase2-cargo-tw-morning', type=float, default=0.55, help='curriculum phase2 cargo morning TW weight')
    parser.add_argument('--curriculum-phase2-cargo-tw-midday', type=float, default=0.28, help='curriculum phase2 cargo midday TW weight')
    parser.add_argument('--curriculum-phase2-cargo-tw-evening', type=float, default=0.17, help='curriculum phase2 cargo evening TW weight')
    parser.add_argument('--resume-path', type=str, default=None, help='从已有 checkpoint 继续训练/微调')
    parser.add_argument('--resume-weights-only', action='store_true', help='仅加载模型权重，不恢复优化器状态')
    parser.add_argument('--distributed', action='store_true', default=None, help='启用单机分布式训练骨架（建议配合 torchrun）')
    parser.add_argument('--dist-backend', type=str, default='nccl', help='distributed backend (default: nccl)')
    parser.add_argument('--local-rank', type=int, default=int(os.environ.get('LOCAL_RANK', 0)), help='当前进程的 local rank')
    parser.add_argument('--amp', action='store_true', help='启用 AMP 自动混合精度训练')
    parser.add_argument('--amp-dtype', choices=['fp16', 'bf16'], default='bf16', help='AMP 精度类型')
    parser.add_argument('--grad-accum-steps', type=int, default=1, help='梯度累积步数')
    parser.add_argument('--dist-eval', action='store_true', help='在 distributed 模式下分布式执行 validation 并做均值归并')
    parser.add_argument('--ddp-find-unused-parameters', action='store_true', help='DistributedDataParallel: find_unused_parameters=True')
    parser.add_argument('--no-cuda', action='store_true')
    return parser.parse_args()



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


def apply_runtime_training_config(args):
    Config.ALPHA_ENERGY = float(args.alpha_energy)
    Config.ALPHA_DELAY = float(args.alpha_delay)
    Config.ALPHA_VEHICLE = float(args.alpha_vehicle)
    Config.ALPHA_REJECT = float(args.alpha_reject)
    Config.ALPHA_UNFULFILLED = float(args.alpha_unfulfilled)
    Config.ALPHA_TRIP_OVERTIME = float(args.alpha_trip_overtime)

    Config.HARD_CARGO_PICKUP_TIMEWINDOW = bool(getattr(args, 'hard_cargo_pickup_timewindow', True))
    tier1 = max(float(getattr(args, 'cargo_delay_tier1_min', Config.CARGO_DELAY_TIER1_MIN)), 0.0)
    tier2 = max(float(getattr(args, 'cargo_delay_tier2_min', Config.CARGO_DELAY_TIER2_MIN)), tier1)
    Config.CARGO_DELAY_TIER1_MIN = tier1
    Config.CARGO_DELAY_TIER2_MIN = tier2
    Config.CARGO_DELAY_COST = max(float(getattr(args, 'cargo_delay_cost_0_30', Config.CARGO_DELAY_COST)), 0.0)
    Config.CARGO_DELAY_COST_30_60 = max(float(getattr(args, 'cargo_delay_cost_30_60', Config.CARGO_DELAY_COST_30_60)), 0.0)
    Config.CARGO_DELAY_COST_60_PLUS = max(float(getattr(args, 'cargo_delay_cost_60_plus', Config.CARGO_DELAY_COST_60_PLUS)), 0.0)


def build_phase_args(cli_args, graph_size):
    if graph_size not in PHASE_CONFIGS:
        raise ValueError(f'Unsupported graph size: {graph_size}')

    phase = PHASE_CONFIGS[graph_size].copy()
    distributed = _infer_distributed_from_env() if cli_args.distributed is None else bool(cli_args.distributed)
    world_size = int(os.environ.get('WORLD_SIZE', '1')) if distributed else 1
    rank = int(os.environ.get('RANK', '0')) if distributed else 0
    local_rank = int(os.environ.get('LOCAL_RANK', cli_args.local_rank)) if distributed else int(cli_args.local_rank)
    use_cuda = torch.cuda.is_available() and not cli_args.no_cuda
    phase1_distance_mix = _normalize_ratio_triplet((
        cli_args.curriculum_phase1_short_ratio,
        cli_args.curriculum_phase1_mid_ratio,
        cli_args.curriculum_phase1_long_ratio,
    ))
    default_passenger_tw = None if cli_args.passenger_tw_period_weights is None else _normalize_ratio_triplet(cli_args.passenger_tw_period_weights)
    default_cargo_tw = None if cli_args.cargo_tw_period_weights is None else _normalize_ratio_triplet(cli_args.cargo_tw_period_weights)
    default_passenger_tw_bounds = _parse_period_bounds(cli_args.passenger_tw_period_bounds)
    default_cargo_tw_bounds = _parse_period_bounds(cli_args.cargo_tw_period_bounds)
    phase2_distance_mix = _normalize_ratio_triplet((
        cli_args.curriculum_phase2_short_ratio,
        cli_args.curriculum_phase2_mid_ratio,
        cli_args.curriculum_phase2_long_ratio,
    ))
    phase1_passenger_tw = _normalize_ratio_triplet((
        cli_args.curriculum_phase1_passenger_tw_morning,
        cli_args.curriculum_phase1_passenger_tw_midday,
        cli_args.curriculum_phase1_passenger_tw_evening,
    ))
    phase1_cargo_tw = _normalize_ratio_triplet((
        cli_args.curriculum_phase1_cargo_tw_morning,
        cli_args.curriculum_phase1_cargo_tw_midday,
        cli_args.curriculum_phase1_cargo_tw_evening,
    ))
    phase2_passenger_tw = _normalize_ratio_triplet((
        cli_args.curriculum_phase2_passenger_tw_morning,
        cli_args.curriculum_phase2_passenger_tw_midday,
        cli_args.curriculum_phase2_passenger_tw_evening,
    ))
    phase2_cargo_tw = _normalize_ratio_triplet((
        cli_args.curriculum_phase2_cargo_tw_morning,
        cli_args.curriculum_phase2_cargo_tw_midday,
        cli_args.curriculum_phase2_cargo_tw_evening,
    ))
    return argparse.Namespace(
        embedding_dim=cli_args.embedding_dim,
        hidden_dim=cli_args.hidden_dim or phase['hidden_dim'],
        n_encode_layers=cli_args.n_encode_layers or phase['n_encode_layers'],
        n_heads=cli_args.n_heads,
        checkpoint_encoder=cli_args.checkpoint_encoder,
        shrink_size=None if (cli_args.shrink_size is not None and cli_args.shrink_size <= 0) else cli_args.shrink_size,
        max_decode_steps=cli_args.max_decode_steps,
        max_consecutive_depot=cli_args.max_consecutive_depot,
        reject_warmup_epochs=cli_args.reject_warmup_epochs,
        reject_init_bias=cli_args.reject_init_bias,
        decode_pickup_urgency_bias=cli_args.decode_pickup_urgency_bias,
        decode_pickup_urgency_horizon_hours=cli_args.decode_pickup_urgency_horizon_hours,
        baseline_mode=cli_args.baseline_mode,
        max_concurrent_open_orders=cli_args.max_concurrent_open_orders,
        min_orders_per_dispatch=cli_args.min_orders_per_dispatch,
        enable_delivery_viability=cli_args.enable_delivery_viability,
        enable_viability_fallback=cli_args.enable_viability_fallback,
        relax_pickup_commitment_trip_time=cli_args.relax_pickup_commitment_trip_time,
        hard_cargo_pickup_timewindow=cli_args.hard_cargo_pickup_timewindow,
        cargo_delay_tier1_min=cli_args.cargo_delay_tier1_min,
        cargo_delay_tier2_min=cli_args.cargo_delay_tier2_min,
        cargo_delay_cost_0_30=cli_args.cargo_delay_cost_0_30,
        cargo_delay_cost_30_60=cli_args.cargo_delay_cost_30_60,
        cargo_delay_cost_60_plus=cli_args.cargo_delay_cost_60_plus,
        alpha_energy=cli_args.alpha_energy,
        alpha_delay=cli_args.alpha_delay,
        alpha_vehicle=cli_args.alpha_vehicle,
        alpha_reject=cli_args.alpha_reject,
        alpha_unfulfilled=cli_args.alpha_unfulfilled,
        alpha_trip_overtime=cli_args.alpha_trip_overtime,
        cargo_delay_cost=cli_args.cargo_delay_cost_0_30,
        passenger_tw_period_weights_override=default_passenger_tw,
        cargo_tw_period_weights_override=default_cargo_tw,
        passenger_tw_period_bounds_override=default_passenger_tw_bounds,
        cargo_tw_period_bounds_override=default_cargo_tw_bounds,
        enable_rideshare_curriculum=cli_args.enable_rideshare_curriculum,
        curriculum_warmup_epochs=cli_args.curriculum_warmup_epochs,
        curriculum_mix_epochs=cli_args.curriculum_mix_epochs,
        curriculum_phase1_passenger_ratio=cli_args.curriculum_phase1_passenger_ratio,
        curriculum_phase1_short_ratio=phase1_distance_mix[0],
        curriculum_phase1_mid_ratio=phase1_distance_mix[1],
        curriculum_phase1_long_ratio=phase1_distance_mix[2],
        curriculum_phase1_passenger_tw_morning=phase1_passenger_tw[0],
        curriculum_phase1_passenger_tw_midday=phase1_passenger_tw[1],
        curriculum_phase1_passenger_tw_evening=phase1_passenger_tw[2],
        curriculum_phase1_cargo_tw_morning=phase1_cargo_tw[0],
        curriculum_phase1_cargo_tw_midday=phase1_cargo_tw[1],
        curriculum_phase1_cargo_tw_evening=phase1_cargo_tw[2],
        curriculum_phase2_passenger_ratio=cli_args.curriculum_phase2_passenger_ratio,
        curriculum_phase2_short_ratio=phase2_distance_mix[0],
        curriculum_phase2_mid_ratio=phase2_distance_mix[1],
        curriculum_phase2_long_ratio=phase2_distance_mix[2],
        curriculum_phase2_passenger_tw_morning=phase2_passenger_tw[0],
        curriculum_phase2_passenger_tw_midday=phase2_passenger_tw[1],
        curriculum_phase2_passenger_tw_evening=phase2_passenger_tw[2],
        curriculum_phase2_cargo_tw_morning=phase2_cargo_tw[0],
        curriculum_phase2_cargo_tw_midday=phase2_cargo_tw[1],
        curriculum_phase2_cargo_tw_evening=phase2_cargo_tw[2],
        tanh_clipping=cli_args.tanh_clipping,
        normalization=cli_args.normalization,
        graph_size=graph_size,
        batch_size=cli_args.batch_size or phase['batch_size'],
        n_epochs=cli_args.epochs or phase['n_epochs'],
        epoch_size=cli_args.epoch_size or phase['epoch_size'],
        val_size=cli_args.val_size,
        pomo_size=cli_args.pomo_size or phase['pomo_size'],
        lr=phase['lr'],
        weight_decay=cli_args.weight_decay,
        max_grad_norm=cli_args.max_grad_norm,
        seed=cli_args.seed,
        no_cuda=cli_args.no_cuda,
        use_cuda=use_cuda,
        distributed=distributed,
        dist_backend=cli_args.dist_backend,
        world_size=world_size,
        rank=rank,
        local_rank=local_rank,
        amp=cli_args.amp,
        amp_dtype=cli_args.amp_dtype,
        grad_accum_steps=cli_args.grad_accum_steps,
        dist_eval=cli_args.dist_eval,
        ddp_find_unused_parameters=cli_args.ddp_find_unused_parameters,
        save_dir=os.path.join(cli_args.output_root, f'pomo_n{graph_size}_optimized'),
        save_interval=cli_args.save_interval,
        log_interval=cli_args.log_interval,
        num_workers=cli_args.num_workers,
        prefetch_factor=cli_args.prefetch_factor,
        disable_persistent_workers=cli_args.disable_persistent_workers,
        no_pin_memory=cli_args.no_pin_memory,
        calibrate_before_train=cli_args.calibrate_before_train,
        normalization_samples=cli_args.normalization_samples,
        normalization_seed=cli_args.normalization_seed,
        normalization_dir=cli_args.normalization_dir,
        num_vehicles=cli_args.num_vehicles,
        collect_mask_diagnostics=cli_args.collect_mask_diagnostics,
        benchmark_mode=cli_args.benchmark_mode,
        benchmark_skip_validation=cli_args.benchmark_skip_validation,
        benchmark_disable_checkpoint=cli_args.benchmark_disable_checkpoint,
        benchmark_disable_log_save=cli_args.benchmark_disable_log_save,
        benchmark_warmup_epochs=cli_args.benchmark_warmup_epochs,
        benchmark_batch_timing=cli_args.benchmark_batch_timing,
        benchmark_json=cli_args.benchmark_json,
        deadlock_limit=cli_args.deadlock_limit,
        resume_path=cli_args.resume_path,
        resume_weights_only=cli_args.resume_weights_only,
    )



def run_calibration_only(cli_args):
    print('=' * 70)
    print('Calibration Only Mode')
    print('=' * 70)
    for graph_size in cli_args.graph_sizes:
        profile = calibrate_normalization(
            graph_size=graph_size,
            num_samples=cli_args.normalization_samples,
            seed=cli_args.normalization_seed,
            num_vehicles=cli_args.num_vehicles,
            output_path=Config.get_normalization_output_path(graph_size, base_dir=cli_args.normalization_dir),
        )
        print(f'[Calibration] N={graph_size}: {json.dumps(profile, ensure_ascii=False)}')



def main():
    cli_args = parse_args()
    if cli_args.calibrate:
        run_calibration_only(cli_args)
        return

    distributed_requested = _infer_distributed_from_env() if cli_args.distributed is None else bool(cli_args.distributed)
    rank = int(os.environ.get('RANK', '0')) if distributed_requested else 0
    world_size = int(os.environ.get('WORLD_SIZE', '1')) if distributed_requested else 1
    local_rank = int(os.environ.get('LOCAL_RANK', cli_args.local_rank)) if distributed_requested else int(cli_args.local_rank)
    if distributed_requested:
        backend = cli_args.dist_backend
        if backend == 'nccl' and (cli_args.no_cuda or not torch.cuda.is_available()):
            backend = 'gloo'
        dist.init_process_group(backend=backend, init_method='env://')
    is_main_process = (not distributed_requested) or rank == 0

    try:
        if is_main_process:
            print('=' * 70)
            print('MCVRP-PDTW with POMO Training (OPTIMIZED VERSION)')
            print('=' * 70)

        summary = {}
        for graph_size in cli_args.graph_sizes:
            phase_args = build_phase_args(cli_args, graph_size)
            apply_runtime_training_config(phase_args)
            if is_main_process:
                print(f"\n[Phase] Training on {graph_size} orders ({graph_size * 2} nodes)...")
            trainer = POMOTrainerOptimized(phase_args)
            best_metrics = trainer.train()
            summary[graph_size] = best_metrics

        if is_main_process:
            print('\n' + '=' * 70)
            print('All requested training phases complete!')
            for graph_size in cli_args.graph_sizes:
                metrics = summary[graph_size]
                if cli_args.benchmark_skip_validation:
                    print(f"N={graph_size}: benchmark-only run (validation skipped)")
                else:
                    print(f"N={graph_size}: best service rate = {metrics['best_service_rate']:.3f}, best objective = {metrics['best_objective']:.2f}")
            print('=' * 70)
    finally:
        if distributed_requested and dist.is_initialized():
            dist.destroy_process_group()


if __name__ == '__main__':
    main()
