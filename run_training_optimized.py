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

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

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


class POMOTrainerOptimized:
    """POMO Trainer with graph-size-aware normalization calibration."""

    def __init__(self, args):
        self.args = args
        set_global_seed(self.args.seed)
        self.device = torch.device('cuda' if torch.cuda.is_available() and not args.no_cuda else 'cpu')
        self.problem = MCVRPPDTW
        self.normalization_profile = self._prepare_normalization_profile()
        self.model = self._create_model().to(self.device)

        self.optimizer = optim.AdamW(
            self.model.parameters(),
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
        self.best_service_key = None
        self.best_service_rate = float('-inf')

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

        print(f"Device: {self.device}")
        print(f"Graph size: {self.args.graph_size} orders")
        print(f"Normalization profile: {json.dumps(self.normalization_profile, ensure_ascii=False)}")
        print(f"Shared env defaults: max_open={self.args.max_concurrent_open_orders}, "
              f"delivery_viability={self.args.enable_delivery_viability}, "
              f"viability_fallback={self.args.enable_viability_fallback}")
        print(f"Reward profile: energy={self.args.alpha_energy}, delay={self.args.alpha_delay}, "
              f"vehicle={self.args.alpha_vehicle}, reject={self.args.alpha_reject}, "
              f"unfulfilled={self.args.alpha_unfulfilled}, overtime={self.args.alpha_trip_overtime}")
        print(f"Passenger pickup TW width: {Config.PASSENGER_TW_WIDTH:.1f} h")
        print(f"Global seed: {self.args.seed}")
        print(f"Model params: {sum(p.numel() for p in self.model.parameters()):,}")

    def _prepare_normalization_profile(self):
        if self.args.calibrate_before_train:
            print(
                f"[Normalization] 开始校准 graph_size={self.args.graph_size}, "
                f"samples={self.args.normalization_samples}, seed={self.args.normalization_seed}"
            )
            return calibrate_normalization(
                graph_size=self.args.graph_size,
                num_samples=self.args.normalization_samples,
                seed=self.args.normalization_seed,
                num_vehicles=self.args.num_vehicles,
                output_path=Config.get_normalization_output_path(
                    self.args.graph_size,
                    base_dir=self.args.normalization_dir,
                ),
            )

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
        )

    def _to_device(self, batch):
        return {key: value.to(self.device) if torch.is_tensor(value) else value for key, value in batch.items()}

    def _build_state_kwargs(self, allow_reject):
        return {
            'allow_reject': allow_reject,
            'deadlock_limit': self.args.deadlock_limit,
            'max_concurrent_open_orders': self.args.max_concurrent_open_orders,
            'enable_delivery_viability': self.args.enable_delivery_viability,
            'enable_viability_fallback': self.args.enable_viability_fallback,
            'relax_pickup_commitment_trip_time': self.args.relax_pickup_commitment_trip_time,
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

    def _pomo_forward(self, batch, state_kwargs=None):
        if self.args.pomo_size <= 1:
            cost, log_likelihood = self.model(batch, state_kwargs=state_kwargs)
            return cost.unsqueeze(1), log_likelihood.unsqueeze(1)

        batch_size = batch['loc'].size(0)
        repeated_batch = self._repeat_for_pomo(batch, self.args.pomo_size)
        cost, log_likelihood = self.model(repeated_batch, state_kwargs=state_kwargs)
        return cost.reshape(batch_size, self.args.pomo_size), log_likelihood.reshape(batch_size, self.args.pomo_size)

    @staticmethod
    def _pomo_loss(costs, log_probs):
        if costs.size(1) <= 1:
            baseline = costs.mean().detach()
            advantage = costs - baseline
        else:
            baseline = costs.mean(dim=1, keepdim=True)
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

    def _augment_service_metrics(self, results):
        graph_size = max(int(self.args.graph_size), 1)
        service_rate = float(results.get('avg_completed_orders', 0.0)) / graph_size
        rejected_rate = float(results.get('avg_rejected_orders', 0.0)) / graph_size
        unfulfilled_rate = float(results.get('avg_unfulfilled_orders', 0.0)) / graph_size
        untouched_unrejected_rate = float(results.get('avg_untouched_unrejected_orders', 0.0)) / graph_size
        served_plus_rejected_rate = min(1.0, service_rate + rejected_rate)
        results['service_rate'] = service_rate
        results['completed_rate'] = service_rate
        results['rejected_rate'] = rejected_rate
        results['unfulfilled_rate'] = unfulfilled_rate
        results['untouched_unrejected_rate'] = untouched_unrejected_rate
        results['served_plus_rejected_rate'] = served_plus_rejected_rate
        results['non_service_rate'] = min(1.0, rejected_rate + unfulfilled_rate)
        return results

    def train_epoch(self, epoch, train_loader):
        self.model.train()
        set_decode_type(self.model, 'sampling')
        allow_reject = epoch > self.args.reject_warmup_epochs
        state_kwargs = self._build_state_kwargs(allow_reject=allow_reject)

        epoch_loss = 0.0
        epoch_objective = 0.0
        n_batches = 0

        current_lr = self.optimizer.param_groups[0]['lr']
        progress = tqdm(train_loader, desc=f"Epoch {epoch} (lr={current_lr:.2e})")
        for batch_idx, batch in enumerate(progress, start=1):
            batch = self._to_device(batch)
            costs, log_probs = self._pomo_forward(batch, state_kwargs=state_kwargs)
            loss, mean_objective = self._pomo_loss(costs, log_probs)

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)
            self.optimizer.step()

            epoch_loss += loss.item()
            epoch_objective += mean_objective.item()
            n_batches += 1

            if (batch_idx % self.log_interval == 0) or (batch_idx == len(train_loader)):
                progress.set_postfix({'loss': f'{loss.item():.4f}', 'objective': f'{mean_objective.item():.2f}'})

        self.lr_scheduler.step()
        return epoch_loss / max(n_batches, 1), epoch_objective / max(n_batches, 1)

    def validate(self, val_loader):
        self.model.eval()
        set_decode_type(self.model, 'greedy')

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

        val_state_kwargs = self._build_state_kwargs(allow_reject=True)
        with torch.no_grad():
            for batch in tqdm(val_loader, desc='Validating'):
                batch = self._to_device(batch)
                if self.args.collect_mask_diagnostics:
                    objective_cost, _, pi, debug = self.model(
                        batch,
                        return_pi=True,
                        state_kwargs=val_state_kwargs,
                        return_debug=True,
                    )
                    for key, value in debug.items():
                        debug_buffers.setdefault(key, []).append(value)
                else:
                    objective_cost, _, pi = self.model(
                        batch,
                        return_pi=True,
                        state_kwargs=val_state_kwargs
                    )
                _, details = self.problem.get_costs(batch, pi, return_details=True)

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
        return results

    def train(self):
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
        print(f"Shared env: {self._build_state_kwargs(allow_reject=True)}")
        print(f"Curriculum enabled: {self.args.enable_rideshare_curriculum}")
        print('=' * 70)

        os.makedirs(self.args.save_dir, exist_ok=True)
        val_dataset = MCVRPPDTWDataset(
            num_samples=self.args.val_size,
            graph_size=self.args.graph_size,
            seed=12345,
            **self._build_default_dataset_kwargs(),
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.args.batch_size,
            collate_fn=collate_fn,
            **self.loader_kwargs,
        )

        best_val_objective = self.best_val_objective
        best_service_key = self.best_service_key
        best_service_rate = self.best_service_rate
        for epoch in range(self.start_epoch, self.args.n_epochs + 1):
            curriculum_kwargs = self._build_curriculum_dataset_kwargs(epoch)
            train_dataset = MCVRPPDTWDataset(
                num_samples=self.args.epoch_size,
                graph_size=self.args.graph_size,
                seed=epoch * 1000,
                **curriculum_kwargs,
            )
            train_loader = DataLoader(
                train_dataset,
                batch_size=self.args.batch_size,
                shuffle=True,
                collate_fn=collate_fn,
                **self.loader_kwargs,
            )

            train_loss, train_objective = self.train_epoch(epoch, train_loader)
            val_results = self.validate(val_loader)

            self.train_log.append({'epoch': epoch, 'loss': train_loss, 'objective': train_objective})
            self.val_log.append({'epoch': epoch, **val_results})

            print(f"\nEpoch {epoch}/{self.args.n_epochs}:")
            print(f"  Train Loss: {train_loss:.4f}")
            print(f"  Train Objective: {train_objective:.2f}")
            print(f"  Data Profile: {self._describe_curriculum(epoch)}")
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

            service_key = self._service_priority_key(val_results)
            if best_service_key is None or service_key > best_service_key:
                best_service_key = service_key
                best_service_rate = float(val_results['service_rate'])
                self.best_service_key = best_service_key
                self.best_service_rate = best_service_rate
                self._save_model(epoch, val_results, 'best')
                print('  [Saved best model by service rate]')

            if val_results['avg_objective'] < best_val_objective:
                best_val_objective = val_results['avg_objective']
                self.best_val_objective = best_val_objective
                self._save_model(epoch, val_results, 'best_objective')
                print('  [Saved best model by objective]')

            if self.args.save_interval > 0 and epoch % self.args.save_interval == 0 and epoch != self.args.n_epochs:
                self._save_model(epoch, val_results, f'epoch_{epoch}')

        self._save_model(self.args.n_epochs, val_results, 'final')
        self._save_logs()

        self.best_service_key = best_service_key
        self.best_service_rate = best_service_rate
        self.best_val_objective = best_val_objective

        print('\n' + '=' * 70)
        print('Training Complete!')
        print(f"Best validation service rate: {best_service_rate:.3f}")
        print(f"Best validation objective: {best_val_objective:.2f}")
        print('=' * 70)
        return {
            'best_service_rate': best_service_rate,
            'best_objective': best_val_objective,
        }

    def _save_model(self, epoch, results, name):
        path = os.path.join(self.args.save_dir, f'model_{name}.pt')
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'results': results,
            'args': vars(self.args),
            'normalization_profile': self.normalization_profile,
        }, path)

    def _save_logs(self):
        log_path = os.path.join(self.args.save_dir, 'training_log.json')
        with open(log_path, 'w', encoding='utf-8') as log_file:
            json.dump({
                'train': self.train_log,
                'val': self.val_log,
                'config': {
                    'AREA_SIZE': Config.AREA_SIZE,
                    'PASSENGER_CAPACITY': Config.PASSENGER_CAPACITY,
                    'CARGO_CAPACITY': Config.CARGO_CAPACITY,
                    'PASSENGER_TW_WIDTH': Config.PASSENGER_TW_WIDTH,
                    'MAX_TRIP_TIME': Config.MAX_TRIP_TIME,
                    'ELECTRICITY_PRICE': Config.ELECTRICITY_PRICE,
                    'PASSENGER_DELAY_COST': Config.PASSENGER_DELAY_COST,
                    'CARGO_DELAY_COST': Config.CARGO_DELAY_COST,
                    'ALPHA_ENERGY': self.args.alpha_energy,
                    'ALPHA_DELAY': self.args.alpha_delay,
                    'ALPHA_VEHICLE': self.args.alpha_vehicle,
                    'ALPHA_REJECT': self.args.alpha_reject,
                    'ALPHA_UNFULFILLED': self.args.alpha_unfulfilled,
                    'ALPHA_TRIP_OVERTIME': self.args.alpha_trip_overtime,
                    'VEHICLE_SPEED': Config.VEHICLE_SPEED,
                    'seed': self.args.seed,
                    'OPERATION_START': Config.OPERATION_START,
                    'reject_warmup_epochs': self.args.reject_warmup_epochs,
                    'reject_init_bias': self.args.reject_init_bias,
                    'best_checkpoint_metric': 'service_rate',
                    'OPERATION_END': Config.OPERATION_END,
                    'max_concurrent_open_orders': self.args.max_concurrent_open_orders,
                    'enable_delivery_viability': self.args.enable_delivery_viability,
                    'enable_viability_fallback': self.args.enable_viability_fallback,
                    'enable_rideshare_curriculum': self.args.enable_rideshare_curriculum,
                },
                'normalization_profile': self.normalization_profile,
            }, log_file, indent=2, ensure_ascii=False)

    def _load_checkpoint(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint['model_state_dict'])

        if self.args.resume_weights_only:
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
            self.best_service_key = self._service_priority_key(saved_results)
            self.best_service_rate = float(saved_results.get('service_rate', float('-inf')))
        print(f"[Resume] Loaded checkpoint: {checkpoint_path}")
        print(f"[Resume] Start epoch: {self.start_epoch}")
        print(f"[Resume] Best objective so far: {self.best_val_objective:.2f}")
        if self.best_service_key is not None:
            print(f"[Resume] Best service rate so far: {self.best_service_rate:.3f}")


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
    parser.add_argument('--collect-mask-diagnostics', action='store_true', help='在验证/评估中收集 mask 与动作可行性诊断指标')
    parser.add_argument('--deadlock-limit', type=int, default=2, help='连续回 depot 且无可服务节点时的终止阈值')
    parser.add_argument('--max-concurrent-open-orders', type=int, default=6, help='共享主线：允许的最大并发 open 单数量')
    parser.add_argument('--enable-delivery-viability', action='store_true', default=True, help='共享主线：启用 delivery viability')
    parser.add_argument('--disable-delivery-viability', action='store_false', dest='enable_delivery_viability', help='关闭 delivery viability（仅对照实验）')
    parser.add_argument('--enable-viability-fallback', action='store_true', default=False, help='启用 delivery viability fallback（仅对照实验）')
    parser.add_argument('--relax-pickup-commitment-trip-time', action='store_true', default=False, help='仅对 pickup_commitment 的 completion proof 放松 trip_time gate（实验开关）')
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


def build_phase_args(cli_args, graph_size):
    if graph_size not in PHASE_CONFIGS:
        raise ValueError(f'Unsupported graph size: {graph_size}')

    phase = PHASE_CONFIGS[graph_size].copy()
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
        max_concurrent_open_orders=cli_args.max_concurrent_open_orders,
        enable_delivery_viability=cli_args.enable_delivery_viability,
        enable_viability_fallback=cli_args.enable_viability_fallback,
        relax_pickup_commitment_trip_time=cli_args.relax_pickup_commitment_trip_time,
        alpha_energy=cli_args.alpha_energy,
        alpha_delay=cli_args.alpha_delay,
        alpha_vehicle=cli_args.alpha_vehicle,
        alpha_reject=cli_args.alpha_reject,
        alpha_unfulfilled=cli_args.alpha_unfulfilled,
        alpha_trip_overtime=cli_args.alpha_trip_overtime,
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

    print('=' * 70)
    print('MCVRP-PDTW with POMO Training (OPTIMIZED VERSION)')
    print('=' * 70)

    summary = {}
    for graph_size in cli_args.graph_sizes:
        phase_args = build_phase_args(cli_args, graph_size)
        apply_runtime_training_config(phase_args)
        print(f"\n[Phase] Training on {graph_size} orders ({graph_size * 2} nodes)...")
        trainer = POMOTrainerOptimized(phase_args)
        best_metrics = trainer.train()
        summary[graph_size] = best_metrics

    print('\n' + '=' * 70)
    print('All requested training phases complete!')
    for graph_size in cli_args.graph_sizes:
        metrics = summary[graph_size]
        print(f"N={graph_size}: best service rate = {metrics['best_service_rate']:.3f}, best objective = {metrics['best_objective']:.2f}")
    print('=' * 70)


if __name__ == '__main__':
    main()
