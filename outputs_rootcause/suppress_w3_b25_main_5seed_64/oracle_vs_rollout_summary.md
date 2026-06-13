# Root-cause oracle-vs-rollout audit

This report compares the trained rollout policy against heuristic baselines on identical instances.

**Important:** `oracle_style` is a heuristic comparator, not a proof of global optimality.

## Config

- graph_size: 25
- num_samples_per_seed: 64
- seeds: [101, 202, 303, 404, 505]
- policies: ['rollout', 'oracle_style', 'random', 'bad']
- checkpoint: /Users/bytedance/Downloads/project-vrp-v6-4-main/outputs_smoke/reject_acceptance_sweep_tw1_v1/suppress_w3_b25/pomo_n25_optimized/model_best.pt

## Overall summary (mean ± std across seeds)

### rollout

- completed_orders: 15.7406 ± 0.1121
- service_rate: 0.6296 ± 0.0045
- served_plus_rejected_orders: 25.0000 ± 0.0000
- served_plus_rejected_rate: 1.0000 ± 0.0000
- rejected_orders: 9.2594 ± 0.1121
- rejected_rate: 0.3704 ± 0.0045
- unfulfilled_orders: 0.0000 ± 0.0000
- unfulfilled_rate: 0.0000 ± 0.0000
- pickup_only_orders: 0.0000 ± 0.0000
- started_not_completed_orders: 0.0000 ± 0.0000
- passenger_pickup_hard_violations: 0.0000 ± 0.0000
- passenger_total_ride_time_violations: 0.0000 ± 0.0000
- passenger_excess_ride_time_violations: 0.0000 ± 0.0000
- trip_overtime_penalty: 0.0000 ± 0.0000
- diag_mask_precedence_pct_of_service_slots: 25.4501 ± 0.2550
- diag_mask_vehicle_limit_pct_of_service_slots: 0.0192 ± 0.0027
- diag_mask_pickup_tw_pct_of_service_slots: 1.5758 ± 0.0668
- diag_mask_trip_time_pct_of_service_slots: 2.0932 ± 0.0919

### oracle_style

- completed_orders: 15.4844 ± 0.1285
- service_rate: 0.6194 ± 0.0051
- served_plus_rejected_orders: 15.4844 ± 0.1285
- served_plus_rejected_rate: 0.6194 ± 0.0051
- rejected_orders: 0.0000 ± 0.0000
- rejected_rate: 0.0000 ± 0.0000
- unfulfilled_orders: 9.5156 ± 0.1285
- unfulfilled_rate: 0.3806 ± 0.0051
- pickup_only_orders: 0.0000 ± 0.0000
- started_not_completed_orders: 0.0000 ± 0.0000
- passenger_pickup_hard_violations: 0.0000 ± 0.0000
- passenger_total_ride_time_violations: 0.0000 ± 0.0000
- passenger_excess_ride_time_violations: 0.0000 ± 0.0000
- trip_overtime_penalty: 0.0000 ± 0.0000
- diag_mask_precedence_pct_of_service_slots: 30.0546 ± 0.4235
- diag_mask_vehicle_limit_pct_of_service_slots: 0.0325 ± 0.0049
- diag_mask_pickup_tw_pct_of_service_slots: 1.2258 ± 0.0406
- diag_mask_trip_time_pct_of_service_slots: 2.0746 ± 0.0712

### random

- completed_orders: 13.9750 ± 0.4096
- service_rate: 0.5590 ± 0.0164
- served_plus_rejected_orders: 13.9750 ± 0.4096
- served_plus_rejected_rate: 0.5590 ± 0.0164
- rejected_orders: 0.0000 ± 0.0000
- rejected_rate: 0.0000 ± 0.0000
- unfulfilled_orders: 11.0250 ± 0.4096
- unfulfilled_rate: 0.4410 ± 0.0164
- pickup_only_orders: 0.0000 ± 0.0000
- started_not_completed_orders: 0.0000 ± 0.0000
- passenger_pickup_hard_violations: 0.0000 ± 0.0000
- passenger_total_ride_time_violations: 0.0000 ± 0.0000
- passenger_excess_ride_time_violations: 0.0000 ± 0.0000
- trip_overtime_penalty: 0.0000 ± 0.0000
- diag_mask_precedence_pct_of_service_slots: 34.0278 ± 0.4847
- diag_mask_vehicle_limit_pct_of_service_slots: 0.1335 ± 0.0288
- diag_mask_pickup_tw_pct_of_service_slots: 4.6349 ± 0.2740
- diag_mask_trip_time_pct_of_service_slots: 3.0654 ± 0.1770

### bad

- completed_orders: 13.7781 ± 0.1970
- service_rate: 0.5511 ± 0.0079
- served_plus_rejected_orders: 13.7781 ± 0.1970
- served_plus_rejected_rate: 0.5511 ± 0.0079
- rejected_orders: 0.0000 ± 0.0000
- rejected_rate: 0.0000 ± 0.0000
- unfulfilled_orders: 11.2219 ± 0.1970
- unfulfilled_rate: 0.4489 ± 0.0079
- pickup_only_orders: 0.0000 ± 0.0000
- started_not_completed_orders: 0.0000 ± 0.0000
- passenger_pickup_hard_violations: 0.0000 ± 0.0000
- passenger_total_ride_time_violations: 0.0000 ± 0.0000
- passenger_excess_ride_time_violations: 0.0000 ± 0.0000
- trip_overtime_penalty: 0.0000 ± 0.0000
- diag_mask_precedence_pct_of_service_slots: 33.9899 ± 0.2974
- diag_mask_vehicle_limit_pct_of_service_slots: 0.1462 ± 0.0119
- diag_mask_pickup_tw_pct_of_service_slots: 4.8653 ± 0.0892
- diag_mask_trip_time_pct_of_service_slots: 3.4448 ± 0.2284

## Per-seed artifact files

- seed 101: `seed_101.json`
- seed 202: `seed_202.json`
- seed 303: `seed_303.json`
- seed 404: `seed_404.json`
- seed 505: `seed_505.json`

