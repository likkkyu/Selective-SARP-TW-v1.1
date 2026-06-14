# Next-state diagnostic and 25-order order-profile summary

This note packages the latest verified findings after the `pickup_commitment` trip-time ablation and the follow-up 25-order order-level profiling run.

## Scope

- checkpoint:
  - `outputs_smoke/reject_acceptance_sweep_tw1_v1/suppress_w3_b25/pomo_n25_optimized/model_best.pt`
- graph size:
  - `25`
- primary evaluation setup:
  - `500` instances
  - `seed = 99999`
- rollout state kwargs:
  - `max_concurrent_open_orders = 6`
  - `enable_delivery_viability = True`
  - `enable_viability_fallback = False`
- generated artifact directories (not versioned by default):
  - `outputs_eval/next_state_diag_compare/`
  - `outputs_eval/next_state_diag_baseline_rootcause/`
  - `outputs_eval/next_state_diag_relaxed_rootcause/`
  - `outputs_eval/order_profile_suppress_w3_b25/`

## Code changes in this package

The state-machine diagnostics were extended again so `diag_pickup_commitment_block_by_next_state` is decomposed into single-label sub-reasons:

- `diag_pickup_commitment_block_by_next_state_precedence`
- `diag_pickup_commitment_block_by_next_state_ride_time`
- `diag_pickup_commitment_block_by_next_state_trip_time`
- `diag_pickup_commitment_block_by_next_state_ops_end`
- `diag_pickup_commitment_block_by_next_state_delivery_viability`
- `diag_pickup_commitment_block_by_next_state_vehicle_limit`
- `diag_pickup_commitment_block_by_next_state_mixed`
- `diag_pickup_commitment_block_by_next_state_other`

Relevant code:
- `state_mcvrptw_v2.py`
- `nets/attention_model.py`
- `evaluate_model.py`
- `rootcause_oracle_vs_rollout.py`

## Headline rollout metrics

Under the current best checkpoint:

- `service_rate = 0.6320`
- `completed_orders = 15.80`
- `rejected_orders = 9.20`
- `unfulfilled_orders = 0.00`
- `used_vehicles = 4.50`

The rollout remains structurally clean:
- no pickup-only orders
- no started-not-completed orders
- no passenger pickup / ride-time hard violations

## Next-state structural diagnostic

A controlled evaluation-only ablation was run that relaxes only the `pickup_commitment` completion-proof `trip_time` gate.

Result:
- rollout outcomes do **not** change
- blocking mass moves from completion-proof `trip_time` into `next_state_feasible`

Baseline vs relaxed summary:
- `completed_orders: 15.7406 -> 15.7406`
- `service_rate: 0.6296 -> 0.6296`
- `rejected_orders: 9.2594 -> 9.2594`
- `unfulfilled_orders: 0.0000 -> 0.0000`
- `diag_pickup_commitment_block_by_completion_trip_time: 4.4170 -> 0.0000`
- `diag_pickup_commitment_block_by_next_state: 0.3285 -> 4.9629`

### next_state subreason shares

Baseline next-state failures:
- `mixed = 100.0%`
- all other next-state subreasons `= 0`

Relaxed next-state failures:
- `trip_time = 80.9%`
- `delivery_viability = 5.6%`
- `mixed = 13.5%`
- `precedence = 0.0%`
- `ride_time = 0.0%`
- `ops_end = 0.0%`
- `vehicle_limit = 0.0%`
- `other = 0.0%`

## 25-order served vs rejected order profile

A new order-level profiling run reconstructed, for each evaluated 25-order instance:
- which orders were completed
- which orders were rejected
- how completed orders were distributed across vehicle routes

Artifacts:
- `outputs_eval/order_profile_suppress_w3_b25/summary.json`
- `outputs_eval/order_profile_suppress_w3_b25/summary.md`

### Vehicle usage and route structure

Across `500` instances:
- average used vehicles = `4.50`
- vehicle count distribution:
  - `5 vehicles: 292`
  - `4 vehicles: 168`
  - `3 vehicles: 39`
  - `2 vehicles: 1`
- average completed orders per route = `3.51`

### Strongest served-vs-rejected signal: time period

Service rate by pickup time-window period:
- `morning = 99.98%`
- `midday = 58.45%`
- `evening = 0.00%`

This is the dominant order-level pattern in the current 25-order regime.

### Type differences are small

Service rate by type:
- `cargo = 64.17%`
- `passenger = 62.53%`

So the main bottleneck is **not** passenger-vs-cargo mix.

### Distance effects exist but are secondary

Service rate by pickup→delivery distance:
- `<5km = 63.96%`
- `5-10km = 62.14%`
- `>=10km = 57.86%`

Service rate by depot→pickup distance is nearly flat:
- `<3km = 63.80%`
- `3-5km = 62.75%`
- `>=5km = 63.29%`

### Mean-feature comparison

Completed orders:
- pickup time-window center = `11.69`
- pickup→delivery distance = `4.55 km`
- depot→pickup distance = `4.04 km`

Rejected orders:
- pickup time-window center = `14.20`
- pickup→delivery distance = `4.64 km`
- depot→pickup distance = `4.09 km`

The biggest gap is clearly the pickup time-window center, not order size or geometry.

## Interpretation

The two diagnostics point in the same direction:

1. The current service ceiling is **not** mainly explained by order type or depot distance.
2. The strongest order-level failure mode is that **late-window orders are much harder**; evening orders are essentially all rejected.
3. The strongest structural state-machine bottleneck remains **trip-time feasibility**, not `precedence`, not `ride_time`, not `vehicle_limit`.
4. Relaxing only the upstream completion-proof `trip_time` gate is insufficient because the bottleneck reappears immediately as **next-step delivery `trip_time`** in the hypothetical next state.

In short:

> The current 25-order service ceiling is best explained as a combination of a late-window-heavy data regime and a trip-time-centered legality structure that becomes especially restrictive for those later orders.

## Recommended next actions

### 1. Data-side action with the highest confidence

Continue making the 25-order data distribution slightly easier by reducing late-window mass:
- reduce the `evening` weight further, or
- shift evening pickup windows earlier, or
- mildly shorten late-window pickup→delivery distances

This has the strongest direct evidence from the new order-profile run.

### 2. Model/state-side action with the highest information value

Run a stronger controlled experiment that relaxes only the **next-state first-delivery `trip_time` gate** after a hypothetical pickup.

Why this is the best next experiment:
- the completion-proof `trip_time` ablation already showed no service improvement,
- the deeper bottleneck is now localized to `next_state_trip_time`,
- this cleanly tests whether the current service ceiling is still driven by trip-time legality at the first post-pickup delivery step.

## Guardrails

- These findings describe the current rollout policy and the current hard-constraint semantics under the present checkpoint.
- The order-profile run reconstructed completed vs rejected orders from the evaluated action sequence and route structure; it is descriptive evidence, not a causal proof by itself.
- The next-state subreasons are a single-label classification of why the hypothetical next state had zero feasible first-step deliveries; they are not a global decomposition of all mask counters.
