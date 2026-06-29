# Pickup commitment structural diagnostic summary

This note packages the latest verified findings from the new `pickup_commitment` structural diagnostic run for the current mainline candidate.

## Scope

- checkpoint:
  - `outputs_smoke/reject_acceptance_sweep_tw1_v1/suppress_w3_b25/pomo_n25_optimized/model_best.pt`
- evaluation setup:
  - `500` instances
  - `seed = 99999`
  - state kwargs resolved from the checkpoint:
    - `max_concurrent_open_orders = 6`
    - `enable_delivery_viability = True`
    - `enable_viability_fallback = False`
- generated artifact directory (not versioned by default):
  - `outputs_eval/pickup_commitment_breakdown_suppress_w3_b25/`

## Code changes in this package

The state-machine diagnostics were extended so `pickup_commitment` blocks are decomposed into explicit sub-reasons.

New debug counters include:
- `diag_pickup_commitment_block_by_k`
- `diag_pickup_commitment_block_by_completion`
- `diag_pickup_commitment_block_by_completion_ride_time`
- `diag_pickup_commitment_block_by_completion_trip_time`
- `diag_pickup_commitment_block_by_completion_ops_end`
- `diag_pickup_commitment_block_by_completion_open_over_6`
- `diag_pickup_commitment_block_by_completion_other`
- `diag_pickup_commitment_block_by_next_state`
- `diag_pickup_commitment_block_by_fallback`

Relevant code:
- `state_mcvrptw_v2.py`
- `nets/attention_model.py`
- `evaluate_model.py`
- `rootcause_oracle_vs_rollout.py`

## Headline metrics

Under the current best checkpoint, the rollout remains unchanged relative to the previous formal evaluation:

- `service_rate = 0.6320`
- `completed_orders = 15.80`
- `rejected_orders = 9.20`
- `unfulfilled_orders = 0.00`

Important structural diagnostics:
- `diag_mask_pickup_commitment = 8.9754`
- `diag_second_pickup_feasible = 0.5813`
- `diag_second_pickup_blocked_by_commitment = 6.1978`

## pickup_commitment block breakdown

Shares below are normalized by `diag_mask_pickup_commitment`, so the top-level categories sum to approximately 100%.

- `K ceiling = 0.0077 / step` (`0.1%`)
- `Completion infeasible = 8.6331 / step` (`96.2%`)
  - `ride time = 1.1664 / step` (`13.0%`)
  - `trip time = 4.4224 / step` (`49.3%`)
  - `ops end = 0.0000 / step` (`0.0%`)
  - `open > 6 = 0.0000 / step` (`0.0%`)
  - `other/mixed = 3.0444 / step` (`33.9%`)
- `Next state no delivery = 0.3346 / step` (`3.7%`)
- `Fallback failed = 0.0000 / step` (`0.0%`)

## Interpretation

- `max_concurrent_open_orders = 6` is not the active bottleneck under the current checkpoint.
- The dominant blocker is the forward-looking completion proof inside `pickup_commitment`, not the K ceiling itself.
- The largest single explicit blocker is `trip_time` inside the completion check.
- `ride_time` also matters, but is smaller than `trip_time`.
- `ops_end` does not materially contribute in the current regime.
- The hidden `open > 6 -> False` cap exists in `_has_feasible_open_completion`, but it is not active for the current checkpoint.
- `next_state_feasible` contributes only a small tail compared with `completion_feasible`.

## Current working hypothesis

The most important remaining structural question is not “should K be larger?”, but rather:

> Is the current forward-looking `pickup_commitment` proof too conservative because it treats single-trip return feasibility (`trip_time`) as a hard gate before allowing a pickup?

This is especially important if the real business semantics are closer to:
- satisfying time windows,
- satisfying passenger ride-time / order-duration constraints,
- and controlling whole-day vehicle usage,

rather than enforcing a strict per-trip return proof before every new pickup.

## Recommended next action

> Historical note: this recommendation has been partially superseded by the follow-up findings in `NEXT_STATE_AND_ORDER_PROFILE_SUMMARY.md`.

当时建议先做：

- 只放松 `pickup_commitment` completion-proof `trip_time` gate 的受控 ablation

后续 follow-up 已确认：

- 只放松 upstream completion-proof `trip_time` 并不会提升 rollout service
- blocking mass 会转移到 `next_state_trip_time`

因此当前更合适的顺序是：

1. 先在**不改变约束语义**的前提下优化 `get_mask()` / `pickup_commitment` / `delivery_viability` 的重复搜索开销
2. 若仍要做语义实验，再单独测试 **next-state first-delivery `trip_time` gate** 是否是当前服务率上限的直接驱动因素

## Guardrails

- These results diagnose the current rollout policy and state-machine gating under the present code and checkpoint.
- They do not prove that all service loss is caused by `trip_time`; `other/mixed` remains sizable and may hide interaction effects across candidate completion permutations.
- The `open > 6` hard cap is a real architectural limit for future scaling, but it is not the currently active bottleneck.
