# Current best checkpoint summary

This note consolidates the current best 25-order DRL checkpoint after the depot dead-end reject-cleanup fix and the follow-up retraining run.

## Current best checkpoint

- checkpoint:
  - `outputs/depot_deadend_cleanup_seed1234_10ep/pomo_n25_optimized/model_best.pt`
- training run directory:
  - `outputs/depot_deadend_cleanup_seed1234_10ep/pomo_n25_optimized/`
- selection rule (historical note for this run):
  - this specific checkpoint was selected under the older `service_rate`-first rule that existed at the time of the run
  - going forward, `model_best.pt` is reserved for the business-first selector: clean residual buckets first, then `service_rate`, then `avg_objective`
  - service-priority snapshots are now kept separately as `model_best_service.pt`
- loaded epoch in the formal evaluation:
  - `10`

## Why this checkpoint matters

The earlier semantic issue was not low service by itself, but that some untouched orders could leak through terminal cleanup as residual `unfulfilled` instead of being converted into explicit `reject`.

That semantic path is now fixed, and the new 10-epoch run shows the model can do both at once:

1. keep the rollout semantically clean
   - `unfulfilled = 0`
   - `untouched_unrejected = 0`
   - `started_not_completed = 0`
2. continue improving service
   - best validation `service_rate = 0.9091`

## Best validation snapshot from `training_log.json`

Best epoch: `10`

- `avg_objective = 1332.1467`
- `avg_cost = 1473.6426`
- `avg_completed_orders = 22.7266`
- `avg_rejected_orders = 2.2734`
- `avg_unfulfilled_orders = 0.0000`
- `avg_untouched_unrejected_orders = 0.0000`
- `avg_started_not_completed_orders = 0.0000`
- `service_rate = 0.9091`
- `rejected_rate = 0.0909`
- `unfulfilled_rate = 0.0000`
- `served_plus_rejected_rate = 1.0000`

Training-run config highlights:
- `graph_size = 25`
- `seed = 1234`
- `reject_warmup_epochs = 3`
- `reject_init_bias = -2.5`
- `max_concurrent_open_orders = 6`
- `enable_delivery_viability = True`
- `enable_viability_fallback = False`

## Formal 500-instance greedy evaluation

Command used:

```bash
python3 evaluate_model.py \
  --graph-size 25 \
  --checkpoint /Users/bytedance/Downloads/project-vrp-v6-4-main/outputs/depot_deadend_cleanup_seed1234_10ep/pomo_n25_optimized/model_best.pt \
  --num-samples 500 \
  --batch-size 32 \
  --seed 99999 \
  --decode greedy \
  --diagnostics
```

Formal evaluation result (`outputs_eval/formal_compare_suppress_w3_b25/current_best_eval.txt`):

- `Total Cost (raw) = 1494.071 ± 1048.812 RMB`
- `Completed Orders = 22.70 ± 1.84`
- `Rejected Before Service = 2.30 ± 1.84`
- `Residual Unfulfilled Orders = 0.00 ± 0.00`
- `Untouched Orders (legacy accounting; includes explicit rejects) = 2.30 ± 1.84`
- `Untouched, Not Explicitly Rejected = 0.00 ± 0.00`
- `Service Rate = 0.908 ± 0.073`
- `Rejected Before Service Rate = 0.092 ± 0.073`
- `Residual Unfulfilled Rate = 0.000 ± 0.000`
- `Served+Rejected Rate = 1.000 ± 0.000`
- `Pickup-only Orders = 0.00 ± 0.00`
- `Started but Not Completed = 0.00 ± 0.00`

Additional formal-eval observations:
- no passenger pickup hard violations
- no passenger ride-time violations
- `Reject predeparture rate = 0.13 ± 0.03`
- `Reject in-route rate = 0.00 ± 0.00`

### Replay-audit conclusion

The evaluator now includes a replay-based order audit that replays the decoded `pi` through `StateMCVRPPDTW` and reconstructs the final per-order partition from state semantics rather than only from count-level accounting.

On the 500-instance formal run, that replay audit reports:

- `Partition-consistent samples = 500 / 500`
- `Core aggregate-match samples = 500 / 500`
- `Audited Rejected Before Service = 2.30 ± 1.84`
- `Audited Residual Unfulfilled Orders = 0.00 ± 0.00`
- `Audited Pickup-only Orders = 0.00 ± 0.00`
- `Audited Started but Not Completed = 0.00 ± 0.00`
- `Audited Untouched, Not Explicitly Rejected = 0.00 ± 0.00`

This directly matches the target business semantics for the current best checkpoint:

- every non-served order is rejected **before service**
- no order is picked up and then left incomplete
- no untouched-but-unrejected residuals remain

Important stability caveat:

- one clean checkpoint proves this training direction can reach an acceptable point
- it does **not** by itself prove that later epochs or other seeds will keep the residual buckets locked at zero
- formal comparison across saved checkpoints, and then longer-run confirmation, is still required before treating a direction as stably converged

Important wording note:

- `Untouched Orders` is now treated as a **legacy accounting metric** only; it is not the business-semantic failure bucket because it can still include orders that were explicitly rejected before service
- the business-semantic residual buckets are instead:
  - `Untouched, Not Explicitly Rejected`
  - `Started but Not Completed`
  - `Residual Unfulfilled Orders`

## Comparison against recent checkpoints

### Versus the 5-epoch post-fix run

5-epoch best validation snapshot (`outputs/depot_deadend_cleanup_seed1234_5ep/pomo_n25_optimized/training_log.json`, epoch 5):

- `service_rate = 0.8591`
- `avg_completed_orders = 21.4766`
- `avg_rejected_orders = 3.5234`
- `avg_unfulfilled_orders = 0.0000`
- `avg_untouched_unrejected_orders = 0.0000`

10-epoch best validation snapshot:

- `service_rate = 0.9091`
- `avg_completed_orders = 22.7266`
- `avg_rejected_orders = 2.2734`
- `avg_unfulfilled_orders = 0.0000`
- `avg_untouched_unrejected_orders = 0.0000`

Interpretation:
- continuing training after the semantic fix did not reintroduce residual failures
- instead, service improved materially while rejects fell

### Versus the earlier high-service 5-epoch smoke under the new defaults

Earlier smoke guidance in this branch established a strong 5-epoch result around:
- `service_rate ≈ 0.884`
- `completed ≈ 22.10`
- `unfulfilled ≈ 1.90`

That run showed the data/business-regime change was effective for service, but it still violated the desired business semantics by leaving residual `unfulfilled` mass.

The current checkpoint is stronger because it improves on both axes simultaneously:
- higher service (`0.909` vs `~0.884`)
- zero `unfulfilled`
- zero `untouched_unrejected`
- zero `started_not_completed`

## Current interpretation

This is now the strongest 25-order checkpoint in the current local line because it achieves all of the following together:

- high service (`~0.91`)
- explicit partition of non-served mass into `reject` rather than silent leftovers
- no pickup-only failures
- no started-but-not-completed failures
- no passenger pickup / ride-time hard violations in the formal evaluation

In short:

> The depot dead-end reject-cleanup fix is not just a semantic cleanup. After retraining, it supports a stronger policy that is both cleaner and better-performing.

## Local workspace notes

During this consolidation pass:
- stale ignored evaluation/root-cause output folders were removed
- the active 10-epoch checkpoint directory was retained
- tracked reference artifacts under `outputs_eval/`, `outputs_rootcause/`, and `outputs/normalization_profiles/` were preserved

If the current checkpoint is accepted as the new stable local baseline, the older 5-epoch local run can be discarded later because its comparison metrics are already captured here.
