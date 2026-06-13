# Root-cause audit and suppress_w3_b25 summary

This note packages the latest verified findings for the current Selective SARP-TW mainline candidate.

## Current candidate

- `reject_warmup_epochs = 3`
- `reject_init_bias = -2.5`
- checkpoint path used for formal verification:
  - `outputs_smoke/reject_acceptance_sweep_tw1_v1/suppress_w3_b25/pomo_n25_optimized/model_best.pt`

## Metric semantics (code truth)

- `completed_orders` means pickup and delivery are both completed.
- `service_rate = completed_orders / graph_size`.
- `rejected_orders` counts active rejects.
- `unfulfilled_orders` counts final not-completed orders that were not actively rejected.

Relevant code:
- `run_training_optimized.py`
- `problem_mcvrptw_v2.py`

## Bug fix included in this package

`problem_mcvrptw_v2.py` had a depot-time drift bug in the cost-side time simulation: depot visits were incorrectly adding `SERVICE_TIME`.

The fix keeps depot time unchanged:

```python
current_time = is_depot * current_time + (1 - is_depot) * (start_service + Config.SERVICE_TIME)
```

This removed spurious pickup-hard-violation / overtime artifacts in evaluation.

## Formal 500-instance evaluation

Candidate (`outputs_eval/formal_compare_suppress_w3_b25/suppress_w3_b25_eval.txt`):
- Total Cost (raw): `5409.106 ± 1236.855`
- Completed Orders: `15.80 ± 2.17`
- Service Rate: `0.632 ± 0.087`
- Rejected Orders: `9.20 ± 2.17`
- Unfulfilled Orders: `0.00 ± 0.00`

Current baseline (`outputs_eval/formal_compare_suppress_w3_b25/current_best_eval.txt`):
- Total Cost (raw): `6220.741 ± 1781.204`
- Completed Orders: `15.32 ± 2.14`
- Service Rate: `0.613 ± 0.085`
- Rejected Orders: `6.62 ± 4.49`
- Unfulfilled Orders: `3.06 ± 5.00`

Conclusion: `suppress_w3_b25` is formally better on service/completion, lower raw cost, and drives unfulfilled orders to zero without introducing hard violations.

## 5-seed root-cause audit (`5 seeds × 64 samples`)

Artifact directory:
- `outputs_rootcause/suppress_w3_b25_main_5seed_64/`

Rollout results:
- `completed_orders = 15.7406 ± 0.1121`
- `service_rate = 0.6296 ± 0.0045`
- `rejected_orders = 9.2594 ± 0.1121`
- `unfulfilled_orders = 0.0000 ± 0.0000`
- `served_plus_rejected_rate = 1.0000 ± 0.0000`
- `diag_mask_precedence_pct_of_service_slots = 25.4501 ± 0.2550`
- `diag_mask_vehicle_limit_pct_of_service_slots = 0.0192 ± 0.0027`

Oracle-style heuristic comparator:
- `completed_orders = 15.4844 ± 0.1285`
- `service_rate = 0.6194 ± 0.0051`
- `unfulfilled_orders = 9.5156 ± 0.1285`

Important guardrail: `oracle_style` is a heuristic comparator, not a proof of global optimality.

## Current interpretation

- `vehicle_limit` remains negligible and is not the main bottleneck.
- The model is not primarily suffering from "accept then fail to finish" behavior anymore.
- The stronger candidate more cleanly partitions orders into either completed or explicitly rejected.
- The next meaningful optimization target is structural serviceability (precedence / pickup commitment / trip-time / delivery-viability interactions), not further large bias-only sweeps.
