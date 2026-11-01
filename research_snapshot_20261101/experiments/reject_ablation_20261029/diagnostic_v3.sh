#!/usr/bin/env bash
set -euo pipefail
for mode in learned service_first; do
 python -u experiments/reject_ablation_20261029/run_v3.py --policy "$mode" --record "diagnostic_${mode}" --graph_sizes 50 --epochs 10 --epoch-size 1024 --val-size 128 --batch-size 16 --pomo-size 1 --seed 1234 --num-workers 0 --reject-warmup-epochs 0 --baseline-mode batch_mean --output-root "diagnostic_${mode}/outputs" --benchmark-mode --benchmark-warmup-epochs 0 --benchmark-json --benchmark-json-output "diagnostic_${mode}/timing.json" > "diagnostic_${mode}.log" 2>&1
 python -u experiments/reject_ablation_20261029/audit_eval_v3.py --audit-output "diagnostic_${mode}/audit.json" --checkpoint "diagnostic_${mode}/outputs/pomo_n50_optimized/model_best.pt" --graph-size 50 --num-samples 128 --batch-size 16 --seed 876543 --max-concurrent-open-orders 6 --enable-delivery-viability --diagnostics --trace-first-untouched-unrejected --json-output "diagnostic_${mode}/eval.json" > "diagnostic_${mode}_eval.log" 2>&1
done
