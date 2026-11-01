#!/usr/bin/env bash
set -euo pipefail
for mode in learned service_first; do
 python -u experiments/reject_ablation_20261029/run_v3.py --policy "$mode" --record "v3check_${mode}" --graph_sizes 50 --epochs 2 --epoch-size 64 --val-size 32 --batch-size 16 --pomo-size 1 --seed 1234 --num-workers 0 --reject-warmup-epochs 0 --baseline-mode batch_mean --output-root "v3check_${mode}/outputs" > "v3check_${mode}.log" 2>&1
 python -u experiments/reject_ablation_20261029/audit_eval_v3.py --audit-output "v3check_${mode}/audit.json" --checkpoint "v3check_${mode}/outputs/pomo_n50_optimized/model_final.pt" --graph-size 50 --num-samples 16 --batch-size 16 --seed 876543 --max-concurrent-open-orders 6 --enable-delivery-viability --json-output "v3check_${mode}/eval.json" > "v3check_${mode}_eval.log" 2>&1
done
