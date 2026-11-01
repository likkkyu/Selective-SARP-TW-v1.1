#!/usr/bin/env bash
set -euo pipefail
for policy in learned service_first; do
 root="diagnostic_${policy}"
 python -u experiments/sensitivity_20261030/corrected_entry.py --stage train --policy "$policy" --record "$root" --graph_sizes 50 --epochs 10 --epoch-size 1024 --val-size 128 --batch-size 16 --pomo-size 1 --seed 1234 --num-workers 0 --reject-warmup-epochs 0 --baseline-mode batch_mean --output-root "$root/outputs" > "$root.log" 2>&1
 python -u experiments/sensitivity_20261030/corrected_entry.py --stage eval --audit-output "$root/audit.json" --checkpoint "$root/outputs/pomo_n50_optimized/model_best.pt" --graph-size 50 --num-samples 128 --batch-size 16 --seed 928632 --max-concurrent-open-orders 6 --enable-delivery-viability --json-output "$root/eval.json" > "${root}_eval.log" 2>&1
done

python - <<'CHECK'
import json
from pathlib import Path
for mode in ['learned','service_first']:
 p=Path('diagnostic_'+mode)
 a=json.loads((p/'audit.json').read_text())
 v=json.loads((p/'outputs/pomo_n50_optimized/training_log.json').read_text())['val']
 assert a['samples']==128 and a['hard_valid']==128 and a['unfulfilled_total']==0
 assert len(v)==10 and all(x['service_rate']>0 for x in v[-3:]),'Persistent rejection collapse; review required'
Path('diagnostic_gate.json').write_text(json.dumps({'operational_gate':True,'note':'Not convergence or superiority proof; review full curves before formal work'}))
CHECK
