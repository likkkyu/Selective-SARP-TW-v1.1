#!/usr/bin/env bash
# Run under a single outer timeout 129600; require a fresh isolated directory.
set -euo pipefail
python - <<'PY'
import json
from pathlib import Path
p=Path('../reject_ablation_v3_20261030')
assert (p/'diagnostic_v3_exit.txt').read_text().strip()=='0'
for mode in ['learned','service_first']:
 d=json.loads((p/f'diagnostic_{mode}/audit.json').read_text())
 assert d['samples']==128 and d['hard_valid']==128 and d['unfulfilled_total']==0
 m=json.loads((p/f'diagnostic_{mode}/manifest.json').read_text());assert m['status']=='completed'
 v=json.loads((p/f'diagnostic_{mode}/outputs/pomo_n50_optimized/training_log.json').read_text())['val']
 assert len(v)==10 and all(x['service_rate']>0 for x in v[-3:])
Path('formal_gate_pass.json').write_text(json.dumps({'passed':True,'diagnostic':'../reject_ablation_v3_20261030','note':'operational gate, not proof of convergence or method benefit'}))
PY
for seed in 1234 4321 8765; do
 for mode in learned service_first; do
  root="formal_${mode}_${seed}"
  python -u experiments/reject_ablation_20261029/run_formal.py --policy "$mode" --record "$root" --graph_sizes 50 --epochs 20 --epoch-size 2048 --val-size 500 --batch-size 16 --pomo-size 1 --seed "$seed" --num-workers 0 --reject-warmup-epochs 0 --baseline-mode batch_mean --output-root "$root/outputs" > "$root.log" 2>&1
  python -u experiments/reject_ablation_20261029/audit_eval_v3.py --audit-output "$root/audit.json" --checkpoint "$root/outputs/pomo_n50_optimized/model_best.pt" --graph-size 50 --num-samples 500 --batch-size 16 --seed 917531 --max-concurrent-open-orders 6 --enable-delivery-viability --json-output "$root/eval.json" > "${root}_eval.log" 2>&1
  python - "$root" <<'PY'
import json,sys
from pathlib import Path
p=Path(sys.argv[1]);d=json.loads((p/'audit.json').read_text())
assert d['samples']==500 and d['hard_valid']==500 and d['unfulfilled_total']==0, 'Formal audit failed; preserve evidence and review before continuing'
PY
 done
done
