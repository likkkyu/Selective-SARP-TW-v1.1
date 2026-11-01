#!/usr/bin/env bash
set -euo pipefail
export PYTHONUNBUFFERED=1
checkpoint=formal_learned_1234/outputs/pomo_n50_optimized/model_best.pt
python -u experiments/sensitivity_20261030/evaluate_v4_guarded.py --scenario guard_regression/failure_single.pt --output-dir guard_regression/single_v2 --checkpoint "$checkpoint" --batch-size 1 --max-concurrent-open-orders 6 --enable-delivery-viability > guard_regression/single_v2.log 2>&1
python -u experiments/sensitivity_20261030/evaluate_v4_guarded.py --scenario guard_regression/failure_batch16.pt --output-dir guard_regression/batch16_v2 --checkpoint "$checkpoint" --batch-size 16 --max-concurrent-open-orders 6 --enable-delivery-viability > guard_regression/batch16_v2.log 2>&1
python -u experiments/sensitivity_20261030/batch_v4_guarded.py --checkpoint "$checkpoint" --output preflight_guarded_v2 --preflight
python - <<'PY'
import json
from pathlib import Path
for path,n in [('guard_regression/single_v2',1),('guard_regression/batch16_v2',16)]:
 p=Path(path);a=json.load(open(p/'audit.json'));g=json.load(open(p/'completion_guard.json'));assert a['samples']==n and a['hard_valid']==n and a['unfulfilled_total']==0 and g['failed_recoveries']==0
m=json.load(open('preflight_guarded_v2/manifest.json'));assert m['status']=='completed' and len(m['completed'])==3
print('GUARDED REGRESSION PASSED: observed failure, containing batch16, 3x4 scenario preflight',flush=True)
Path('guard_regression/gate_pass.json').write_text(json.dumps({'status':'passed','guard_version':1,'lookahead_margin_seconds':.036,'audit_tolerance_hours':.0001,'requires_same_guard_for_comparisons':True},indent=2))
PY
echo "GUARDED FORMAL START $(date -u -Iseconds)"
python -u experiments/sensitivity_20261030/batch_v4_guarded.py --checkpoint "$checkpoint" --output learned_1234_v4_guarded
