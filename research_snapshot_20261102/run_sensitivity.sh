#!/usr/bin/env bash
set -euo pipefail
export PYTHONUNBUFFERED=1
checkpoint=formal_learned_1234/outputs/pomo_n50_optimized/model_best.pt
echo "PREFLIGHT START $(date -u -Iseconds)"
timeout 1200 python -u experiments/sensitivity_20261030/batch_v4.py --checkpoint "$checkpoint" --output preflight_v4 --preflight
python - <<'PY'
import json,pathlib
p=pathlib.Path('preflight_v4');d=json.load(open(p/'manifest.json'));assert d['status']=='completed' and len(d['completed'])==3
for folder in p.glob('r*'):
 if not folder.is_dir():continue
 m=json.load(open(folder/'scenario.json'));c=json.load(open(folder/'effective_config.json'));a=json.load(open(folder/'audit.json'))
 assert c['PASSENGER_MAX_EXCESS_RIDE_TIME_MINUTES']==m['excess_minutes']
 assert a['hard_valid']==4 and a['unfulfilled_total']==0
print('PREFLIGHT PASS: 3 scenarios x 4, cost-v4 enforced, configuration and independent constraints verified',flush=True)
PY
echo "FORMAL SENSITIVITY START $(date -u -Iseconds)"
timeout --signal=TERM --kill-after=60 28800 python -u experiments/sensitivity_20261030/batch_v4.py --checkpoint "$checkpoint" --output learned_1234_v4
echo "COMPLETED $(date -u -Iseconds)"
