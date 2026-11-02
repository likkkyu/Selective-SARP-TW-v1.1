#!/usr/bin/env bash
set -uo pipefail
cd /root/autodl-tmp/cost_v4_formal_pair1234_20261031
printf 'Waiting for bounded tolerance replay; training starts only after reviewed numeric gate.\n'
for attempt in $(seq 1 180); do
 if test -f tolerance_review.json; then break; fi
 if ! kill -0 1393 2>/dev/null; then echo 'Replay ended without report; STOP'; exit 1; fi
 sleep 10
done
python - <<'PY'
import json
r=json.load(open('tolerance_review.json'));a=json.load(open('tolerance_audit.json'))
assert r['max_seconds_over']<=0.36 and a['hard_valid']==500 and a['unfulfilled_total']==0
print('TOLERANCE GATE PASS',r,flush=True)
PY
if test $? -ne 0; then exit 1; fi
cd /root/autodl-tmp/cost_v4_remaining_seeds_20261101
date -u -Iseconds > start_utc.txt
timeout --signal=TERM --kill-after=60 129600 bash remaining.sh
code=$?
echo "$code" > exit_status.txt
exit "$code"
