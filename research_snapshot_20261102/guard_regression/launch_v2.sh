#!/usr/bin/env bash
set -uo pipefail
for i in $(seq 1 24); do
 if ! kill -0 7497 2>/dev/null; then break; fi
 sleep 5
done
if kill -0 7497 2>/dev/null; then echo 'Previous regression still active; no duplicate launch'; exit 1; fi
timeout --signal=TERM --kill-after=60 30000 bash guard_regression/check_and_resume_v2.sh
code=$?
echo "$code" > guarded_v2_exit.txt
exit "$code"
