"""Dependency audit for completed/in-progress experiment, without mutating its plan."""
import hashlib,json,sys
from importlib.metadata import version
from validate import P,ROOT,SNAP,C,PROFILE
out=P/(sys.argv[1] if len(sys.argv)>1 else 'common_n50_15s_v1')
plan=json.loads((out/'plan.json').read_text())
assert all(hashlib.sha256((P/k).read_bytes()).hexdigest()==v for k,v in plan['source_sha256'].items())
files=[SNAP/'problem_mcvrptw_v2.py',SNAP/'baseline_utils.py',SNAP/'experiments/reject_ablation_20261029/route_audit.py',SNAP/'experiments/sensitivity_20261030/timing_candidate.py',SNAP/'outputs/normalization_profiles/normalization_n50.json',P/'validate.py']
r=dict(scope='dependency/parameter capture; main solver and initialization hashes checked against pre-run plan',physical_and_cost_parameters=C,profile=PROFILE,packages={p:version(p) for p in ['pyscipopt','torch','numpy']},dependencies={str(f.relative_to(ROOT)):hashlib.sha256(f.read_bytes()).hexdigest() for f in files})
(out/'dependencies.json').write_text(json.dumps(r,indent=2));print('Dependency and parameter capture saved; frozen implementation hashes match')
