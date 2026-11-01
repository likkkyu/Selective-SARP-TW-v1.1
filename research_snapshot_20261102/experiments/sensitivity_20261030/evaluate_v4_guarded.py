"""Frozen-scenario evaluation with mandatory cost-v4 checkpoint validation."""
import runpy
from pathlib import Path
from scenarios import ROOT
import sys
sys.path.insert(0,str(ROOT/'experiments/reject_ablation_20261029'))
from timing_candidate import install
install()
from completion_guard import install as install_guard, STATS, VERSION
install_guard()
import evaluate_v3
original=evaluate_v3.evaluate_model.build_model_from_checkpoint
def build(checkpoint,*args,**kwargs):
    if checkpoint.get('args',{}).get('cost_semantics_version')!=4:
        raise ValueError('Sensitivity requires cost-v4 checkpoint')
    return original(checkpoint,*args,**kwargs)
evaluate_v3.evaluate_model.build_model_from_checkpoint=build
out=Path(sys.argv[sys.argv.index('--output-dir')+1])
try:
    runpy.run_path(str(Path(__file__).with_name('evaluate.py')),run_name='__main__')
finally:
    import json
    if out.exists():(out/'completion_guard.json').write_text(json.dumps({'version':VERSION,**STATS,'physical_tolerance_hours':1e-5,'lookahead_margin_seconds':.036},indent=2))
