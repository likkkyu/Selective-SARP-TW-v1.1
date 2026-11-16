"""Frozen-scenario evaluation with mandatory cost-v4 checkpoint validation."""
import runpy
from pathlib import Path
from scenarios import ROOT
import sys
sys.path.insert(0,str(ROOT/'experiments/reject_ablation_20261029'))
from timing_candidate import install
install()
import evaluate_v3
original=evaluate_v3.evaluate_model.build_model_from_checkpoint
def build(checkpoint,*args,**kwargs):
    if checkpoint.get('args',{}).get('cost_semantics_version')!=4:
        raise ValueError('Sensitivity requires cost-v4 checkpoint')
    return original(checkpoint,*args,**kwargs)
evaluate_v3.evaluate_model.build_model_from_checkpoint=build
runpy.run_path(str(Path(__file__).with_name('evaluate.py')),run_name='__main__')
