"""Isolated cost-v4 smoke launcher; v3 policy and no-op semantics retained."""
import argparse,runpy,sys
from pathlib import Path
from scenarios import ROOT
sys.path.insert(0,str(ROOT/'experiments/reject_ablation_20261029'))
from timing_candidate import install
install()
p=argparse.ArgumentParser();p.add_argument('--stage',choices=['train','eval'],required=True);a,rest=p.parse_known_args()
if a.stage=='train':
 import run_training_optimized as training
 original=training.POMOTrainerOptimized._save_model
 def save(self,*args,**kwargs):
  self.args.cost_semantics_version=4
  return original(self,*args,**kwargs)
 training.POMOTrainerOptimized._save_model=save
 target='run_v3.py'
else:
 import evaluate_v3
 original=evaluate_v3.evaluate_model.build_model_from_checkpoint
 def build(checkpoint,*args,**kwargs):
  if checkpoint.get('args',{}).get('cost_semantics_version')!=4:raise ValueError('Expected cost-v4 checkpoint')
  return original(checkpoint,*args,**kwargs)
 evaluate_v3.evaluate_model.build_model_from_checkpoint=build
 target='audit_eval_v3.py'
sys.argv=[target,*rest]
runpy.run_path(str(ROOT/'experiments/reject_ablation_20261029'/target),run_name='__main__')
