import json
from pathlib import Path
import torch
from run_v2 import install_policy
from evaluate_v2 import build
from nets.attention_model import AttentionModel
for mode in ['learned','service_first']:
 p=Path(f'preflight_{mode}')
 checkpoint=torch.load(p/'outputs/pomo_n50_optimized/model_final.pt',map_location='cpu',weights_only=False)
 assert checkpoint['args']['ablation_policy']==mode
 assert checkpoint['args']['ablation_policy_version']==2
 evaluation=json.loads((p/'eval.json').read_text())
 assert evaluation['audit']['partition_consistent_samples']==16
 assert evaluation['audit']['core_aggregate_match_samples']==16
 assert 'Restored ablation policy: '+mode in Path(f'preflight_{mode}_eval.log').read_text()
# Installation can switch back to learned in the same interpreter.
install_policy('service_first'); install_policy('learned')
original=AttentionModel._ablation_original_logits
# Unknown/legacy policies must fail before constructing/loading a model.
try: build({'args':{}})
except ValueError: pass
else: raise AssertionError('untagged checkpoint accepted')
Path('preflight_pass.json').write_text(json.dumps({'passed':True,'checks':['checkpoint policy metadata','eval automatic restoration','partition and aggregate audit','legacy checkpoint rejected']}))
print('PREFLIGHT PASS')
