"""Run tagged v3 checkpoint on frozen scenarios with independent route audit."""
import argparse, hashlib, json, runpy, sys
from pathlib import Path
import torch
from scenarios import ROOT
sys.path.insert(0,str(ROOT/'experiments/reject_ablation_20261029'))
import evaluate_v3 as entry
from problem_mcvrptw_v2 import Config
p=argparse.ArgumentParser();p.add_argument('--scenario',required=True);p.add_argument('--output-dir',required=True)
a,rest=p.parse_known_args()
if any(x.split('=')[0] in ('--graph-size','--num-samples','--seed','--json-output','--audit-output') for x in rest):
    p.error('Scenario controls size/seed; output directory controls result paths')
source=Path(a.scenario);bundle=torch.load(source,map_location='cpu',weights_only=False)
m=bundle['meta']; out=Path(a.output_dir);out.mkdir(parents=True,exist_ok=False)

def frozen_dataset(**kwargs):
    if Config.PASSENGER_MAX_RIDE_TIME_MINUTES != m['total_ride_minutes']:
        raise ValueError('Total ride baseline mismatch')
    Config.PASSENGER_MAX_EXCESS_RIDE_TIME_MINUTES=m['excess_minutes']
    (out/'effective_config.json').write_text(json.dumps({k:getattr(Config,k) for k in dir(Config) if k.isupper() and isinstance(getattr(Config,k),(int,float,str,bool,tuple))},indent=2))
    return bundle['data']
entry.evaluate_model.MCVRPPDTWDataset=frozen_dataset
(out/'scenario.json').write_text(json.dumps(dict(m,sha256=hashlib.sha256(source.read_bytes()).hexdigest(),argv=rest),indent=2))
sys.argv=['audit_eval_v3.py','--audit-output',str(out/'audit.json'),'--json-output',str(out/'eval.json'),
          '--graph-size',str(m['graph_size']),'--num-samples',str(m['samples']),'--seed',str(m['seed']),*rest]
original_replay=entry.evaluate_model._replay_order_audit
instance_count=0
def traced_replay(batch, pi, state_kwargs):
    global instance_count
    with (out/'routes.jsonl').open('a') as trace:
        for i in range(pi.shape[0]):
            sample={k:(v[i].detach().cpu().tolist() if hasattr(v,'shape') and len(v.shape)>0 and v.shape[0]==pi.shape[0] else v) for k,v in batch.items()}
            trace.write(json.dumps(dict(instance=instance_count,sample=sample,actions=pi[i].tolist(),state_kwargs=state_kwargs))+'\n')
            instance_count+=1
    return original_replay(batch,pi,state_kwargs)
entry.evaluate_model._replay_order_audit=traced_replay
runpy.run_path(str(ROOT/'experiments/reject_ablation_20261029/audit_eval_v3.py'),run_name='__main__')
r=json.loads((out/'audit.json').read_text())
if r['samples']!=m['samples'] or r['hard_valid']!=m['samples'] or r['unfulfilled_total']:
    raise RuntimeError('Scenario audit failed; preserved evidence')
