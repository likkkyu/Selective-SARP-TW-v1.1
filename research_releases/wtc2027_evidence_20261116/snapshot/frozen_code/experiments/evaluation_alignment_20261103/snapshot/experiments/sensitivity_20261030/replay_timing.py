import sys,json
from pathlib import Path
import torch
from scenarios import ROOT
from problem_mcvrptw_v2 import Config,MCVRPPDTW
from evaluate_model import collate_fn
from timing_candidate import install
p=Path(__file__).parent;root=p/'results_1234/learned_1234'
records=[]
for folder in sorted(root.iterdir()):
 if not (folder/'routes.jsonl').exists():continue
 cfg=json.loads((folder/'effective_config.json').read_text())
 records.append((folder,cfg,[json.loads(x) for x in (folder/'routes.jsonl').read_text().splitlines()]))
report={}
for mode in ['original','candidate']:
 if mode=='candidate':install()
 for folder,cfg,rows in records:
  for k,v in cfg.items():
   if hasattr(Config,k):setattr(Config,k,v)
  sums={}
  for start in range(0,len(rows),16):
   chunk=rows[start:start+16];samples=[]
   for r in chunk:
    samples.append({k:torch.tensor(v,dtype=(torch.long if k in ['node_type','pickup_delivery_pairs','n_orders'] else torch.bool if k=='pd_pair_mask' else torch.float32)) if isinstance(v,list) else v for k,v in r['sample'].items()})
   length=max(len(r['actions']) for r in chunk);pi=torch.zeros(len(chunk),length,dtype=torch.long)
   for i,r in enumerate(chunk):pi[i,:len(r['actions'])]=torch.tensor(r['actions'])
   _,d=MCVRPPDTW.get_costs(collate_fn(samples),pi,return_details=True)
   for k in ['trip_overtime_penalty','passenger_excess_ride_time_violations','passenger_total_ride_time_violations','total_distance','energy_cost_raw','total_cost_raw']:
    if k in d:sums[k]=sums.get(k,0.)+d[k].sum().item()
  report.setdefault(folder.name,{})[mode]={k:v/len(rows) for k,v in sums.items()}
(p/'timing_replay.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
