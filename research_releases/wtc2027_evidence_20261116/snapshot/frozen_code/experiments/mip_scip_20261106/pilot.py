"""Bounded SCIP pilot on existing common instances; not a final MIP benchmark."""
import hashlib,json,sys,time
from pathlib import Path
from validate import ROOT,P,SNAP,C,PROFILE
import torch
from route_audit import audit
from problem_mcvrptw_v2 import Config,MCVRPPDTW
from baseline_utils import sample_to_batch
from scip_selective import SelectiveSCIP,audit_config

def main():
    out=P/'pilot_v1';out.mkdir(exist_ok=False)
    path=ROOT/'experiments/paper_delivery_20261104/confirmation_n50_seed803219.pt'
    bundle=torch.load(path,map_location='cpu',weights_only=False)
    plan=dict(scope='SCIP implementation and bounded-budget pilot on previously read common test instances; not tuning or fresh confirmation',instances=[0,1],time_limit_per_instance_seconds=30,threads=1,solver_seed=1234,profile=PROFILE,scenario_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),no_neural_warm_start=True)
    (out/'plan.json').write_text(json.dumps(plan,indent=2));rows=[]
    Config.set_normalization_profile(50,PROFILE)
    for i in plan['instances']:
        print('RUN SCIP N50 instance',i,'budget 30 s, CPU 1 thread',flush=True)
        s=bundle['data'][i];solver=SelectiveSCIP(C,PROFILE,time_limit=30,verbose=False)
        row=solver.solve(s,logfile=out/f'instance_{i}.log')
        if row['actions'] is not None:
            a=audit(s,row['actions'],audit_config(C,50,9));row['audit']=a
            assert a['unfulfilled']==0 and not any(v['kind']!='min_dispatch' for v in a['violations']),a
            with torch.no_grad():j,d=MCVRPPDTW.get_costs(sample_to_batch(s),torch.tensor([row['actions']],dtype=torch.long),return_details=True)
            row['tensor_objective_error']=abs(float(j.item())-row['replayed']['objective']);row['tensor_raw_error']=abs(float(d['total_cost_raw'].item())-row['replayed']['raw_cost'])
            assert row['tensor_objective_error']<.05 and row['tensor_raw_error']<.05,row
        row['instance']=i;rows.append(row)
        (out/f'instance_{i}.json').write_text(json.dumps(row,indent=2))
        print('DONE',i,'status',row['status'],'served',row.get('completed'),'J',row.get('objective'),'bound',row['dual_bound'],'gap',row.get('gap'),flush=True)
    (out/'report.json').write_text(json.dumps(dict(plan=plan,status='completed',rows=rows),indent=2))
if __name__=='__main__':main()
