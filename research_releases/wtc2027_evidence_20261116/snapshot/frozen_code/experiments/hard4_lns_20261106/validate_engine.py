"""Cross-check independent engine against archived audit and tensor C4.

Includes real valid plans, randomized corruptions and aggregate cargo-tier case.
"""
from bootstrap import *
from route_engine import RouteEngine,split_actions
import json,random,signal
P=Path(__file__).resolve().parent


def main():
 out=P/'validation_v1';out.mkdir(exist_ok=False)
 signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('180s validation budget')));signal.alarm(180)
 data=torch.load(ROOT/'experiments/paper_delivery_20261104/development_n50_seed614729.pt',weights_only=False)['data']
 rows=[];mutations=0;validmutations=0;rng=random.Random(61106)
 Config.set_normalization_profile(50,dict(PROFILE,graph_size=50))
 for source in sorted((ROOT/'experiments/hard4_admission_20261106/dev4_v1').glob('*_*.json')):
  for row in json.loads(source.read_text())['rows']:
   s=data[row['instance']];engine=RouteEngine(s,C,PROFILE);routes=split_actions(row['actions'],50)
   cost=engine.cost(routes);assert cost is not None
   reference=scalar_cost(s,row['actions'],C,PROFILE)
   assert max(abs(cost[k]-reference[k]) for k in reference)<1e-7,(source,row['instance'],cost,reference)
   with torch.no_grad():j,d=MCVRPPDTW.get_costs(sample_to_batch(s),torch.tensor([row['actions']],dtype=torch.long),return_details=True)
   assert abs(float(j)-cost['objective'])<.02 and abs(float(d['total_cost_raw'])-cost['raw_cost'])<.02
   for trial_id in range(30):
    trial=[list(r) for r in routes];r=rng.randrange(len(trial));route=trial[r]
    mode=trial_id%5
    if mode==0:
     i,jj=rng.sample(range(len(route)),2);route[i],route[jj]=route[jj],route[i]
    elif mode==1:route.insert(rng.randrange(len(route)),rng.choice(route))
    elif mode==2:route.pop(rng.randrange(len(route)))
    elif mode==3:
     order=rng.choice([x for x in route if x<=50]);trial[r]=[x for x in route if x not in (order,order+50)]
    else:
     q=rng.randrange(len(trial));trial[q].extend(route[:]);trial[r]=[] if q!=r else trial[q]
    trial=[tuple(r) for r in trial if r];actions=engine.actions(trial)
    # Invalid duplicate-node plans can serialize negative reject counts; audit
    # still catches their duplicate status. Compare feasibility, not their costs.
    a=audit(s,actions,audit_config(C,50,9));valid=not a['violations'] and not a['unfulfilled']
    result=engine.cost(trial);assert (result is not None)==valid,(mode,a,trial,result)
    mutations+=1;validmutations+=int(valid)
   rows.append(dict(model=source.stem,instance=row['instance'],objective_error=abs(float(j)-cost['objective']),raw_error=abs(float(d['total_cost_raw'])-cost['raw_cost'])))
   print('CHECK',len(rows),'/24',source.stem,row['instance'],'passed',flush=True)
 # Two complete four-cargo tours, each 40+ minutes tardy: enforce the
 # instance-level tier rather than separately applying the cheap first tier.
 n=8;s=dict(depot=torch.tensor([0.,0.]),loc=torch.tensor([[.1,0.]]*n+[[.2,0.]]*n),time_windows=torch.tensor([[10.,16.]]*n+[[10.,10.01]]*n),node_type=torch.zeros(2*n,dtype=torch.long),demand_passenger=torch.zeros(2*n),demand_cargo=torch.tensor([.1]*n+[-.1]*n))
 engine=RouteEngine(s,C,PROFILE,2);routes=[tuple(list(range(1,5))+list(range(9,13))),tuple(list(range(5,9))+list(range(13,17)))];r=engine.cost(routes);reference=scalar_cost(s,engine.actions(routes),C,PROFILE)
 assert r is not None and abs(r['cargo_delay_cost']-reference['cargo_delay_cost'])<1e-8
 assert sum(engine.route(x)['cargo_minutes'] for x in routes)>60
 assert engine.cost([routes[0][:3]+routes[0][4:]]) is None
 report=dict(status='passed',real_plans=len(rows),mutated_plans=mutations,feasible_mutations=validmutations,rows=rows,cargo_tier_case=r,physical_tolerance_hours=1e-4,scope='implementation equivalence on tested plans; not universal proof or final performance benchmark')
 (out/'report.json').write_text(json.dumps(report,indent=2));signal.alarm(0)
 print('PASS independent engine: 24 real plans,',mutations,'perturbations, aggregate cargo tiers',flush=True)
if __name__=='__main__':main()
