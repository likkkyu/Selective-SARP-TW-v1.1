"""Adversarial feasibility and upper-tier cost checks, separate from benchmarks."""
import json
from validate import P,C,PROFILE,torch,audit,exhaustive
from scip_selective import SelectiveSCIP,audit_config

def toy(n):
 return dict(depot=torch.tensor([0.,0.]),loc=torch.tensor([[.1,0.]]*n+[[.2,0.]]*n),time_windows=torch.tensor([[10.,15.]]*(2*n)),node_type=torch.zeros(2*n,dtype=torch.long),demand_passenger=torch.zeros(2*n),demand_cargo=torch.tensor([.01]*n+[-.01]*n),n_orders=torch.tensor(n))

def force_routes(s,routes,k):
 solver=SelectiveSCIP(C,PROFILE,vehicles=k,time_limit=10,verbose=False);m=solver.build(s);wanted=set()
 for route in routes:
  path=[0]+route+[0];wanted.update(zip(path,path[1:]))
 for edge,var in solver.x.items():m.addCons(var==int(edge in wanted))
 for var in solver.z.values():m.addCons(var==1)
 m.optimize();return str(m.getStatus())

out=P/'constraint_validation_v1';out.mkdir(exist_ok=False);rows=[]
s=toy(2);status=force_routes(s,[[1,4],[2,3]],2);assert status=='infeasible',status
rows.append(dict(case='split_pair_across_routes',status=status,expected='infeasible'))
s=toy(7);status=force_routes(s,[list(range(1,15))],1);assert status=='infeasible',status
rows.append(dict(case='seven_simultaneously_open',status=status,expected='infeasible'))
s=toy(2);s['loc']=torch.tensor([[.1,0.],[.2,0.],[.99,0.],[.99,0.]])
s['time_windows']=torch.tensor([[10.,11.],[10.,11.],[9.,9.5],[9.,9.5]])
s['demand_cargo']=torch.tensor([.6,.6,-.6,-.6])
oracle=exhaustive(s,2);sol=SelectiveSCIP(C,PROFILE,vehicles=2,time_limit=30,verbose=False).solve(s)
assert sol['status']=='optimal' and abs(sol['objective']-oracle['best']['objective'])<.02
assert sol['replayed']['cargo_delay_cost']>C['CARGO_DELAY_TIER1_MIN']*C['CARGO_DELAY_COST']+(C['CARGO_DELAY_TIER2_MIN']-C['CARGO_DELAY_TIER1_MIN'])*C['CARGO_DELAY_COST_30_60']
rows.append(dict(case='cargo_aggregate_lateness_above_60_minutes',result=sol,oracle=oracle))
(out/'report.json').write_text(json.dumps(dict(status='passed',rows=rows),indent=2));print('PASS split-pair, max-open, cargo upper-tier tests')
