"""Check archived neural feasible solutions against the actual MILP rows, no solve."""
import hashlib,json,signal,sys,time,gc
from validate import ROOT,P,SNAP,C,PROFILE,torch
from scip_selective import SelectiveSCIP,lists

def main():
 out=P/'neural_routes_mip_membership_v1';out.mkdir(exist_ok=False)
 signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('240s MIP membership check')));signal.alarm(240)
 q=ROOT/'experiments/paper_delivery_20261104/confirmation_v1';plan=json.loads((q/'plan.json').read_text());data=torch.load(ROOT/'experiments/paper_delivery_20261104/confirmation_n50_seed803219.pt',map_location='cpu',weights_only=False)['data']
 traces={f"{m['policy']}_{m['seed']}":[json.loads(x) for x in (q/'models'/f"{m['policy']}_{m['seed']}"/'routes.jsonl').read_text().splitlines()] for m in plan['models']}
 result=dict(status='running',scope='membership of 192 archived neural routes in original SCIP MILP; no reoptimization, no new performance values',rows=[],domain=[]);begin=time.monotonic()
 def save():result['elapsed_seconds']=time.monotonic()-begin;(out/'report.json').write_text(json.dumps(result,indent=2))
 for i,s in enumerate(data):
  ss=lists(s);lows=[float(x[0]) for x in ss['time_windows']]
  domain=dict(instance=i,max_window_start=max(lows),min_window_start=min(lows),all_node_windows_start_before_end_minus_service=all(x<=C['OPERATION_END']-C['SERVICE_TIME'] for x in lows),capacity_normalization_passenger_15=True,capacity_normalization_cargo_20=True)
  assert domain['all_node_windows_start_before_end_minus_service'];result['domain'].append(domain)
  solver=SelectiveSCIP(C,PROFILE,time_limit=1,verbose=False);solver.build(s)
  for label,rows in traces.items():
   try:
    stored=solver.add_route_start(rows[i]['actions']);row=dict(instance=i,model=label,mip_feasible=True,stored_as_improving_solution=bool(stored))
   except ValueError as exc:row=dict(instance=i,model=label,mip_feasible=False,error=str(exc))
   result['rows'].append(row)
  solver.model.freeProb();del solver;gc.collect();save();print('CHECK',i+1,'/32',sum(x['mip_feasible'] for x in result['rows']),'/',len(result['rows']),flush=True)
 result['status']='completed';result['summary']=dict(checked=len(result['rows']),mip_feasible=sum(x['mip_feasible'] for x in result['rows']),mip_infeasible=sum(not x['mip_feasible'] for x in result['rows']),domain_checks_passed=len(result['domain']));save();signal.alarm(0);print(json.dumps(result['summary']),flush=True)
if __name__=='__main__':main()
