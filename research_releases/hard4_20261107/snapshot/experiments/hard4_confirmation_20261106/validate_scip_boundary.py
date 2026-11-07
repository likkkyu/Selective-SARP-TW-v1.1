"""Optional requests outside operating hours must not invalidate all-reject.

Old model's actual failure is retained; new version tested in isolation.
"""
from common import *
from scip_boundary_v2 import BoundarySafeSCIP
from strict_scip import HardMinDispatchSCIP
import copy,json,time,signal
P=Path(__file__).resolve().parent


def toy(n):
 return dict(depot=torch.tensor([0.,0.]),loc=torch.tensor([[.1,0.]]*n+[[.2,0.]]*n),time_windows=torch.tensor([[10.,11.]]*n+[[10.,16.]]*n),node_type=torch.zeros(2*n,dtype=torch.long),demand_passenger=torch.zeros(2*n),demand_cargo=torch.tensor([.1]*n+[-.1]*n))


def main():
 out=P/'scip_boundary_validation_v1';out.mkdir(exist_ok=False);signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('120s boundary validation cap')));signal.alarm(120)
 rows=[]
 for label,n,changes,expected in [('four_feasible',4,{},4),('one_late_pickup_below_minimum',4,{0:[16.5,17.]},0),('all_late_pickups',4,{i:[30.,31.] for i in range(4)},0),('late_delivery_earliest',4,{4:[17.,18.]},0),('past_hard_pickup',4,{0:[8.,9.]},0),('four_feasible_plus_optional_late_pair',5,{4:[20.,21.],9:[21.,22.]},4)]:
  s=toy(n)
  for j,tw in changes.items():s['time_windows'][j]=torch.tensor(tw)
  actions=list(range(1,5))+list(range(n+1,n+5))+[0]+[2*n+1]*(n-4) if expected==4 else [2*n+1]*n
  old=None
  if label=='all_late_pickups':
   try:old=HardMinDispatchSCIP(C,PROFILE,vehicles=1,time_limit=2,verbose=False).solve(s)
   except Exception as exc:old=dict(error=repr(exc))
  solver=BoundarySafeSCIP(C,PROFILE,vehicles=1,time_limit=10,verbose=False)
  r=solver.solve(s,warm_actions=actions);assert r['status']=='optimal' and r['completed']==expected,(label,r)
  Config.set_normalization_profile(n,dict(PROFILE,graph_size=n))
  with torch.no_grad():j,d=MCVRPPDTW.get_costs(sample_to_batch(s),torch.tensor([r['actions']],dtype=torch.long),return_details=True)
  assert abs(float(j)-r['objective'])<.02 and abs(float(d['total_cost_raw'])-r['replayed']['raw_cost'])<.02
  rows.append(dict(case=label,result=r,old_model_observation=old));(out/'report.json').write_text(json.dumps(dict(status='running',rows=rows),indent=2));print('PASS',label,'served',expected,'forced',r['forced_declined_order_ids'],flush=True)
 # Regression on a real N50 development start: necessary presolve is a no-op.
 sample=torch.load(ROOT/'experiments/paper_delivery_20261104/development_n50_seed614729.pt',weights_only=False)['data'][0]
 warm=json.loads((ROOT/'experiments/hard4_lns_20261106/development_v2/initial_0.json').read_text())['actions']
 r=BoundarySafeSCIP(C,PROFILE,vehicles=9,time_limit=2,verbose=False).solve(sample,warm_actions=warm)
 assert r['forced_declined_order_ids']==[] and r['completed']==50
 rows.append(dict(case='real_n50_noop',result=r))
 # Warm start must be rejected by the real MILP if a trip has only 3 pairs.
 s=toy(8);bad=[1,2,3,9,10,11,0,4,5,6,7,8,12,13,14,15,16,0]
 rejected=False
 try:BoundarySafeSCIP(C,PROFILE,vehicles=2,time_limit=1,verbose=False).solve(s,warm_actions=bad)
 except ValueError as exc:rejected='SCIP rejected complete route start' in str(exc)
 assert rejected
 (out/'report.json').write_text(json.dumps(dict(status='passed',rows=rows,invalid_three_plus_five_warm_start_rejected=True),indent=2));signal.alarm(0);print('PASS SCIP boundary and real-instance regression',flush=True)
if __name__=='__main__':main()
