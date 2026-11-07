"""Small exact reference: enumerate ALL legal accepted subsets and PD orders.

N<=5 has one independent dispatch under ceil(N/6). Every nonempty accepted
subset must contain >=4 complete requests. No neural masks or SCIP constraints
are used by enumeration. Oracle warm starts validate representation, not speed.
Run only after timed main methods stop competing for the CPU.
"""
from pathlib import Path
import sys,json,time,itertools,signal,hashlib
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'experiments/hard4_confirmation_20261106'))
from common import *
from scip_boundary_v2 import BoundarySafeSCIP
P=Path(__file__).resolve().parent


def pd_orders(ids,n):
 def visit(unpicked,opened,path):
  if not unpicked and not opened:yield path;return
  for i in sorted(unpicked):yield from visit(unpicked-{i},opened|{i},path+[i])
  for i in sorted(opened):yield from visit(unpicked,opened-{i},path+[i+n])
 yield from visit(set(ids),set(),[])


def exact(sample):
 s=lists(sample);n=len(s['loc'])//2;best=scalar_cost(s,[2*n+1]*n,C,PROFILE);actions=[2*n+1]*n;checked=1;feasible=1
 for size in range(4,n+1):
  for chosen in itertools.combinations(range(1,n+1),size):
   for route in pd_orders(chosen,n):
    candidate=route+[0]+[2*n+1]*(n-size);checked+=1;a=audit(s,candidate,audit_config(C,n,1))
    if a['violations'] or a['unfulfilled']:continue
    feasible+=1;score=scalar_cost(s,candidate,C,PROFILE)
    if score['objective']<best['objective']:best=score;actions=candidate
 return dict(actions=actions,cost=best,enumerated_plans=checked,feasible_plans=feasible,audit=audit(s,actions,audit_config(C,n,1)))


def toy(n,late=False,mixed=False):
 start=14. if late else 10.
 types=[1,1]+[0]*(n-2) if mixed else [0]*n
 dp=[.6 if t else 0. for t in types];dc=[0. if t else .2 for t in types]
 return dict(depot=torch.tensor([0.,0.]),loc=torch.tensor([[.1,0.]]*n+[[.2,0.]]*n),time_windows=torch.tensor([[start,start+1.]]*n+[[start,start+.02]]*n),node_type=torch.tensor(types*2),demand_passenger=torch.tensor(dp+[-x for x in dp]),demand_cargo=torch.tensor(dc+[-x for x in dc]),n_orders=torch.tensor(n))


def subset(sample,n=5):
 old_n=len(sample['loc'])//2;idx=list(range(n))+list(range(old_n,old_n+n));s={}
 for k,v in sample.items():
  if k in ['loc','time_windows','node_type','demand_passenger','demand_cargo']:s[k]=v[idx].clone()
  elif k=='n_orders':s[k]=torch.tensor(n)
  else:s[k]=v.clone() if hasattr(v,'clone') else v
 return s


def main():
 out=P/'reference_v1';out.mkdir(exist_ok=False);begin=time.monotonic()
 signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('300s exact-reference total cap')));signal.alarm(300)
 data=torch.load(ROOT/'experiments/paper_delivery_20261104/development_n50_seed614729.pt',map_location='cpu',weights_only=False)['data']
 cases=[('three_requests_no_dispatch',toy(3)),('four_late_dispatch',toy(4,late=True)),('five_cargo_piecewise_delay',toy(5)),('five_mixed_capacity_ride',toy(5,mixed=True)),('development0_first_five',subset(data[0])),('development1_first_five',subset(data[1]))]
 plan=dict(scope='small-instance exact model verification, not main benchmark, not untouched test or cold-solve speed claim',cases=[dict(label=label,orders=len(s['loc'])//2) for label,s in cases],enumeration='all accepted subsets of size>=4 plus all-reject; all pickup-before-delivery permutations, one trip',cold_solver_seconds=5,oracle_warm_solver_seconds=2,total_timeout_seconds=300,physical_config=C,normalization_profile=PROFILE,source_sha256={str(f.relative_to(ROOT)):hashlib.sha256(f.read_bytes()).hexdigest() for f in [Path(__file__),ROOT/'experiments/hard4_confirmation_20261106/scip_boundary_v2.py',ROOT/'experiments/min_dispatch_hard_20261106/strict_scip.py',ROOT/'experiments/mip_scip_20261106/scip_selective.py']})
 (out/'plan.json').write_text(json.dumps(plan,indent=2));torch.save(dict(cases=cases,scope=plan['scope']),out/'cases.pt');report=dict(status='running',scope=plan['scope'],rows=[])
 def save():report['seconds']=time.monotonic()-begin;(out/'report.json').write_text(json.dumps(report,indent=2))
 save()
 try:
  for label,s in cases:
   n=len(s['loc'])//2;Config.set_normalization_profile(n,dict(PROFILE,graph_size=n));t=time.monotonic();oracle=exact(s);enum_seconds=time.monotonic()-t
   assert not oracle['audit']['violations'] and oracle['audit']['unfulfilled']==0
   with torch.no_grad():j,d=MCVRPPDTW.get_costs(sample_to_batch(s),torch.tensor([oracle['actions']],dtype=torch.long),return_details=True)
   assert abs(float(j)-oracle['cost']['objective'])<.02 and abs(float(d['total_cost_raw'])-oracle['cost']['raw_cost'])<.02
   cold=BoundarySafeSCIP(C,PROFILE,vehicles=1,time_limit=5,threads=1,verbose=False).solve(s,logfile=out/f'{label}_cold.log')
   warm=BoundarySafeSCIP(C,PROFILE,vehicles=1,time_limit=2,threads=1,verbose=False).solve(s,warm_actions=oracle['actions'],logfile=out/f'{label}_oracle_warm.log')
   assert cold['actions'] is not None and warm['actions'] is not None
   for result in [cold,warm]:
    assert result['objective']>=oracle['cost']['objective']-.02
    assert result['dual_bound']<=oracle['cost']['objective']+.02
    assert not result['original_instance_audit']['violations'] and not result['original_instance_audit']['unfulfilled']
   assert abs(warm['objective']-oracle['cost']['objective'])<.02
   report['rows'].append(dict(case=label,orders=n,oracle=oracle,enumeration_seconds=enum_seconds,cold_solver=cold,oracle_warm_solver=warm,tensor_objective_error=abs(float(j)-oracle['cost']['objective']),tensor_raw_error=abs(float(d['total_cost_raw'])-oracle['cost']['raw_cost'])));save()
   print('EXACT',label,'enumerated',oracle['enumerated_plans'],'served',oracle['audit']['completed'],'J',round(oracle['cost']['objective'],6),'cold_status',cold['status'],'warm_status',warm['status'],flush=True)
  report['status']='passed'
 except BaseException as exc:report.update(status='failed',error=repr(exc));raise
 finally:save();signal.alarm(0)
if __name__=='__main__':main()
