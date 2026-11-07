"""Adversarial waiting/deletion and exhaustive tiny-instance cross checks."""
from bootstrap import *
from route_engine import RouteEngine
import itertools,json,time
P=Path(__file__).resolve().parent


def main():
 out=P/'boundary_v1';out.mkdir(exist_ok=False);begin=time.monotonic()
 n=4;s=dict(depot=torch.tensor([0.,0.]),loc=torch.tensor([[.1,0.]]*n+[[.2,0.]]*n),time_windows=torch.tensor([[10.,12.],[10.65,12.],[10.,12.],[10.,12.],[11.2,16.],[10.,16.],[10.,16.],[10.,16.]]),node_type=torch.tensor([1,0,0,0,1,0,0,0]),demand_passenger=torch.tensor([.1,0.,0.,0.,-.1,0.,0.,0.]),demand_cargo=torch.tensor([0.,.1,.1,.1,0.,-.1,-.1,-.1]))
 engine=RouteEngine(s,C,PROFILE,1);r=(2,1,5,6,3,7,4,8);assert engine.cost([r]) is not None
 shorter=tuple(x for x in r if x not in (2,6));a=audit(s,list(shorter)+[0,9],audit_config(C,4,1));assert any(v['kind']=='passenger_excess' for v in a['violations']) and engine.route(shorter) is None
 checked=valid=0;best=None
 for perm in itertools.permutations(range(1,9)):
  pos={v:i for i,v in enumerate(perm)}
  if any(pos[i]>pos[i+4] for i in range(1,5)):continue
  a2=audit(s,list(perm)+[0],audit_config(C,4,1));ok=not a2['violations'] and not a2['unfulfilled'];m=engine.cost([perm]);assert (m is not None)==ok,(perm,a2,m)
  checked+=1
  if ok:
   valid+=1;reference=scalar_cost(s,list(perm)+[0],C,PROFILE);assert abs(m['objective']-reference['objective'])<1e-8
   best=min(best,m['objective']) if best is not None else m['objective']
 assert engine.cost([])['completed']==0
 assert engine.cost([r,r]) is None
 report=dict(status='passed',checked_pair_precedence_permutations=checked,valid_permutations=valid,best_objective=best,waiting_deletion_counterexample=dict(original=list(r),removed_pair=2,audit=a),seconds=time.monotonic()-begin,scope='tiny exhaustive implementation check; no general optimality claim')
 (out/'report.json').write_text(json.dumps(report,indent=2));print('PASS boundary checks',checked,'permutations;',valid,'feasible',flush=True)
if __name__=='__main__':main()
