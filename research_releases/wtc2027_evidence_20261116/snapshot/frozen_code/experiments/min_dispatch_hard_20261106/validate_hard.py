"""Strict >=4 oracle, forced invalid partitions and acceptance/decline tests."""
import itertools,json,signal
from bootstrap import *
from strict_scip import HardMinDispatchSCIP

def toy(n,split=False):
 starts=[10.]*n
 if split:starts=[10.]*(n//2)+[14.]*(n-n//2)
 return dict(depot=torch.tensor([0.,0.]),loc=torch.tensor([[.1,0.]]*n+[[.2,0.]]*n),time_windows=torch.tensor([[x,x+.2] for x in starts]+[[10.,16.]]*n),node_type=torch.zeros(2*n,dtype=torch.long),demand_passenger=torch.zeros(2*n),demand_cargo=torch.tensor([.1]*n+[-.1]*n),n_orders=torch.tensor(n))

def oracle4(s):
 n=len(s['loc'])//2;best=scalar_cost(s,[2*n+1]*n,C,PROFILE);count=1
 for perm in itertools.permutations(range(1,2*n+1)):
  pos={v:i for i,v in enumerate(perm)}
  if any(pos[i]>pos[i+n] for i in range(1,n+1)):continue
  actions=list(perm)+[0];a=audit(s,actions,audit_config(C,n,1))
  if a['violations'] or a['unfulfilled']:continue
  count+=1;r=scalar_cost(s,actions,C,PROFILE)
  if r['objective']<best['objective']:best=r
 return best,count

def main():
 out=P/'validation_v4';out.mkdir(exist_ok=False);signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('180s strict MIP validation budget')));signal.alarm(180);rows=[]
 for label,n,k,split in [('under_four_all_declined',3,1,False),('exact_four',4,1,False),('four_split_incompatible_two_plus_two',4,2,True),('eight_two_four_order_trips',8,2,True)]:
  s=toy(n,split);solver=HardMinDispatchSCIP(C,PROFILE,vehicles=k,time_limit=5 if n==8 else 20,verbose=False)
  warm=[1,2,3,4,9,10,11,12,0,5,6,7,8,13,14,15,16,0] if n==8 else None
  r=solver.solve(s,warm_actions=warm);a=audit(s,r['actions'],audit_config(C,n,k));assert not a['violations'] and not a['unfulfilled'],a
  if n==3:assert r['completed']==0 and r['status']=='optimal'
  elif n==4:
   o,count=oracle4(s);assert abs(o['objective']-r['objective'])<.02 and r['status']=='optimal'
   assert r['completed']==(0 if split else 4)
  else:assert r['completed']==8 and sorted(r['completed_per_trip'])==[4,4]
  Config.set_normalization_profile(n,dict(PROFILE,graph_size=n))
  with torch.no_grad():j,d=MCVRPPDTW.get_costs(sample_to_batch(s),torch.tensor([r['actions']],dtype=torch.long),return_details=True)
  assert abs(float(j)-r['objective'])<.02 and abs(float(d['total_cost_raw'])-r['replayed']['raw_cost'])<.02
  rows.append(dict(case=label,result=r,audit=a));print('PASS',label,r['completed'],r['completed_per_trip'],flush=True)
 # All eight accepted but split 3+5 is forbidden, including the last trip.
 s=toy(8);solver=HardMinDispatchSCIP(C,PROFILE,vehicles=2,time_limit=10,verbose=False);m=solver.build(s);routes=[[1,2,3,9,10,11],[4,5,6,7,8,12,13,14,15,16]];edges=set()
 for route in routes:
  path=[0]+route+[0];edges.update(zip(path,path[1:]))
 for edge,v in solver.x.items():m.addCons(v==int(edge in edges))
 for v in solver.z.values():m.addCons(v==1)
 m.optimize();assert str(m.getStatus())=='infeasible';rows.append(dict(case='three_plus_five_forbidden',status='infeasible'))
 (out/'report.json').write_text(json.dumps(dict(status='passed',rows=rows),indent=2));signal.alarm(0);print('PASS all hard-minimum tests',flush=True)
if __name__=='__main__':main()
