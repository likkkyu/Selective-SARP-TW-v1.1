"""Hard-four certificates on tight windows, late dispatch, capacity and batches."""
from pathlib import Path
import json,sys,signal
P=Path(__file__).resolve().parent;ROOT=P.parents[1]
sys.path.insert(0,str(ROOT/'experiments/dispatch_clock_fix_20261106'))
from bootstrap import *
P=Path(__file__).resolve().parent
from clock_patch import install as clock
clock()
from completion_guard import install as guard
guard()
from admission_targeted import install,clear,STATS
from route_audit import audit

def toy(n,start=10.,width=.2,split=False,passenger=False,demand=.1):
 starts=[start]*n
 if split:starts=[10.]*(n//2)+[14.]*(n-n//2)
 return dict(depot=torch.tensor([0.,0.]),loc=torch.tensor([[.1,0.]]*n+[[.2,0.]]*n),time_windows=torch.tensor([[x,x+width] for x in starts]+[[10.,16.]]*n),node_type=torch.full((2*n,),int(passenger)),demand_passenger=torch.tensor([demand]*n+[-demand]*n) if passenger else torch.zeros(2*n),demand_cargo=torch.zeros(2*n) if passenger else torch.tensor([demand]*n+[-demand]*n),n_orders=torch.tensor(n))

def main():
 out=P/'validation_v1';out.mkdir(exist_ok=False);signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('180s admission validation')));signal.alarm(180);install();rows=[]
 cases=[('tight_four',toy(4),4),('late_four',toy(4,start=14.),4),('only_three_decline',toy(3),0),('two_plus_two_cannot_dispatch',toy(4,split=True),0),('eight_two_trips',toy(8,split=True),8),('capacity_impossible',toy(4,passenger=True,demand=1.1),0)]
 for name,s,expected in cases:
  clear();n=len(s['loc'])//2;st=State.initialize(sample_to_batch(s),**KW);actions=[]
  for step in range(4*n+20):
   if st.all_finished():break
   mask=st.get_mask();available=(~mask[0,0]).nonzero().flatten().tolist();real=[x for x in available if 1<=x<=2*n]
   # Arbitrary deterministic preference, never directly execute the witness.
   node=real[-1] if real else 0 if 0 in available else 2*n+1
   assert node in available;actions.append(node);st=st.update(torch.tensor([node]),current_mask=mask)
  assert st.all_finished(),name
  cfg=dict(start=10.,end=16.,area=10.,speed=25.,trip=3.,service=.05,min_dispatch=4,hard_cargo=True,ride=70/60,excess=.5,max_open=6,vehicles=max(1,(n+5)//6))
  a=audit(s,actions,cfg);assert not a['violations'] and not a['unfulfilled'] and a['completed']==expected,(name,a,actions)
  rows.append(dict(case=name,audit=a,actions=actions,stats=dict(STATS)));print('PASS',name,a['completed'],flush=True)
 # Mixed rows exercise global ids and row slicing independently.
 from torch.utils.data import default_collate
 clear();batch=default_collate([toy(4),toy(4,start=14.)]);st=State.initialize(batch,**KW);mask=st.get_mask();assert (~mask[0,0,1:5]).any() and (~mask[1,0,1:5]).any()
 report=dict(status='passed',scope='bounded constructive suffix validation; not completeness or model quality proof',rows=rows,mixed_rows_initial_dispatch_certified=True)
 (out/'report.json').write_text(json.dumps(report,indent=2));signal.alarm(0);print('PASS admission cases and mixed rows')
if __name__=='__main__':main()
