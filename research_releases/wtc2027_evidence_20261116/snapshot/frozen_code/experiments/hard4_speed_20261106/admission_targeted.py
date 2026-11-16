"""Bounded constructive certificates for hard >=4 trips, no learned parameters.

Certificates prove a suffix exists; failure to find one means uncertified, not
mathematically infeasible. A stored, actually replayed suffix preserves progress.
"""
from collections import OrderedDict
from dataclasses import dataclass,replace
import math,inspect,textwrap
from pathlib import Path
AUDIT_TARGET_QUERIES=False
VERSION=3
STATIC={};WITNESS=OrderedDict()
STATS={k:0 for k in ['calls','candidates','certified','uncertified','search_nodes','verification_failures','cache_hits','fails_closed','target_queries','target_full_fallbacks','target_verified']}
MAX_SEARCH_NODES=2000

def clear():
 STATIC.clear();WITNESS.clear()
 for k in STATS:STATS[k]=0

@dataclass(frozen=True)
class Position:
 node:int
 now:float
 departure:float
 completed:int
 passenger:float
 cargo:float
 status:tuple
 pickup_finish:tuple

class Problem:
 def __init__(self,state,b):
  from problem_mcvrptw_v2 import Config as C
  idx=int(state.ids[b]);self.n=state.n_orders
  self.xy=state.coords[idx].detach().cpu().tolist();self.tw=state.time_windows[idx].detach().cpu().tolist()
  self.types=state.node_type[idx].detach().cpu().tolist();self.dp=state.demand_passenger[idx].detach().cpu().tolist();self.dc=state.demand_cargo[idx].detach().cpu().tolist()
  self.travel=[[math.dist(x,y)*state.AREA_SIZE/state.VEHICLE_SPEED for y in self.xy] for x in self.xy]
  self.start=state.OPERATION_START;self.end=state.OPERATION_END;self.trip=state.MAX_TRIP_TIME;self.service=state.SERVICE_TIME
  self.ride=C.PASSENGER_MAX_RIDE_TIME_MINUTES/60;self.excess=C.PASSENGER_MAX_EXCESS_RIDE_TIME_MINUTES/60;self.open_limit=state.max_concurrent_open_orders
  self.hard_cargo=C.HARD_CARGO_PICKUP_TIMEWINDOW
  # Retain tensors so their identities cannot be reused for a different batch.
  self.owner=state.coords;self.token=(id(state.coords),idx)
 def capture(self,state,b):
  n=self.n;vis=state.visited_[b,0].detach().cpu().tolist();picked=state.picked_up_[b,0].detach().cpu().tolist();rej=state.rejected_[b,0].detach().cpu().tolist()
  status=tuple(3 if rej[i] else 2 if vis[n+i+1] else 1 if picked[i] else 0 for i in range(n))
  return Position(int(state.prev_a[b]),float(state.current_time[b]),float(state.trip_start_time[b]),int(state.served_orders_since_dispatch[b]),float(state.used_capacity_passenger[b]),float(state.used_capacity_cargo[b]),status,tuple(state.passenger_pickup_time[b,0].detach().cpu().tolist()))
 def step(self,s,node):
  # Tight reserve for scalar certificates; every suffix is also replayed through
  # the actual float32 State + common masks before it can certify a candidate.
  eps=1e-7;n=self.n
  if node==0:
   if s.node==0 or 1 in s.status or s.completed<4:return None
   back=s.now+self.travel[s.node][0]
   return s if back<=self.end-eps and back-s.departure<=self.trip-eps else None
  i=(node-1)%n;pickup=node<=n
  if pickup and s.status[i]!=0:return None
  if not pickup and s.status[i]!=1:return None
  depart=s.departure;now=s.now
  if s.node==0:
   if not pickup:return None
   depart=max(now,self.tw[node][0]-self.travel[0][node]);now=depart
  start=max(now+self.travel[s.node][node],self.tw[node][0]);finish=start+self.service
  if pickup and (self.types[node]==1 or self.hard_cargo) and start>self.tw[node][1]-eps:return None
  if not pickup and self.types[node]==1:
   duration=start-s.pickup_finish[i]
   if duration>self.ride-eps or duration-self.travel[i+1][i+n+1]>self.excess-eps:return None
  lp=s.passenger+self.dp[node];lc=s.cargo+self.dc[node]
  if lp< -1e-6 or lc< -1e-6 or lp>1+1e-7 or lc>1+1e-7:return None
  status=list(s.status);status[i]=1 if pickup else 2
  if status.count(1)>self.open_limit:return None
  back=finish+self.travel[node][0]
  if back>self.end-eps or back-depart>self.trip-eps:return None
  pt=list(s.pickup_finish)
  if pickup:pt[i]=finish
  return Position(node,finish,depart,s.completed+int(not pickup),lp,lc,tuple(status),tuple(pt))
 def search(self,s,budget):
  budget[0]+=1;STATS['search_nodes']+=1
  if budget[0]>MAX_SEARCH_NODES:return None
  opened=[i for i,x in enumerate(s.status) if x==1]
  if not opened and s.completed>=4:return (0,) if self.step(s,0) is not None else None
  untouched=[i for i,x in enumerate(s.status) if x==0]
  if s.completed+len(opened)+len(untouched)<4:return None
  # Explore mixed pickup/delivery suffixes, so four tight-window pickups may
  # precede delivery. Search is bounded; an exhausted budget is uncertified.
  choices=[]
  for i in opened:
   deadline=min(self.end,s.pickup_finish[i]+min(self.ride,self.travel[i+1][i+self.n+1]+self.excess)) if self.types[i+1]==1 else self.end
   choices.append((deadline,1,i+self.n+1))
  if s.completed+len(opened)<4:
   choices.extend((self.tw[i+1][1],0,i+1) for i in untouched)
  for _,_,node in sorted(choices):
   after=self.step(s,node)
   if after is None:continue
   tail=self.search(after,budget)
   if tail is not None:return (node,)+tail
  return None

def _remember(key,value):
 WITNESS[key]=value;WITNESS.move_to_end(key)
 while len(WITNESS)>50000:WITNESS.popitem(last=False)

def install():
 import torch
 from state_mcvrptw_v2 import StateMCVRPPDTW as State
 if getattr(State,'_hard4_admission_version',None)==VERSION:return
 if not hasattr(State,'_before_completion_guard'):raise RuntimeError('Install corrected clock and completion_guard before admission')
 original=State.get_mask
 # A certificate needs one action bit, not the commitments of every other
 # pickup. Preserve basic masks and selected pickup commitment exactly.
 raw=State._before_completion_guard
 code=textwrap.dedent(inspect.getsource(raw))
 before='def get_mask(self, return_debug=False, skip_pickup_commitment=False, benchmark_stats=None, return_urgency_slack=False, aux_outputs=None):'
 after='def target_mask(self, target_node, return_debug=False, skip_pickup_commitment=False, benchmark_stats=None, return_urgency_slack=False, aux_outputs=None):'
 if code.count(before)!=1:raise RuntimeError('Unexpected frozen mask signature')
 code=code.replace(before,after)
 anchor='        pickup_physical_feasible = ~service_mask[:, :n_orders]'
 inserted=anchor+'\n        selected_only = torch.zeros_like(pickup_physical_feasible)\n        if 1 <= target_node <= n_orders:\n            selected_only[:, target_node-1] = pickup_physical_feasible[:, target_node-1]\n        pickup_physical_feasible = selected_only'
 if code.count(anchor)!=1:raise RuntimeError('Unexpected commitment filter')
 code=code.replace(anchor,inserted)
 path=Path(__file__).resolve().parent/'generated_target_mask_v3.py'
 if path.exists():
  if path.read_text()!=code:raise RuntimeError('Refuse overwrite changed target helper')
 else:path.write_text(code)
 namespace=dict(raw.__globals__);exec(compile(code,str(path),'exec'),namespace);target=namespace['target_mask']
 def selected_mask(current,node):
  STATS['target_queries']+=1;mask=target(current,node)
  # The completion guard can recover a masked delivery. Fall back to the full
  # common function on a negative answer rather than reject a valid recovery.
  if bool(mask[0,0,node]) or (node in (0,current.reject_index) and bool(current.has_open_started_orders().any())):
   STATS['target_full_fallbacks']+=1;mask=original(current)
  if AUDIT_TARGET_QUERIES:
   full=original(current)
   if bool(mask[0,0,node])!=bool(full[0,0,node]):raise AssertionError('Selected-action predicate differs from full mask')
   STATS['target_verified']+=1
  return mask
 def problem(state,b):
  key=(id(state.coords),int(state.ids[b]))
  if key not in STATIC:STATIC[key]=Problem(state,b)
  return STATIC[key]
 def verify(state,sequence,prob):
  current=state;cache=[]
  for offset,node in enumerate(sequence):
   mask=selected_mask(current,node)
   if bool(mask[0,0,node]):return False
   snap=prob.capture(current,0)
   if prob.step(snap,node) is None:return False
   cache.append(((prob.token,snap),tuple(sequence[offset:])))
   if node==0:
    for key,tail in cache:_remember(key,tail)
    return True
   current=current.update(torch.tensor([node],device=mask.device),current_mask=mask)
  return False
 def certified(next_state,prob):
  snap=prob.capture(next_state,0);key=(prob.token,snap)
  if key in WITNESS:
   STATS['cache_hits']+=1;return True
  suffix=prob.search(snap,[0])
  if suffix is None:return False
  if not verify(next_state,suffix,prob):STATS['verification_failures']+=1;return False
  return True
 def guarded(self,*args,**kwargs):
  # Internal debug calls request this to explain original commitment pruning.
  if kwargs.get('skip_pickup_commitment',False):return original(self,*args,**kwargs)
  result=original(self,*args,**kwargs);mask=result[0] if isinstance(result,tuple) else result;mask=mask.clone();STATS['calls']+=1
  finished=self.get_finished()
  for b in range(mask.shape[0]):
   if bool(finished[b]):continue
   one=self[b:b+1];prob=problem(self,b);snap=prob.capture(one,0)
   if snap.node!=0:
    mask[b,:,one.reject_index]=True
    if snap.completed<4 or 1 in snap.status:mask[b,:,0]=True
    elif prob.step(snap,0) is None:mask[b,:,0]=True
   else:
    # Prevent an idle depot escape; pre-execution refusal remains available.
    mask[b,:,0]=True
   base=mask[b:b+1].clone()
   for node in (~base[0,0,1:2*one.n_orders+1]).nonzero().flatten().tolist():
    node+=1;STATS['candidates']+=1
    if prob.step(snap,node) is None:ok=False
    else:
     nxt=one.update(torch.tensor([node],device=mask.device),current_mask=base)
     ok=certified(nxt,prob)
    if ok:STATS['certified']+=1
    else:mask[b,:,node]=True;STATS['uncertified']+=1
   if not (~mask[b]).any():
    STATS['fails_closed']+=1
    raise RuntimeError('No certified hard-four continuation; do not emit an invalid under-four trip')
  return (mask,*result[1:]) if isinstance(result,tuple) else mask
 State._before_hard4_admission=original;State.get_mask=guarded;State._hard4_admission_version=VERSION
