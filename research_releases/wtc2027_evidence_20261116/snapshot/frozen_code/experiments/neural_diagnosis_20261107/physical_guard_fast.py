"""Float32-faithful scalar acceleration of physical-certificate guard v4.

CPU/float32 prototype only. Physics and DFS are unchanged from v4; virtual
updates emulate the relevant State fields. Actual real-node updates are always
compared with the predicted float32 state, and fail closed on any mismatch.
Validation can additionally shadow every mask against the slow v4.
"""
import math,struct
from collections import OrderedDict
import torch
import physical_guard as slow
import admission_targeted as legacy
VERSION=5
AUDIT_MASKS=False
STATIC={};WITNESS=OrderedDict()
STATS={k:0 for k in ['calls','candidates','certified','uncertified','cache_hits','verification_steps','verification_failures','actual_transition_checks','transition_mismatches','shadow_masks','fails_closed']}


def f32(x):return struct.unpack('f',struct.pack('f',x))[0]


class Problem(legacy.Problem):
 def __init__(self,state,b):
  super().__init__(state,b)
  if state.coords.device.type!='cpu' or state.coords.dtype!=torch.float32:raise ValueError('Fast v5 requires validated CPU float32 backend')
  xy=state.coords[int(state.ids[b])]
  self.travel32=((xy[:,None,:]-xy[None,:,:]).norm(p=2,dim=-1)*state.AREA_SIZE/state.VEHICLE_SPEED).tolist()
  self.service32=f32(self.service)
 def actual_step(self,s,node):
  if not 1<=node<=2*self.n:raise ValueError('Only real-node transitions are emulated')
  i=(node-1)%self.n;pickup=node<=self.n;travel=self.travel32[s.node][node]
  arrival=f32(s.now+travel);start=max(arrival,self.tw[node][0]);finish=f32(start+self.service32)
  departure=max(s.now,f32(self.tw[node][0]-travel)) if s.node==0 else s.departure
  lp=max(0.,f32(s.passenger+self.dp[node]));lc=max(0.,f32(s.cargo+self.dc[node]))
  status=list(s.status);status[i]=1 if pickup else 2;pt=list(s.pickup_finish)
  if pickup and self.types[node]==1:pt[i]=finish
  completed=0 if s.node==0 else s.completed+int(not pickup)
  return legacy.Position(node,finish,departure,completed,lp,lc,tuple(status),tuple(pt))


def clear():
 STATIC.clear();WITNESS.clear();slow.clear()
 for k in STATS:STATS[k]=0


def problem(state,b):
 key=(id(state.coords),int(state.ids[b]))
 if key not in STATIC:STATIC[key]=Problem(state,b)
 return STATIC[key]


def verify(snap,sequence,prob):
 cache=[]
 for offset,node in enumerate(sequence):
  STATS['verification_steps']+=1
  if prob.step(snap,node) is None:return False
  cache.append(((prob.token,snap),tuple(sequence[offset:])))
  if node==0:
   for key,tail in cache:
    WITNESS[key]=tail;WITNESS.move_to_end(key)
    while len(WITNESS)>50000:WITNESS.popitem(last=False)
   return True
  snap=prob.actual_step(snap,node)
 return False


def certified(snap,prob):
 if (prob.token,snap) in WITNESS:STATS['cache_hits']+=1;return True
 sequence=prob.search(snap,[0])
 if sequence is None:return False
 if not verify(snap,sequence,prob):STATS['verification_failures']+=1;return False
 return True


def install():
 from state_mcvrptw_v2 import StateMCVRPPDTW as State
 from problem_mcvrptw_v2 import Config
 if getattr(State,'_physical_certificate_version',None)==VERSION:return
 slow.install();reference=State.get_mask;original_update=State.update
 def guarded(self,return_debug=False,skip_pickup_commitment=False,benchmark_stats=None,return_urgency_slack=False,aux_outputs=None):
  if return_debug or return_urgency_slack:raise ValueError('Fast v5 does not supply legacy debug/urgency diagnostics')
  STATS['calls']+=1;n=self.n_orders;mask=torch.ones((self.ids.size(0),1,2*n+2),dtype=torch.bool,device=self.coords.device)
  finished=self.get_finished()
  for b in range(self.ids.size(0)):
   if bool(finished[b]):mask[b,0,0]=False;continue
   prob=problem(self,b);snap=prob.capture(self,b)
   if snap.node==0:
    if bool(self.allow_reject) and 0 in snap.status:mask[b,0,2*n+1]=False
    can_depart=float(self.used_vehicles[b])<max(1,math.ceil(n*Config.DEFAULT_NUM_VEHICLE_RATIO)) and snap.status.count(0)>=4
   else:
    can_depart=True
    if prob.step(snap,0) is not None:mask[b,0,0]=False
   if can_depart:
    for node in range(1,2*n+1):
     if prob.step(snap,node) is None:continue
     STATS['candidates']+=1;after=prob.actual_step(snap,node)
     if certified(after,prob):mask[b,0,node]=False;STATS['certified']+=1
     else:STATS['uncertified']+=1
   if bool(mask[b].all()):STATS['fails_closed']+=1;raise RuntimeError('No certified hard4 continuation')
  if AUDIT_MASKS:
   expected=reference(self,return_debug=False);STATS['shadow_masks']+=self.ids.size(0)
   if not torch.equal(mask,expected):raise AssertionError('Fast v5 mask differs from slow v4')
  return mask
 def update(self,selected,current_mask=None):
  predictions=[]
  for b in range(self.ids.size(0)):
   node=int(selected[b]);prob=problem(self,b)
   predictions.append(prob.actual_step(prob.capture(self,b),node) if 1<=node<=2*self.n_orders else None)
  result=original_update(self,selected,current_mask=current_mask)
  for b,prediction in enumerate(predictions):
   if prediction is None:continue
   STATS['actual_transition_checks']+=1
   if prediction!=problem(result,b).capture(result,b):STATS['transition_mismatches']+=1;raise RuntimeError('Float32 state transition mismatch; do not continue')
  return result
 State.get_mask=guarded;State.update=update;State._physical_certificate_version=VERSION;State._hard4_admission_version=VERSION
