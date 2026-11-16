"""Isolated hard4 physical-certificate deployment prototype, version 4.

This replaces legacy heuristic mask pruning. It is NOT claimed equivalent to
the old mask. Physical parameters and weights stay fixed. Every allowed real
action has a same-trip >=4 completion certificate verified on float32 State
transitions using independent scalar physical checks, without calling the old
full masks at virtual steps. Final plans still require external full audits.
"""
import math
from collections import OrderedDict
import torch
import admission_targeted as legacy

VERSION=4
STATIC={};WITNESS=OrderedDict()
STATS={k:0 for k in ['calls','candidates','certified','uncertified','cache_hits','physical_verification_steps','physical_verification_failures','fails_closed']}


def clear():
 STATIC.clear();WITNESS.clear();legacy.clear()
 for k in STATS:STATS[k]=0


def remember(key,tail):
 WITNESS[key]=tail;WITNESS.move_to_end(key)
 while len(WITNESS)>50000:WITNESS.popitem(last=False)


def problem(state,b):
 key=(id(state.coords),int(state.ids[b]))
 if key not in STATIC:STATIC[key]=legacy.Problem(state,b)
 return STATIC[key]


def verify(state,sequence,prob,update):
 current=state;cache=[];n=prob.n
 blank=torch.zeros((1,1,2*n+2),dtype=torch.bool,device=state.coords.device)
 for offset,node in enumerate(sequence):
  snap=prob.capture(current,0);STATS['physical_verification_steps']+=1
  if prob.step(snap,node) is None:return False
  cache.append(((prob.token,snap),tuple(sequence[offset:])))
  if node==0:
   for key,tail in cache:remember(key,tail)
   return True
  # Real-node transitions do not use other mask bits. Rejection and return
  # are never updated with this placeholder mask during virtual verification.
  current=update(current,torch.tensor([node],device=state.coords.device),current_mask=blank)
 return False


def certified(state,prob,update):
 snap=prob.capture(state,0);key=(prob.token,snap)
 if key in WITNESS:STATS['cache_hits']+=1;return True
 sequence=prob.search(snap,[0])
 if sequence is None:return False
 if not verify(state,sequence,prob,update):STATS['physical_verification_failures']+=1;return False
 return True


def install():
 from state_mcvrptw_v2 import StateMCVRPPDTW as State
 from problem_mcvrptw_v2 import Config
 if getattr(State,'_physical_certificate_version',None)==VERSION:return
 original_update=State.update;State._before_physical_certificate_mask=State.get_mask
 State._before_physical_certificate_update=original_update
 def guarded(self,return_debug=False,skip_pickup_commitment=False,benchmark_stats=None,return_urgency_slack=False,aux_outputs=None):
  if return_debug or return_urgency_slack:raise ValueError('Physical v4 prototype does not provide legacy debug/urgency diagnostics')
  # skip_pickup_commitment never weakens the hard physical-certificate check.
  STATS['calls']+=1;n=self.n_orders;batch=self.ids.size(0)
  mask=torch.ones((batch,1,2*n+2),dtype=torch.bool,device=self.coords.device)
  finished=self.get_finished()
  for b in range(batch):
   if bool(finished[b]):mask[b,0,0]=False;continue
   one=self[b:b+1];prob=problem(self,b);snap=prob.capture(one,0)
   if snap.node==0:
    if bool(self.allow_reject) and 0 in snap.status:mask[b,0,2*n+1]=False
    limit=max(1,math.ceil(n*Config.DEFAULT_NUM_VEHICLE_RATIO))
    can_depart=float(one.used_vehicles[0])<limit and snap.status.count(0)>=4
   else:
    can_depart=True
    if prob.step(snap,0) is not None:mask[b,0,0]=False
   if can_depart:
    blank=torch.zeros((1,1,2*n+2),dtype=torch.bool,device=self.coords.device)
    for node in range(1,2*n+1):
     if prob.step(snap,node) is None:continue
     STATS['candidates']+=1
     after=original_update(one,torch.tensor([node],device=self.coords.device),current_mask=blank)
     if certified(after,prob,original_update):mask[b,0,node]=False;STATS['certified']+=1
     else:STATS['uncertified']+=1
   if bool(mask[b].all()):STATS['fails_closed']+=1;raise RuntimeError('No physical hard4 continuation; refuse invalid or residual plan')
  return mask
 def update(self,selected,current_mask=None):
  result=original_update(self,selected,current_mask=current_mask)
  # Legacy deadlock heuristics are not physical termination conditions. With
  # no idle depot action and certified real actions, progress is monotonic;
  # finish only after all requests are completed/declined and vehicle is back.
  return result._replace(deadlock_count=torch.zeros_like(result.deadlock_count),terminal_=torch.zeros_like(result.terminal_))
 State.get_mask=guarded;State.update=update;State._physical_certificate_version=VERSION;State._hard4_admission_version=VERSION
