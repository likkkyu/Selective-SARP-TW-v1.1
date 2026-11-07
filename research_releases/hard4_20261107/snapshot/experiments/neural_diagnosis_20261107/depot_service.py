"""Separate policy variant: prefer a certified service action over returning.

No physical action is unmasked. This heuristic is separate from learned
rejection, shared with both neural arms, and may affect route quality.
"""
import torch
STATS={'depot_suppressed':0}


def install():
 from nets.attention_model import AttentionModel
 if hasattr(AttentionModel,'_before_depot_service_priority'):return
 original=AttentionModel._get_log_p;AttentionModel._before_depot_service_priority=original
 def controlled(self,fixed,state,*args,**kwargs):
  logp,mask,debug=original(self,fixed,state,*args,**kwargs)
  service=torch.isfinite(logp[...,1:-1]).any(-1);change=service & torch.isfinite(logp[...,0])
  STATS['depot_suppressed']+=int(change.sum())
  result=logp.clone();result[...,0]=result[...,0].masked_fill(change,-torch.inf)
  result=result-torch.logsumexp(result,dim=-1,keepdim=True)
  return result,mask,debug
 AttentionModel._get_log_p=controlled
