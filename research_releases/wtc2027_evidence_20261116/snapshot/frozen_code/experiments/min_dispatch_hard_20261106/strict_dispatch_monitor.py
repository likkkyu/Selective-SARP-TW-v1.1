"""Fail-closed execution monitor, not an admission/completion planning algorithm."""
VERSION=1
class IncompleteDispatchError(RuntimeError):pass

def install():
 import state_mcvrptw_v2 as module
 State=module.StateMCVRPPDTW
 if getattr(State,'_hard_minimum_monitor_version',None)==VERSION:return
 original=State.get_mask
 def guarded(self,*args,**kwargs):
  result=original(self,*args,**kwargs);mask=result[0] if isinstance(result,tuple) else result;mask=mask.clone()
  active=(self.prev_a!=0).squeeze(1);below=(self.served_orders_since_dispatch<4).squeeze(1);restricted=active&below
  mask[restricted,:,0]=True
  opened=self.get_open_started_count().squeeze(1)
  for b in range(mask.shape[0]):
   if not bool(restricted[b]) or int(opened[b])>0:continue
   if not (~mask[b,:,1:2*self.n_orders+1]).any():
    raise IncompleteDispatchError(f'Hard minimum four cannot be completed from this prefix: row={b}, served={int(self.served_orders_since_dispatch[b])}, used_vehicles={float(self.used_vehicles[b])}; reject-only/depot escape is not a feasible solution')
  return (mask,*result[1:]) if isinstance(result,tuple) else mask
 State._before_hard_minimum_monitor=original;State.get_mask=guarded;State._hard_minimum_monitor_version=VERSION
