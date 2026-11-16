"""Versioned inference safety recovery; does not alter training/checkpoints.

On an open-order dead end, search complete delivery-only routes in float64
using the existing state physical tolerance (1e-5 hours). Fail closed if none
exists. Never return to depot or reject while carrying open orders.
"""
import math, inspect, textwrap
from problem_mcvrptw_v2 import Config

VERSION = 1
STATS = {'recoveries': 0, 'failed_recoveries': 0}

def completion_order(coords, current, now, trip_start, orders, pickup_finish,
                     passenger, windows, n, area, speed, service, trip, end,
                     ride, excess, tolerance=1e-5):
    def travel(a,b):return math.dist(coords[a],coords[b])*area/speed
    def search(node,time,remaining):
        if not remaining:
            back=time+travel(node,0)
            return () if back<=end+tolerance and back-trip_start<=trip+tolerance else None
        for i in sorted(remaining):
            target=n+i+1
            start=max(time+travel(node,target),windows[target-1][0])
            finish=start+service
            if passenger[i]:
                duration=start-pickup_finish[i]
                if duration>ride+tolerance or duration-travel(i+1,target)>excess+tolerance:continue
            if finish+travel(target,0)>end+tolerance or finish+travel(target,0)-trip_start>trip+tolerance:continue
            tail=search(target,finish,remaining-{i})
            if tail is not None:return (i,)+tail
        return None
    return search(current,now,set(orders))

def install():
    from state_mcvrptw_v2 import StateMCVRPPDTW as State
    if hasattr(State,'_before_completion_guard'):return
    # Reserve 0.036 seconds in scalar lookahead to prevent float32 state
    # rounding from invalidating a promised completion. Audit unchanged.
    import state_mcvrptw_v2 as module
    for name in ['_delivery_sequence_feasible','_delivery_step_feasible','_completion_step_feasible_from_cache','_cached_completion_problem_step_feasible']:
        source=textwrap.dedent(inspect.getsource(getattr(State,name)))
        assert '+ 1e-5' in source
        ns=dict(module.__dict__)
        exec(source.replace('+ 1e-5','- 1e-5'),ns)
        setattr(State,name,ns[name])
    original=State.get_mask
    State._before_completion_guard=original
    def guarded(self,*args,**kwargs):
        result=original(self,*args,**kwargs)
        mask=result[0] if isinstance(result,tuple) else result
        opened=self.get_open_started_mask()
        if not opened.any():return result
        mask=mask.clone()
        n=self.n_orders
        for b in range(mask.shape[0]):
            orders=opened[b].nonzero().flatten().tolist()
            if not orders:continue
            mask[b,:,0]=True
            mask[b,:,self.reject_index]=True
            if (~mask[b,:,1:2*n+1]).any():continue
            if len(orders)>6:raise RuntimeError('Guard supports at most six open orders')
            idx=int(self.ids[b].item())
            coords=self.coords[idx].detach().cpu().tolist()
            route=completion_order(coords,int(self.prev_a[b].item()),float(self.current_time[b].item()),float(self.trip_start_time[b].item()),orders,self.passenger_pickup_time[b,0].detach().cpu().tolist(),(self.node_type[idx,1:n+1]==1).tolist(),self.time_windows[idx,1:].detach().cpu().tolist(),n,self.AREA_SIZE,self.VEHICLE_SPEED,self.SERVICE_TIME,self.MAX_TRIP_TIME,self.OPERATION_END,Config.PASSENGER_MAX_RIDE_TIME_MINUTES/60,Config.PASSENGER_MAX_EXCESS_RIDE_TIME_MINUTES/60)
            if route is None:
                STATS['failed_recoveries']+=1
                raise RuntimeError(f'No safe open-order completion: row={b},node={int(self.prev_a[b].item())},open={orders}')
            mask[b,:,n+route[0]+1]=False
            STATS['recoveries']+=1
        if isinstance(result,tuple):return (mask,*result[1:])
        return mask
    State.get_mask=guarded
