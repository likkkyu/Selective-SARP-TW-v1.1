"""Exact necessary-condition presolve for unreachable optional time windows.

Only pairs that cannot finish service in the operating horizon are fixed out.
Their inactive timing placeholders are made benign before the inherited MILP.
No feasible accepted pair is changed. Original sample retained for route audit.
"""
from common import *
from strict_scip import HardMinDispatchSCIP
import copy


class BoundarySafeSCIP(HardMinDispatchSCIP):
    def solve(self,sample,*args,**kwargs):
        original=lists(sample);safe=copy.deepcopy(original);n=len(original['loc'])//2
        st=self.c['OPERATION_START'];latest=self.c['OPERATION_END']-self.c['SERVICE_TIME']
        forced=[];reasons={}
        for i in range(n):
            why=[]
            if any(original['time_windows'][j][0]>latest for j in (i,i+n)):why.append('earliest service completion beyond operating horizon')
            hard_pickup=original['node_type'][i]==1 or self.c['HARD_CARGO_PICKUP_TIMEWINDOW']
            if hard_pickup and original['time_windows'][i][1]<st:why.append('hard pickup deadline before operating start')
            if why:
                forced.append(i+1);reasons[i+1]=why
                for j in (i,i+n):safe['time_windows'][j]=[st,self.c['OPERATION_END']]
        self.forced_declined=forced
        # Refuse unsupported arbitrary historical soft delivery deadlines rather
        # than silently applying the parent's horizon-sized lateness bounds.
        for i in range(n):
            if i+1 in forced:continue
            if original['time_windows'][i+n][1]<st:
                raise ValueError('Soft delivery deadlines before operation start require separate enlarged lateness bounds; input outside supported domain')
        result=super().solve(safe,*args,**kwargs)
        result.update(optional_window_presolve_version=2,forced_declined_order_ids=forced,forced_decline_reasons=reasons,presolve_equivalence='only analytically unserviceable pairs fixed out; inactive placeholders sanitized; feasible pair timing and all cost coefficients unchanged')
        if result.get('actions') is not None:
            check=scalar_cost(original,result['actions'],self.c,self.profile)
            if max(abs(check[k]-result['replayed'][k]) for k in check)>1e-7:raise RuntimeError('Presolve changed replay cost')
            a=audit(original,result['actions'],audit_config(self.c,n,self.K))
            if a['violations'] or a['unfulfilled']:raise RuntimeError(('Original-instance audit failed',a))
            result['original_instance_audit']=a
        return result

    def build(self,sample):
        m=super().build(sample)
        for i in self.forced_declined:m.addCons(self.z[i]==0)
        self.original_constraints=m.getNConss()
        return m
