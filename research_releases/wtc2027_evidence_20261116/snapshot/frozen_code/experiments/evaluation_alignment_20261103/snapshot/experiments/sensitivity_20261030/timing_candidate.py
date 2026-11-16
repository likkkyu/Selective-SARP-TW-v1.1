"""Isolated candidate; never imported by active training."""
import inspect,textwrap
import torch
import problem_mcvrptw_v2 as problem

def install():
    cls=problem.MCVRPPDTW
    source=textwrap.dedent(inspect.getsource(cls._compute_time_and_delay)).replace('@staticmethod\n','')
    old='current_time = torch.where(leaving_depot, dispatch_time, current_time)'
    assert old in source
    source=source.replace(old,'current_time = torch.where(leaving_depot, dispatch_time + depart_travel, current_time)')
    scope={};exec(source,vars(problem),scope)
    cls._compute_time_and_delay=staticmethod(scope['_compute_time_and_delay'])
    original=cls.get_costs
    def costs(dataset,pi,*args,**kwargs):
        # Explicit initial depot accounts for initial travel/energy and dispatch.
        pi=torch.cat((pi.new_zeros(pi.shape[0],1),pi),dim=1)
        return original(dataset,pi,*args,**kwargs)
    cls.get_costs=staticmethod(costs)
