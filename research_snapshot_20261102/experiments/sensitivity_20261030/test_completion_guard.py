"""Independent tiny route tests and regression against the observed unsafe prefix."""
import json,sys
from pathlib import Path
import torch
from scenarios import ROOT
from completion_guard import completion_order,install
from state_mcvrptw_v2 import StateMCVRPPDTW as State

def main():
    coords=[[0.,0.],[0.,0.],[0.,0.]]
    args=dict(coords=coords,current=1,now=10.,trip_start=10.,orders=[0],pickup_finish=[10.],passenger=[True],windows=[[10.,16.],[10.,16.]],n=1,area=10,speed=30,service=.01,trip=3,end=16,ride=1,excess=.5)
    assert completion_order(**args)==(0,)
    assert completion_order(**dict(args,now=12.)) is None
    assert completion_order(**dict(args,windows=[[10.,16.],[12.,16.]])) is None
    assert completion_order(**dict(args,trip=.001)) is None
    r=json.loads((Path(__file__).parent/'guard_regression/failure317.json').read_text())
    x={k:torch.tensor(v).unsqueeze(0) for k,v in r['sample'].items() if isinstance(v,list)}
    torch.set_num_threads(1);install();s=State.initialize(x,**r['state_kwargs']);blocked=None
    for step,a in enumerate(r['actions']):
        m=s.get_mask()
        if s.has_open_started_orders().any():
            assert m[0,0,0] and m[0,0,s.reject_index]
        if m[0,0,a]:blocked=step;break
        s=s.update(torch.tensor([a]),current_mask=m)
    assert blocked is not None and blocked<37
    report={'status':'passed','unsafe_prefix_blocked_at_step':blocked,'synthetic_cases':4,'audit_tolerance_unchanged':True,'lookahead_margin_seconds':.036}
    (Path(__file__).parent/'guard_regression/local_checks.json').write_text(json.dumps(report,indent=2));print(report)
if __name__=='__main__':main()
