import copy, json, sys
from pathlib import Path
import torch
from scenarios import make, save, ROOT
sys.path.insert(0,str(ROOT/'experiments/reject_ablation_20261029'))
from route_audit import audit
from state_mcvrptw_v2 import StateMCVRPPDTW
from evaluate_model import collate_fn
from problem_mcvrptw_v2 import Config

base=make(); tight=make(tw=.5); wide=make(tw=1.5)
for a,b,c in zip(base['data'],tight['data'],wide['data']):
    for k in a:
        if k!='time_windows' and torch.is_tensor(a[k]):
            assert torch.equal(a[k],b[k]) and torch.equal(a[k],c[k]),k
    n=50; m=a['node_type'][:n]==1
    w=a['time_windows'][:n,1]-a['time_windows'][:n,0]
    assert torch.allclose(b['time_windows'][:n,1][m]-b['time_windows'][:n,0][m],w[m]*.5)
    assert torch.equal(a['time_windows'][:,0],b['time_windows'][:,0])
    assert torch.equal(a['time_windows'][n:],c['time_windows'][n:])
counts=[]
for r in (0.,.3,.9,1.):
    d=make(ratio=r);counts.append(sum(int((s['node_type'][:50]==1).sum()) for s in d['data']))
assert counts[0]==0 and counts[-1]==200 and counts[1]<counts[2]
# Verify model environment consumes frozen time windows and rejects a late pickup.
s=copy.deepcopy(base['data'][0]);s['time_windows'][:50]=torch.tensor([8.,9.])
state=StateMCVRPPDTW.initialize(collate_fn([s]),max_concurrent_open_orders=6,enable_delivery_viability=True)
mask=state.get_mask()[0]
assert mask[...,1:51].all()
# Independent audit must respond to excess limit, with total ride fixed.
cfg=dict(start=10.,end=16.,area=10.,speed=25.,trip=3.,service=.05,min_dispatch=1,hard_cargo=True,ride=70/60,excess=.25,max_open=6,vehicles=1)
s=dict(depot=[0.,0.],loc=[[0.,0.],[0.,0.]],time_windows=[[10.,16.],[10.5,16.]],node_type=[1,1],demand_passenger=[.5,-.5],demand_cargo=[0.,0.])
strict=audit(s,[1,2,0],cfg);loose=audit(s,[1,2,0],dict(cfg,excess=.75))
assert any(v['kind']=='passenger_excess' for v in strict['violations'])
assert not loose['violations']
assert make(excess=.5)['meta']['excess_minutes']==15
assert make(excess=1.5)['meta']['excess_minutes']==45
# State uses the shared Config object: all three components see the intervention.
import state_mcvrptw_v2
assert state_mcvrptw_v2.Config is Config
out=Path(__file__).parent/'smoke_evidence';out.mkdir(exist_ok=True)
for name,b in [('baseline',base),('tw_tight',tight),('tw_wide',wide)]:
    path=out/(name+'.pt')
    if not path.exists():save(b,path)
    loaded=torch.load(path,weights_only=False)
    assert torch.equal(loaded['data'][0]['time_windows'],b['data'][0]['time_windows'])
report=dict(status='passed',instances_per_scenario=4,orders_per_instance=50,passenger_counts=counts,
    checks=['paired coordinates/demands','fixed pickup starts','pickup width scaling','unchanged delivery windows','ratio extremes','environment late-pickup mask','independent excess-limit boundary','shared Config','serialization'],
    limitation='No trained v3 checkpoint inference performed; not research results')
(out/'checks.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
