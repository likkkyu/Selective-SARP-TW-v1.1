"""Minimal meaningful boundaries and all 32 recorded counterexamples."""
import hashlib,json,math,signal,time
from bootstrap import *
from clock_patch import install

def toy(pickup_start=10.,pickup_end=15.,delivery=(.2,0.),passenger=True,demand=.1):
    # Four orders keep the unchanged min-dispatch gate out of single-order tests.
    n=4;types=[1 if passenger else 0]*8;dp=[demand]*4+[-demand]*4 if passenger else [0.]*8;dc=[0.]*8 if passenger else [demand]*4+[-demand]*4
    return dict(depot=torch.tensor([0.,0.]),loc=torch.tensor([[.1,0.]]*4+[list(delivery)]*4),node_type=torch.tensor(types),demand_passenger=torch.tensor(dp),demand_cargo=torch.tensor(dc),time_windows=torch.tensor([[pickup_start,pickup_end]]*4+[[10.,16.]]*4),n_orders=torch.tensor(n))

def main():
 out=P/'regression_v2';out.mkdir(exist_ok=False);signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('240s regression budget')));signal.alarm(240)
 frozen=State.get_mask;cases={
 'early_feasible':toy(), 'late_feasible':toy(14.,15.),
 'expired_pickup':toy(9.,9.5), 'beyond_operation_end':toy(15.9,16.),
 'capacity_violation':toy(demand=1.1),
 'passenger_ride_violation':toy(delivery=(4.,0.)),
 'cargo_trip_violation':toy(delivery=(4.,0.),passenger=False)}
 states={k:State.initialize(sample_to_batch(v),**KW) for k,v in cases.items()}
 # Existing trip must retain its true trip_start, never restart on a new pickup.
 base=states['early_feasible'];ones=torch.ones((1,1),dtype=torch.long)
 states['inroute'] = base.update(torch.tensor([1]),current_mask=base.get_mask())
 states['trip_already_exhausted']=states['inroute']._replace(current_time=torch.full((1,1),13.1),trip_start_time=torch.full((1,1),10.))
 states['open_limit']=states['inroute']._replace(max_concurrent_open_orders=1)
 # Different candidates at the same depot have distinct departure clocks.
 mix=toy();mix['time_windows'][1]=torch.tensor([14.,15.]);states['mixed_early_late']=State.initialize(sample_to_batch(mix),**KW)
 # Different rows must not share a candidate's departure clock.
 batch={k:torch.stack([cases['early_feasible'][k],cases['late_feasible'][k]]) for k in ('depot','loc','node_type','demand_passenger','demand_cargo','time_windows')}
 batch['n_orders']=torch.tensor([4,4]);states['mixed_batch']=State.initialize(batch,**KW)
 old={k:st.get_mask().clone() for k,st in states.items()}
 original_hash=hashlib.sha256((SNAP/'state_mcvrptw_v2.py').read_bytes()).hexdigest();install()
 from completion_guard import install as guard
 guard()
 rows=[]
 expected={'early_feasible':False,'late_feasible':False,'expired_pickup':True,'beyond_operation_end':True,'capacity_violation':True,'passenger_ride_violation':True,'cargo_trip_violation':True}
 for label,st in states.items():
  if label=='trip_already_exhausted':
   try:st.get_mask()
   except RuntimeError as exc:
    assert 'No safe open-order completion' in str(exc)
    rows.append(dict(case=label,status='passed',guard_fail_closed=True));print('PASS boundary',label,'guard fails closed',flush=True);continue
   raise AssertionError('Unrecoverable open orders must fail closed')
  mask=st.get_mask();debugmask,_=st.get_mask(return_debug=True);assert torch.equal(mask,debugmask),label
  if label in expected:assert bool(mask[0,0,1])==expected[label],(label,mask)
  if label=='early_feasible':assert torch.equal(old[label],mask)
  if label in ('inroute','trip_already_exhausted','open_limit'):assert torch.equal(old[label],mask),label
  if label=='trip_already_exhausted':assert bool(mask[0,0,2])
  if label=='open_limit':assert bool(mask[0,0,2])
  if label=='mixed_early_late':assert not bool(mask[0,0,1]) and not bool(mask[0,0,2])
  if label=='mixed_batch':assert not bool(mask[0,0,1]) and not bool(mask[1,0,1])
  rows.append(dict(case=label,old_mask=old[label].tolist(),new_mask=mask.tolist(),status='passed'))
  print('PASS boundary',label,flush=True)
 diagnosis=json.loads((ROOT/'experiments/mip_scip_20261106/dispatch_clock_diagnosis_v1/report.json').read_text());audit=json.loads((ROOT/'experiments/mip_scip_20261106/comparability_audit_v2/report.json').read_text());data=torch.load(ROOT/'experiments/paper_delivery_20261104/confirmation_n50_seed803219.pt',map_location='cpu',weights_only=False)['data'];replays=[]
 for entry in audit['new_routes']:
  i=entry['instance'];failure=entry['checks']['neural_state'];st=State.initialize(sample_to_batch(data[i]),**KW)
  for node in failure['prefix'][:-1]:
   # Check actual prefix still legal under the corrected rule.
   mask=st.get_mask();assert not bool(mask[0,0,node]),(i,node)
   st=st.update(torch.tensor([node]),current_mask=mask)
  mask=st.get_mask();assert not bool(mask[0,0,failure['node']]),i
  replays.append(dict(instance=i,step=failure['step'],node=failure['node'],corrected_action_allowed=True))
 assert hashlib.sha256((SNAP/'state_mcvrptw_v2.py').read_bytes()).hexdigest()==original_hash
 report=dict(status='passed',boundary_cases=rows,recorded_counterexamples=replays,unblocked_counterexamples=len(replays),production_snapshot_unchanged=True,clock_patch_version=1,cost_version=4,guard_version=1,min_orders_per_dispatch=4,scope='boundary regressions and recorded-prefix checks; no claim of full corrected-model performance')
 (out/'report.json').write_text(json.dumps(report,indent=2));signal.alarm(0);print('PASS all boundaries and',len(replays),'recorded prefixes; frozen source unchanged',flush=True)
if __name__=='__main__':main()
