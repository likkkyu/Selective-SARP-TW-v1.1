"""Candidate-local counterfactual only; no production patch or checkpoint change."""
import json,math,signal
from validate import P,ROOT,SNAP,C,torch,sample_to_batch
import sys
sys.path.insert(0,str(SNAP/'experiments/sensitivity_20261030'))
from completion_guard import install
from state_mcvrptw_v2 import StateMCVRPPDTW as State
install()
def main():
 out=P/'dispatch_clock_diagnosis_v1';out.mkdir(exist_ok=False)
 signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('120s local diagnosis limit')));signal.alarm(120)
 base=json.loads((P/'comparability_audit_v2/report.json').read_text());data=torch.load(ROOT/'experiments/paper_delivery_20261104/confirmation_n50_seed803219.pt',map_location='cpu',weights_only=False)['data'];kw=base['models'][0]['state_kwargs'];rows=[]
 for record in base['new_routes']:
  i=record['instance'];fail=record['checks']['neural_state'];s=data[i];n=50
  state=State.initialize(sample_to_batch(s),**kw)
  for node in fail['prefix'][:-1]:
   mask=state.get_mask();assert not bool(mask[0,0,node]);state=state.update(torch.tensor([node]),current_mask=mask)
  node=fail['node'];at_depot=int(state.prev_a.item())==0;is_pickup=1<=node<=n
  result=dict(instance=i,node=node,at_depot=at_depot,is_pickup=is_pickup)
  if at_depot and is_pickup:
   j=node-1;xy=[s['depot'].tolist()]+s['loc'].tolist();tw=s['time_windows'].tolist()
   travel=lambda a,b:math.dist(xy[a],xy[b])*C['AREA_SIZE']/C['VEHICLE_SPEED']
   now=float(state.current_time.item());depart=max(now,tw[j][0]-travel(0,node));first_start=max(now+travel(0,node),tw[j][0]);delivery_start=max(first_start+C['SERVICE_TIME']+travel(node,node+n),tw[j+n][0]);back=delivery_start+C['SERVICE_TIME']+travel(node+n,0)
   original=state.get_mask();assert bool(original[0,0,node])
   counter=state._replace(current_time=torch.full_like(state.current_time,depart),trip_start_time=torch.full_like(state.trip_start_time,depart))
   after=counter.get_mask()
   result.update(original_current=now,correct_candidate_dispatch=depart,pickup_service_start=first_start,solo_return=back,solo_duration_from_old_clock=back-now,solo_duration_from_dispatch=back-depart,target_allowed_with_candidate_dispatch=not bool(after[0,0,node]),counterfactual_scope='only this candidate at the recorded prefix; not a global fix and not policy evaluation')
  rows.append(result)
  (out/'report.json').write_text(json.dumps(dict(status='running',rows=rows),indent=2))
 summary=dict(instances=len(rows),depot_pickup_first_failures=sum(x['at_depot'] and x['is_pickup'] for x in rows),candidate_dispatch_unblocks=sum(x.get('target_allowed_with_candidate_dispatch',False) for x in rows),old_clock_over_3h_correct_clock_within_3h=sum(x.get('solo_duration_from_old_clock',0)>3 and x.get('solo_duration_from_dispatch',4)<=3 for x in rows))
 (out/'report.json').write_text(json.dumps(dict(status='completed',summary=summary,rows=rows),indent=2));signal.alarm(0);print(json.dumps(summary));print('EXAMPLE',json.dumps(rows[0]))
if __name__=='__main__':main()
