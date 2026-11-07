"""Explain fixed-policy rejections without changing min-dispatch or policy."""
import json,signal,time
from bootstrap import *
from clock_patch import install

def main():
 out=P/'rejection_trace_v1';out.mkdir(exist_ok=False);signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('240s rejection diagnosis budget')));signal.alarm(240)
 install()
 from completion_guard import install as guard
 guard()
 data=torch.load(P/'development_n50_seed660611.pt',map_location='cpu',weights_only=False)['data'];rows=[];start=time.monotonic()
 for path in sorted((P/'fixed_dev16_v1').glob('*.json')):
  if path.name=='report.json':continue
  trace=json.loads(path.read_text());events=[]
  for row in trace['rows']:
   st=State.initialize(sample_to_batch(data[row['instance']]),**KW)
   for step,node in enumerate(row['actions']):
    if st.all_finished():break
    mask=st.get_mask();assert not bool(mask[0,0,node]),(path.name,row['instance'],step)
    if node==101:
     at_depot=int(st.prev_a.item())==0;served=int(st.served_orders_since_dispatch.item());opened=int(st.get_open_started_count().item());used=int(st.used_vehicles.item());service_available=int((~mask[0,0,1:101]).sum());only_reject=int((~mask).sum())==1
     e=dict(instance=row['instance'],step=step,at_depot=at_depot,served_in_trip=served,open_orders=opened,used_vehicles=used,service_actions=service_available,forced_only_reject=only_reject,depot_blocked=bool(mask[0,0,0]))
     if not at_depot and opened==0 and served<4 and used<9:
      # Hypothetical legal return/new dispatch state, preserving visited orders
      # and vehicle count. Only diagnose whether the 4-order return gate matters.
      fresh=st._replace(prev_a=torch.zeros_like(st.prev_a),cur_coord=st.coords[st.ids.squeeze(-1),0:1,:],current_time=torch.full_like(st.current_time,10.),trip_start_time=torch.full_like(st.trip_start_time,10.),served_orders_since_dispatch=torch.zeros_like(st.served_orders_since_dispatch),used_capacity_passenger=torch.zeros_like(st.used_capacity_passenger),used_capacity_cargo=torch.zeros_like(st.used_capacity_cargo))
      fm=fresh.get_mask();e['new_vehicle_service_actions']=int((~fm[0,0,1:101]).sum())
     events.append(e)
    st=st.update(torch.tensor([node]),current_mask=mask)
  summary=dict(model=path.stem,rejected=len(events),forced_only_reject=sum(e['forced_only_reject'] for e in events),inroute_below4_no_service=sum(not e['at_depot'] and e['served_in_trip']<4 and e['service_actions']==0 for e in events),unused_vehicle_could_serve=sum(e.get('new_vehicle_service_actions',0)>0 for e in events),events=events)
  rows.append(summary);print('TRACE',path.stem,'reject',summary['rejected'],'only_reject',summary['forced_only_reject'],'fresh_vehicle_service',summary['unused_vehicle_could_serve'],flush=True)
  (out/'report.json').write_text(json.dumps(dict(status='running',rows=rows),indent=2))
 summary={k:sum(r[k] for r in rows) for k in ['rejected','forced_only_reject','inroute_below4_no_service','unused_vehicle_could_serve']}
 (out/'report.json').write_text(json.dumps(dict(status='completed',scope='observational trajectory counts and hypothetical fresh-vehicle availability, not a changed policy evaluation or causal performance estimate',summary=summary,seconds=time.monotonic()-start,rows=rows),indent=2));signal.alarm(0);print(json.dumps(summary),flush=True)
if __name__=='__main__':main()
