import csv,json,hashlib,sys
from pathlib import Path
p=Path(__file__).parent;root=p/'results_1234/learned_1234';sys.path.insert(0,str(p.parent/'reject_ablation_20261029'))
from route_audit import audit
rows=[];base=None;paired={}
for folder in sorted(root.iterdir()):
 if not folder.is_dir() or not (folder/'audit.json').exists():continue
 a=json.loads((folder/'audit.json').read_text());ev=json.loads((folder/'eval.json').read_text());c=json.loads((folder/'effective_config.json').read_text());meta=json.loads((folder/'scenario.json').read_text())
 r=dict(scenario=folder.name,service_rate=ev['aggregate']['service_rate_mean'],raw_cost=ev['aggregate']['total_cost_raw_mean'],trip_penalty=ev['aggregate']['trip_overtime_penalty_mean'],internal_excess_violations=ev['aggregate']['passenger_excess_ride_time_violations_mean']*500,hard_valid=a['hard_valid'],unfulfilled=a['unfulfilled_total'],excess_minutes=c['PASSENGER_MAX_EXCESS_RIDE_TIME_MINUTES'])
 cnt={1:[0,0],0:[0,0]};hashes=[];rerun=0
 for line in (folder/'routes.jsonl').open():
  t=json.loads(line);s=t['sample'];n=len(s['loc'])//2;done=set(x-n-1 for x in t['actions'] if n<x<=2*n)
  for j in range(n):
   kind=int(s['node_type'][j]);cnt[kind][0]+=1;cnt[kind][1]+=j in done
  hashes.append(hashlib.sha256(json.dumps({k:v for k,v in s.items() if k!='time_windows'},sort_keys=True).encode()).hexdigest())
  cfg=dict(start=c['OPERATION_START'],end=c['OPERATION_END'],area=c['AREA_SIZE'],speed=c['VEHICLE_SPEED'],trip=c['MAX_TRIP_TIME'],service=c['SERVICE_TIME'],min_dispatch=t['state_kwargs']['min_orders_per_dispatch'],hard_cargo=c['HARD_CARGO_PICKUP_TIMEWINDOW'],ride=c['PASSENGER_MAX_RIDE_TIME_MINUTES']/60,excess=c['PASSENGER_MAX_EXCESS_RIDE_TIME_MINUTES']/60,max_open=t['state_kwargs']['max_concurrent_open_orders'],vehicles=9)
  check=audit(s,t['actions'],cfg)
  assert check['completed']==a['rows'][t['instance']]['completed']
  rerun+=not any(v['kind']!='min_dispatch' for v in check['violations'])
 r.update(passenger_rate=cnt[1][1]/cnt[1][0],cargo_rate=cnt[0][1]/cnt[0][0],actual_passenger_ratio=cnt[1][0]/25000,replay_hard_valid=rerun)
 paired[folder.name]=hashes;rows.append(r)
b=paired['r0.6_tw1_ex1'];assert all(h==b for k,h in paired.items() if k.startswith('r0.6_'))
with (p/'diagnosis_metrics.csv').open('w') as f:
 w=csv.DictWriter(f,fieldnames=rows[0].keys());w.writeheader();w.writerows(rows)
print(json.dumps(rows,indent=2))
