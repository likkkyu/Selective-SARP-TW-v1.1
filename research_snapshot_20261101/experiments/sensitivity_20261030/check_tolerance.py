import json,math
from pathlib import Path
p=Path(__file__).parent;root=p/'results_1234/learned_1234';summary={}
for folder in root.iterdir():
 if not (folder/'routes.jsonl').exists():continue
 c=json.loads((folder/'effective_config.json').read_text());over=[]
 for line in (folder/'routes.jsonl').open():
  r=json.loads(line);s=r['sample'];n=len(s['loc'])//2;coords=[s['depot']]+s['loc'];time=c['OPERATION_START'];cur=0;picks={}
  for node in r['actions']:
   if node==2*n+1:continue
   if node==0:time=c['OPERATION_START'];cur=0;continue
   idx=node-1;travel=math.dist(coords[cur],coords[node])*c['AREA_SIZE']/c['VEHICLE_SPEED'];start=max(time+travel,s['time_windows'][idx][0]);finish=start+c['SERVICE_TIME']
   if node<=n:picks[idx]=finish
   elif s['node_type'][idx]==1:
    i=idx-n;direct=math.dist(coords[i+1],coords[node])*c['AREA_SIZE']/c['VEHICLE_SPEED'];excess=(start-picks[i]-direct)*60-c['PASSENGER_MAX_EXCESS_RIDE_TIME_MINUTES'];over.append(excess)
   time=finish;cur=node
 summary[folder.name]={'max_excess_over_limit_minutes':max(over),'count_over_1e_5_minutes':sum(x>1e-5 for x in over),'count_over_1e_4_hours':sum(x>.006 for x in over)}
(p/'tolerance_replay.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2))
