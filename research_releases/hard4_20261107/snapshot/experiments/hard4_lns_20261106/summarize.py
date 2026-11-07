"""Descriptive paired-instance development summary; no significance claims."""
from pathlib import Path
import json,csv,hashlib,statistics
P=Path(__file__).resolve().parent;ROOT=P.parents[1]


def main():
 groups=[('seed614729','development_v2','dev4_v1'),('seed660611','followup_dev4_v1','followup_dev4_v1')]
 allrows=[]; rows=[]; checks=[]
 for group,lns,neural in groups:
  report=json.loads((P/lns/'report.json').read_text());assert report['status']=='passed' and len(report['rows'])==12
  plan=json.loads((P/lns/'plan.json').read_text())
  assert hashlib.sha256(Path(plan['scenario']).read_bytes()).hexdigest()==plan['scenario_sha256']
  runner=P/lns/'runner_frozen.py';assert hashlib.sha256(runner.read_bytes()).hexdigest()==plan['source_sha256']['run_development.py']
  for name in ['route_engine.py','search.py']:assert hashlib.sha256((P/name).read_bytes()).hexdigest()==plan['source_sha256'][name]
  for instance in range(4):
   selected=[r for r in report['rows'] if r['instance']==instance];assert len(selected)==3
   for r in selected:
    assert not r['audit']['violations'] and not r['audit']['unfulfilled']
    allrows.append(dict(group=group,**r))
   rec=dict(group=group,instance=instance,initial_completed=50-selected[0]['initial_cost']['reject_penalty']/575,LNS_mean_completed=statistics.mean(r['audit']['completed'] for r in selected),LNS_mean_J=statistics.mean(r['cost']['objective'] for r in selected),LNS_mean_total_seconds=statistics.mean(r['total_seconds'] for r in selected))
   for policy in ['learned','service_first']:
    items=[json.loads((ROOT/'experiments/hard4_admission_20261106'/neural/f'{policy}_{seed}.json').read_text())['rows'][instance] for seed in [1234,4321,8765]]
    assert all(not r['audit']['violations'] and not r['audit']['unfulfilled'] for r in items)
    rec[policy+'_mean_completed']=statistics.mean(r['audit']['completed'] for r in items)
    rec[policy+'_mean_J']=statistics.mean(r['objective'] for r in items)
   rows.append(rec)
 summary=dict(status='passed',scope='8 previously used development instances; 3 LNS search repeats and 3 independent trained checkpoints per neural policy; not held-out or inferential evidence',unique_development_instances=8,LNS_runs=24,LNS_completed=sum(r['audit']['completed'] for r in allrows),LNS_requests_across_repeats=1200,LNS_service_rate=sum(r['audit']['completed'] for r in allrows)/1200,LNS_mean_objective=statistics.mean(r['cost']['objective'] for r in allrows),LNS_mean_total_seconds=statistics.mean(r['total_seconds'] for r in allrows),LNS_hard_invalid=0,LNS_unfulfilled=0,LNS_trips=sum(r['cost']['vehicles'] for r in allrows),LNS_min_completed_per_trip=min(len(route)//2 for r in allrows for route in r['routes']),initial_service_rate=statistics.mean(r['initial_completed'] for r in rows)/50,learned_service_rate=statistics.mean(r['learned_mean_completed'] for r in rows)/50,rule_service_rate=statistics.mean(r['service_first_mean_completed'] for r in rows)/50,rows=rows,statistical_note='Independent instances n=8. Runs, trips and requests are nested observations. Training seeds are not equivalent to search seeds. No confidence interval or p-value reported from this small reused development sample.',timing_note='LNS total includes strict insertion initialization charged in full to each search repeat; excludes final independent audit/tensor verification and imports. Neural prior batch timings differ: no speedup claim.',failed_pilot='development_v1 failed before producing a result; removal can invalidate downstream ride time; repaired in source and isolated v2; original failure evidence retained')
 (P/'SUMMARY.json').write_text(json.dumps(summary,indent=2))
 with (P/'development_paired_instances.csv').open('w') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 print(json.dumps({k:v for k,v in summary.items() if k not in ['rows']},indent=2))
if __name__=='__main__':main()
