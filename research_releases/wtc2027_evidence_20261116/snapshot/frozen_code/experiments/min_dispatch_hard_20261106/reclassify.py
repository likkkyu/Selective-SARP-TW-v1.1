"""Never overwrite historical observations; apply clarified hard validity separately."""
import csv,json
from pathlib import Path
P=Path(__file__).resolve().parent;ROOT=P.parents[1]
def main():
 rows=list(csv.DictReader((ROOT/'experiments/paper_delivery_20261104/confirmation_v1/instance_seed_results.csv').open()));groups={}
 for method in ['learned','service_first','SA','GA']:
  r=[x for x in rows if x['method']==method];bad=sum(x['dispatch_target_shortfall']=='True' for x in r)
  groups[method]=dict(solutions=len(r),violating_minimum_four=bad,hard_valid_under_clarified_definition=len(r)-bad,scope='all prior base physical audits passed; min-four failures newly count as hard violations; no conditional performance ranking')
 prior=ROOT/'experiments/mip_scip_20261106/common_n50_15s_v1';new=[]
 for f in sorted(prior.glob('instance_*.json')):
  r=json.loads(f.read_text())
  for method,a in [('old_insertion',r['insertion_audit']),('old_SCIP',r['scip']['audit'])]:new.append(dict(method=method,instance=r['instance'],hard_valid=not a['violations'] and a['unfulfilled']==0,min4_violations=sum(v['kind']=='min_dispatch' for v in a['violations'])))
 newgroups={m:dict(solutions=sum(x['method']==m for x in new),hard_valid=sum(x['method']==m and x['hard_valid'] for x in new),invalid=sum(x['method']==m and not x['hard_valid'] for x in new)) for m in ['old_insertion','old_SCIP']}
 diagnostic={}
 for size in [4,16]:
  for mode in ['old','fixed']:
   d=ROOT/'experiments/dispatch_clock_fix_20261106'/f'{mode}_dev{size}_v1';a=[]
   for f in sorted(d.glob('*.json')):
    if f.name=='report.json':continue
    a.extend(x['audit'] for x in json.loads(f.read_text())['rows'])
   diagnostic[f'{mode}_dev{size}']=dict(solutions=len(a),hard_valid=sum(not x['violations'] and not x['unfulfilled'] for x in a),invalid=sum(bool(x['violations']) or bool(x['unfulfilled']) for x in a))
 result=dict(status='completed',definition=json.loads((P/'DEFINITION.json').read_text()),previous_common_comparison=groups,previous_MIP_comparison=newgroups,clock_development_diagnostics=diagnostic,raw_results_unchanged=True)
 (P/'RECLASSIFICATION.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
if __name__=='__main__':main()
