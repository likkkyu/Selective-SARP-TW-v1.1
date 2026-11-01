import sys,runpy,inspect,json
from pathlib import Path
sys.path.insert(0,str(Path('experiments/reject_ablation_20261029').resolve()))
sys.path.insert(0,str(Path('experiments/sensitivity_20261030').resolve()))
import route_audit
ns=dict(route_audit.__dict__);exec(inspect.getsource(route_audit.audit).replace('tol=1e-4','tol=1e-9'),ns);strict=ns['audit'];original=route_audit.audit;details=[]
def checked(sample,actions,cfg):
 r=strict(sample,actions,cfg)
 for v in r['violations']:
  if v['kind']=='passenger_excess':details.append({'seconds_over':(v['value']-cfg['excess'])*3600,'step':v['step']})
 return original(sample,actions,cfg)
route_audit.audit=checked
sys.argv=['corrected_entry.py','--stage','eval','--audit-output','tolerance_audit.json','--checkpoint','formal_learned_1234/outputs/pomo_n50_optimized/model_best.pt','--graph-size','50','--num-samples','500','--batch-size','16','--seed','917531','--max-concurrent-open-orders','6','--enable-delivery-viability','--json-output','tolerance_eval.json']
runpy.run_path('experiments/sensitivity_20261030/corrected_entry.py',run_name='__main__')
report={'strict_excess_events':details,'max_seconds_over':max([x['seconds_over'] for x in details],default=0),'declared_audit_tolerance_seconds':0.36}
Path('tolerance_review.json').write_text(json.dumps(report,indent=2));print('TOLERANCE_REVIEW',report,flush=True)
assert report['max_seconds_over']<=0.36
