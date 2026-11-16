"""Diagnostic independent constraint audit, alongside original evaluation."""
import argparse,json,sys
from pathlib import Path
import evaluate_v3 as entry
from route_audit import audit
from depot_noop import install
from problem_mcvrptw_v2 import Config
p=argparse.ArgumentParser();p.add_argument('--audit-output',required=True);p.add_argument('--no-op',action='store_true');args,rest=p.parse_known_args()
if args.no_op:install()
rows=[]
original=entry.evaluate_model._replay_order_audit

def audited(batch,pi,state_kwargs):
    cfg=dict(start=Config.OPERATION_START,end=Config.OPERATION_END,area=Config.AREA_SIZE,speed=Config.VEHICLE_SPEED,trip=Config.MAX_TRIP_TIME,service=Config.SERVICE_TIME,min_dispatch=state_kwargs['min_orders_per_dispatch'],hard_cargo=Config.HARD_CARGO_PICKUP_TIMEWINDOW,ride=Config.PASSENGER_MAX_RIDE_TIME_MINUTES/60,excess=Config.PASSENGER_MAX_EXCESS_RIDE_TIME_MINUTES/60,max_open=state_kwargs['max_concurrent_open_orders'],vehicles=Config.get_default_num_vehicles(pi.shape[0] and batch['loc'].shape[1]//2))
    for i in range(pi.shape[0]):
        sample={k:(v[i] if hasattr(v,'shape') and len(v.shape)>0 and v.shape[0]==pi.shape[0] else v) for k,v in batch.items()}
        result=audit(sample,pi[i].tolist(),cfg);result['sample_index']=len(rows);rows.append(result)
    return original(batch,pi,state_kwargs)
entry.evaluate_model._replay_order_audit=audited
sys.argv=['evaluate_model.py',*rest]
entry.evaluate_model.evaluate()
report={'samples':len(rows),'strict_all_settings_valid':sum(not r['violations'] for r in rows),'hard_valid':sum(not any(v['kind']!='min_dispatch' for v in r['violations']) for r in rows),'dispatch_target_shortfalls':sum(v['kind']=='min_dispatch' for r in rows for v in r['violations']),'unfulfilled_total':sum(r['unfulfilled'] for r in rows),'rows':rows,'intervention_noop':args.no_op}
Path(args.audit_output).write_text(json.dumps(report,indent=2))
print('INDEPENDENT_AUDIT',report['hard_valid'],report['samples'])
