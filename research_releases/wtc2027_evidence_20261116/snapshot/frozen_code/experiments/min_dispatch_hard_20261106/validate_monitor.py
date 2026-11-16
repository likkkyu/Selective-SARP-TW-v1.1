import json,sys
from bootstrap import *
sys.path.insert(0,str(ROOT/'experiments/dispatch_clock_fix_20261106'))
from clock_patch import install as clock
clock()
from completion_guard import install as guard
guard()
from state_mcvrptw_v2 import StateMCVRPPDTW as State
from strict_dispatch_monitor import install,IncompleteDispatchError
from validate_hard import toy

def main():
 out=P/'monitor_validation_v1';out.mkdir(exist_ok=False);s=toy(4)
 kw=dict(max_concurrent_open_orders=6,min_orders_per_dispatch=4,enable_delivery_viability=True,enable_viability_fallback=False,relax_pickup_commitment_trip_time=False)
 # Build valid bookkeeping with exactly three completed and the fourth declined;
 # this is precisely the old final-trip all_done escape, not a valid hard plan.
 st=State.initialize(sample_to_batch(s),**kw)
 for node in [1,5,2,6,3,7,9]:st=st.update(torch.tensor([node]),current_mask=torch.zeros((1,1,10),dtype=torch.bool))
 assert int(st.served_orders_since_dispatch)==3 and int(st.prev_a)!=0
 before=st.get_mask();assert not bool(before[0,0,0])
 install()
 try:st.get_mask()
 except IncompleteDispatchError:under=True
 else:raise AssertionError('Under-four all_done return must fail closed')
 # Four actually completed: ordinary depot return remains available.
 full=State.initialize(sample_to_batch(s),**kw)
 for node in [1,5,2,6,3,7,4,8]:full=full.update(torch.tensor([node]),current_mask=torch.zeros((1,1,10),dtype=torch.bool))
 assert not bool(full.get_mask()[0,0,0]);full=full.update(torch.tensor([0]),current_mask=full.get_mask());assert full.all_finished()
 # All rejected without any departure is allowed by user definition.
 none=State.initialize(sample_to_batch(s),**kw)
 for _ in range(4):none=none.update(torch.tensor([9]),current_mask=none.get_mask())
 assert none.all_finished() and int(none.used_vehicles)==0
 result=dict(status='passed',under_four_final_return_prevented=True,four_order_return_allowed=True,all_declined_without_dispatch_allowed=True,scope='failure detection only; does not prove decoder can construct every feasible >=4 plan')
 (out/'report.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
if __name__=='__main__':main()
