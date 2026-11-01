import torch
from scenarios import ROOT
from problem_mcvrptw_v2 import MCVRPPDTW,Config
from timing_candidate import install
install()
s={'depot':torch.tensor([[0.,0.]]),'loc':torch.tensor([[[.1,0.],[.2,0.]]]),'node_type':torch.tensor([[1,1]]),'demand_passenger':torch.tensor([[.5,-.5]]),'demand_cargo':torch.tensor([[0.,0.]]),'time_windows':torch.tensor([[[13.,14.],[13.,15.]]]),'n_orders':1}
_,d=MCVRPPDTW.get_costs(s,torch.tensor([[1,2,0]]),return_details=True)
assert abs(d['trip_overtime_penalty'].item())<1e-5
# Explicit leading depot must not change the result.
a,_=MCVRPPDTW.get_costs(s,torch.tensor([[1,2,0]]),return_details=True)
b,_=MCVRPPDTW.get_costs(s,torch.tensor([[0,1,2,0]]),return_details=True)
assert torch.allclose(a,b)
print('PASS delayed dispatch and leading depot invariance')
# Closed-form complete route length: 0 -> .1 -> .2 -> 0.
assert abs(d['total_distance'].item()-.4*Config.AREA_SIZE)<1e-5
# A depot wait until 14:00 followed by late delivery at 18:00 is real overtime.
late={**s,'time_windows':torch.tensor([[[14.,15.],[18.,19.]]])}
_,bad=MCVRPPDTW.get_costs(late,torch.tensor([[1,2,0]]),return_details=True)
assert bad['trip_overtime_penalty'].item()>0
# Reject-only routes incur rejection cost, no physical travel or overtime.
_,rej=MCVRPPDTW.get_costs(s,torch.tensor([[3,0]]),return_details=True)
assert rej['total_distance'].item()==0 and rej['trip_overtime_penalty'].item()==0
print('PASS analytic initial distance, genuine overtime, reject-only route')
