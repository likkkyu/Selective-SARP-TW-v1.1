import copy
from route_audit import audit
cfg=dict(start=10.,end=16.,area=10.,speed=25.,trip=3.,service=.05,min_dispatch=1,hard_cargo=True,ride=70/60,excess=.5,max_open=6,vehicles=1)
s=dict(depot=[0.,0.],loc=[[0.,0.],[0.,0.]],time_windows=[[10.,16.],[10.,16.]],node_type=[1,1],demand_passenger=[.5,-.5],demand_cargo=[0.,0.])
def kinds(sample,actions,config=cfg):return {v['kind'] for v in audit(sample,actions,config)['violations']}
assert not kinds(s,[1,2,0])
assert not kinds(s,[3,0])
assert 'delivery_precedence' in kinds(s,[2,0])
t=copy.deepcopy(s);t['demand_passenger']=[2.,-2.];assert 'passenger_capacity' in kinds(t,[1,2,0])
t=copy.deepcopy(s);t['time_windows'][0]=[9.,9.5];assert 'pickup_window' in kinds(t,[1,2,0])
t=copy.deepcopy(s);t['time_windows'][1]=[12.,16.];assert 'passenger_ride' in kinds(t,[1,2,0]) and 'passenger_excess' in kinds(t,[1,2,0])
assert 'fleet_size' in kinds(s,[1,2,0],dict(cfg,vehicles=0))
assert 'min_dispatch' in kinds(s,[1,2,0],dict(cfg,min_dispatch=2))
assert 'return_with_open_orders' in kinds(s,[1,0])
assert 'final_depot' in kinds(s,[1,2])
assert 'operation_end' in kinds(s,[1,2,0],dict(cfg,end=10.08))
assert 'trip_duration' in kinds(s,[1,2,0],dict(cfg,trip=.08))
print('INDEPENDENT_AUDIT_SYNTHETIC_TESTS_PASS')
