"""Independent scalar route replay. No State.get_mask/update or cost evaluator.
Reports residuals separately; hard checks use declared physical parameters.
"""
import math

def audit(sample, actions, cfg):
    def values(x): return x.detach().cpu().tolist() if hasattr(x,'detach') else x
    s={k:values(v) for k,v in sample.items()}
    n=len(s['loc'])//2; coords=[s['depot']]+s['loc']; tw=s['time_windows']
    status=[0]*n; pickup_finish={}; current=0; time=cfg['start']; dispatch=time
    loadp=loadc=0.; vehicles=0; served=0; violations=[]
    tol=1e-4
    def fail(kind,step,value): violations.append({'kind':kind,'step':step,'value':value})
    def travel(a,b): return math.dist(coords[a],coords[b])*cfg['area']/cfg['speed']
    for step,node in enumerate(actions):
        node=int(node)
        if node==2*n+1:
            candidates=[i for i,x in enumerate(status) if x==0]
            if not candidates: fail('reject_without_candidate',step,1);continue
            i=min(candidates,key=lambda i:(tw[i][1],i));status[i]=3;continue
        if not 0<=node<=2*n: fail('invalid_node',step,node);continue
        if node==0 and current==0: continue
        arrival=time+travel(current,node)
        if node==0:
            if any(x==1 for x in status):fail('return_with_open_orders',step,1)
            if served<cfg['min_dispatch']:fail('min_dispatch',step,served)
            if arrival-dispatch>cfg['trip']+tol:fail('trip_duration',step,arrival-dispatch)
            if arrival>cfg['end']+tol:fail('operation_end',step,arrival)
            current=0;time=cfg['start'];loadp=loadc=0.;served=0;continue
        idx=node-1;i=idx%n;is_pickup=idx<n
        if current==0:
            vehicles+=1;dispatch=max(cfg['start'],tw[idx][0]-travel(0,node))
        start=max(arrival,tw[idx][0]);finish=start+cfg['service']
        if is_pickup:
            if status[i]!=0:fail('pickup_status',step,status[i])
            if (s['node_type'][idx]==1 or cfg['hard_cargo']) and start>tw[idx][1]+tol:fail('pickup_window',step,start-tw[idx][1])
            status[i]=1;pickup_finish[i]=finish
        else:
            if status[i]!=1:fail('delivery_precedence',step,status[i])
            if s['node_type'][idx]==1 and i in pickup_finish:
                ride=start-pickup_finish[i];direct=travel(i+1,i+n+1)
                if ride>cfg['ride']+tol:fail('passenger_ride',step,ride)
                if ride-direct>cfg['excess']+tol:fail('passenger_excess',step,ride-direct)
            status[i]=2;served+=1
        loadp+=s['demand_passenger'][idx];loadc+=s['demand_cargo'][idx]
        if not -tol<=loadp<=1+tol:fail('passenger_capacity',step,loadp)
        if not -tol<=loadc<=1+tol:fail('cargo_capacity',step,loadc)
        if sum(x==1 for x in status)>cfg['max_open']:fail('max_open',step,sum(x==1 for x in status))
        if finish>cfg['end']+tol:fail('operation_end',step,finish)
        time=finish;current=node
    if current!=0:fail('final_depot',len(actions),current)
    if vehicles>cfg['vehicles']:fail('fleet_size',len(actions),vehicles)
    return {'violations':violations,'completed':status.count(2),'rejected':status.count(3),'unfulfilled':status.count(0)+status.count(1),'vehicles':vehicles}
