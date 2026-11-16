"""Selective SARP MILP with SCIP; no Gurobi dependency or neural masks.

Arc formulation: route labels equal the first pickup ID, so pair labels enforce
same-trip pickup/delivery without a vehicle-indexed arc expansion. Exact earliest
service times prohibit voluntary waiting not represented by the route evaluator.
Four orders/trip remains a reported construction target, never a hard constraint.
"""
from __future__ import annotations
import math,time
from pyscipopt import Model, quicksum


def lists(sample):
    return {k:v.detach().cpu().tolist() if hasattr(v,'detach') else v for k,v in sample.items()}


def parameters(config):
    names=['OPERATION_START','OPERATION_END','AREA_SIZE','VEHICLE_SPEED','MAX_TRIP_TIME',
           'SERVICE_TIME','PASSENGER_MAX_RIDE_TIME_MINUTES','PASSENGER_MAX_EXCESS_RIDE_TIME_MINUTES',
           'HARD_CARGO_PICKUP_TIMEWINDOW','PASSENGER_CAPACITY','CARGO_CAPACITY','PASSENGER_WEIGHT_KG',
           'CARGO_UNIT_WEIGHT_KG','ENERGY_BASE','ENERGY_WEIGHT_REF','ELECTRICITY_PRICE',
           'PASSENGER_DELAY_COST','CARGO_DELAY_COST','CARGO_DELAY_TIER1_MIN','CARGO_DELAY_TIER2_MIN',
           'CARGO_DELAY_COST_30_60','CARGO_DELAY_COST_60_PLUS','VEHICLE_COST',
           'ALPHA_ENERGY','ALPHA_DELAY','ALPHA_VEHICLE','ALPHA_REJECT','ALPHA_UNFULFILLED','ALPHA_TRIP_OVERTIME']
    return {k:getattr(config,k) for k in names}


def scalar_cost(sample,actions,c,profile):
    """Arrival-based tardiness; service-start-based physical ride audit is separate."""
    s=lists(sample);n=len(s['loc'])//2;xy=[s['depot']]+s['loc']
    cur=0;t=c['OPERATION_START'];lp=lc=0.;energy=distance=pd=cd=0.;dispatches=reject=0
    for node in actions:
        if node==2*n+1:reject+=1;continue
        if node==0 and cur==0:continue
        km=math.dist(xy[cur],xy[node])*c['AREA_SIZE'];travel=km/c['VEHICLE_SPEED']
        energy+=km*c['ENERGY_BASE']*(1+(lp*c['PASSENGER_CAPACITY']*c['PASSENGER_WEIGHT_KG']+lc*c['CARGO_CAPACITY']*c['CARGO_UNIT_WEIGHT_KG'])/c['ENERGY_WEIGHT_REF'])*c['ELECTRICITY_PRICE'];distance+=km
        if cur==0:
            dispatches+=1;t=max(c['OPERATION_START'],s['time_windows'][node-1][0]-travel)
        arrival=t+travel
        if node==0:cur=0;t=c['OPERATION_START'];lp=lc=0.;continue
        j=node-1
        if node>n:
            late=max(0.,arrival-s['time_windows'][j][1])*60
            if s['node_type'][j]==1:pd+=late
            else:cd+=late
        t=max(arrival,s['time_windows'][j][0])+c['SERVICE_TIME']
        lp+=s['demand_passenger'][j];lc+=s['demand_cargo'][j];cur=node
    cp=pd*c['PASSENGER_DELAY_COST'];b1=c['CARGO_DELAY_TIER1_MIN'];b2=c['CARGO_DELAY_TIER2_MIN']
    cc=min(cd,b1)*c['CARGO_DELAY_COST']+max(0.,min(cd,b2)-b1)*c['CARGO_DELAY_COST_30_60']+max(0.,cd-b2)*c['CARGO_DELAY_COST_60_PLUS']
    vc=dispatches*c['VEHICLE_COST'];rp=reject*c['ALPHA_REJECT']
    J=c['ALPHA_ENERGY']*energy/profile['energy_cost_raw']+c['ALPHA_DELAY']*(cp/profile['passenger_delivery_delay_cost_raw']+cc/profile['cargo_delay_cost_raw'])+c['ALPHA_VEHICLE']*vc/max(1.,profile['vehicle_cost_norm'])+rp
    return dict(objective=J,raw_cost=energy+cp+cc+vc+rp,energy=energy,distance=distance,passenger_delay_cost=cp,cargo_delay_cost=cc,vehicle_cost=vc,reject_penalty=rp)


def audit_config(c,n,vehicles):
    return dict(start=c['OPERATION_START'],end=c['OPERATION_END'],area=c['AREA_SIZE'],speed=c['VEHICLE_SPEED'],trip=c['MAX_TRIP_TIME'],service=c['SERVICE_TIME'],min_dispatch=4,hard_cargo=c['HARD_CARGO_PICKUP_TIMEWINDOW'],ride=c['PASSENGER_MAX_RIDE_TIME_MINUTES']/60,excess=c['PASSENGER_MAX_EXCESS_RIDE_TIME_MINUTES']/60,max_open=6,vehicles=vehicles)


class SelectiveSCIP:
    def __init__(self,c,profile,vehicles=None,time_limit=60.,gap=0.,threads=1,seed=1234,verbose=True):
        if time_limit<=0 or threads<1:raise ValueError('Positive finite budget and threads required')
        self.c=c;self.profile=profile;self.vehicles=vehicles;self.time_limit=time_limit;self.gap=gap;self.threads=threads;self.seed=seed;self.verbose=verbose

    def build(self,sample):
        start=time.monotonic();s=lists(sample);self.s=s;n=len(s['loc'])//2;self.n=n
        c=self.c;p=self.profile;V=list(range(1,2*n+1));K=self.vehicles or math.ceil(n/6)
        self.K=K;xy=[s['depot']]+s['loc'];lo={i:float(s['time_windows'][i-1][0]) for i in V};hi={i:float(s['time_windows'][i-1][1]) for i in V}
        st=float(c['OPERATION_START']);en=float(c['OPERATION_END']);srv=float(c['SERVICE_TIME']);M=2*(en-st)+2*srv+2
        d={(i,j):math.dist(xy[i],xy[j])*c['AREA_SIZE'] for i in [0]+V for j in [0]+V if i!=j}
        tr={a:v/c['VEHICLE_SPEED'] for a,v in d.items()}
        # Safe pruning uses only necessary time-window/precedence conditions.
        arcs=[]
        for i,j in d:
            if i==0 and j>n:continue
            if j==0 and i<=n:continue
            if i>n and j==i-n:continue
            earliest_i=max(st+tr.get((0,i),0),lo.get(i,st)) if i else st
            if j and i and earliest_i+srv+tr[i,j]>en-srv+1e-9:continue
            arcs.append((i,j))
        m=Model('Selective_SARP_cost4_MILP');self.model=m
        m.setRealParam('limits/time',float(self.time_limit));m.setRealParam('limits/gap',float(self.gap));m.setIntParam('parallel/maxnthreads',self.threads);m.setIntParam('randomization/randomseedshift',self.seed);m.setRealParam('numerics/feastol',1e-8)
        if not self.verbose:m.hideOutput()
        x={a:m.addVar(f'x_{a[0]}_{a[1]}',vtype='B') for a in arcs};self.x=x
        z={i:m.addVar(f'accept_{i}',vtype='B') for i in range(1,n+1)};self.z=z
        a={i:m.addVar(f'arrival_{i}',lb=st,ub=en-srv) for i in V}
        t={i:m.addVar(f'service_{i}',lb=st,ub=en-srv) for i in V}
        w={i:m.addVar(f'wait_{i}',vtype='B') for i in V}
        label={i:m.addVar(f'route_label_{i}',lb=0,ub=n) for i in V}
        ds={i:m.addVar(f'dispatch_time_{i}',lb=st,ub=en) for i in V}
        loads=[{i:m.addVar(f'load_{r}_{i}',lb=0,ub=cap) for i in V} for r,cap in enumerate((1.,1.,6.))]
        dem=[{i:float(s[key][i-1]) for i in V} for key in ('demand_passenger','demand_cargo')]+[{i:1. if i<=n else -1. for i in V}]
        def choose(i):return z[(i-1)%n+1]
        for i in V:
            m.addCons(quicksum(x[u,v] for u,v in arcs if v==i)==choose(i))
            m.addCons(quicksum(x[u,v] for u,v in arcs if u==i)==choose(i))
            # max(arrival, earliest): upper envelopes prevent arbitrary waiting.
            m.addCons(t[i]>=a[i]);m.addCons(t[i]>=lo[i]);m.addCons(t[i]<=a[i]+M*w[i]);m.addCons(t[i]<=lo[i]+M*(1-w[i]))
            if i<=n and (s['node_type'][i-1]==1 or c['HARD_CARGO_PICKUP_TIMEWINDOW']):m.addCons(t[i]<=hi[i]+M*(1-z[i]))
        m.addCons(quicksum(v for (i,j),v in x.items() if i==0)<=K)
        m.addCons(quicksum(v for (i,j),v in x.items() if i==0)==quicksum(v for (i,j),v in x.items() if j==0))
        for i in range(1,n+1):
            j=i+n;m.addCons(label[i]==label[j]);m.addCons(t[j]>=t[i]+srv-M*(1-z[i]))
            if s['node_type'][i-1]==1:
                ride=min(c['PASSENGER_MAX_RIDE_TIME_MINUTES']/60,tr[i,j]+c['PASSENGER_MAX_EXCESS_RIDE_TIME_MINUTES']/60)
                m.addCons(t[j]-t[i]-srv<=ride+M*(1-z[i]))
        energy_terms=[];arc_loads={}
        for (i,j),v in x.items():
            if i==0:
                first=max(st+tr[i,j],lo[j]);depart=max(st,lo[j]-tr[i,j])
                for var,value,bound in ((a[j],first,M),(ds[j],depart,M),(label[j],j,n)):
                    m.addCons(var-value<=bound*(1-v));m.addCons(var-value>=-bound*(1-v))
                for r,cap in enumerate((1.,1.,6.)):
                    m.addCons(loads[r][j]-dem[r][j]<=2*cap*(1-v));m.addCons(loads[r][j]-dem[r][j]>=-2*cap*(1-v))
            elif j==0:
                m.addCons(t[i]+srv+tr[i,j]<=en+M*(1-v));m.addCons(t[i]+srv+tr[i,j]-ds[i]<=c['MAX_TRIP_TIME']+M*(1-v))
                for r,cap in enumerate((1.,1.,6.)):m.addCons(loads[r][i]<=cap*(1-v))
            else:
                for lhs,big in ((a[j]-t[i]-srv-tr[i,j],M),(ds[j]-ds[i],M),(label[j]-label[i],n)):
                    m.addCons(lhs<=big*(1-v));m.addCons(lhs>=-big*(1-v))
                for r,cap in enumerate((1.,1.,6.)):
                    delta=loads[r][j]-loads[r][i]-dem[r][j]
                    m.addCons(delta<=2*cap*(1-v));m.addCons(delta>=-2*cap*(1-v))
            # Exact product load_after_i * arc, McCormick with binary arc.
            load_terms=[]
            if i:
                for r,weight in ((0,c['PASSENGER_CAPACITY']*c['PASSENGER_WEIGHT_KG']),(1,c['CARGO_CAPACITY']*c['CARGO_UNIT_WEIGHT_KG'])):
                    g=m.addVar(f'arc_load_{r}_{i}_{j}',lb=0,ub=1)
                    arc_loads[r,i,j]=g
                    m.addCons(g<=v);m.addCons(g<=loads[r][i]);m.addCons(g>=loads[r][i]-(1-v));load_terms.append(weight*g)
            energy_terms.append(d[i,j]*c['ENERGY_BASE']*c['ELECTRICITY_PRICE']*(v+quicksum(load_terms)/c['ENERGY_WEIGHT_REF']))
        late={i:m.addVar(f'late_minutes_{i}',lb=0,ub=60*(en-st)) for i in range(n+1,2*n+1)}
        for i,l in late.items():m.addCons(l>=60*(a[i]-hi[i])-60*M*(1-choose(i)));m.addCons(l<=60*(en-st)*choose(i))
        pd=quicksum(l for i,l in late.items() if s['node_type'][i-1]==1)*c['PASSENGER_DELAY_COST']
        cd=quicksum(l for i,l in late.items() if s['node_type'][i-1]==0)
        cc=m.addVar('cargo_instance_piecewise_cost',lb=0)
        b1=c['CARGO_DELAY_TIER1_MIN'];b2=c['CARGO_DELAY_TIER2_MIN'];r1=c['CARGO_DELAY_COST'];r2=c['CARGO_DELAY_COST_30_60'];r3=c['CARGO_DELAY_COST_60_PLUS']
        if not 0<=r1<=r2<=r3:raise ValueError('Convex tier rates required')
        m.addCons(cc>=r1*cd);m.addCons(cc>=r2*cd+(r1-r2)*b1);m.addCons(cc>=r3*cd+(r1-r2)*b1+(r2-r3)*b2)
        energy=quicksum(energy_terms);vc=quicksum(v for (i,j),v in x.items() if i==0)*c['VEHICLE_COST'];rp=c['ALPHA_REJECT']*(n-quicksum(z.values()))
        objective=c['ALPHA_ENERGY']*energy/p['energy_cost_raw']+c['ALPHA_DELAY']*(pd/p['passenger_delivery_delay_cost_raw']+cc/p['cargo_delay_cost_raw'])+c['ALPHA_VEHICLE']*vc/max(1.,p['vehicle_cost_norm'])+rp
        m.setObjective(objective,'minimize');self.build_seconds=time.monotonic()-start
        self.variables=dict(a=a,t=t,w=w,label=label,ds=ds,loads=loads,late=late,cc=cc,arc_loads=arc_loads)
        self.original_variables=m.getNVars();self.original_constraints=m.getNConss()
        return m

    def add_route_start(self,actions):
        """Complete primal start; checked by SCIP, independently audited by caller."""
        m=self.model;s=self.s;n=self.n;c=self.c;v=self.variables;sol=m.createSol()
        active_edges=set();values={};accepted=set();cur=0;route_label=0;lp=lc=opened=0.;t=c['OPERATION_START'];depart=t
        xy=[s['depot']]+s['loc']
        for node in actions:
            if node==2*n+1:continue
            if cur==0 and node==0:continue
            active_edges.add((cur,node))
            travel=math.dist(xy[cur],xy[node])*c['AREA_SIZE']/c['VEHICLE_SPEED']
            if node==0:cur=0;t=c['OPERATION_START'];lp=lc=opened=0.;continue
            lo=s['time_windows'][node-1][0]
            if cur==0:
                depart=max(c['OPERATION_START'],lo-travel);t=depart;route_label=node
            arrival=t+travel;service=max(arrival,lo);t=service+c['SERVICE_TIME']
            lp+=s['demand_passenger'][node-1];lc+=s['demand_cargo'][node-1];opened+=1 if node<=n else -1
            accepted.add((node-1)%n+1)
            values[node]=dict(a=arrival,t=service,w=int(arrival<lo),label=route_label,ds=depart,loads=(lp,lc,opened))
            cur=node
        if not active_edges.issubset(self.x):raise ValueError('Warm start uses a pruned arc')
        for edge,var in self.x.items():m.setSolVal(sol,var,int(edge in active_edges))
        for i,var in self.z.items():m.setSolVal(sol,var,int(i in accepted))
        for i in range(1,2*n+1):
            value=values.get(i,dict(a=max(c['OPERATION_START'],s['time_windows'][i-1][0]),t=max(c['OPERATION_START'],s['time_windows'][i-1][0]),w=0,label=0,ds=c['OPERATION_START'],loads=(0.,0.,0.)))
            for key in ('a','t','w','label','ds'):m.setSolVal(sol,v[key][i],value[key])
            for r in range(3):m.setSolVal(sol,v['loads'][r][i],value['loads'][r])
            if i>n:m.setSolVal(sol,v['late'][i],max(0.,value['a']-s['time_windows'][i-1][1])*60 if i in values else 0.)
        for (r,i,j),var in v['arc_loads'].items():m.setSolVal(sol,var,values[i]['loads'][r] if (i,j) in active_edges else 0.)
        m.setSolVal(sol,v['cc'],scalar_cost(s,actions,c,self.profile)['cargo_delay_cost'])
        if not m.checkSol(sol,printreason=False,completely=True):raise ValueError('SCIP rejected complete route start')
        return m.addSol(sol,free=True)

    def solve(self,sample,logfile=None,model_file=None,warm_actions=None):
        m=self.build(sample)
        if logfile:m.setLogfile(str(logfile))
        if model_file:m.writeProblem(str(model_file))
        # Feasible fallback is always available; it is not evidence of good quality.
        sol=m.createSol();n=self.n;s=self.s;c=self.c;vrs=self.variables
        for i in range(1,2*n+1):
            earliest=max(c['OPERATION_START'],s['time_windows'][i-1][0])
            m.setSolVal(sol,vrs['a'][i],earliest);m.setSolVal(sol,vrs['t'][i],earliest)
            m.setSolVal(sol,vrs['ds'][i],c['OPERATION_START'])
        fallback_added=m.addSol(sol,free=True)
        warm_added=self.add_route_start(warm_actions) if warm_actions is not None else False
        begin=time.monotonic();m.optimize();elapsed=time.monotonic()-begin
        result=dict(solver='SCIP',version=f'{m.getMajorVersion()}.{m.getMinorVersion()}.{m.getTechVersion()}',status=str(m.getStatus()),build_seconds=self.build_seconds,solve_seconds=elapsed,variables=self.original_variables,constraints=self.original_constraints,nodes=m.getNNodes(),time_limit_seconds=self.time_limit,threads=self.threads,seed=self.seed,relative_gap_limit=self.gap,all_reject_fallback_added=fallback_added,cost_version=4,physical_constraints='no residual; max_open6; min_dispatch4 reported only; earliest-service deterministic timing',search_space='physical route MILP; no neural masks or one-step viability pruning; not identical to neural construction search space')
        result['dual_bound']=float(m.getDualbound())
        result['warm_start_added']=warm_added
        if m.getNSols()==0:result.update(actions=None,objective=None,primal_bound=None,gap=None);return result
        best=m.getBestSol();edges=[e for e,v in self.x.items() if m.getSolVal(best,v)>.5]
        successor={i:j for i,j in edges if i};routes=[];visited=set()
        for _,first in sorted(e for e in edges if e[0]==0):
            route=[];cur=first
            while cur:
                if cur in visited:raise RuntimeError('Cycle or duplicate in extracted routes')
                visited.add(cur);route.append(cur);cur=successor[cur]
            routes.append(route)
        if len(visited)!=sum(i!=0 for i,j in edges):raise RuntimeError('Disconnected solution')
        rejected=sum(m.getSolVal(best,z)<.5 for z in self.z.values());actions=[]
        for route in routes:actions.extend(route+[0])
        actions.extend([2*n+1]*rejected)
        score=scalar_cost(sample,actions,c,self.profile)
        result.update(actions=actions,routes=routes,completed=n-rejected,rejected=rejected,service_rate=(n-rejected)/n,objective=float(m.getObjVal()),primal_bound=float(m.getPrimalbound()),gap=float(m.getGap()),replayed=score)
        result['relative_gap_to_primal']=max(0.,result['primal_bound']-result['dual_bound'])/max(1e-10,abs(result['primal_bound']))
        result['objective_replay_error']=abs(result['objective']-score['objective'])
        if result['objective_replay_error']>max(.02,1e-6*abs(score['objective'])):raise RuntimeError(f'MILP objective differs from route replay: {result}')
        return result
