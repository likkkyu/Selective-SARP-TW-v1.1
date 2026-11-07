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
    completed_since_dispatch=0
    for node in actions:
        if node==0:
            completed_since_dispatch=0
        elif 1<=node<=2*n:
            completed_since_dispatch+=int(node>n)
            m.setSolVal(sol,v['trip_completed'][node],completed_since_dispatch)
    if not m.checkSol(sol,printreason=False,completely=True):raise ValueError('SCIP rejected complete route start')
    return m.addSol(sol,free=True)
