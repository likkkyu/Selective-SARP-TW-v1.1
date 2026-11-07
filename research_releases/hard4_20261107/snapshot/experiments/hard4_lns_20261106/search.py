"""Bounded destroy/repair LNS, not an implementation of published ALNS.

Intermediate repair routes may have <4 pairs, final proposals never may.
Returned best cannot worsen the supplied audited initial plan.
"""
import random,time,math


class SearchTimeout(Exception): pass


def insertion_options(engine, routes, order, deadline):
    choices=[]
    for rid in range(len(routes)+(len(routes)<engine.k)):
        route=routes[rid] if rid<len(routes) else ()
        for p in range(len(route)+1):
            first=route[:p]+(order,)+route[p:]
            for q in range(p+1,len(first)+1):
                if time.monotonic()>deadline: raise SearchTimeout()
                new=first[:q]+(order+engine.n,)+first[q:]
                if engine.route(new) is None: continue
                trial=routes.copy()
                if rid==len(routes): trial.append(new)
                else: trial[rid]=new
                score=engine.cost(trial,strict=False)
                if score is not None: choices.append((score['objective'],rid,new))
    return sorted(choices,key=lambda x:(x[0],x[1],x[2]))


def repair(engine,routes,pool,rng,deadline,regret=True):
    # Deleting an earlier stop can create additional downstream waiting and
    # violate another passenger's ride limit. Do not assume deletion preserves
    # physical feasibility: dissolve any such provisional tour before repair.
    pending=set(pool); valid=[]
    for r in routes:
        if not r: continue
        if engine.route(r) is None: pending.update(x for x in r if x<=engine.n)
        else: valid.append(r)
    routes=valid
    while pending:
        options=[]
        for order in sorted(pending):
            choices=insertion_options(engine,routes,order,deadline)
            if choices:
                # Distinct node placements, deterministic ties; objective includes
                # the instance-level cargo tier, also for provisional short tours.
                delta=(choices[1][0]-choices[0][0]) if len(choices)>1 else engine.c['ALPHA_REJECT']
                options.append(((-delta if regret else choices[0][0]),rng.random(),order,choices[0]))
        if not options: break
        _,_,order,(_,rid,new)=min(options)
        if rid==len(routes): routes.append(new)
        else: routes[rid]=new
        pending.remove(order)
    # Dissolve short tours, try to place their complete pairs on valid tours.
    dropped=[x for r in routes if len(r)//2<4 for x in r if x<=engine.n]
    routes=[r for r in routes if len(r)//2>=4]
    for order in sorted(dropped):
        choices=insertion_options(engine,routes,order,deadline)
        choices=[v for v in choices if v[1]<len(routes)]
        if choices:
            _,rid,new=choices[0]; routes[rid]=new
    assert engine.cost(routes) is not None
    return routes


def solve(engine,initial,seed=1234,seconds=10.,max_iterations=1000,progress=None):
    if seconds<=0 or not math.isfinite(seconds): raise ValueError('Finite positive budget required')
    rng=random.Random(seed); begin=time.monotonic(); deadline=begin+seconds
    current=[tuple(r) for r in initial]; best=current.copy(); bestcost=engine.cost(best)
    if bestcost is None: raise ValueError('Initial plan is not hard4 feasible')
    currentcost=bestcost; history=[dict(iteration=0,seconds=0.,**bestcost)]; iterations=0; partial=False
    for it in range(1,max_iterations+1):
        if time.monotonic()>=deadline: break
        ids=[x for r in current for x in r if x<=engine.n]
        if not ids: break
        mode=(it-1)%3; count=min(len(ids),rng.randint(4,10))
        if mode==0: removed=set(rng.sample(ids,count))
        elif mode==1:
            anchor=rng.choice(ids)
            removed=set(sorted(ids,key=lambda x:engine.distance[anchor][x]+abs(engine.s['time_windows'][anchor-1][0]-engine.s['time_windows'][x-1][0]))[:count])
        else: removed={x for x in rng.choice(current) if x<=engine.n}
        trial=[tuple(x for x in r if (x-1)%engine.n+1 not in removed) for r in current]
        assigned={x for r in trial for x in r if x<=engine.n}
        pool=set(range(1,engine.n+1))-assigned
        try: trial=repair(engine,trial,pool,rng,deadline,regret=it%2==1)
        except SearchTimeout: partial=True; break
        score=engine.cost(trial); iterations=it
        # Cooling scaled by time; saved best always independent of acceptance.
        temperature=25.*max(.01,1-(time.monotonic()-begin)/seconds)
        if score['objective']<=currentcost['objective'] or rng.random()<math.exp(min(0.,(currentcost['objective']-score['objective'])/temperature)):
            current=trial; currentcost=score
        if score['objective']<bestcost['objective']-1e-10:
            best=trial.copy(); bestcost=score; record=dict(iteration=it,seconds=time.monotonic()-begin,**score); history.append(record)
            if progress: progress(record)
    return dict(routes=best,actions=engine.actions(best),cost=bestcost,history=history,iterations=iterations,seconds=time.monotonic()-begin,partial_last_iteration_discarded=partial,cache_hits=engine.hits,route_evaluations=engine.evaluations)
