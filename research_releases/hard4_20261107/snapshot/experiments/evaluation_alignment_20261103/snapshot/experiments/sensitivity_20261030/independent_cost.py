"""Scalar physical-cost replay, independent of tensor cost/state implementation."""
import math

def replay(s,actions,c):
 n=len(s['loc'])//2;coords=[s['depot']]+s['loc'];cur=0;t=c['OPERATION_START'];dispatch=t;lp=lc=0.;distance=energy=pd=cd=overtime=0.;vehicles=reject=0;completed=set()
 for node in actions:
  if node==2*n+1:reject+=1;continue
  if node==0 and cur==0:continue
  km=math.dist(coords[cur],coords[node])*c['AREA_SIZE'];travel=km/c['VEHICLE_SPEED']
  distance+=km;weight=lp*c['PASSENGER_CAPACITY']*c['PASSENGER_WEIGHT_KG']+lc*c['CARGO_CAPACITY']*c['CARGO_UNIT_WEIGHT_KG']
  energy+=km*c['ENERGY_BASE']*(1+weight/c['ENERGY_WEIGHT_REF'])*c['ELECTRICITY_PRICE']
  if cur==0:
   vehicles+=1;dispatch=max(c['OPERATION_START'],s['time_windows'][node-1][0]-travel);t=dispatch
  arrival=t+travel
  if node==0:
   overtime+=max(0.,arrival-dispatch-c['MAX_TRIP_TIME']);t=c['OPERATION_START'];lp=lc=0.;cur=0;continue
  i=node-1;tw=s['time_windows'][i]
  if node>n:
   completed.add(i-n);delay=max(0.,arrival-tw[1])*60
   if s['node_type'][i]==1:pd+=delay
   else:cd+=delay
  t=max(arrival,tw[0])+c['SERVICE_TIME'];lp+=s['demand_passenger'][i];lc+=s['demand_cargo'][i];cur=node
 if cur: overtime+=max(0.,t-dispatch-c['MAX_TRIP_TIME'])
 cargo=min(cd,c['CARGO_DELAY_TIER1_MIN'])*c['CARGO_DELAY_COST']+max(0.,min(cd,c['CARGO_DELAY_TIER2_MIN'])-c['CARGO_DELAY_TIER1_MIN'])*c['CARGO_DELAY_COST_30_60']+max(0.,cd-c['CARGO_DELAY_TIER2_MIN'])*c['CARGO_DELAY_COST_60_PLUS']
 return dict(total_distance=distance,energy_cost_raw=energy,total_cost_raw=energy+pd*c['PASSENGER_DELAY_COST']+cargo+vehicles*c['VEHICLE_COST']+reject*c['ALPHA_REJECT']+max(0,n-len(completed)-reject)*c['ALPHA_UNFULFILLED']+overtime*c['ALPHA_TRIP_OVERTIME'],trip_overtime_penalty=overtime*c['ALPHA_TRIP_OVERTIME'])
