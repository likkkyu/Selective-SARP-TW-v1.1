"""Explain exact MILP rejection of physically tolerance-valid neural traces."""
import inspect,json,textwrap
from validate import P,ROOT,C,PROFILE,torch
import scip_selective as module
from scip_selective import SelectiveSCIP

def main():
 out=P/'membership_tolerance_diagnosis_v1';out.mkdir(exist_ok=False)
 old=json.loads((P/'neural_routes_mip_membership_v1/report.json').read_text());bad=[x for x in old['rows'] if not x['mip_feasible']]
 data=torch.load(ROOT/'experiments/paper_delivery_20261104/confirmation_n50_seed803219.pt',map_location='cpu',weights_only=False)['data']
 source=textwrap.dedent(inspect.getsource(SelectiveSCIP.add_route_start)).replace('printreason=False','printreason=True')
 scope={};exec(source,vars(module),scope);SelectiveSCIP.add_route_start=scope['add_route_start'];rows=[]
 for record in bad:
  i=record['instance'];label=record['model'];trace=[json.loads(x) for x in (ROOT/'experiments/paper_delivery_20261104/confirmation_v1/models'/label/'routes.jsonl').read_text().splitlines()][i]
  solver=SelectiveSCIP(C,PROFILE,time_limit=1,verbose=True);m=solver.build(data[i]);m.setLogfile(str(out/f'{label}_{i}.log'))
  tests=[]
  for tol in (1e-8,1e-7,1e-6,1e-5):
   m.setRealParam('numerics/feastol',tol)
   try:solver.add_route_start(trace['actions']);valid=True
   except ValueError:valid=False
   tests.append(dict(feastol=tol,mip_membership=valid));print('TOLERANCE',label,i,tol,valid,flush=True)
  rows.append(dict(instance=i,model=label,checks=tests));m.freeProb()
 (out/'report.json').write_text(json.dumps(dict(status='completed',scope='membership sensitivity only; does not change original solve tolerances or results',rows=rows),indent=2))
if __name__=='__main__':main()
