"""Sequential bounded inference; stops on first failed independent audit."""
import argparse, hashlib, json, subprocess, sys, time
from pathlib import Path
from scenarios import make, save, ROOT
p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True);p.add_argument('--output',required=True);p.add_argument('--preflight',action='store_true');a=p.parse_args()
out=Path(a.output);out.mkdir(parents=True,exist_ok=False)
# Explicit base ratio matches N50 default .6. Deduplicate factorial intersections.
settings=[(.6,1.,1.)]
settings += [(r,1.,1.) for r in (.3,.5,.7,.9)]
settings += [(.6,w,1.) for w in (.5,.75,1.25,1.5)]
settings += [(.6,1.,e) for e in (.5,.75,1.25,1.5)]
settings += [(r,w,1.) for r in (.3,.6,.9) for w in (.75,1.,1.25)]
settings=list(dict.fromkeys(settings))
if a.preflight:settings=[(.6,1.,1.),(.3,.5,.5),(.9,1.5,1.5)]
n=4 if a.preflight else 500
manifest=dict(status='running',samples=n,seed=928631,settings=settings,checkpoint_sha256=hashlib.sha256(Path(a.checkpoint).read_bytes()).hexdigest(),scope='single-model frozen-policy sensitivity; multi-seed comparison pending',started=time.time(),completed=[])
manifest['source_sha256']={str(x.relative_to(ROOT)):hashlib.sha256(x.read_bytes()).hexdigest() for x in [ROOT/'evaluate_model.py', ROOT/'state_mcvrptw_v2.py',ROOT/'problem_mcvrptw_v2.py',*Path(__file__).parent.glob('*.py')]}
def record(): (out/'manifest.json').write_text(json.dumps(manifest,indent=2))
record()
try:
 for i,(r,w,e) in enumerate(settings):
    name=f'r{r:g}_tw{w:g}_ex{e:g}'
    path=out/'scenarios'/(name+'.pt'); save(make(samples=n,ratio=r,tw=w,excess=e),path)
    cmd=[sys.executable,str(Path(__file__).parent/'evaluate.py'),'--scenario',str(path),'--output-dir',str(out/name),'--checkpoint',a.checkpoint,'--batch-size','16','--max-concurrent-open-orders','6','--enable-delivery-viability']
    t=time.time()
    with (out/(name+'.log')).open('w') as log:subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,check=True)
    manifest['completed'].append(dict(scenario=name,seconds=time.time()-t));record()
 manifest['status']='completed'
except BaseException as exc:
 manifest['status']='failed';manifest['error']=repr(exc);raise
finally:
 manifest['ended']=time.time();record()
