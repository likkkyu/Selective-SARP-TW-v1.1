"""Freeze methods/seeds before generating one new common confirmation set."""
from common import *
import json,hashlib,time,platform,sys
P=Path(__file__).resolve().parent


def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
 pre=json.loads((P/'PREFLIGHT_REVIEW.json').read_text());assert pre['status']=='passed'
 base=json.loads((P/'baseline_entry_preflight_v2/report.json').read_text());assert base['status']=='completed' and len(base['rows'])==1
 dev=json.loads((ROOT/'experiments/hard4_lns_20261106/SUMMARY.json').read_text());assert dev['status']=='passed' and dev['LNS_runs']==24
 out=P/'confirmation_v1';out.mkdir(exist_ok=False)
 scenario=out/'common_n50_seed9416377.pt';assert not scenario.exists()
 models=json.loads((ROOT/'experiments/paper_delivery_20261104/confirmation_v1/plan.json').read_text())['models']
 for m in models:assert digest(Path(m['checkpoint']))==m['sha256']
 files=set(SNAP.rglob('*.py'))
 for folder in [P,ROOT/'experiments/dispatch_clock_fix_20261106',ROOT/'experiments/hard4_speed_20261106',ROOT/'experiments/min_dispatch_hard_20261106',ROOT/'experiments/mip_scip_20261106',ROOT/'experiments/hard4_lns_20261106']:
  files.update(folder.glob('*.py'))
 source={str(f.relative_to(ROOT)):digest(f) for f in sorted(files) if '__pycache__' not in str(f)}
 plan=dict(scope='new common 32-instance confirmation; user-approved main comparison: pretrained learned/rule deployments, strict insertion, LNS and SCIP; no retraining or pure training-component ablation',created_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),user_decision='explicit confirmation: strong LNS and SCIP as main traditional controls; SA/GA reviewed auxiliary evidence, no short-run ranking',scenario=str(scenario.resolve()),scenario_seed=9416377,scenario_sha256=None,samples=32,graph_size=50,models=models,search_seeds=[1234,4321,8765],lns_seconds=10.,lns_max_iterations=1000,scip_seconds=10.,scip_seed=1234,scip_warm_start='same strict insertion initial plan as LNS, not best-of-three LNS outputs',neural_batch_size=1,neural_timeout_seconds=3600,baseline_timeout_seconds=3600,device='cpu',threads=1,host_platform=platform.platform(),physical_config=C,normalization_profile=PROFILE,hard_minimum=4,cost_version=4,source_sha256=source,statistics='32 common instances; retain three trained seeds and three search seeds separately; instance bootstrap conditional on frozen models, not training-population inference; no optional stopping or outcome-based exclusions',timing='sequential methods on same CPU; loading, algorithm and audit separately; all construction/certificate overhead charged to neural; initialization charged in full to each baseline run; LNS 10s is search-only, report actual end-to-end time',audit_failure='stop and preserve original evidence; do not drop instances; any changed algorithm invalidates this as untouched method-selection data and requires a new protocol')
 # This immutable record precedes the first call to the generator.
 (out/'selection_before_generation.json').write_text(json.dumps(plan,indent=2))
 sys.path.insert(0,str(SNAP/'experiments/sensitivity_20261030'))
 from scenarios import make,save
 bundle=make(n=50,samples=32,seed=plan['scenario_seed'],ratio=None,tw=1.,excess=1.)
 save(bundle,scenario);plan['scenario_sha256']=digest(scenario)
 (out/'plan.json').write_text(json.dumps(plan,indent=2));(out/'STATUS.json').write_text(json.dumps(dict(status='prepared',neural='not_started',baselines='not_started'),indent=2))
 print('FROZEN before generation:',len(source),'source files and six checkpoints; generated exactly 32 new common N50 instances; seed',plan['scenario_seed'],'sha',plan['scenario_sha256'],flush=True)
if __name__=='__main__':main()
