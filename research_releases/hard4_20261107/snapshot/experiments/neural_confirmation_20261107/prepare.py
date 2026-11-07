"""Record candidate selection before creating new independent common data."""
from pathlib import Path
import hashlib,json,platform,sys,time
ROOT=Path(__file__).resolve().parents[2];P=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'experiments/hard4_confirmation_20261106'))
from common import *


def main():
    assert json.loads((P/'preflight_neural/report.json').read_text())['status']=='completed'
    assert json.loads((P/'preflight_controls/report.json').read_text())['status']=='completed'
    out=P/'confirmation_v1';out.mkdir(exist_ok=False)
    old=json.loads((ROOT/'experiments/neural_diagnosis_20261107/FAST_DEVELOPMENT_PLAN.json').read_text())
    files=dict(old['source_sha256'])
    for f in P.glob('*.py'):files[str(f.relative_to(ROOT))]=hashlib.sha256(f.read_bytes()).hexdigest()
    for f,h in files.items():assert hashlib.sha256((ROOT/f).read_bytes()).hexdigest()==h,f
    for m in old['models']:assert hashlib.sha256(Path(m['checkpoint']).read_bytes()).hexdigest()==m['sha256']
    plan=dict(scope='Fresh common N50 confirmation of frozen CPU-v5/native neural starts with and without matched short LNS; strict insertion and matched short/strong LNS controls. No retraining. First-round 32-case results remain archived; this is a distinct protocol, not selective replacement.',
      created_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
      selection='OLD eight-case development: native chosen for common hybrid protocol because learned mean J with short LNS is lower than service-priority variant. Both arms receive same native policy and exact same LNS. No new confirmation results used for selection.',
      scenario=str((out/'common_n50_seed20711369.pt').resolve()),scenario_seed=20711369,scenario_sha256=None,samples=32,graph_size=50,models=old['models'],source_sha256=files,
      physical_config=C,normalization_profile=PROFILE,device='cpu',threads=1,host=platform.platform(),guard=5,depot_policy='native',hard_minimum=4,cost_version=4,
      hybrid_search_seconds=.5,hybrid_search_seed=1234,strong_search_seconds=10.,lns_max_iterations=1000,
      neural_timeout_seconds=1800,baseline_timeout_seconds=1800,
      search_replication='One fixed search seed for every matched postprocessor and strong control, not three independent algorithm runs. Three pretrained checkpoint seeds per neural arm. Do not infer search-seed stability.',
      statistics='32 paired instances; average fixed checkpoints within each instance, bootstrap common instance indices 10000 times, seed8162049. Conditional intervals, not inference over training populations. Seed-specific results separately; no p-values or optional stopping.',
      timing='Sequential pipelines, actual measured integrated time includes fresh engine and all initialization. Loading and final audit separately. LNS budgets search-only. Wall-clock limits imply host/load dependence.',
      safety='Fail on any audit, cost, metadata or hash mismatch; preserve all outcomes, never drop/replace cases. No training changes. No further tuning on this set.')
    (out/'selection_before_generation.json').write_text(json.dumps(plan,indent=2))
    sys.path.insert(0,str(SNAP/'experiments/sensitivity_20261030'))
    from scenarios import make,save
    save(make(n=50,samples=32,seed=plan['scenario_seed'],ratio=None,tw=1.,excess=1.),Path(plan['scenario']))
    plan['scenario_sha256']=hashlib.sha256(Path(plan['scenario']).read_bytes()).hexdigest()
    (out/'plan.json').write_text(json.dumps(plan,indent=2))
    print('FROZEN',len(files),'sources, six unchanged checkpoints, 32 fresh common instances',plan['scenario_sha256'],flush=True)


if __name__=='__main__':main()
