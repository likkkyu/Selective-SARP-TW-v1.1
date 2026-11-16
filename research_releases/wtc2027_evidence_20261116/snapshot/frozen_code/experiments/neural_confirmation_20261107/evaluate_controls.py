"""Integrated timing for strict insertion + matched short/strong LNS controls."""
from pathlib import Path
import argparse, hashlib, json, signal, sys, time
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'experiments/hard4_confirmation_20261106'))
from common import *
from evaluate_baselines import verify, make_start, solve, RouteEngine, split_actions


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--plan',required=True);parser.add_argument('--output',required=True);args=parser.parse_args()
    plan=json.loads(Path(args.plan).read_text());out=Path(args.output);out.mkdir(parents=True,exist_ok=False)
    for file, sha in plan['source_sha256'].items():
        assert hashlib.sha256((ROOT/file).read_bytes()).hexdigest()==sha,file
    source=Path(plan['scenario']);assert hashlib.sha256(source.read_bytes()).hexdigest()==plan['scenario_sha256']
    data=torch.load(source,weights_only=False)['data'][:plan['samples']];assert len(data)==plan['samples']
    Config.set_normalization_profile(50,PROFILE)
    started=time.monotonic();signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('Finite control budget')));signal.alarm(plan['baseline_timeout_seconds'])
    report=dict(status='running',scope=plan['scope'],scenario_sha256=plan['scenario_sha256'],rows=[],timing='Actual pipeline includes fresh insertion initialization, fresh RouteEngine and complete search. Audits/replay separately; no summed historical timings.')
    def save():
        report['seconds']=time.monotonic()-started;(out/'report.json').write_text(json.dumps(report,indent=2))
    save()
    try:
        for i,sample in enumerate(data):
            entry=dict(instance=i)
            for label,budget in [('short',plan['hybrid_search_seconds']),('strong',plan['strong_search_seconds'])]:
                begin=time.monotonic();initial=make_start(sample);initial_seconds=time.monotonic()-begin
                engine=RouteEngine(sample,C,PROFILE)
                result=solve(engine,split_actions(initial['actions'],50),plan['hybrid_search_seed'],budget,plan['lns_max_iterations'])
                integrated=time.monotonic()-begin
                audit_begin=time.monotonic();checked=verify(sample,result['actions']);initial_check=verify(sample,initial['actions']);audit_seconds=time.monotonic()-audit_begin
                assert checked['cost']['objective']<=initial_check['cost']['objective']+1e-8
                if 'initial' in entry:assert entry['initial']['actions']==initial['actions']
                else:entry['initial']=dict(actions=initial['actions'],algorithm_seconds=initial_seconds,**initial_check)
                entry[label]=dict(result,integrated_algorithm_seconds=integrated,initialization_seconds=initial_seconds,final_audit_seconds=audit_seconds,**checked)
                (out/f'instance_{i}.json').write_text(json.dumps(entry,indent=2))
                print('CONTROL',i+1,'/',len(data),label,'served',checked['audit']['completed'],'J',round(checked['cost']['objective'],4),'integrated_s',round(integrated,3),flush=True)
            report['rows'].append(dict(instance=i,initial_completed=entry['initial']['audit']['completed'],short_completed=entry['short']['audit']['completed'],strong_completed=entry['strong']['audit']['completed']));save()
        report['status']='completed'
    except BaseException as exc:
        report.update(status='failed',error=repr(exc));raise
    finally:
        save();signal.alarm(0)


if __name__=='__main__':main()
