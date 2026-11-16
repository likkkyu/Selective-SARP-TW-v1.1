import ast,hashlib,json
from pathlib import Path
P=Path(__file__).resolve().parent;ROOT=P.parents[1]
def main():
 audit=json.loads((P/'target_audit_v1/report.json').read_text());bench=json.loads((P/'paired_benchmark_v1/report.json').read_text());search=json.loads((P/'search_smoke_v1/report.json').read_text());model=json.loads((P/'optimized_dev4_v1/report.json').read_text());unit=json.loads((P/'validation_v1/report.json').read_text())
 assert audit['status']==bench['status']==search['status']==unit['status']=='passed' and model['status']=='completed'
 assert len(model['models'])==6 and audit['stats']['target_verified']==6310
 cases=0
 for m in model['models']:
  rows=json.loads((P/'optimized_dev4_v1'/f"{m['model']}.json").read_text())['rows'];reference=json.loads((ROOT/'experiments/hard4_admission_20261106/dev4_v1'/f"{m['model']}.json").read_text())['rows']
  for a,b in zip(rows,reference):
   assert a['actions']==b['actions'] and abs(a['objective']-b['objective'])<.02 and not a['audit']['violations'] and not a['audit']['unfulfilled'];cases+=1
 for r in search['rows']:
  assert r['evaluations']==8 and not r['best']['independent_audit']['violations'] and r['best']['independent_audit']['unfulfilled']==0
  assert hashlib.sha256((P/'admission_targeted.py').read_bytes()).hexdigest()==json.loads((P/'search_smoke_v1/plan.json').read_text())['source_sha256']
 old=json.loads((ROOT/'experiments/hard4_admission_20261106/dev4_v1/report.json').read_text())
 result=dict(status='cpu_optimization_and_short_search_validation_passed',optimization_version=3,search_budget_unchanged=2000,full_mask_target_bit_checks=6310,mask_bit_mismatches=0,model_instance_exact_action_matches=cases,strict_model_audit_invalid=0,unfulfilled=0,paired_candidate_timings=bench['summary'],observed_model_reference_seconds=old['seconds'],observed_model_optimized_seconds=model['seconds'],model_timing_scope='single before/after engineering observation; not publication algorithm speedup',short_search=[dict(method=r['method'],evaluations=r['evaluations'],cache_hits=r['cache_hits'],cache_misses=r['cache_misses'],completed=r['best']['independent_audit']['completed'],seconds=r['seconds']) for r in search['rows']],short_search_is_final_benchmark=False,weights_updated=False,gpu_used=False,automation_created=False,source_sha256={f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in [P/'admission_targeted.py',P/'generated_target_mask_v3.py',P/'eval_optimized.py',P/'search_smoke.py']})
 (P/'SUMMARY.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
 oldsource=json.loads((ROOT/'experiments/dispatch_clock_fix_20261106/PATCH_MANIFEST.json').read_text())
 for f in [ROOT/'state_mcvrptw_v2.py',ROOT/'experiments/evaluation_alignment_20261103/snapshot/state_mcvrptw_v2.py']:assert hashlib.sha256(f.read_bytes()).hexdigest()==oldsource['base_module_sha256']
 for f in P.glob('*.py'):ast.parse(f.read_text())
 statuspath=ROOT/'experiments/min_dispatch_hard_20261106/ACTIVE_STATUS.json';status=json.loads(statuspath.read_text());status.update(status='targeted_cpu_optimization_and_complete_short_searches_validated',latest_evidence='../hard4_speed_20261106/REVIEW.md',formal_comparison_complete=False,next_required='freeze bounded full baseline/strict insertion/SCIP comparison protocol on development data; independent final confirmation; no immediate GPU training')
 statuspath.write_text(json.dumps(status,indent=2))
 review=P/'REVIEW.md';text=review.read_text();text=text.replace('预算内最终结果和完成状态以search_smoke_v1/report.json为准。未完成或失败不得称正式对比完成。',f"两方法均完成8/8次逻辑评价，所有候选与最终方案完整硬审计通过、残留0。SA最终完成{search['rows'][0]['best']['independent_audit']['completed']}/50单，GA完成{search['rows'][1]['best']['independent_audit']['completed']}/50单；只是短预检，不能作为最终方法排名。详见search_smoke_v1/report.json。")
 review.write_text(text)
if __name__=='__main__':main()
