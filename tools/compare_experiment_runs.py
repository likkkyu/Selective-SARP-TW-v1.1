import argparse
import json
from pathlib import Path


EPS = 1e-9


BENCHMARK_PROTOCOL_FIELDS = [
    ('run_context', 'graph_size'),
    ('run_context', 'seed'),
    ('run_context', 'world_size'),
    ('run_context', 'distributed'),
    ('run_context', 'device'),
    ('run_context', 'benchmark_config', 'warmup_epochs'),
    ('run_context', 'benchmark_config', 'skip_validation'),
    ('run_context', 'benchmark_config', 'disable_checkpoint'),
    ('run_context', 'benchmark_config', 'disable_log_save'),
    ('run_context', 'benchmark_config', 'batch_timing'),
    ('run_context', 'training_config', 'batch_size'),
    ('run_context', 'training_config', 'epoch_size'),
    ('run_context', 'training_config', 'pomo_size'),
    ('run_context', 'training_config', 'amp'),
    ('run_context', 'training_config', 'amp_dtype'),
    ('run_context', 'training_config', 'checkpoint_encoder'),
    ('run_context', 'training_config', 'num_workers'),
    ('run_context', 'training_config', 'prefetch_factor'),
    ('run_context', 'training_config', 'persistent_workers'),
    ('run_context', 'training_config', 'pin_memory'),
    ('run_context', 'state_kwargs', 'max_concurrent_open_orders'),
    ('run_context', 'state_kwargs', 'min_orders_per_dispatch'),
    ('run_context', 'state_kwargs', 'enable_delivery_viability'),
    ('run_context', 'state_kwargs', 'enable_viability_fallback'),
    ('run_context', 'state_kwargs', 'relax_pickup_commitment_trip_time'),
]

EVAL_PROTOCOL_FIELDS = [
    ('run', 'graph_size'),
    ('run', 'num_samples'),
    ('run', 'seed'),
    ('run', 'decode'),
    ('run', 'state_kwargs', 'max_concurrent_open_orders'),
    ('run', 'state_kwargs', 'min_orders_per_dispatch'),
    ('run', 'state_kwargs', 'enable_delivery_viability'),
    ('run', 'state_kwargs', 'enable_viability_fallback'),
    ('run', 'state_kwargs', 'relax_pickup_commitment_trip_time'),
]


def _load_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def _safe_get(data, *path, default=None):
    cur = data
    for key in path:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def _pct_delta(candidate, baseline):
    if candidate is None or baseline is None or abs(float(baseline)) <= 1e-12:
        return None
    return (float(candidate) - float(baseline)) / float(baseline) * 100.0


def _fmt(value, digits=3):
    if value is None:
        return 'n/a'
    return f'{float(value):.{digits}f}'


def _collect_missing_fields(data, field_paths):
    missing = []
    for field_path in field_paths:
        if _safe_get(data, *field_path) is None:
            missing.append('.'.join(field_path))
    return missing


def _collect_mismatches(candidate, baseline, field_paths):
    mismatches = {}
    for field_path in field_paths:
        candidate_value = _safe_get(candidate, *field_path)
        baseline_value = _safe_get(baseline, *field_path)
        if candidate_value != baseline_value:
            mismatches['.'.join(field_path)] = {
                'candidate': candidate_value,
                'baseline': baseline_value,
            }
    return mismatches


def _analyze_benchmark(candidate, baseline):
    candidate_missing = _collect_missing_fields(candidate, BENCHMARK_PROTOCOL_FIELDS)
    baseline_missing = _collect_missing_fields(baseline, BENCHMARK_PROTOCOL_FIELDS)
    protocol_mismatches = _collect_mismatches(candidate, baseline, BENCHMARK_PROTOCOL_FIELDS)

    epoch_delta = _pct_delta(_safe_get(candidate, 'key_metrics', 'epoch_total_s'), _safe_get(baseline, 'key_metrics', 'epoch_total_s'))
    train_delta = _pct_delta(_safe_get(candidate, 'key_metrics', 'train_s'), _safe_get(baseline, 'key_metrics', 'train_s'))
    speed_delta = _pct_delta(_safe_get(candidate, 'key_metrics', 'samples_per_s'), _safe_get(baseline, 'key_metrics', 'samples_per_s'))
    decode_delta = _pct_delta(_safe_get(candidate, 'key_metrics', 'decode_get_mask_ms'), _safe_get(baseline, 'key_metrics', 'decode_get_mask_ms'))
    pickup_delta = _pct_delta(_safe_get(candidate, 'key_metrics', 'mask_pickup_commitment_ms'), _safe_get(baseline, 'key_metrics', 'mask_pickup_commitment_ms'))
    viability_delta = _pct_delta(_safe_get(candidate, 'key_metrics', 'mask_delivery_viability_ms'), _safe_get(baseline, 'key_metrics', 'mask_delivery_viability_ms'))
    completion_delta = _pct_delta(_safe_get(candidate, 'key_metrics', 'pc_has_feasible_open_completion_ms'), _safe_get(baseline, 'key_metrics', 'pc_has_feasible_open_completion_ms'))

    measured_epochs_candidate = _safe_get(candidate, 'measured_epochs')
    measured_epochs_baseline = _safe_get(baseline, 'measured_epochs')
    blockers = []

    if candidate_missing or baseline_missing:
        blockers.append('missing_benchmark_provenance')
    if protocol_mismatches:
        blockers.append('benchmark_protocol_mismatch')
    if measured_epochs_candidate is None or measured_epochs_baseline is None or measured_epochs_candidate < 1 or measured_epochs_baseline < 1:
        blockers.append('benchmark_without_measured_epochs')
    if None in (epoch_delta, train_delta, speed_delta):
        blockers.append('missing_benchmark_key_metrics')

    if speed_delta is not None and speed_delta <= -2.0:
        return {
            'verdict': 'reject',
            'blockers': blockers + ['samples_per_s_regression'],
            'protocol_mismatches': protocol_mismatches,
            'missing_fields': {
                'candidate': candidate_missing,
                'baseline': baseline_missing,
            },
        }
    if epoch_delta is not None and epoch_delta >= 2.0:
        return {
            'verdict': 'reject',
            'blockers': blockers + ['epoch_wall_clock_regression'],
            'protocol_mismatches': protocol_mismatches,
            'missing_fields': {
                'candidate': candidate_missing,
                'baseline': baseline_missing,
            },
        }
    if train_delta is not None and train_delta >= 2.0:
        return {
            'verdict': 'reject',
            'blockers': blockers + ['train_time_regression'],
            'protocol_mismatches': protocol_mismatches,
            'missing_fields': {
                'candidate': candidate_missing,
                'baseline': baseline_missing,
            },
        }
    if pickup_delta is not None and pickup_delta >= 5.0:
        return {
            'verdict': 'reject',
            'blockers': blockers + ['pickup_commitment_hotspot_regression'],
            'protocol_mismatches': protocol_mismatches,
            'missing_fields': {
                'candidate': candidate_missing,
                'baseline': baseline_missing,
            },
        }

    if blockers:
        verdict = 'hold'
    elif (
        speed_delta >= 3.0
        and epoch_delta <= -3.0
        and train_delta <= -3.0
        and (pickup_delta is None or pickup_delta <= -1.0)
        and (decode_delta is None or decode_delta <= 0.0)
        and (viability_delta is None or viability_delta <= 0.0)
        and (completion_delta is None or completion_delta <= 0.0)
    ):
        verdict = 'promote_to_short_train'
    else:
        verdict = 'hold'

    return {
        'verdict': verdict,
        'blockers': blockers,
        'protocol_mismatches': protocol_mismatches,
        'missing_fields': {
            'candidate': candidate_missing,
            'baseline': baseline_missing,
        },
    }


def _analyze_eval(candidate_eval, baseline_eval, eval_gate_profile='strict', research_max_unfulfilled_delta=0.015, research_max_unfulfilled_abs=0.015):
    candidate_missing = _collect_missing_fields(candidate_eval, EVAL_PROTOCOL_FIELDS)
    baseline_missing = _collect_missing_fields(baseline_eval, EVAL_PROTOCOL_FIELDS)
    protocol_mismatches = _collect_mismatches(candidate_eval, baseline_eval, EVAL_PROTOCOL_FIELDS)

    cand_clean = bool(_safe_get(candidate_eval, 'business_acceptance', 'clean', default=False))
    base_clean = bool(_safe_get(baseline_eval, 'business_acceptance', 'clean', default=False))
    cand_partition = _safe_get(candidate_eval, 'audit', 'partition_consistent_samples')
    cand_total = _safe_get(candidate_eval, 'audit', 'total_samples')
    cand_match = _safe_get(candidate_eval, 'audit', 'core_aggregate_match_samples')
    base_partition = _safe_get(baseline_eval, 'audit', 'partition_consistent_samples')
    base_total = _safe_get(baseline_eval, 'audit', 'total_samples')
    base_match = _safe_get(baseline_eval, 'audit', 'core_aggregate_match_samples')
    cand_service = _safe_get(candidate_eval, 'aggregate', 'service_rate_mean')
    base_service = _safe_get(baseline_eval, 'aggregate', 'service_rate_mean')
    cand_unfulfilled = _safe_get(candidate_eval, 'aggregate', 'unfulfilled_rate_mean')
    base_unfulfilled = _safe_get(baseline_eval, 'aggregate', 'unfulfilled_rate_mean')

    candidate_audit_ok = (cand_partition == cand_total) and (cand_match == cand_total)
    baseline_audit_ok = (base_partition == base_total) and (base_match == base_total)
    blockers = []

    if candidate_missing or baseline_missing:
        blockers.append('missing_eval_provenance')
    if protocol_mismatches:
        blockers.append('eval_protocol_mismatch')
    if not baseline_audit_ok:
        blockers.append('baseline_eval_audit_not_clean')

    common_payload = {
        'protocol_mismatches': protocol_mismatches,
        'missing_fields': {
            'candidate': candidate_missing,
            'baseline': baseline_missing,
        },
        'audit_ok': {
            'candidate': candidate_audit_ok,
            'baseline': baseline_audit_ok,
        },
        'soft_gate_applied': False,
        'soft_gate_meta': None,
    }

    if not candidate_audit_ok:
        return {
            'verdict': 'reject',
            'blockers': blockers + ['candidate_eval_audit_not_clean'],
            **common_payload,
        }
    if blockers:
        return {
            'verdict': 'hold',
            'blockers': blockers,
            **common_payload,
        }

    semantic_blockers = []
    if base_clean and not cand_clean:
        semantic_blockers.append('business_clean_regression')
    unfulfilled_delta = None
    if cand_unfulfilled is not None and base_unfulfilled is not None:
        unfulfilled_delta = float(cand_unfulfilled) - float(base_unfulfilled)
        if unfulfilled_delta > EPS:
            semantic_blockers.append('unfulfilled_rate_regression')
    if cand_service is not None and base_service is not None and float(cand_service) + EPS < float(base_service):
        semantic_blockers.append('service_rate_regression')

    if semantic_blockers:
        if eval_gate_profile == 'research':
            soft_blocker_set = {'business_clean_regression', 'unfulfilled_rate_regression'}
            only_soft_blockers = all(item in soft_blocker_set for item in semantic_blockers)
            within_unfulfilled_delta = (
                unfulfilled_delta is not None
                and unfulfilled_delta <= float(research_max_unfulfilled_delta) + EPS
            )
            within_unfulfilled_abs = (
                cand_unfulfilled is not None
                and float(cand_unfulfilled) <= float(research_max_unfulfilled_abs) + EPS
            )
            if only_soft_blockers and within_unfulfilled_delta and within_unfulfilled_abs:
                return {
                    'verdict': 'hold',
                    'blockers': semantic_blockers,
                    'soft_gate_applied': True,
                    'soft_gate_meta': {
                        'profile': 'research',
                        'research_max_unfulfilled_delta': float(research_max_unfulfilled_delta),
                        'research_max_unfulfilled_abs': float(research_max_unfulfilled_abs),
                        'unfulfilled_delta': unfulfilled_delta,
                        'candidate_unfulfilled_rate': None if cand_unfulfilled is None else float(cand_unfulfilled),
                    },
                    **common_payload,
                }

        return {
            'verdict': 'reject',
            'blockers': semantic_blockers,
            **common_payload,
        }

    if cand_clean and ((not base_clean) or (cand_service is not None and base_service is not None and float(cand_service) > float(base_service) + EPS)):
        verdict = 'promote_to_formal_eval'
    else:
        verdict = 'hold'

    return {
        'verdict': verdict,
        'blockers': [],
        **common_payload,
    }


def compare_runs(args):
    candidate = _load_json(args.candidate_benchmark)
    baseline = _load_json(args.baseline_benchmark)
    benchmark_analysis = _analyze_benchmark(candidate, baseline)

    report = {
        'type': 'experiment_comparison',
        'candidate_benchmark': str(Path(args.candidate_benchmark)),
        'baseline_benchmark': str(Path(args.baseline_benchmark)),
        'run_context': _safe_get(candidate, 'run_context', default={}),
        'benchmark_delta': {
            'epoch_total_s_pct': _pct_delta(_safe_get(candidate, 'key_metrics', 'epoch_total_s'), _safe_get(baseline, 'key_metrics', 'epoch_total_s')),
            'train_s_pct': _pct_delta(_safe_get(candidate, 'key_metrics', 'train_s'), _safe_get(baseline, 'key_metrics', 'train_s')),
            'samples_per_s_pct': _pct_delta(_safe_get(candidate, 'key_metrics', 'samples_per_s'), _safe_get(baseline, 'key_metrics', 'samples_per_s')),
            'decode_get_mask_ms_pct': _pct_delta(_safe_get(candidate, 'key_metrics', 'decode_get_mask_ms'), _safe_get(baseline, 'key_metrics', 'decode_get_mask_ms')),
            'mask_pickup_commitment_ms_pct': _pct_delta(_safe_get(candidate, 'key_metrics', 'mask_pickup_commitment_ms'), _safe_get(baseline, 'key_metrics', 'mask_pickup_commitment_ms')),
            'mask_delivery_viability_ms_pct': _pct_delta(_safe_get(candidate, 'key_metrics', 'mask_delivery_viability_ms'), _safe_get(baseline, 'key_metrics', 'mask_delivery_viability_ms')),
            'pc_has_feasible_open_completion_ms_pct': _pct_delta(_safe_get(candidate, 'key_metrics', 'pc_has_feasible_open_completion_ms'), _safe_get(baseline, 'key_metrics', 'pc_has_feasible_open_completion_ms')),
        },
        'benchmark_guardrails': {
            'blockers': benchmark_analysis['blockers'],
            'protocol_mismatches': benchmark_analysis['protocol_mismatches'],
            'missing_fields': benchmark_analysis['missing_fields'],
        },
        'benchmark_verdict': benchmark_analysis['verdict'],
    }

    if args.candidate_eval and args.baseline_eval:
        candidate_eval = _load_json(args.candidate_eval)
        baseline_eval = _load_json(args.baseline_eval)
        eval_analysis = _analyze_eval(
            candidate_eval,
            baseline_eval,
            eval_gate_profile=args.eval_gate_profile,
            research_max_unfulfilled_delta=args.research_max_unfulfilled_delta,
            research_max_unfulfilled_abs=args.research_max_unfulfilled_abs,
        )
        report['candidate_eval'] = str(Path(args.candidate_eval))
        report['baseline_eval'] = str(Path(args.baseline_eval))
        report['eval_gate_profile'] = args.eval_gate_profile
        report['eval_delta'] = {
            'service_rate_mean_delta': None if None in (_safe_get(candidate_eval, 'aggregate', 'service_rate_mean'), _safe_get(baseline_eval, 'aggregate', 'service_rate_mean')) else float(_safe_get(candidate_eval, 'aggregate', 'service_rate_mean')) - float(_safe_get(baseline_eval, 'aggregate', 'service_rate_mean')),
            'unfulfilled_rate_mean_delta': None if None in (_safe_get(candidate_eval, 'aggregate', 'unfulfilled_rate_mean'), _safe_get(baseline_eval, 'aggregate', 'unfulfilled_rate_mean')) else float(_safe_get(candidate_eval, 'aggregate', 'unfulfilled_rate_mean')) - float(_safe_get(baseline_eval, 'aggregate', 'unfulfilled_rate_mean')),
            'business_clean_candidate': cand_clean if (cand_clean := bool(_safe_get(candidate_eval, 'business_acceptance', 'clean', default=False))) is not None else False,
            'business_clean_baseline': base_clean if (base_clean := bool(_safe_get(baseline_eval, 'business_acceptance', 'clean', default=False))) is not None else False,
        }
        report['eval_guardrails'] = {
            'blockers': eval_analysis['blockers'],
            'protocol_mismatches': eval_analysis['protocol_mismatches'],
            'missing_fields': eval_analysis['missing_fields'],
            'audit_ok': eval_analysis['audit_ok'],
            'soft_gate_applied': eval_analysis.get('soft_gate_applied', False),
            'soft_gate_meta': eval_analysis.get('soft_gate_meta'),
        }
        report['eval_verdict'] = eval_analysis['verdict']
    else:
        report['eval_verdict'] = None

    print('=' * 72)
    print('Experiment Comparison')
    print('=' * 72)
    print(f"candidate benchmark : {report['candidate_benchmark']}")
    print(f"baseline benchmark  : {report['baseline_benchmark']}")
    print(f"experiment_label    : {_safe_get(report, 'run_context', 'experiment_label', default='n/a')}")
    print(f"single variable     : {_safe_get(report, 'run_context', 'single_variable_under_test', default='n/a')}")
    print('-' * 72)
    for key, value in report['benchmark_delta'].items():
        print(f"{key:36s}: {_fmt(value)}%")
    if report['benchmark_guardrails']['blockers']:
        print(f"benchmark blockers                    : {', '.join(report['benchmark_guardrails']['blockers'])}")
    print(f"benchmark verdict                     : {report['benchmark_verdict']}")
    if report['eval_verdict'] is not None:
        print('-' * 72)
        print(f"eval gate profile                     : {_safe_get(report, 'eval_gate_profile', default='strict')}")
        print(f"eval service delta                    : {_fmt(_safe_get(report, 'eval_delta', 'service_rate_mean_delta'))}")
        print(f"eval unfulfilled delta                : {_fmt(_safe_get(report, 'eval_delta', 'unfulfilled_rate_mean_delta'))}")
        if report.get('eval_guardrails', {}).get('blockers'):
            print(f"eval blockers                         : {', '.join(report['eval_guardrails']['blockers'])}")
        if report.get('eval_guardrails', {}).get('soft_gate_applied'):
            print('eval soft-gate applied                : True (converted reject -> hold)')
        print(f"eval verdict                          : {report['eval_verdict']}")
    print('-' * 72)
    print(json.dumps(report, ensure_ascii=False, indent=2))

    if args.json_output:
        with open(args.json_output, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Compare benchmark/eval evidence for one candidate vs one baseline')
    parser.add_argument('--candidate-benchmark', required=True, help='candidate benchmark_final JSON path')
    parser.add_argument('--baseline-benchmark', required=True, help='baseline benchmark_final JSON path')
    parser.add_argument('--candidate-eval', default=None, help='optional candidate eval JSON path')
    parser.add_argument('--baseline-eval', default=None, help='optional baseline eval JSON path')
    parser.add_argument('--eval-gate-profile', choices=['strict', 'research'], default='strict',
                        help='strict: production gate; research: convert limited semantic regressions from reject to hold')
    parser.add_argument('--research-max-unfulfilled-delta', type=float, default=0.015,
                        help='research profile only: max allowed unfulfilled_rate delta for soft-gate hold')
    parser.add_argument('--research-max-unfulfilled-abs', type=float, default=0.015,
                        help='research profile only: max allowed candidate unfulfilled_rate for soft-gate hold')
    parser.add_argument('--json-output', default=None, help='optional output report JSON path')
    compare_runs(parser.parse_args())
