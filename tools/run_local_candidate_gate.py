import argparse
import json
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path


METRIC_LINE_RE = re.compile(r'^(?P<name>[A-Za-z0-9_]+)\s+(?P<calls>\d+)\s+(?P<total_ms>[0-9.]+)\s+(?P<per_top_ms>[0-9.]+)$')
AVG_TOP_RE = re.compile(r'^avg top-level get_mask\s*:\s*([0-9.]+) ms/call$')
AVG_UPDATE_RE = re.compile(r'^avg rollout update\s*:\s*([0-9.]+) ms/call$')


def parse_args():
    parser = argparse.ArgumentParser(description='Run local candidate gates before GPU short-train')
    parser.add_argument('--graph-size', type=int, required=True, help='target graph size')
    parser.add_argument('--output-dir', type=str, default=None, help='artifact directory; default under outputs_cmp/local_gate_runs')
    parser.add_argument('--seed', type=int, default=1234, help='shared seed for profile/benchmark')
    parser.add_argument('--profile-runs', type=int, default=3, help='number of repeated hotspot profile runs')
    parser.add_argument('--profile-batch-size', type=int, default=8, help='batch size for hotspot profiling')
    parser.add_argument('--profile-rollout-repeats', type=int, default=5, help='repeat count inside profile_mask_hotspots.py')
    parser.add_argument('--profile-steps', type=int, default=20, help='rollout steps inside profile_mask_hotspots.py')
    parser.add_argument('--benchmark-epochs', type=int, default=1, help='epochs for local benchmark preflight')
    parser.add_argument('--benchmark-epoch-size', type=int, default=32, help='epoch size for local benchmark preflight')
    parser.add_argument('--benchmark-batch-size', type=int, default=2, help='batch size for local benchmark preflight')
    parser.add_argument('--benchmark-val-size', type=int, default=8, help='val size for local benchmark preflight')
    parser.add_argument('--benchmark-warmup-epochs', type=int, default=1, help='warmup epochs for benchmark summary')
    parser.add_argument('--benchmark-batch-timing', action='store_true', help='enable batch timing in local benchmark')
    parser.add_argument('--baseline-benchmark', type=str, default=None, help='optional baseline benchmark JSON for compare step')
    parser.add_argument('--candidate-eval', type=str, default=None, help='optional candidate eval JSON for compare step')
    parser.add_argument('--baseline-eval', type=str, default=None, help='optional baseline eval JSON for compare step')
    parser.add_argument('--eval-gate-profile', choices=['strict', 'research'], default='strict',
                        help='strict: production gate; research: convert limited semantic regressions from reject to hold')
    parser.add_argument('--research-max-unfulfilled-delta', type=float, default=0.015,
                        help='research profile only: max allowed unfulfilled_rate delta for soft-gate hold')
    parser.add_argument('--research-max-unfulfilled-abs', type=float, default=0.015,
                        help='research profile only: max allowed candidate unfulfilled_rate for soft-gate hold')
    parser.add_argument('--experiment-label', type=str, default=None, help='experiment label stored in benchmark artifacts')
    parser.add_argument('--baseline-ref', type=str, default=None, help='baseline reference stored in artifacts')
    parser.add_argument('--candidate-ref', type=str, default='local-working-tree', help='candidate reference stored in artifacts')
    parser.add_argument('--single-variable-under-test', type=str, default=None, help='single variable under test')
    parser.add_argument('--no-cuda', action='store_true', help='force benchmark to run on CPU')
    return parser.parse_args()


def _timestamp_slug():
    return datetime.now().strftime('%Y%m%d_%H%M%S')


def _default_output_dir(repo_root, args):
    label = args.experiment_label or args.single_variable_under_test or args.candidate_ref or 'candidate'
    safe_label = re.sub(r'[^A-Za-z0-9_.-]+', '-', label).strip('-') or 'candidate'
    return repo_root / 'outputs_cmp' / 'local_gate_runs' / f'n{args.graph_size}_{safe_label}_{_timestamp_slug()}'


def _write_text(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding='utf-8')


def _run_step(name, argv, cwd, log_path):
    completed = subprocess.run(argv, cwd=str(cwd), text=True, capture_output=True)
    combined = ''
    if completed.stdout:
        combined += completed.stdout
    if completed.stderr:
        if combined and not combined.endswith('\n'):
            combined += '\n'
        combined += completed.stderr
    _write_text(log_path, combined)
    return {
        'name': name,
        'argv': argv,
        'returncode': int(completed.returncode),
        'ok': completed.returncode == 0,
        'log_path': str(log_path),
    }, combined


def _parse_profile_output(text):
    summary = {
        'avg_top_level_get_mask_ms': None,
        'avg_rollout_update_ms': None,
        'metrics': {},
    }
    for raw_line in text.splitlines():
        line = raw_line.strip()
        match = AVG_TOP_RE.match(line)
        if match:
            summary['avg_top_level_get_mask_ms'] = float(match.group(1))
            continue
        match = AVG_UPDATE_RE.match(line)
        if match:
            summary['avg_rollout_update_ms'] = float(match.group(1))
            continue
        match = METRIC_LINE_RE.match(line)
        if match:
            summary['metrics'][match.group('name')] = {
                'calls': int(match.group('calls')),
                'total_ms': float(match.group('total_ms')),
                'per_top_ms': float(match.group('per_top_ms')),
            }
    return summary


def _mean(values):
    values = [float(v) for v in values if v is not None]
    if not values:
        return None
    return sum(values) / len(values)


def _min_or_none(values):
    values = [float(v) for v in values if v is not None]
    if not values:
        return None
    return min(values)


def _max_or_none(values):
    values = [float(v) for v in values if v is not None]
    if not values:
        return None
    return max(values)


def _load_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def _overall_verdict(report):
    steps = report['steps']
    if not steps['compile']['ok']:
        return 'failed_compile'
    if not steps['dry_run']['ok']:
        return 'failed_dry_run'
    if any((not step['ok']) for step in steps['profile_runs']):
        return 'failed_profile'
    if not steps['benchmark']['ok']:
        return 'failed_benchmark'
    compare_step = steps.get('compare')
    if compare_step is None:
        return 'local_preflight_passed_no_baseline_compare'
    if not compare_step['ok']:
        return 'failed_compare'
    benchmark_verdict = report.get('compare_report', {}).get('benchmark_verdict')
    eval_verdict = report.get('compare_report', {}).get('eval_verdict')
    if benchmark_verdict == 'reject' or eval_verdict == 'reject':
        return 'rejected_locally'
    if benchmark_verdict == 'promote_to_short_train':
        if eval_verdict in (None, 'hold', 'promote_to_formal_eval'):
            return 'ready_for_gpu_short_train'
    return 'hold_local_candidate'


def main():
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    output_dir = Path(args.output_dir).resolve() if args.output_dir else _default_output_dir(repo_root, args)
    output_dir.mkdir(parents=True, exist_ok=True)

    candidate_benchmark_path = output_dir / 'candidate_benchmark_final.json'
    compare_json_path = output_dir / 'candidate_vs_baseline_compare.json'
    report_path = output_dir / 'local_gate_report.json'

    report = {
        'type': 'local_candidate_gate',
        'repo_root': str(repo_root),
        'graph_size': int(args.graph_size),
        'seed': int(args.seed),
        'output_dir': str(output_dir),
        'artifacts': {
            'candidate_benchmark': str(candidate_benchmark_path),
            'compare_report': str(compare_json_path),
            'final_report': str(report_path),
        },
        'metadata': {
            'experiment_label': args.experiment_label,
            'baseline_ref': args.baseline_ref,
            'candidate_ref': args.candidate_ref,
            'single_variable_under_test': args.single_variable_under_test,
        },
        'steps': {
            'compile': None,
            'dry_run': None,
            'profile_runs': [],
            'benchmark': None,
        },
        'profile_summary': None,
        'benchmark_summary': None,
        'compare_report': None,
        'overall_verdict': None,
    }

    compile_targets = [
        repo_root / 'state_mcvrptw_v2.py',
        repo_root / 'dry_run_test.py',
        repo_root / 'profile_mask_hotspots.py',
        repo_root / 'run_training_optimized.py',
        repo_root / 'evaluate_model.py',
        repo_root / 'tools' / 'compare_experiment_runs.py',
        repo_root / 'tools' / 'run_local_candidate_gate.py',
    ]
    compile_argv = [sys.executable, '-m', 'py_compile', *[str(path) for path in compile_targets]]
    compile_step, _ = _run_step('compile', compile_argv, repo_root, output_dir / 'compile.log')
    report['steps']['compile'] = compile_step
    if not compile_step['ok']:
        report['overall_verdict'] = _overall_verdict(report)
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
        print(json.dumps(report, ensure_ascii=False, indent=2))
        raise SystemExit(1)

    dry_run_step, _ = _run_step(
        'dry_run',
        [sys.executable, str(repo_root / 'dry_run_test.py')],
        repo_root,
        output_dir / 'dry_run.log',
    )
    report['steps']['dry_run'] = dry_run_step
    if not dry_run_step['ok']:
        report['overall_verdict'] = _overall_verdict(report)
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
        print(json.dumps(report, ensure_ascii=False, indent=2))
        raise SystemExit(1)

    parsed_profiles = []
    for run_idx in range(1, args.profile_runs + 1):
        profile_argv = [
            sys.executable,
            str(repo_root / 'profile_mask_hotspots.py'),
            '--graph-sizes', str(args.graph_size),
            '--batch-size', str(args.profile_batch_size),
            '--seed', str(args.seed),
            '--repeats', str(args.profile_rollout_repeats),
            '--steps', str(args.profile_steps),
        ]
        step, stdout_text = _run_step(
            f'profile_run_{run_idx}',
            profile_argv,
            repo_root,
            output_dir / f'profile_run_{run_idx}.log',
        )
        parsed = _parse_profile_output(stdout_text)
        step['parsed_summary'] = parsed
        parsed_profiles.append(parsed)
        report['steps']['profile_runs'].append(step)
        if not step['ok']:
            report['overall_verdict'] = _overall_verdict(report)
            report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
            print(json.dumps(report, ensure_ascii=False, indent=2))
            raise SystemExit(1)

    report['profile_summary'] = {
        'runs': len(parsed_profiles),
        'avg_top_level_get_mask_ms_mean': _mean([item['avg_top_level_get_mask_ms'] for item in parsed_profiles]),
        'avg_top_level_get_mask_ms_min': _min_or_none([item['avg_top_level_get_mask_ms'] for item in parsed_profiles]),
        'avg_top_level_get_mask_ms_max': _max_or_none([item['avg_top_level_get_mask_ms'] for item in parsed_profiles]),
        'pc_has_feasible_open_completion_ms_mean': _mean([
            item['metrics'].get('_has_feasible_open_completion', {}).get('per_top_ms') for item in parsed_profiles
        ]),
        'pc_has_feasible_open_completion_ms_min': _min_or_none([
            item['metrics'].get('_has_feasible_open_completion', {}).get('per_top_ms') for item in parsed_profiles
        ]),
        'pc_has_feasible_open_completion_ms_max': _max_or_none([
            item['metrics'].get('_has_feasible_open_completion', {}).get('per_top_ms') for item in parsed_profiles
        ]),
        'build_completion_scalar_caches_ms_mean': _mean([
            item['metrics'].get('_build_completion_scalar_caches', {}).get('per_top_ms') for item in parsed_profiles
        ]),
    }

    benchmark_argv = [
        sys.executable,
        str(repo_root / 'run_training_optimized.py'),
        '--graph_sizes', str(args.graph_size),
        '--epochs', str(args.benchmark_epochs),
        '--epoch-size', str(args.benchmark_epoch_size),
        '--batch-size', str(args.benchmark_batch_size),
        '--val-size', str(args.benchmark_val_size),
        '--seed', str(args.seed),
        '--benchmark-mode',
        '--benchmark-skip-validation',
        '--benchmark-disable-checkpoint',
        '--benchmark-disable-log-save',
        '--benchmark-json',
        '--benchmark-json-output', str(candidate_benchmark_path),
        '--benchmark-warmup-epochs', str(args.benchmark_warmup_epochs),
        '--output-root', str(output_dir / 'training_outputs'),
        '--candidate-ref', str(args.candidate_ref),
    ]
    if args.experiment_label is not None:
        benchmark_argv.extend(['--experiment-label', args.experiment_label])
    if args.baseline_ref is not None:
        benchmark_argv.extend(['--baseline-ref', args.baseline_ref])
    if args.single_variable_under_test is not None:
        benchmark_argv.extend(['--single-variable-under-test', args.single_variable_under_test])
    if args.benchmark_batch_timing:
        benchmark_argv.append('--benchmark-batch-timing')
    if args.no_cuda:
        benchmark_argv.append('--no-cuda')

    benchmark_step, _ = _run_step('benchmark', benchmark_argv, repo_root, output_dir / 'benchmark.log')
    report['steps']['benchmark'] = benchmark_step
    if benchmark_step['ok'] and candidate_benchmark_path.exists():
        report['benchmark_summary'] = _load_json(candidate_benchmark_path)
    else:
        report['overall_verdict'] = _overall_verdict(report)
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
        print(json.dumps(report, ensure_ascii=False, indent=2))
        raise SystemExit(1)

    if args.baseline_benchmark is not None:
        compare_argv = [
            sys.executable,
            str(repo_root / 'tools' / 'compare_experiment_runs.py'),
            '--candidate-benchmark', str(candidate_benchmark_path),
            '--baseline-benchmark', str(Path(args.baseline_benchmark).resolve()),
            '--json-output', str(compare_json_path),
        ]
        if args.candidate_eval is not None and args.baseline_eval is not None:
            compare_argv.extend([
                '--candidate-eval', str(Path(args.candidate_eval).resolve()),
                '--baseline-eval', str(Path(args.baseline_eval).resolve()),
                '--eval-gate-profile', args.eval_gate_profile,
                '--research-max-unfulfilled-delta', str(args.research_max_unfulfilled_delta),
                '--research-max-unfulfilled-abs', str(args.research_max_unfulfilled_abs),
            ])
        compare_step, _ = _run_step('compare', compare_argv, repo_root, output_dir / 'compare.log')
        report['steps']['compare'] = compare_step
        if compare_step['ok'] and compare_json_path.exists():
            report['compare_report'] = _load_json(compare_json_path)

    report['overall_verdict'] = _overall_verdict(report)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')

    print('=' * 72)
    print('Local Candidate Gate')
    print('=' * 72)
    print(f'output_dir        : {output_dir}')
    print(f'overall_verdict   : {report["overall_verdict"]}')
    profile_summary = report.get('profile_summary') or {}
    print(f'profile get_mask  : {profile_summary.get("avg_top_level_get_mask_ms_mean")} ms mean')
    benchmark_summary = report.get('benchmark_summary') or {}
    key_metrics = benchmark_summary.get('key_metrics', {})
    print(f'benchmark epoch_s : {key_metrics.get("epoch_total_s")}')
    print(f'benchmark sps     : {key_metrics.get("samples_per_s")}')
    if report.get('compare_report') is not None:
        print(f'benchmark verdict : {report["compare_report"].get("benchmark_verdict")}')
        print(f'eval verdict      : {report["compare_report"].get("eval_verdict")}')
    print(f'final report      : {report_path}')
    print('-' * 72)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
