"""Dependency-free archive statistics check; no inference, solver or route replay.

Reads only the extracted archive, never paths embedded in old metadata.
Recomputes means/class rates, not confidence intervals or physical audits.
"""
from pathlib import Path
import argparse
import csv
import hashlib
import json
import math
from collections import defaultdict
from statistics import mean


def load(path):
    return json.loads(path.read_text())


def compare(actual, expected):
    assert math.isfinite(actual) and math.isclose(actual, expected, rel_tol=1e-11, abs_tol=1e-9), (actual, expected)


def common_means(rows, fields, n=32, repeats=None):
    buckets = defaultdict(list)
    identities = set()
    for row in rows:
        key = (row['method'], int(row['instance']), str(row['repeat_seed']))
        assert key not in identities, key
        identities.add(key)
        buckets[(row['method'], int(row['instance']))].append(row)
    methods = sorted({r['method'] for r in rows})
    result = {}
    for method in methods:
        assert {i for m, i in buckets if m == method} == set(range(n)), method
        if repeats:
            assert all(len(buckets[method, i]) == repeats[method] for i in range(n))
        result[method] = {field: mean(mean(float(x[field]) for x in buckets[method, i])
                                      for i in range(n)) for field in fields}
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[2])
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    root = args.root.resolve()
    manifest = load(root / 'CANDIDATE_MANIFEST.json')
    for row in manifest['files']:
        rel = Path(row['path'])
        assert not rel.is_absolute() and '..' not in rel.parts
        path = root / rel
        assert path.is_file() and not path.is_symlink(), rel
        assert hashlib.sha256(path.read_bytes()).hexdigest() == row['sha256'], rel
    base1 = root / 'experiments/hard4_confirmation_20261106/confirmation_v1'
    base2 = root / 'experiments/neural_confirmation_20261107/confirmation_v1'
    window = root / 'experiments/neural_sensitivity_20261107/window_v1'
    for base, filename, key, digest in [(base1, 'plan.json', 'scenario', 'scenario_sha256'),
                                       (base2, 'plan.json', 'scenario', 'scenario_sha256'),
                                       (window, 'protocol.json', 'base_scenario', 'base_sha256')]:
        protocol = load(base / filename)
        # Historical absolute paths are not opened. Locate the immutable basename inside the archive.
        matches = list(root.rglob(Path(protocol[key]).name))
        assert len(matches) == 1
        assert hashlib.sha256(matches[0].read_bytes()).hexdigest() == protocol[digest]
        for rel, expected in protocol['source_sha256'].items():
            assert hashlib.sha256((root / rel).read_bytes()).hexdigest() == expected
    with (base1 / 'raw_final_plans.csv').open() as handle:
        rows1 = list(csv.DictReader(handle))
    assert len(rows1) == 352
    for row in rows1:
        assert int(row['hard_invalid']) == int(row['unfulfilled']) == 0
        assert int(row['vehicles']) == 0 or int(row['min_completed_per_trip']) >= 4
    table1 = common_means(rows1, ['service_pct', 'objective', 'algorithm_seconds'],
                          repeats=dict(learned=3, service_first=3, insertion=1, LNS=3, SCIP=1))
    reference1 = load(base1 / 'analysis.json')['aggregates']
    for method, metrics in table1.items():
        for metric, value in metrics.items():
            compare(value, reference1[method][metric]['mean'])
    rows2 = []
    plan2 = load(base2 / 'plan.json')
    for spec in plan2['models']:
        record = load(base2 / 'neural' / f"{spec['policy']}_{spec['seed']}.json")
        assert record['model'] == spec
        assert len(record['rows']) == 32
        for index, raw in enumerate(record['rows']):
            assert raw['instance'] == index
            for phase in ['raw', 'hybrid']:
                row = raw if phase == 'raw' else raw['hybrid']
                assert not row['audit']['violations'] and row['audit']['unfulfilled'] == 0
                rows2.append(dict(method=f"{spec['policy']}_{phase}", instance=index,
                                  repeat_seed=spec['seed'], service_percent=2 * row['audit']['completed'],
                                  mean_J=raw['objective'] if phase == 'raw' else row['cost']['objective'],
                                  mean_algorithm_seconds=raw['inference_seconds'] if phase == 'raw' else row['integrated_algorithm_seconds']))
    for index in range(32):
        entry = load(base2 / 'controls' / f'instance_{index}.json')
        assert entry['instance'] == index
        for phase in ['initial', 'short', 'strong']:
            row = entry[phase]
            assert not row['audit']['violations'] and row['audit']['unfulfilled'] == 0
            rows2.append(dict(method=f'insertion_{phase}', instance=index, repeat_seed=1234,
                              service_percent=2 * row['audit']['completed'], mean_J=row['cost']['objective'],
                              mean_algorithm_seconds=row['algorithm_seconds'] if phase == 'initial' else row['integrated_algorithm_seconds']))
    assert len(rows2) == 480
    reps = {f'{m}_{p}': 3 for m in ['learned', 'service_first'] for p in ['raw', 'hybrid']}
    reps.update({f'insertion_{p}': 1 for p in ['initial', 'short', 'strong']})
    table2 = common_means(rows2, ['service_percent', 'mean_J', 'mean_algorithm_seconds'], repeats=reps)
    reference2 = {x['method']: x for x in load(base2 / 'analysis.json')['summary']}
    for method, metrics in table2.items():
        for metric, value in metrics.items():
            compare(value, reference2[method][metric])
    with (window / 'raw_window_plans.csv').open() as handle:
        rows3 = list(csv.DictReader(handle))
    assert len(rows3) == 672
    reference3 = load(window / 'analysis.json')
    table3, windows = [], {}
    for factor in [.5, 1., 1.5]:
        subset = [x for x in rows3 if float(x['window_multiplier']) == factor]
        assert len(subset) == 224
        aggregate = common_means(subset, ['service_pct', 'objective', 'energy_fee', 'cargo_delay_fee', 'vehicles'],
                                 repeats=dict(learned_raw=3, learned_hybrid=3, LNS=1))
        windows[str(factor)] = aggregate
        for method, metrics in aggregate.items():
            for metric, value in metrics.items():
                compare(value, reference3['summary'][method][str(factor)][metric]['mean'])
        for method in ['learned_raw', 'learned_hybrid', 'LNS']:
            selected = [x for x in subset if x['method'] == method]
            rates = {kind: 100 * sum(int(x[f'{kind}_completed']) for x in selected) /
                     sum(int(x[f'{kind}_total']) for x in selected) for kind in ['passenger', 'cargo']}
            expected = next(x for x in reference3['class_coverage'] if x['method'] == method and x['window_multiplier'] == factor)
            for kind, value in rates.items():
                compare(value, expected[f'{kind}_service_percent'])
            if method == 'learned_hybrid':
                table3.append(dict(window_multiplier=factor, **rates,
                                   vehicles=aggregate[method]['vehicles']))
    result = dict(status='passed_archive_only_reanalysis', verified_archive_files=len(manifest['files']),
                  table1=table1, table2=table2, table3=table3, window_means=windows,
                  record_counts=[352, 480, 672],
                  scope='All means/class rates rechecked against original analyses with instance-first aggregation, from archive-local records only; no inference, route replay, solver run, training or bootstrap interval recomputation',
                  independence='Second-confirmation baseline plans reused in window data; counts are not additive independent replications')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2))
    print('PASS archive-only tables1–3 and all9 window cells; files', len(manifest['files']))


if __name__ == '__main__':
    main()
