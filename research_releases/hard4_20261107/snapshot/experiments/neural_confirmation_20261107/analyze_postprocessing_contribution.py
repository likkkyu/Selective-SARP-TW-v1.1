"""Post hoc, complete paired contrasts; never modify frozen experiment records."""
from pathlib import Path
import csv
import hashlib
import json
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
BASE = Path(__file__).resolve().parent / 'confirmation_v1'
OUT = Path(__file__).resolve().parent / 'postprocessing_contribution_v1'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    plan = json.loads((BASE / 'plan.json').read_text())
    assert (BASE / 'exit_status.txt').read_text().strip() == '0'
    assert json.loads((BASE / 'analysis.json').read_text())['status'] == 'passed'
    assert plan['samples'] == 32
    for path, expected in plan['source_sha256'].items():
        assert sha(ROOT / path) == expected, path
    assert sha(Path(plan['scenario'])) == plan['scenario_sha256']
    inputs = [BASE / 'plan.json', BASE / 'analysis.json']
    groups = {'learned': [], 'service_first': [], 'insertion': []}
    per_seed = []
    cases = []
    for model in plan['models']:
        path = BASE / 'neural' / f"{model['policy']}_{model['seed']}.json"
        inputs.append(path)
        data = json.loads(path.read_text())
        assert data['model'] == model and len(data['rows']) == 32
        deltas = []
        for index, raw in enumerate(data['rows']):
            assert raw['instance'] == index
            after = raw['hybrid']
            for phase in [raw, after]:
                assert not phase['audit']['violations']
                assert phase['audit']['unfulfilled'] == 0
            delta = np.array([
                2.0 * (after['audit']['completed'] - raw['audit']['completed']),
                after['cost']['objective'] - raw['objective'],
            ])
            assert np.isfinite(delta).all()
            # Dominant refusal penalty makes accepted count monotonic when J improves.
            assert delta[0] >= 0 and delta[1] <= 1e-8
            deltas.append(delta)
            cases.append(dict(initializer=model['policy'], training_seed=model['seed'],
                              instance=index, service_gain_pp=float(delta[0]),
                              objective_delta=float(delta[1])))
        values = np.array(deltas)
        groups[model['policy']].append(values)
        per_seed.append(dict(initializer=model['policy'], training_seed=model['seed'],
                             service_gain_pp=float(values[:, 0].mean()),
                             objective_delta=float(values[:, 1].mean())))
    for index in range(32):
        path = BASE / 'controls' / f'instance_{index}.json'
        inputs.append(path)
        data = json.loads(path.read_text())
        assert data['instance'] == index
        before, after = data['initial'], data['short']
        for phase in [before, after]:
            assert not phase['audit']['violations'] and phase['audit']['unfulfilled'] == 0
        delta = [2.0 * (after['audit']['completed'] - before['audit']['completed']),
                 after['cost']['objective'] - before['cost']['objective']]
        assert delta[0] >= 0 and delta[1] <= 1e-8
        groups['insertion'].append(np.array([delta]))
        cases.append(dict(initializer='insertion', training_seed='', instance=index,
                          service_gain_pp=delta[0], objective_delta=delta[1]))
    neural_means = {k: np.mean(v, axis=0) for k, v in groups.items() if k != 'insertion'}
    neural_means['insertion'] = np.concatenate(groups['insertion'], axis=0)
    indices = np.random.default_rng(8162049).integers(0, 32, size=(10000, 32))
    summaries = []
    for name, values in neural_means.items():
        boot = values[indices].mean(axis=1)
        summaries.append(dict(initializer=name, estimate=values.mean(axis=0).tolist(),
                              conditional_instance_CI95=np.quantile(boot, [.025, .975], axis=0).T.tolist(),
                              instances_improved_service=int((values[:, 0] > 0).sum()),
                              instances_unchanged_service=int((values[:, 0] == 0).sum())))
    report = dict(status='passed', analysis_type='Post hoc descriptive contrasts of already frozen complete results',
                  independent_instances=32, neural_training_seeds=[1234, 4321, 8765],
                  search_seed=1234, search_budget_seconds=0.5,
                  columns=['service_gain_percentage_points', 'objective_after_minus_before'],
                  intervals='Pointwise paired-instance percentile bootstrap, 10000 resamples; fixed checkpoints/search seed; not multiplicity-adjusted or a new training replication',
                  summaries=summaries, seed_specific=per_seed,
                  no_plan_lost_service_or_increased_J=True,
                  interpretation='All three initializers retained; improvement is attributable to adding the implemented manual search pipeline, not evidence of learning-rejection superiority or a learned repair operator',
                  timing='Use integrated times in original analysis; do not infer causal runtime differences from separately timed initializers',
                  input_sha256={str(p.relative_to(ROOT)): sha(p) for p in inputs},
                  analysis_source_sha256=sha(Path(__file__)))
    OUT.mkdir(exist_ok=True)
    (OUT / 'analysis.json').write_text(json.dumps(report, ensure_ascii=False, indent=2))
    with (OUT / 'paired_cases.csv').open('w') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(cases[0]))
        writer.writeheader()
        writer.writerows(cases)
    print(json.dumps({'summaries': summaries, 'seed_specific': per_seed}, indent=2))


if __name__ == '__main__':
    main()
