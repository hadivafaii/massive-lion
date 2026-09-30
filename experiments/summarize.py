"""Validation-selected summaries and seed-level inference from local run files."""
import argparse
import itertools
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from experiments.sweep import expand

HYPERPARAMETERS = ['optim', 'label', 'section', 'lr', 'weight_decay', 'beta1', 'beta2',
                   'mass', 'adaptive_mass', 'kinematics', 'grad_clip']
CIFAR_METRICS = ['validation_accuracy', 'validation_loss', 'test_accuracy', 'test_loss']


def best_checkpoint(history, cutoff=None):
    """Select by validation correct count, then earliest step; never use test data."""
    candidates = [r for r in history if 'validation_accuracy' in r
                  and (cutoff is None or r['step'] <= cutoff)]
    if not candidates:
        raise ValueError('No validation checkpoint available at this cutoff')
    return min(candidates, key=lambda r: (-r['validation_correct'], r['step']))


def load_runs(root, config, allow_incomplete=False):
    jobs = expand(config)
    rows, histories, missing = [], {}, []
    for index, job in enumerate(jobs):
        directory = root / f'{index:05d}'
        if not (directory / 'final.json').exists():
            missing.append(index)
            continue
        actual = json.loads((directory / 'config.json').read_text())
        if actual.get('smoke'):
            raise ValueError(f'Smoke run cannot be used as experimental evidence: {directory}')
        mismatches = [key for key, value in job.items() if actual.get(key) != value]
        if mismatches:
            raise ValueError(f'{directory}: config differs from grid in {mismatches}')
        final = json.loads((directory / 'final.json').read_text())
        if final['step'] != job['steps']:
            raise ValueError(f'{directory}: incomplete final checkpoint')
        history = [json.loads(line) for line in (directory / 'history.jsonl').read_text().splitlines()]
        selected = best_checkpoint(history) if config['experiment'] == 'cifar' else final
        row = {**job, **selected, 'run_index': index, 'selected_step': selected['step']}
        rows.append(row)
        histories[index] = history
    if missing and not allow_incomplete:
        raise ValueError(f'{len(missing)}/{len(jobs)} runs missing; finish the grid or pass --allow-incomplete')
    if not rows:
        raise ValueError('No complete runs found')
    return pd.DataFrame(rows), histories, {'expected_runs': len(jobs), 'completed_runs': len(rows),
                                        'missing_indices': missing}


def aggregate(frame, keys, metrics):
    records = []
    for values, group in frame.groupby(keys, sort=False, dropna=False):
        if not isinstance(values, tuple):
            values = (values,)
        record = dict(zip(keys, values))
        if group['seed'].duplicated().any():
            raise ValueError('A configuration has duplicate seeds')
        record['n_seeds'] = len(group)
        for metric in metrics:
            record['mean_' + metric] = group[metric].mean()
            record['sd_' + metric] = group[metric].std(ddof=1)
        records.append(record)
    return pd.DataFrame(records)


def holm(p_values):
    """Holm correction, including every predeclared contrast in the family."""
    p_values = np.asarray(p_values, dtype=float)
    order = np.argsort(p_values)
    adjusted = np.empty_like(p_values)
    adjusted[order] = np.minimum(1, np.maximum.accumulate(
        p_values[order] * np.arange(len(order), 0, -1)))
    return adjusted


def paired_test(left, right, metric, fresh_seeds):
    left, right = left.set_index('seed'), right.set_index('seed')
    if set(left.index) != set(fresh_seeds) or set(right.index) != set(fresh_seeds):
        raise ValueError('Paired inference requires every planned fresh seed')
    differences = left.loc[fresh_seeds, metric] - right.loc[fresh_seeds, metric]
    result = stats.ttest_rel(left.loc[fresh_seeds, metric], right.loc[fresh_seeds, metric])
    return {'n_seeds': len(fresh_seeds), 'mean_difference': differences.mean(),
            't_statistic': result.statistic, 'p_value': result.pvalue}


def selected_summary(frame):
    keys = [k for k in HYPERPARAMETERS if k in frame]
    summary = aggregate(frame, keys, CIFAR_METRICS)
    comparisons = []
    fresh = frame[frame.seed.isin(range(1, 7))]
    # Each section is a complete pairwise family; all 13 tests share Holm correction.
    for section, section_rows in fresh.groupby('section', sort=False):
        for left, right in itertools.combinations(section_rows.label.unique(), 2):
            result = paired_test(section_rows[section_rows.label == left],
                                 section_rows[section_rows.label == right], 'test_accuracy', list(range(1, 7)))
            comparisons.append(dict(section=section, left=left, right=right, metric='test_accuracy', **result))
    comparisons = pd.DataFrame(comparisons)
    comparisons['p_holm'] = holm(comparisons.p_value)
    return summary, comparisons


def choose_by_validation(frame, keys):
    # Deterministic tie order; test metrics never participate.
    ordered = frame.sort_values(['validation_accuracy', 'selected_step', 'mass', 'run_index'],
                                ascending=[False, True, True, True], kind='stable')
    return ordered.groupby(keys, sort=False, dropna=False).head(1)


def language_summary(frame):
    frame = frame.copy()
    frame['zeta'] = 1 - frame.beta1 / frame.beta2
    keys = [k for k in HYPERPARAMETERS if k in frame] + ['zeta']
    summary = aggregate(frame, keys, ['validation_loss', 'validation_perplexity'])
    fresh_summary = aggregate(frame[frame.seed.isin(range(101, 107))], keys,
                              ['validation_loss', 'validation_perplexity'])
    fresh_summary = fresh_summary.rename(columns={column: 'fresh_' + column
                                                 for column in fresh_summary if column not in keys})
    summary = summary.merge(fresh_summary, on=keys, validate='one_to_one')
    half_width = stats.t.ppf(.975, summary.n_seeds - 1) * summary.sd_validation_perplexity / np.sqrt(summary.n_seeds)
    summary['ci95_low'] = summary.mean_validation_perplexity - half_width
    summary['ci95_high'] = summary.mean_validation_perplexity + half_width
    # Ten planned contrasts, each against the same-beta/LR/clipping zero-zeta control.
    comparisons = []
    fresh = frame[frame.seed.isin(range(101, 107))]
    for (beta, lr, clip), group in fresh.groupby(['beta2', 'lr', 'grad_clip'], sort=False):
        control = group[group.beta1 == group.beta2]
        for beta1, candidate in group[group.beta1 != group.beta2].groupby('beta1', sort=False):
            result = paired_test(candidate, control, 'validation_loss', list(range(101, 107)))
            comparisons.append(dict(beta1=beta1, beta2=beta, lr=lr, grad_clip=clip,
                                    zeta=1 - beta1 / beta, metric='validation_loss', **result))
    comparisons = pd.DataFrame(comparisons)
    comparisons['p_holm'] = holm(comparisons.p_value)
    return summary, comparisons


def kinematics_summary(frame):
    keys = ['kinematics', 'mass', 'lr', 'weight_decay', 'beta1', 'beta2']
    complete = [group for _, group in frame.groupby(keys, sort=False)
                if set(group.seed) == set(range(5)) and len(group) == 5]
    if not complete:
        raise ValueError('No complete five-seed kinematics configurations')
    repeated = pd.concat(complete)
    all_summary = aggregate(repeated, keys, CIFAR_METRICS)
    selected = all_summary.sort_values('mean_validation_accuracy', ascending=False, kind='stable')
    selected = selected.groupby('kinematics', sort=False).head(1)
    paired = []
    match = ['mass', 'lr', 'weight_decay', 'beta1', 'beta2', 'seed']
    for left, right in itertools.combinations(['minkowski', 'arctan', 'tanh'], 2):
        joined = repeated[repeated.kinematics == left].merge(
            repeated[repeated.kinematics == right], on=match, suffixes=('_left', '_right'))
        joined['difference'] = joined.test_accuracy_left - joined.test_accuracy_right
        differences = joined.groupby('seed').difference.mean()
        if len(differences) != 5:
            raise ValueError('Kinematics comparison requires all five seeds')
        half = stats.t.ppf(.975, 4) * differences.std(ddof=1) / math.sqrt(5)
        paired.append(dict(left=left, right=right, mean_difference=differences.mean(),
                           ci95_low=differences.mean()-half, ci95_high=differences.mean()+half,
                           n_seeds=5))
    return selected, pd.DataFrame(paired), all_summary


def cutoff_summary(frame, histories):
    records = []
    for _, row in frame.iterrows():
        for cutoff in range(int(row.eval_every), int(row.steps) + 1, int(row.eval_every)):
            best = best_checkpoint(histories[row.run_index], cutoff)
            records.append({**row.to_dict(), **best, 'cutoff': cutoff,
                            'selected_step': best['step'], 'family': row.section,
                            'epochs': cutoff * row.batch_size / 45000})
    result = pd.DataFrame(records)
    result['selected_mass'] = False
    for _, group in result.groupby(['family', 'cutoff'], sort=False):
        winner = group.sort_values(['validation_accuracy', 'selected_step', 'mass'],
                                  ascending=[False, True, True], kind='stable').index[0]
        result.loc[winner, 'selected_mass'] = True
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['selected', 'adaptive-mass', 'fixed-mass', 'kinematics', 'cutoffs', 'language'])
    parser.add_argument('--runs', type=Path, required=True)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--allow-incomplete', action='store_true', help='Descriptive summaries only; no paired tests')
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    frame, histories, coverage = load_runs(args.runs, config, args.allow_incomplete)
    args.output.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output / 'per_seed.csv', index=False)
    (args.output / 'coverage.json').write_text(json.dumps(coverage, indent=2) + '\n')
    comparisons = None
    if args.mode == 'selected':
        if args.allow_incomplete:
            summary = aggregate(frame, [k for k in HYPERPARAMETERS if k in frame], CIFAR_METRICS)
        else:
            summary, comparisons = selected_summary(frame)
    elif args.mode == 'adaptive-mass':
        summary = choose_by_validation(frame, ['mass', 'beta2'])
    elif args.mode == 'fixed-mass':
        summary = choose_by_validation(frame, ['section', 'mass', 'lr'])
    elif args.mode == 'kinematics':
        if args.allow_incomplete:
            raise ValueError('Kinematics inference requires the full grid')
        summary, comparisons, all_summary = kinematics_summary(frame)
        all_summary.to_csv(args.output / 'all_configurations.csv', index=False)
    elif args.mode == 'cutoffs':
        summary = cutoff_summary(frame, histories)
    elif args.mode == 'language':
        if args.allow_incomplete:
            frame['zeta'] = 1 - frame.beta1 / frame.beta2
            summary = aggregate(frame, [k for k in HYPERPARAMETERS if k in frame] + ['zeta'],
                                ['validation_loss', 'validation_perplexity'])
        else:
            summary, comparisons = language_summary(frame)
    summary.to_csv(args.output / 'summary.csv', index=False)
    if comparisons is not None:
        comparisons.to_csv(args.output / 'comparisons.csv', index=False)
    print(f'Wrote {len(summary)} summary rows from {len(frame)} completed runs to {args.output}')


if __name__ == '__main__':
    main()
