"""Inspect or execute explicit reproduction grids without a cluster dependency."""
import argparse
import itertools
import json
import shlex
import subprocess
import sys
from pathlib import Path


def expand(config):
    """Cross each stage's grid with its explicit cases; preserve authored order."""
    jobs = []
    for stage in config['stages']:
        grid = stage.get('grid', {})
        for values in itertools.product(*grid.values()):
            for case in stage.get('cases', [{}]):
                jobs.append({**config.get('fixed', {}), **dict(zip(grid, values)), **case})
    return jobs


def command(experiment, job, output, extra=()):
    result = [sys.executable, '-m', f'experiments.{experiment}', '--output', str(output)]
    for key, value in job.items():
        flag = key.replace('_', '-')
        if isinstance(value, bool):
            result.append('--' + ('' if value else 'no-') + flag)
        else:
            result.extend(['--' + flag, str(value)])
    return result + list(extra)


def main():
    argv = sys.argv[1:]
    extra = []
    if '--' in argv:
        split = argv.index('--')
        argv, extra = argv[:split], argv[split + 1:]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('config', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--index', type=int, help='Run just this zero-based grid index')
    parser.add_argument('--execute', action='store_true', help='Without this flag, only show the plan')
    args = parser.parse_args(argv)
    config = json.loads(args.config.read_text())
    jobs = expand(config)
    indices = range(len(jobs)) if args.index is None else [args.index]
    if args.index is not None and not 0 <= args.index < len(jobs):
        parser.error(f'index must be in [0, {len(jobs) - 1}]')
    print(f'{config["experiment"]}: {len(jobs)} jobs. {config["description"]}')
    for index in indices:
        cmd = command(config['experiment'], jobs[index], args.output / f'{index:05d}', extra)
        if args.execute:
            print(shlex.join(cmd), flush=True)
            subprocess.run(cmd, check=True)
        elif args.index is not None or index < 3:
            print(f'[{index}] {shlex.join(cmd)}')
    if not args.execute:
        print('Preview only. Add --execute to train; use --index to distribute jobs externally.')


if __name__ == '__main__':
    main()
