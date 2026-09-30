"""Small shared helpers; experiment-specific arithmetic stays in each trainer."""
import argparse
import contextlib
import json
import math
import platform
import random
from pathlib import Path

import numpy as np
import torch
from massive_lion import create_optimizer


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def write_json(path, data):
    Path(path).write_text(json.dumps(data, indent=2, default=str) + '\n')


def start_run(args, model_config=None):
    # Avoid mixing previous metrics with a new run.
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output / 'config.json').exists():
        raise FileExistsError(f'{args.output} already contains a run')
    write_json(args.output / 'config.json', vars(args))
    write_json(args.output / 'environment.json', {
        'python': platform.python_version(), 'torch': torch.__version__,
        'numpy': np.__version__, 'platform': platform.platform(),
        'cuda': torch.version.cuda,
        'device': str(args.device),
    })
    if model_config is not None:
        write_json(args.output / 'model.json', model_config)


def log_metrics(output, metrics):
    with (output / 'history.jsonl').open('a') as handle:
        handle.write(json.dumps(metrics) + '\n')
    print(' | '.join(f'{k}: {v:.6g}' if isinstance(v, float) else f'{k}: {v}'
                     for k, v in metrics.items()), flush=True)


def amp_context(device, dtype='bfloat16'):
    if device.type != 'cuda' or dtype == 'float32':
        return contextlib.nullcontext(), torch.amp.GradScaler(enabled=False)
    precision = {'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
    return (torch.amp.autocast('cuda', dtype=precision),
            torch.amp.GradScaler(enabled=precision == torch.float16))


def add_optimizer_args(parser, *, mass=1.0, beta1=0.7, beta2=0.95):
    parser.add_argument('--optim', default='m_lion',
                        choices=['m_lion', 'lion', 'signum', 'm_signum', 'ss_adamw', 'qhm', 'sgd'])
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--weight-decay', type=float, default=0.1)
    parser.add_argument('--beta1', type=float, default=None)
    parser.add_argument('--beta2', type=float, default=beta2)
    parser.add_argument('--mass', type=float, default=None)
    parser.add_argument('--adaptive-mass', action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument('--kinematics', choices=['minkowski', 'arctan', 'tanh'], default='minkowski')
    parser.add_argument('--foreach', action=argparse.BooleanOptionalAction, default=True)
    parser.set_defaults(_default_beta1=beta1, _default_mass=mass)


def resolve_optimizer_args(args):
    """Record the actual reduction settings and reject incompatible explicit flags."""
    tied = args.optim in {'signum', 'm_signum', 'ss_adamw', 'sgd'}
    if args.beta1 is None:
        args.beta1 = args.beta2 if tied else args._default_beta1
    if args.mass is None:
        args.mass = 0.0 if args.optim in {'lion', 'signum', 'ss_adamw', 'sgd', 'qhm'} else args._default_mass
    if args.adaptive_mass is None:
        args.adaptive_mass = args.optim in {'m_lion', 'ss_adamw'}
    if tied and args.beta1 != args.beta2:
        raise ValueError(f'{args.optim} requires beta1 == beta2')
    if args.optim in {'lion', 'signum'} and (args.mass != 0 or args.adaptive_mass):
        raise ValueError(f'{args.optim} requires mass=0 and no adaptive mass')
    if args.optim == 'ss_adamw' and (args.mass != 0 or not args.adaptive_mass or args.kinematics != 'minkowski'):
        raise ValueError('ss_adamw requires mass=0, adaptive mass and Minkowski kinematics')
    if args.optim in {'sgd', 'qhm'} and (args.mass != 0 or args.adaptive_mass):
        raise ValueError('SGD/QHM do not use rest mass or adaptive mass')
    if args.label is None:
        args.label = {'m_lion': 'M-Lion', 'lion': 'Lion', 'signum': 'Signum',
                      'm_signum': 'M-Signum', 'ss_adamw': 'SS AdamW',
                      'qhm': 'QHM', 'sgd': 'SGD'}[args.optim]
    if hasattr(args, 'section') and args.section is None:
        args.section = ('SGD' if args.optim in {'sgd', 'qhm'} else
                        'Signum' if args.optim in {'signum', 'm_signum', 'ss_adamw'} else 'Lion')
    del args._default_beta1, args._default_mass
    return args


def optimizer_for(groups, args):
    common = dict(lr=args.lr, betas=(args.beta1, args.beta2), weight_decay=args.weight_decay)
    if args.optim in {'qhm', 'sgd'}:
        from massive_lion.baselines import QHM
        return QHM(groups, **common)
    return create_optimizer(args.optim, groups, foreach=args.foreach, mass=args.mass,
                            adaptive_mass=args.adaptive_mass, kinematics=args.kinematics, **common)


def save_checkpoint(path, model, optimizer, scaler, step, args):
    torch.save({'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
                'scaler': scaler.state_dict(), 'step': step, 'config': vars(args)}, path)


def vision_lr(step, lr, min_lr, steps):
    """The original zero-warmup CIFAR schedule, indexed before each update."""
    return min_lr + 0.5 * (1.0 + np.cos(np.pi * step / steps)) * (lr - min_lr)


def language_lr(step, lr, min_lr, warmup_steps, steps):
    """The original LM schedule: first update at zero, scheduler after update."""
    if step <= warmup_steps:
        return lr * step / warmup_steps
    progress = (step - warmup_steps) / (steps - warmup_steps)
    return min_lr + 0.5 * (lr - min_lr) * (1 + math.cos(math.pi * progress))
