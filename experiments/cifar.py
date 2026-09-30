"""Train the paper's ResNet-18 on CIFAR-10 with local metrics and checkpoints."""
import argparse
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from experiments.common import (add_optimizer_args, amp_context, log_metrics,
    optimizer_for, resolve_optimizer_args, save_checkpoint, seed_everything, start_run, vision_lr, write_json)
from experiments.vision_data import CIFAR10Loader, load_splits
from experiments.vision_groups import parameter_groups
from experiments.vision_model import ResNet18


@torch.no_grad()
def evaluate(model, loader, ctx):
    model.eval()
    loss_sum, correct, count = 0.0, 0, 0
    for x, y in loader:
        with ctx:
            logits = model(x)
        loss_sum += F.cross_entropy(logits.float(), y, reduction='sum').item()
        correct += (logits.argmax(1) == y).sum().item()
        count += len(y)
    model.train()
    return {'accuracy': correct / count * 100, 'loss': loss_sum / count,
            'correct': correct, 'count': count}


def evaluate_preserving_rng(model, loader, ctx, device):
    """Keep test evaluation from changing the subsequent training RNG state."""
    cpu_rng = torch.get_rng_state()
    cuda_rng = torch.cuda.get_rng_state(device) if device.type == 'cuda' else None
    numpy_rng, python_rng = np.random.get_state(), random.getstate()
    try:
        return evaluate(model, loader, ctx)
    finally:
        torch.set_rng_state(cpu_rng)
        if cuda_rng is not None:
            torch.cuda.set_rng_state(cuda_rng, device)
        np.random.set_state(numpy_rng)
        random.setstate(python_rng)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=Path('data/cifar10/processed'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--steps', type=int, default=80000)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--eval-every', type=int, default=5000)
    parser.add_argument('--log-every', type=int, default=1000)
    parser.add_argument('--min-lr', type=float, default=1e-5)
    parser.add_argument('--compile', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--amp', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--save-checkpoint', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--smoke', action='store_true', help='Two CPU steps on generated data; no download')
    parser.add_argument('--label', default=None, help='Label carried into result summaries')
    parser.add_argument('--section', default=None, help='Comparison family in the manuscript table')
    add_optimizer_args(parser)
    args = parser.parse_args(argv)
    if args.smoke:
        args.device, args.steps, args.batch_size = 'cpu', 2, 2
        args.eval_every, args.log_every, args.compile, args.amp = 1, 1, False, False
    if min(args.steps, args.batch_size, args.eval_every, args.log_every) <= 0:
        parser.error('steps, batch size, and logging intervals must be positive')
    return resolve_optimizer_args(args)


def train(args):
    if args.smoke:
        torch.set_num_threads(1)
    seed_everything(args.seed)
    device = torch.device(args.device)
    if args.smoke:
        generator = torch.Generator().manual_seed(1123)
        splits = [(torch.randn(n, 3, 32, 32, generator=generator),
                   torch.randint(10, (n,), generator=generator)) for n in [8, 4, 4]]
    else:
        splits = load_splits(args.data, device)
    train_loader, val_loader, test_loader = [
        CIFAR10Loader(x, y, args.batch_size, train=i == 0, seed=args.seed)
        for i, (x, y) in enumerate(splits)]
    raw_model = ResNet18().to(device)
    model = torch.compile(raw_model) if args.compile else raw_model
    optimizer = optimizer_for(parameter_groups(raw_model, args.weight_decay), args)
    precision = ('bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
                 else 'float16') if args.amp else 'float32'
    ctx, scaler = amp_context(device, precision)
    start_run(args, {'model': 'ResNet18', 'num_classes': 10})
    step, best = 0, None
    model.train()
    while step < args.steps:
        for images, targets in train_loader:
            if step == args.steps:
                break
            lr = vision_lr(step, args.lr, args.min_lr, args.steps)
            for group in optimizer.param_groups:
                group['lr'] = lr
            optimizer.zero_grad(set_to_none=True)
            with ctx:
                loss = F.cross_entropy(model(images), targets)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            scaler.step(optimizer)
            scaler.update()
            step += 1
            record = {'step': step, 'lr': lr, 'train_loss': loss.item()}
            if step % args.eval_every == 0 or step == args.steps:
                validation = evaluate(model, val_loader, ctx)
                test = evaluate_preserving_rng(model, test_loader, ctx, device)
                record.update({f'validation_{k}': v for k, v in validation.items()})
                record.update({f'test_{k}': v for k, v in test.items()})
                # Strict comparison keeps the earliest checkpoint in a tie.
                if best is None or record['validation_correct'] > best['validation_correct']:
                    best = dict(record)
                    write_json(args.output / 'best.json', best)
                    if args.save_checkpoint:
                        save_checkpoint(args.output / 'best.pt', raw_model, optimizer, scaler, step, args)
            if step % args.log_every == 0 or 'validation_accuracy' in record:
                log_metrics(args.output, record)
    write_json(args.output / 'final.json', record)
    if args.save_checkpoint:
        save_checkpoint(args.output / 'final.pt', raw_model, optimizer, scaler, step, args)
    return record


if __name__ == '__main__':
    train(parse_args())
