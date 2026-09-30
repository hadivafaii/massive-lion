"""The 160M SlimPajama confirmation recipe with local logging."""
import argparse
import dataclasses
import math
from pathlib import Path

import torch
from torch.utils.data import Dataset
from experiments.common import (add_optimizer_args, amp_context, language_lr,
    log_metrics, optimizer_for, resolve_optimizer_args, save_checkpoint, seed_everything, start_run, write_json)
from experiments.language_data import make_secret_sauce_dataloaders, make_dataloader, move_batch_to_device
from experiments.language_groups import parameter_groups
from experiments.language_model import ModelConfig, Transformer


class SmokeTokens(Dataset):
    """Fixed synthetic token rows; generation never changes model-initialization RNG."""
    def __init__(self, rows, length, vocab):
        generator = torch.Generator().manual_seed(1123)
        self.tokens = torch.randint(vocab, (rows, length + 1), generator=generator)

    def __len__(self):
        return len(self.tokens)

    def __getitem__(self, index):
        return {'input_ids': self.tokens[index]}


@torch.no_grad()
def evaluate(model, loader, args, device, ctx):
    model.eval()
    loss_sum, count = 0.0, 0
    for batch in loader:
        inputs, targets, mask = move_batch_to_device(batch, args.seq_len, device)
        with ctx:
            _, loss = model(inputs, targets=targets, attn_mask=mask)
        loss_sum += loss.item() * len(inputs)
        count += len(inputs)
    model.train()
    if count == 0:
        raise ValueError('Validation set has no full microbatch')
    return loss_sum / count, count * args.seq_len


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    root = Path('data/slimpajama/tokenized_EleutherAI_gpt-neox-20b/ctx_2048')
    parser.add_argument('--trainset-path', type=Path, default=root / 'train')
    parser.add_argument('--validset-path', type=Path, default=root / 'valid')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seed', type=int, default=100)
    parser.add_argument('--steps', type=int, default=6200)
    parser.add_argument('--warmup-steps', type=int, default=620)
    parser.add_argument('--min-lr', type=float, default=1e-5)
    parser.add_argument('--micro-batch-size', type=int, default=32)
    parser.add_argument('--accumulation', type=int, default=8)
    parser.add_argument('--target-batch-size', type=int, default=256)
    parser.add_argument('--grad-clip', type=float, default=1.0)
    parser.add_argument('--valid-tokens', type=int, default=100000000)
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--seq-len', type=int, default=2048)
    parser.add_argument('--vocab-size', type=int, default=50280)
    parser.add_argument('--d-model', type=int, default=768)
    parser.add_argument('--n-layers', type=int, default=12)
    parser.add_argument('--n-heads', type=int, default=12)
    parser.add_argument('--dtype', choices=['bfloat16', 'float16', 'float32'], default='bfloat16')
    parser.add_argument('--compile', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--save-checkpoint', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--log-every', type=int, default=1)
    parser.add_argument('--smoke', action='store_true', help='Two small CPU steps on generated tokens')
    parser.add_argument('--label', default=None)
    add_optimizer_args(parser, mass=0.0, beta1=0.92625, beta2=0.975)
    parser.set_defaults(lr=0.016)
    args = parser.parse_args(argv)
    if args.smoke:
        args.device, args.steps, args.warmup_steps, args.dtype = 'cpu', 2, 1, 'float32'
        args.micro_batch_size, args.accumulation, args.target_batch_size = 2, 2, 4
        args.seq_len, args.vocab_size, args.d_model, args.n_layers, args.n_heads = 16, 64, 32, 1, 2
        args.num_workers, args.compile = 0, False
    if args.steps <= args.warmup_steps or args.warmup_steps <= 0:
        parser.error('require 0 < warmup-steps < steps')
    if min(args.micro_batch_size, args.accumulation, args.log_every) <= 0:
        parser.error('microbatch, accumulation and log interval must be positive')
    if args.micro_batch_size * args.accumulation != args.target_batch_size:
        parser.error('micro-batch-size times accumulation must equal target-batch-size')
    return resolve_optimizer_args(args)


def train(args):
    if args.smoke:
        torch.set_num_threads(1)
    seed_everything(args.seed)
    device = torch.device(args.device)
    if device.type == 'cuda':
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = True
    cfg = ModelConfig(vocab_size=args.vocab_size, seq_len=args.seq_len,
                      dim=args.d_model, expand=8 / 3, n_layers=args.n_layers,
                      n_heads=args.n_heads, mlp='glu', rmsnorm_eps=1e-6,
                      tie_embeddings=False, qk_norm=True, embed_norm=True, rope_theta=500000.0)
    # Loaders precede model initialization, matching the source trainer.
    if args.smoke:
        train_loader = make_dataloader(SmokeTokens(8, args.seq_len, args.vocab_size), 2, 0)
        valid_loader = make_dataloader(SmokeTokens(4, args.seq_len, args.vocab_size), 2, 0, drop_last=True)
    else:
        args.sampler, args.sampler_seed = 'sequential', None
        train_loader, valid_loader = make_secret_sauce_dataloaders(args)
    raw_model = Transformer(cfg).to(device)
    model = torch.compile(raw_model) if args.compile else raw_model
    optimizer = optimizer_for(parameter_groups(raw_model, args.weight_decay), args)
    ctx, scaler = amp_context(device, args.dtype)
    start_run(args, dataclasses.asdict(cfg))
    for group in optimizer.param_groups:
        group['lr'] = 0.0
    optimizer.zero_grad(set_to_none=True)
    model.train()
    step, micro_step, tokens = 0, 0, 0
    pending_losses = []
    for batch in train_loader:
        inputs, targets, mask = move_batch_to_device(batch, args.seq_len, device)
        tokens += len(inputs) * args.seq_len
        micro_step += 1
        with ctx:
            _, loss = model(inputs, targets=targets, attn_mask=mask)
            scaled_loss = loss / args.accumulation
        pending_losses.append(loss.detach())
        scaler.scale(scaled_loss).backward()
        if micro_step < args.accumulation:
            continue
        micro_step = 0
        if args.grad_clip:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        lr_used = optimizer.param_groups[0]['lr']
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
        step += 1
        next_lr = language_lr(step, args.lr, args.min_lr, args.warmup_steps, args.steps)
        for group in optimizer.param_groups:
            group['lr'] = next_lr
        if step % args.log_every == 0 or step == args.steps:
            log_metrics(args.output, {'step': step, 'tokens': tokens, 'lr': lr_used,
                                     'next_lr': next_lr, 'train_loss': torch.stack(pending_losses).mean().item()})
            pending_losses = []
        if step == args.steps:
            break
    if step != args.steps:
        raise RuntimeError(f'Training data exhausted at step {step}, expected {args.steps}')
    validation_loss, validation_tokens = evaluate(model, valid_loader, args, device, ctx)
    final = {'step': step, 'tokens': tokens, 'validation_tokens': validation_tokens,
             'validation_loss': validation_loss, 'validation_perplexity': math.exp(validation_loss)}
    log_metrics(args.output, final)
    write_json(args.output / 'final.json', final)
    if args.save_checkpoint:
        save_checkpoint(args.output / 'final.pt', raw_model, optimizer, scaler, step, args)
    return final


if __name__ == '__main__':
    train(parse_args())
