"""Paired 9D quadratic experiments using the public MassiveLion optimizer."""
import argparse
import hashlib
import itertools
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import torch

from massive_lion import MassiveLion
import massive_lion.optimizer as optimizer_module

SPECTRA = {
    "heterogeneous": [[1, 2, 3], [99, 100, 101], [4998, 4999, 5000]],
    "homogeneous": [[1, 99, 4998], [2, 100, 4999], [3, 101, 5000]],
}


def dataset(seeds, landscape, steps, batch_size, factor="eigenvector"):
    hs, xs, batches = [], [], []
    for seed in seeds:
        rng = np.random.default_rng(seed)
        h = np.zeros((9, 9))
        for b, eig in enumerate(SPECTRA[landscape]):
            a = rng.normal(size=(3, 6)) / 6
            _, u = np.linalg.eigh(a @ a.T)
            h[3*b:3*b+3, 3*b:3*b+3] = u.T @ np.diag(eig) @ u
        eig, u = np.linalg.eigh(h)
        x = 3 * u @ np.diag(np.sqrt(eig))
        if factor == "symmetric":
            x = x @ u.T
        sample_rng = np.random.default_rng(np.random.SeedSequence([seed, 20260924]))
        batch = np.array([sample_rng.choice(9, batch_size, replace=False)
                          for _ in range(steps)])
        hs.append(h)
        xs.append(x)
        batches.append(batch)
    return np.array(hs), np.array(xs), np.array(batches).transpose(1, 0, 2)


def optimizer(params, configs):
    groups = []
    for p, c in zip(params, configs):
        beta, zeta = c["beta"], c["zeta"]
        groups.append(dict(params=[p], lr=c["lr"],
                           betas=((1-zeta)*beta, beta),
                           beta_gravity=beta, kappa=beta,
                           mass_mode=c.get("mass_mode", "momentum_diff")))
    return MassiveLion(groups, mass=0, weight_decay=0,
                                    mass_mode="momentum_diff",
                                    update_mode="coordinate", kinematics="minkowski")


def schedule(steps, warmup=0.05):
    n = int(warmup * steps)
    return np.r_[np.linspace(0, 1, n),
                 0.5 * (1 + np.cos(np.pi * np.arange(steps-n)/(steps-n)))]


def validate():
    """Check actual optimizer states, batching, and objective construction."""
    h, x, _ = dataset([17, 23], "heterogeneous", 10, 3)
    assert np.allclose(x @ x.transpose(0, 2, 1)/9, h)
    assert np.allclose(np.linalg.eigvalsh(h), sorted(sum(SPECTRA["heterogeneous"], [])))
    all_batches = np.array(list(itertools.combinations(range(9), 3)))
    xb = x[0, :, all_batches].transpose(0, 2, 1)
    assert np.allclose((xb @ xb.transpose(0, 2, 1)/3).mean(0), h[0])
    c = dict(beta=0.95, zeta=0.0, lr=0.03)
    p = torch.nn.Parameter(torch.ones((2, 9), dtype=torch.float64))
    opt = optimizer([p], [c])
    q = [torch.nn.Parameter(torch.ones(9, dtype=torch.float64)) for _ in range(2)]
    separate = [optimizer([v], [c]) for v in q]
    v = torch.zeros_like(p)
    rng = np.random.default_rng(100)
    max_error = 0.0
    for t in range(25):
        g = torch.tensor(rng.normal(size=(2, 9)), dtype=torch.float64)
        g[:, -1] = 0  # exercise exact zero/zero handling without epsilon
        before = p.detach().clone()
        p.grad = g
        opt.step()
        v = 0.95*v + 0.05*g.square()
        m = opt.state[p]["exp_avg"]
        mass = opt.state[p]["metric_diag"]
        assert torch.allclose(mass+m.square(), v, atol=1e-14, rtol=1e-14)
        denominator = torch.sqrt(v)
        expected = before - c["lr"]*m/denominator.masked_fill(denominator == 0, 1)
        max_error = max(max_error, float((p.detach()-expected).abs().max()))
        assert torch.allclose(p, expected, atol=1e-14, rtol=1e-14)
        for j, (qq, oo) in enumerate(zip(q, separate)):
            qq.grad = g[j].clone()
            oo.step()
            assert torch.equal(p[j], qq)
    return dict(dataset_hessian=True, unbiased_batch_gradient=True,
                batched_matches_individual=True, zero_epsilon=True,
                adam_second_moment_identity=True, max_update_error=max_error)


@torch.no_grad()
def run(job, output_dir, input_fixture=None):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    name, configs = job["name"], job["configs"]
    seeds = np.arange(job["seed_start"], job["seed_start"]+job["n_seeds"])
    steps, bs = job.get("steps", 500), job["batch_size"]
    h, x, batches = dataset(seeds, job["landscape"], steps, bs,
                            job.get("factor", "eigenvector"))
    initial = None
    if input_fixture is not None:
        # Freeze the mathematical problem, not optimizer outputs. Eigensolver
        # eigenvector signs are not portable across LAPACK implementations.
        fixture = json.loads(Path(input_fixture).read_text())
        if (job["landscape"] != "heterogeneous"
                or job.get("initialization") != "gaussian"
                or job.get("factor", "eigenvector") != "eigenvector"):
            raise ValueError("The confirmation fixture requires heterogeneous Gaussian-initialized runs with the eigenvector factor")
        indices = {seed: i for i, seed in enumerate(fixture["seeds"])}
        selected = [indices[int(seed)] for seed in seeds]
        h = np.asarray(fixture["h"], dtype=np.float64)[selected]
        x = np.asarray(fixture["x"], dtype=np.float64)[selected]
        initial = np.asarray(fixture["w0"], dtype=np.float64)[selected]
    ht, xt = torch.from_numpy(h), torch.from_numpy(x)
    weights = torch.ones((len(configs), len(seeds), 9), dtype=torch.float64)
    if initial is not None:
        weights[:] = torch.from_numpy(initial)
    elif job.get("initialization", "ones") == "gaussian":
        initial = np.array([np.random.default_rng(np.random.SeedSequence([int(s), 831])).normal(size=9)
                            for s in seeds])
        initial *= 3/np.linalg.norm(initial, axis=1, keepdims=True)
        weights[:] = torch.from_numpy(initial)
    np.savez_compressed(output_dir/f"{name}_data.npz", seeds=seeds, h=h, x=x, batches=batches,
                        w0=weights[0].numpy())
    params = [torch.nn.Parameter(weights[i]) for i in range(len(configs))]
    opt = optimizer(params, configs)
    rates = schedule(steps, job.get("warmup", 0.05))
    if job.get("schedule") == "constant":
        rates[:] = 1
    losses = np.empty((steps+1, len(configs), len(seeds)))
    failed = np.zeros((len(configs), len(seeds)), dtype=bool)
    dynamic = job.get("dynamics", False)
    details = {}
    if dynamic:
        for key in ("block_loss", "linear_work", "quadratic_cost", "step_sq",
                    "uphill", "mass_sq", "innovation_sq"):
            details[key] = np.empty((steps, len(configs), len(seeds), 3))
        details["weights"] = np.empty((steps+1, len(configs), len(seeds), 9))

    for t in range(steps+1):
        true_g = torch.einsum("sij,csj->csi", ht, weights)
        losses[t] = (0.5*(weights*true_g).sum(-1)).numpy()
        failed |= ~np.isfinite(losses[t]) | (losses[t] > 1e8)
        if dynamic:
            details["weights"][t] = weights.numpy()
        if t == steps:
            break
        if bs == 9:
            grad = true_g
        else:
            idx = torch.from_numpy(batches[t])[:, None, :].expand(-1, 9, -1)
            xb = torch.gather(xt, 2, idx)
            hb = xb @ xb.transpose(1, 2) / bs
            grad = torch.einsum("sij,csj->csi", hb, weights)
        if dynamic:
            before = weights.clone()
            m = torch.stack([opt.state[p]["exp_avg"] if t else torch.zeros_like(p)
                             for p in params])
            details["innovation_sq"][t] = (grad-m).square().reshape(len(configs), -1, 3, 3).mean(-1)
            for i, c in enumerate(configs):
                if c.get("mass_mode") == "gradient_diff":
                    shock = grad[i] - opt.state[params[i]]["prev_grad"] if t else torch.zeros_like(grad[i])
                    details["innovation_sq"][t, i] = shock.square().reshape(-1, 3, 3).mean(-1)
            details["block_loss"][t] = (0.5*weights*true_g).reshape(len(configs), -1, 3, 3).sum(-1)
        for i, (p, group) in enumerate(zip(params, opt.param_groups)):
            p.grad = grad[i]
            group["lr"] = configs[i]["lr"]*rates[t]
        opt.step()
        if dynamic:
            delta = weights-before
            hdelta = torch.einsum("sij,csj->csi", ht, delta)
            for key, arr in (("linear_work", true_g*delta),
                             ("quadratic_cost", 0.5*delta*hdelta),
                             ("step_sq", delta.square()),
                             ("uphill", (true_g*delta > 0).double())):
                details[key][t] = arr.reshape(len(configs), -1, 3, 3).sum(-1).numpy()
            mass = torch.stack([opt.state[p]["metric_diag"] for p in params])
            details["mass_sq"][t] = mass.reshape(len(configs), -1, 3, 3).mean(-1).numpy()
        if t % 100 == 99:
            print(f"{name}: {t+1}/{steps}, {time.monotonic()-start:.1f}s", flush=True)

    np.savez_compressed(output_dir/f"{name}_results.npz", losses=losses, failed=failed,
                        rates=rates, seeds=seeds, **details)
    receipt = dict(job, seconds=time.monotonic()-start,
                   optimizer="massive_lion.MassiveLion",
                   optimizer_sha256=hashlib.sha256(Path(optimizer_module.__file__).read_bytes()).hexdigest(),
                   runner_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                   epsilon=0, mass=0, weight_decay=0, bias_correction=False,
                   mass_mode=(configs[0].get("mass_mode", "momentum_diff")
                              if len({c.get("mass_mode", "momentum_diff") for c in configs}) == 1
                              else sorted({c.get("mass_mode", "momentum_diff") for c in configs})),
                   update_mode="coordinate", kinematics="minkowski",
                   zeta_definition="1 - beta1 / beta2", beta3_mapping="beta_gravity",
                   python=sys.version, torch=torch.__version__, numpy=np.__version__,
                   platform=platform.platform(), failures=int(failed.sum()))
    receipt["input_fixture_sha256"] = (
        hashlib.sha256(Path(input_fixture).read_bytes()).hexdigest() if input_fixture else None)
    (output_dir/f"{name}_receipt.json").write_text(json.dumps(receipt, indent=2)+"\n")
    print(f"DONE {name}: {len(configs)*len(seeds)} runs; failures={failed.sum()}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("plan", type=Path)
    parser.add_argument("--output", type=Path, default=Path("outputs/quadratics/9d"))
    parser.add_argument("--smoke", action="store_true", help="Two seeds, two settings, 10 steps; not paper results")
    parser.add_argument("--inputs", type=Path, help="Override a plan's fixed-input fixture")
    args = parser.parse_args()
    torch.set_num_threads(1)
    args.output.mkdir(parents=True, exist_ok=True)
    checks = validate()
    (args.output/"validation.json").write_text(json.dumps(checks, indent=2)+"\n")
    jobs = json.loads(args.plan.read_text())
    if args.smoke:
        jobs = [dict(jobs[0], name="smoke_9d", n_seeds=2, steps=10, configs=jobs[0]["configs"][:2])]
    for job in jobs:
        fixture = args.inputs
        if fixture is None and job.get("input_fixture"):
            fixture = args.plan.parent / job["input_fixture"]
        run(job, args.output, fixture)


if __name__ == "__main__":
    main()
