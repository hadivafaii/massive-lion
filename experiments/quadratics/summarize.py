"""Freeze tuning selections, then summarize untouched confirmation seeds."""
import csv
import json
from pathlib import Path
import numpy as np



def write_csv(path, rows):
    with path.open("w") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def metrics(loss):
    rel = loss/loss[0]
    logs = np.log10(np.maximum(rel, 1e-12))
    times = np.arange(len(loss))[:, None, None]
    return dict(auc=logs[1:].mean(0), endpoint=logs[-1], final=loss[-1],
                t6=np.where(rel > 1e-6, times, -1).max(0)+1,
                t10=np.where(rel > 1e-10, times, -1).max(0)+1)


def ci(values):
    rng = np.random.default_rng(824)
    means = values[rng.integers(len(values), size=(10000, len(values)))].mean(1)
    return [float(v) for v in np.quantile(means, [.025, .975])]


def select(output_dir, prefix="random"):
    tune_stem = prefix+"_tune" if prefix else "refine"
    confirm_stem = prefix+"_confirm" if prefix else "confirm"
    file_prefix = prefix+"_" if prefix else ""
    jobs, choices, rows = [], [], []
    for rpath in sorted(output_dir.glob(tune_stem+"_*_receipt.json")):
        r = json.loads(rpath.read_text())
        data = np.load(output_dir/f"{r['name']}_results.npz")
        m, cs = metrics(data["losses"]), r["configs"]
        if data["failed"].any():
            raise RuntimeError("Tuning failures require explicit penalization")
        selected = []
        for criterion in ("auc", "endpoint"):
            winners = []
            for z in sorted({c["zeta"] for c in cs}):
                ids = [i for i,c in enumerate(cs) if c["zeta"] == z]
                i = min(ids, key=lambda i: (m[criterion][i].mean(), m["auc"][i].mean()))
                winners.append(i)
                selected.append(dict(cs[i], selection=criterion))
                rows.append(dict(condition=r["name"], criterion=criterion, **cs[i],
                                 mean_auc=m["auc"][i].mean(), mean_endpoint=m["endpoint"][i].mean(),
                                 median_final=np.median(m["final"][i])))
            best = min(winners[1:], key=lambda i:(m[criterion][i].mean(), m["auc"][i].mean()))
            choices.append(dict(condition=r["name"].replace(tune_stem, confirm_stem),
                                criterion=criterion, baseline=cs[winners[0]], best_positive=cs[best]))
        baseline_lr = selected[0]["lr"]
        for z in sorted({c["zeta"] for c in cs}):
            selected.append(dict(beta=cs[0]["beta"], zeta=z, lr=baseline_lr, selection="matched_lr"))
        jobs.append(dict(name=r["name"].replace(tune_stem, confirm_stem),
                         landscape=r["landscape"], batch_size=r["batch_size"], steps=500,
                         initialization=r.get("initialization", "ones"), factor=r.get("factor", "eigenvector"),
                         seed_start=1000, n_seeds=128, configs=selected))
    if not rows:
        raise FileNotFoundError(f"No {tune_stem} tuning receipts in {output_dir}")
    write_csv(output_dir/(file_prefix+"tuning_summary.csv"), rows)
    (output_dir/(file_prefix+"selection.json")).write_text(json.dumps(choices, indent=2)+"\n")
    (output_dir/(file_prefix+"confirm_plan.json")).write_text(json.dumps(jobs, indent=2)+"\n")
    print(json.dumps(choices, indent=2))


def confirm(output_dir, prefix="random"):
    file_prefix = prefix+"_" if prefix else ""
    rows, pairs, per_seed = [], [], []
    choices = json.loads((output_dir/(file_prefix+"selection.json")).read_text())
    for job in json.loads((output_dir/(file_prefix+"confirm_plan.json")).read_text()):
        receipt = json.loads((output_dir/f"{job['name']}_receipt.json").read_text())
        if receipt["configs"] != job["configs"]:
            raise ValueError(f"{job['name']}: results do not match this plan; rerun confirmation after selection")
        d = np.load(output_dir/f"{job['name']}_results.npz")
        m, cs = metrics(d["losses"]), job["configs"]
        for i,c in enumerate(cs):
            rows.append(dict(condition=job["name"], **c, mean_auc=m["auc"][i].mean(),
                             mean_endpoint=m["endpoint"][i].mean(), median_final=np.median(m["final"][i]),
                             mean_t6=m["t6"][i].mean(), success6=np.mean(m["t6"][i]<=500),
                             mean_t10=m["t10"][i].mean(), success10=np.mean(m["t10"][i]<=500),
                             failures=int(d["failed"][i].sum())))
            for s,seed in enumerate(d["seeds"]):
                per_seed.append(dict(condition=job["name"], **c, seed=int(seed),
                                     **{k:float(v[i,s]) for k,v in m.items()}, failed=bool(d["failed"][i,s])))
        for choice in [v for v in choices if v["condition"]==job["name"]]:
            criterion = choice["criterion"]
            base = next(i for i,c in enumerate(cs) if c["selection"]==criterion and c["zeta"]==0)
            z = choice["best_positive"]["zeta"]
            target = next(i for i,c in enumerate(cs) if c["selection"]==criterion and c["zeta"]==z)
            pairs.append(compare(job["name"], criterion, base, target, cs, m))
            if criterion == "auc":
                target = next(i for i,c in enumerate(cs) if c["selection"]=="matched_lr" and c["zeta"]==z)
                pairs.append(compare(job["name"], "matched_lr", base, target, cs, m))
    write_csv(output_dir/(file_prefix+"confirmation_summary.csv"), rows)
    write_csv(output_dir/(file_prefix+"confirmation_per_seed.csv"), per_seed)
    write_csv(output_dir/(file_prefix+"paired_comparisons.csv"), pairs)
    for row in pairs:
        print(row)


def compare(name, criterion, base, target, cs, m):
    delta = m["auc"][target]-m["auc"][base]
    lo, hi = ci(delta)
    endpoint = m["endpoint"][target]-m["endpoint"][base]
    elo, ehi = ci(endpoint)
    time_delta = m["t6"][target]-m["t6"][base]
    tlo, thi = ci(time_delta)
    return dict(condition=name, criterion=criterion, zeta=cs[target]["zeta"],
                baseline_lr=cs[base]["lr"], ca_lr=cs[target]["lr"],
                auc_delta=delta.mean(), auc_ci_low=lo, auc_ci_high=hi,
                auc_win_fraction=np.mean(delta<0), endpoint_delta=endpoint.mean(),
                endpoint_ci_low=elo, endpoint_ci_high=ehi,
                median_baseline_final=np.median(m["final"][base]),
                median_ca_final=np.median(m["final"][target]),
                mean_baseline_t6=m["t6"][base].mean(), mean_ca_t6=m["t6"][target].mean(),
                t6_delta=time_delta.mean(), t6_ci_low=tlo, t6_ci_high=thi,
                baseline_success6=np.mean(m["t6"][base]<=500), ca_success6=np.mean(m["t6"][target]<=500))


def joint(output_dir):
    rows, selections = [], []
    for bs in [3, 9]:
        candidates = []
        for tp, cp in [("random_tune_heterogeneous", "random_confirm_heterogeneous"),
                       ("beta_tune_0.9", "beta_confirm_0.9"),
                       ("beta_tune_0.975", "beta_confirm_0.975")]:
            r = json.loads((output_dir/f"{tp}_{bs}_receipt.json").read_text())
            m = metrics(np.load(output_dir/f"{tp}_{bs}_results.npz")["losses"])
            candidates.extend((float(m["auc"][i].mean()), c, f"{cp}_{bs}")
                              for i,c in enumerate(r["configs"]))
        chosen = [min([c for c in candidates if (c[1]["zeta"]>0)==pos], key=lambda c:c[0])
                  for pos in [False, True]]
        losses, configs = [], []
        for score,c,name in chosen:
            r = json.loads((output_dir/f"{name}_receipt.json").read_text())
            idx = next(i for i,a in enumerate(r["configs"]) if a["selection"]=="auc"
                       and all(a[k]==v for k,v in c.items()))
            losses.append(np.load(output_dir/f"{name}_results.npz")["losses"][:,idx])
            configs.append(c)
        row = compare(f"joint_beta_heterogeneous_{bs}", "auc", 0, 1, configs,
                      metrics(np.stack(losses, axis=1)))
        row.update(baseline_beta=configs[0]["beta"], ca_beta=configs[1]["beta"])
        rows.append(row)
        selections.append(dict(batch_size=bs, baseline=dict(configs[0],source=chosen[0][2]),
                               ca=dict(configs[1],source=chosen[1][2])))
    write_csv(output_dir/"joint_beta_comparisons.csv", rows)
    (output_dir/"joint_selection.json").write_text(json.dumps(selections, indent=2)+"\n")


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("select", "confirm", "joint"))
    parser.add_argument("--input", type=Path, default=Path("outputs/quadratics/9d"))
    parser.add_argument("--prefix", choices=("random", "beta"), default="random")
    args = parser.parse_args()
    if args.command == "joint":
        joint(args.input)
    else:
        {"select": select, "confirm": confirm}[args.command](args.input, args.prefix)


if __name__ == "__main__":
    main()
