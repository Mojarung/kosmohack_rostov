"""Интегрирование неоднозначных агрегатов цепями на сфере с сохранением суммы и дисперсии."""

import argparse
import json

import numpy as np
import pandas as pd

from gapfill.config import ARTIFACTS_DIR
from gapfill.data import load_all, make_mask
from gapfill.research_blend import metric
from gapfill import research_calibrate as calibration
from gapfill.research_constraints import published_norms


def propose(z, rng):
    """Симметричные повороты и перестановки позволяют переносить выброс между датами."""
    if rng.random() < .2:
        out = z.copy()
        i, j = rng.choice(z.shape[1], 2, replace=False)
        out[:, [i, j]] = out[:, [j, i]]
        return out
    tangent = rng.normal(size=z.shape)
    tangent -= tangent.mean(1, keepdims=True)
    tangent -= (tangent * z).sum(1, keepdims=True) * z
    tangent /= np.maximum(np.linalg.norm(tangent, axis=1, keepdims=True), 1e-12)
    angle = rng.normal(size=(len(z), 1)) * rng.choice([.03, .1, .3, 1.])
    return z * np.cos(angle) + tangent * np.sin(angle)


def integrate(prior, center, radius, sigma, density, seed, steps):
    """32 независимых цепи после разогрева; ошибки между цепями сохраняются для диагностики."""
    rng = np.random.default_rng(seed)
    candidates = calibration.sphere_directions(len(prior))
    ll = calibration.log_likelihood(center + radius * candidates - prior, sigma, density).sum(1)
    w = np.exp(ll - ll.max())
    w /= w.sum()
    z = candidates[rng.choice(len(candidates), 32, p=w)].copy()
    lp = calibration.log_likelihood(center + radius * z - prior, sigma, density).sum(1)
    total = np.zeros_like(z)
    n = 0
    for step in range(steps + 1000):
        proposed = propose(z, rng)
        value = center + radius * proposed
        new_lp = calibration.log_likelihood(value - prior, sigma, density).sum(1)
        accept = np.log(rng.random(len(z))) < new_lp - lp
        z[accept], lp[accept] = proposed[accept], new_lp[accept]
        if step >= 1000 and step % 5 == 0:
            total += center + radius * z
            n += 1
    means = total / n
    error = float(np.max(means.std(0) / np.sqrt(len(z))))
    return means.mean(0), {"n": len(prior), "ess_sobol": float(1 / (w @ w)), "max_chain_se": error}


def main():
    """Заменяет только численное интегрирование, сохраняя модель и контрольную маску."""
    p = argparse.ArgumentParser()
    p.add_argument("--components", required=True)
    p.add_argument("--val-seed", type=int, default=777)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--steps", type=int, default=4000)
    p.add_argument("--out", default="mcmc777")
    args = p.parse_args()
    obs, grid, _ = load_all()
    norms = published_norms(obs.loc[make_mask(obs, seed=args.val_seed)])
    base = pd.read_parquet(args.components)
    source = base.assign(prior=.6 * base.prior + .4 * base.neural)
    diagnostics = []

    def estimator(prior, center, radius, sigma, density=None):
        estimate, info = integrate(prior, center, radius, sigma, density, args.seed, args.steps)
        diagnostics.append(info)
        return estimate

    calibration.robust_group_mean = estimator
    pred, _ = calibration.calibrate(source, grid, norms, "robust", .04)
    out = ARTIFACTS_DIR / "research" / args.out
    out.mkdir(parents=True, exist_ok=True)
    pred.to_parquet(out / "pred.parquet")
    result = {"args": vars(args), **metric(pred, "pred"), "chains": diagnostics}
    (out / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k != "chains"}), flush=True)


if __name__ == "__main__":
    main()
