"""Условное среднее Student-t при известных суммах: уменьшение размазывания выбросов."""

import argparse
import json

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from gapfill.config import ARTIFACTS_DIR
from gapfill.data import load_all, make_mask
from gapfill.research import calibrate as calibration
from gapfill.research.blend import metric
from gapfill.research.constraints import linear_projection, published_norms


def reflection_moves(x, center, null, prior, sigma, rng, repeats):
    """Симметричные отражения переносят большие остатки, не меняя известных сумм."""
    for _ in range(repeats):
        i, j = rng.choice(len(prior), 2, replace=False)
        direction = null[:, i] - null[:, j]
        norm = direction @ direction
        if norm < 1e-10:
            continue
        candidate = x - 2 * ((x - center) @ direction)[:, None] * direction / norm
        old_lp = calibration.log_likelihood(x - prior, sigma).sum(1)
        new_lp = calibration.log_likelihood(candidate - prior, sigma).sum(1)
        accept = np.log(rng.random(len(x))) < new_lp - old_lp
        x[accept] = candidate[accept]
    return x


def posterior(prior, a, b, scale, steps, seed, chains=12, reflections=0):
    """Гиббс: Student-t представлен смесью нормальных распределений с гамма-точностью."""
    u, s, vt = np.linalg.svd(a, full_matrices=False)
    rank = int((s > 1e-9).sum())
    basis = vt[:rank]
    sums = u[:, :rank].T @ b / s[:rank]
    rng = np.random.default_rng(seed)
    sigma = np.full(len(prior), .04) if scale is None else scale
    x = np.tile(linear_projection(prior, a, b, scale=sigma), (chains, 1))
    center = x[0].copy()
    null = np.eye(len(prior)) - basis.T @ basis
    total = np.zeros_like(x)
    rhs_mean = sums - basis @ prior
    for iteration in range(steps + 300):
        precision = rng.gamma(2., 2. / (3. + ((x - prior) / sigma) ** 2))
        variance = sigma ** 2 / np.maximum(precision, 1e-8)
        cross = variance[:, :, None] * basis.T[None]
        cov = basis[None] @ cross
        noise = rng.normal(size=x.shape) * np.sqrt(variance)
        rhs = np.stack([np.broadcast_to(rhs_mean, (len(x), rank)), noise @ basis.T], axis=-1)
        correction = cross @ np.linalg.solve(cov, rhs)
        mean = prior + correction[:, :, 0]
        x = mean + noise - correction[:, :, 1]
        x = reflection_moves(x, center, null, prior, sigma, rng, reflections)
        if iteration >= 300:
            total += mean
    means = total / steps
    return means.mean(0), float(np.max(means.std(0) / np.sqrt(len(means))))


def main():
    """Сравнивает только распределение поправки; остальные части прогнозирования фиксированы."""
    p = argparse.ArgumentParser()
    p.add_argument("--components", required=True)
    p.add_argument("--val-seed", type=int, default=777)
    p.add_argument("--steps", type=int, default=700)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--chains", type=int, default=12)
    p.add_argument("--reflections", type=int, default=0)
    p.add_argument("--out", default="robust_linear777")
    args = p.parse_args()
    obs, grid, _ = load_all()
    norms = published_norms(obs.loc[make_mask(obs, seed=args.val_seed)])
    base = pd.read_parquet(args.components)
    source = base.assign(prior=.6 * base.prior + .4 * base.neural)
    diagnostics = []

    def estimator(prior, a, b, strength=1., ridge=0., scale=None):
        pred, error = posterior(prior, a, b, scale, args.steps, args.seed, args.chains, args.reflections)
        diagnostics.append({"n": len(prior), "max_chain_se": error})
        print(f"Условное среднее: {len(prior)} точек, ошибка между цепями {error:.5f}", flush=True)
        return pred

    calibration.linear_projection = estimator
    with threadpool_limits(limits=1):
        pred, _ = calibration.calibrate(source, grid, norms, "robust", .04)
    out = ARTIFACTS_DIR / "research" / args.out
    out.mkdir(parents=True, exist_ok=True)
    pred.to_parquet(out / "pred.parquet")
    result = {"args": vars(args), **metric(pred, "pred"), "chains": diagnostics}
    (out / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k != "chains"}), flush=True)


if __name__ == "__main__":
    main()
