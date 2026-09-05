"""Устойчивая калибровка сумм с границами, выведенными из опубликованных дисперсий."""

import argparse
import json

import numpy as np
import pandas as pd
from scipy.optimize import Bounds, LinearConstraint, minimize
from threadpoolctl import threadpool_limits

from gapfill import research_calibrate as calibration
from gapfill.config import ARTIFACTS_DIR
from gapfill.data import load_all, make_mask
from gapfill.research.blend import metric
from gapfill.research.constraints import linear_projection, published_norms


def value_bounds(a, b, c):
    """Для n чисел с суммой b и суммой квадратов c каждый ответ лежит в вычислимом интервале."""
    count = a.sum(1)
    useful = count > 0
    count, aa, bb, cc = count[useful], a[useful], b[useful], c[useful]
    center = bb / count
    radius = np.sqrt(np.maximum((count - 1) / count * (cc - bb ** 2 / count), 0.))
    low = np.where(aa > 0, (center - radius)[:, None], -np.inf).max(0) - 1e-6
    high = np.where(aa > 0, (center + radius)[:, None], np.inf).min(0) + 1e-6
    return low, high


def feasible_start(prior, a, b, basis, sums, sigma, low, high):
    """Ближайшая начальная точка внутри границ и на плоскости известных сумм."""
    initial = linear_projection(prior, a, b, scale=sigma)
    if np.all((initial >= low) & (initial <= high)):
        return initial
    fit = minimize(lambda x: .5 * np.sum(((x - prior) / sigma) ** 2), initial.clip(low, high),
                   jac=lambda x: (x - prior) / sigma ** 2, method="SLSQP", bounds=Bounds(low, high),
                   constraints=LinearConstraint(basis, sums, sums), options={"maxiter": 250, "ftol": 1e-9})
    if np.max(np.abs(basis @ fit.x - sums)) > 1e-5:
        raise ValueError(f"Не удалось найти допустимую начальную точку: {fit.message}")
    return fit.x


def ellipse_step(x, mean, noise, low, high, rng):
    """Срез нормального распределения внутри границ с сохранением линейных равенств."""
    angle = rng.uniform(0, 2 * np.pi, (len(x), 1))
    left, right = angle - 2 * np.pi, angle.copy()
    selected = x.copy()
    pending = np.ones(len(x), bool)
    for _ in range(100):
        candidate = mean + (x - mean) * np.cos(angle) + noise * np.sin(angle)
        valid = np.all((candidate >= low) & (candidate <= high), axis=1) & pending
        selected[valid] = candidate[valid]
        pending &= ~valid
        if not pending.any():
            break
        left = np.where((angle < 0) & pending[:, None], angle, left)
        right = np.where((angle >= 0) & pending[:, None], angle, right)
        angle = rng.uniform(left, right)
    return selected


def posterior(prior, a, b, c, sigma, steps, chains, seed):
    """Латентная гамма-точность и усечённое условное нормальное распределение."""
    u, s, vt = np.linalg.svd(a, full_matrices=False)
    rank = int((s > 1e-9).sum())
    basis, sums = vt[:rank], u[:, :rank].T @ b / s[:rank]
    low, high = value_bounds(a, b, c)
    initial = feasible_start(prior, a, b, basis, sums, sigma, low, high)
    x = np.tile(initial, (chains, 1))
    total, rng = np.zeros_like(x), np.random.default_rng(seed)
    rhs_mean = sums - basis @ prior
    for iteration in range(steps + 500):
        precision = rng.gamma(2., 2. / (3. + ((x - prior) / sigma) ** 2))
        variance = sigma ** 2 / np.maximum(precision, 1e-8)
        cross = variance[:, :, None] * basis.T[None]
        cov = basis[None] @ cross
        noise = rng.normal(size=x.shape) * np.sqrt(variance)
        rhs = np.stack([np.broadcast_to(rhs_mean, (len(x), rank)), noise @ basis.T], axis=-1)
        correction = cross @ np.linalg.solve(cov, rhs)
        mean = prior + correction[:, :, 0]
        x = ellipse_step(x, mean, noise - correction[:, :, 1], low, high, rng)
        if iteration >= 500:
            total += x
    means = total / steps
    return means.mean(0), float(np.max(means.std(0) / np.sqrt(chains)))


def main():
    """Сохраняет эксперимент отдельно; конечная метрика включает все отложенные выбросы."""
    p = argparse.ArgumentParser()
    p.add_argument("--components", required=True)
    p.add_argument("--val-seed", type=int, default=777)
    p.add_argument("--steps", type=int, default=1200)
    p.add_argument("--chains", type=int, default=24)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", default="truncated777")
    args = p.parse_args()
    obs, grid, _ = load_all()
    norms = published_norms(obs.loc[make_mask(obs, seed=args.val_seed)])
    base = pd.read_parquet(args.components)
    source = base.assign(prior=.6 * base.prior + .4 * base.neural)
    original = calibration.grouped_moments
    diagnostics = []

    def grouped(prior, a, b, c, **kwargs):
        pred, error = posterior(prior, a, b, c, kwargs["scale"], args.steps, args.chains, args.seed)
        calibration.linear_projection = lambda *a, **kw: pred.copy()
        diagnostics.append({"n": len(prior), "max_chain_se": error})
        print(f"Усечённая калибровка: {len(prior)}, ошибка между цепями {error:.5f}", flush=True)
        return original(prior, a, b, c, **kwargs)

    calibration.grouped_moments = grouped
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
