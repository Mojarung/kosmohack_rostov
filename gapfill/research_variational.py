"""Детерминированная вариационная аппроксимация устойчивой поправки по известным суммам."""

import argparse
import json

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from gapfill import research_calibrate as calibration
from gapfill.config import ARTIFACTS_DIR
from gapfill.data import load_all, make_mask
from gapfill.research_blend import metric
from gapfill.research_constraints import published_norms


def posterior(prior, a, b, sigma, df=3., steps=50):
    """Чередует условное нормальное распределение и ожидаемую точность Student-t."""
    u, s, vt = np.linalg.svd(a, full_matrices=False)
    rank = int((s > 1e-9).sum())
    basis, sums = vt[:rank], u[:, :rank].T @ b / s[:rank]
    variance = sigma ** 2
    for _ in range(steps):
        cross = variance[:, None] * basis.T
        cov = basis @ cross
        correction = cross @ np.linalg.solve(cov, sums - basis @ prior)
        mean = prior + correction
        conditional_var = np.maximum(variance - (cross * np.linalg.solve(cov, cross.T).T).sum(1), 0.)
        new_var = (df * sigma ** 2 + correction ** 2 + conditional_var) / (df + 1.)
        variance = .5 * variance + .5 * new_var
    return mean


def main():
    """Сравнивает заранее перечисленные формы устойчивости на фиксированном прогнозе."""
    p = argparse.ArgumentParser()
    p.add_argument("--components", required=True)
    p.add_argument("--val-seed", type=int, default=777)
    p.add_argument("--dfs", type=float, nargs="+", default=[1., 3., 10.])
    p.add_argument("--out", default="variational777")
    args = p.parse_args()
    obs, grid, _ = load_all()
    norms = published_norms(obs.loc[make_mask(obs, seed=args.val_seed)])
    base = pd.read_parquet(args.components)
    source = base.assign(prior=.6 * base.prior + .4 * base.neural)
    out = ARTIFACTS_DIR / "research" / args.out
    out.mkdir(parents=True, exist_ok=True)
    results = []
    for df in args.dfs:
        def projection(prior, a, b, strength=1., ridge=0., scale=None):
            return posterior(prior, a, b, scale, df)
        calibration.linear_projection = projection
        with threadpool_limits(limits=1):
            pred, _ = calibration.calibrate(source, grid, norms, "robust", .04)
        pred.to_parquet(out / f"df{df}.parquet")
        result = {"df": df, **metric(pred, "pred")}
        results.append(result)
        print(json.dumps(result), flush=True)
    (out / "result.json").write_text(json.dumps({"args": vars(args), "results": results}, indent=2),
                                     encoding="utf-8")


if __name__ == "__main__":
    main()
