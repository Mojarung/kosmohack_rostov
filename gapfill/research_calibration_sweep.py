"""Небольшой перебор формы шума на разработочной маске, без переобучения моделей."""

import argparse
import json

import pandas as pd

from gapfill.config import ARTIFACTS_DIR
from gapfill.data import load_all, make_mask
from gapfill.research_blend import metric
from gapfill.research_calibrate import calibrate
from gapfill.research_constraints import published_norms


def main():
    """Оценка вариантов использует ровно те же отложенные точки и модельные прогнозы."""
    p = argparse.ArgumentParser()
    p.add_argument("--components", required=True)
    p.add_argument("--val-seed", type=int, default=777)
    p.add_argument("--dfs", type=float, nargs="+", default=[1.5, 3., 6., 12.])
    p.add_argument("--scales", type=float, nargs="+", default=[.75, 1., 1.25])
    p.add_argument("--out", default="calibration_sweep777")
    args = p.parse_args()
    obs, grid, _ = load_all()
    norms = published_norms(obs.loc[make_mask(obs, seed=args.val_seed)])
    base = pd.read_parquet(args.components)
    source = base.assign(prior=.6 * base.prior + .4 * base.neural)
    out = ARTIFACTS_DIR / "research" / args.out
    out.mkdir(parents=True, exist_ok=True)
    entries = []
    for df in args.dfs:
        for scale in args.scales:
            pred, _ = calibrate(source.assign(sigma=source.sigma * scale), grid, norms,
                                 "robust", .04, {"family": "student", "df": df})
            entry = {"df": df, "scale": scale, **metric(pred, "pred")}
            pred.to_parquet(out / f"df{df}_scale{scale}.parquet")
            entries.append(entry)
            print(json.dumps(entry), flush=True)
    (out / "result.json").write_text(json.dumps({"args": vars(args), "results": entries}, indent=2),
                                     encoding="utf-8")


if __name__ == "__main__":
    main()
