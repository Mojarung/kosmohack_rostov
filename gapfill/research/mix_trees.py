"""Проверка разнообразия двух табличных моделей при зафиксированной доле нейросети."""

import argparse
import json

import pandas as pd

from gapfill.config import ARTIFACTS_DIR
from gapfill.data import load_all, make_mask
from gapfill.research.blend import metric
from gapfill.research.calibrate import calibrate, get_predictions
from gapfill.research.constraints import published_norms


def main():
    """Сохраняет каждую смесь для парного сравнения на той же маске."""
    p = argparse.ArgumentParser()
    p.add_argument("--components", required=True)
    p.add_argument("--other", required=True)
    p.add_argument("--val-seed", type=int, default=777)
    p.add_argument("--out", default="mixed_trees777")
    args = p.parse_args()
    obs, grid, _ = load_all()
    held = obs.loc[make_mask(obs, seed=args.val_seed)]
    base = pd.read_parquet(args.components)
    directory = ARTIFACTS_DIR / "research" / args.out
    directory.mkdir(parents=True, exist_ok=True)
    other = get_predictions(obs, grid, held, args.other, directory)
    merged = base.merge(other[["pid", "date", "prior"]], on=["pid", "date"], suffixes=("", "_other"))
    norms = published_norms(held)
    report = {"args": vars(args), "results": []}
    for weight in (.25, .5, 1.):
        prior = .6 * ((1 - weight) * merged.prior + weight * merged.prior_other) + .4 * merged.neural
        source = merged.assign(prior=prior)
        calibrated, _ = calibrate(source, grid, norms, "robust", .04)
        calibrated.to_parquet(directory / f"mix_{weight}.parquet")
        result = {"weight_other_tree": weight, "raw": metric(source, "prior"), "calibrated": metric(calibrated, "pred")}
        report["results"].append(result)
        print(json.dumps(result), flush=True)
    (directory / "result.json").write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
