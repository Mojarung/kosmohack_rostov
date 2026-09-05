"""Оценка неодинаковой неопределённости пропусков для взвешенной калибровки."""

import argparse
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
from sklearn.mixture import GaussianMixture

from gapfill.config import ARTIFACTS_DIR, TARGET
from gapfill.data import load_all
from gapfill.dataset import groups_of
from gapfill.research_data import examples
from gapfill.train import holdout_groups


def main():
    """Обучает оценку ошибки только на внутренних группах, которых не видел регрессор."""
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--n-masks", type=int, default=30)
    p.add_argument("--val-seed", type=int, default=777)
    p.add_argument("--share", type=float, default=.15)
    p.add_argument("--analog", action="store_true")
    p.add_argument("--kriging", action="store_true")
    p.add_argument("--out", default="uncertainty777")
    args = p.parse_args()
    obs, grid, _ = load_all()
    x, meta, _, _ = examples(obs, grid, args.val_seed, args.n_masks, share=args.share,
                             analog=args.analog or args.kriging, kriging=args.kriging)
    model = lgb.Booster(model_file=args.model)
    selected = holdout_groups(groups_of(meta), .1, 42)
    error = meta.loc[selected, TARGET].to_numpy() - model.predict(x.loc[selected, model.feature_name()], num_threads=4)
    # Ограничение применяется только к обучающей оценке масштаба шума, не к конкурсной метрике.
    label = np.clip(np.abs(error), .003, .3)
    out = ARTIFACTS_DIR / "research" / args.out
    out.mkdir(parents=True, exist_ok=True)
    params = {"objective": "regression", "learning_rate": .025, "num_leaves": 15,
              "min_data_in_leaf": 200, "lambda_l2": 20., "feature_fraction": .7,
              "num_threads": 4, "verbosity": -1, "seed": 42, "force_col_wise": True}
    booster = lgb.train(params, lgb.Dataset(x.loc[selected], label), num_boost_round=700)
    booster.save_model(str(out / "model.txt"))
    scale = np.clip(.5 * booster.predict(x.loc[selected], num_threads=4) + .5 * .035, .015, .15)
    normalized = np.clip(error / scale, -15., 15.)[:, None]
    mixture = GaussianMixture(3, covariance_type="diag", random_state=42, reg_covar=.02).fit(normalized)
    density = {"weights": mixture.weights_.tolist(), "means": mixture.means_.ravel().tolist(),
               "stds": np.sqrt(mixture.covariances_).ravel().tolist()}
    (out / "density.json").write_text(json.dumps(density, indent=2), encoding="utf-8")
    (out / "result.json").write_text(json.dumps({"args": vars(args), "n": int(selected.sum()),
                                                "mean_abs_error": float(label.mean())}, indent=2), encoding="utf-8")
    print(f"Модель неопределённости: {selected.sum()} внутренних контрольных примеров", flush=True)


if __name__ == "__main__":
    main()
