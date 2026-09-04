"""CatBoost (GPU) на тех же кэшированных признаках — вторая табличная модель для смеси.

Запуск: uv run python -m gapfill.train_cb --n-masks 20 --out cb_v1 [--final --rounds N]
Без --final: валидация на маске val_seed, предсказания в artifacts/<out>/val_pred.parquet.
С --final: обучение на всех известных точках и предсказание контрольных точек → gap_pred.parquet.
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor, Pool

from gapfill.config import ARTIFACTS_DIR, RANDOM_SEED, TARGET
from gapfill.data import load_all
from gapfill.dataset import groups_of, train_examples, val_examples
from gapfill.predict import gap_features
from gapfill.train import ES_ROUNDS, ES_SHARE, evaluate, holdout_groups

CB_PARAMS = {"loss_function": "RMSE", "learning_rate": 0.05, "depth": 8, "l2_leaf_reg": 3.0,
             "bootstrap_type": "Bernoulli", "subsample": 0.8, "random_seed": RANDOM_SEED,
             "task_type": "GPU", "verbose": 0, "border_count": 254}
CLIP = (-0.1, 1.0)


def fit_cb(X: pd.DataFrame, y: np.ndarray, groups: np.ndarray | None, rounds: int, seed: int) -> CatBoostRegressor:
    """Обучение с ранней остановкой на отложенных полигон-годах (groups) или фиксированным числом итераций."""
    model = CatBoostRegressor(**(CB_PARAMS | {"iterations": rounds, "random_seed": seed}))
    if groups is None:
        model.fit(Pool(X, y))
        return model
    es = holdout_groups(groups, ES_SHARE, RANDOM_SEED)
    model.fit(Pool(X.loc[~es], y[~es]), eval_set=Pool(X.loc[es], y[es]), early_stopping_rounds=ES_ROUNDS,
              use_best_model=True)
    return model


def main() -> None:
    parser = argparse.ArgumentParser(description="CatBoost для восстановления primary_ndvi")
    parser.add_argument("--n-masks", type=int, default=20)
    parser.add_argument("--val-seed", type=int, default=777)
    parser.add_argument("--rounds", type=int, default=20000)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0])
    parser.add_argument("--final", action="store_true")
    parser.add_argument("--out", type=str, default="cb_v1")
    args = parser.parse_args()
    obs, grid, gaps = load_all()
    out_dir = ARTIFACTS_DIR / args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.final:
        X, meta = train_examples(obs, grid, None, args.n_masks, 0)
        X_pred = gap_features(obs, grid, gaps)
    else:
        X, meta = train_examples(obs, grid, args.val_seed, args.n_masks, 0)
        X_pred, meta_val = val_examples(obs, grid, args.val_seed)
    y = np.clip(meta[TARGET].to_numpy(), *CLIP)
    groups = None if args.final else groups_of(meta)
    preds, info = [], {}
    for s in args.seeds:
        model = fit_cb(X, y, groups, args.rounds, s)
        preds.append(model.predict(X_pred))
        info[f"seed{s}_best_iteration"] = int(model.get_best_iteration() or args.rounds)
        model.save_model(str(out_dir / f"cb_seed{s}.cbm"))
    pred = np.mean(preds, axis=0)
    if args.final:
        gaps[["pid", "date"]].assign(pred=pred).to_parquet(out_dir / "gap_pred.parquet")
        result = info | {"n_train": int(len(X))}
    else:
        result = evaluate(pred, meta_val, X_pred) | info
        meta_val.assign(pred=pred).to_parquet(out_dir / "val_pred.parquet")
    (out_dir / "result.json").write_text(json.dumps(result | {"args": vars(args)}, ensure_ascii=False, indent=2),
                                         encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k != "args"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
