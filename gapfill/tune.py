"""Подбор гиперпараметров LightGBM (Optuna) на кэшированных признаках.

Запуск: uv run python -m gapfill.tune --trials 25 --n-masks 20
Критерий — усечённый RMSE на валидационной маске (устойчивее к выбросам target, чем общий RMSE).
"""

from __future__ import annotations

import argparse
import json

import lightgbm as lgb
import numpy as np
import optuna

from gapfill.config import ARTIFACTS_DIR, RANDOM_SEED, TARGET
from gapfill.data import load_all, rmse
from gapfill.dataset import groups_of, train_examples, val_examples
from gapfill.train import ES_ROUNDS, LGB_PARAMS, MAX_ROUNDS, TRIM, holdout_groups

CLIP = (-0.1, 1.0)


def objective_factory(X, y, es, X_val, y_val):
    """Замыкание objective для Optuna: обучение с ранней остановкой, оценка на валидации."""
    dtrain = lgb.Dataset(X.loc[~es], y[~es], free_raw_data=False)
    dvalid = lgb.Dataset(X.loc[es], y[es], reference=dtrain, free_raw_data=False)

    def objective(trial: optuna.Trial) -> float:
        params = LGB_PARAMS | {
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.06, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 15, 255, log=True),
            "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 10, 300, log=True),
            "feature_fraction": trial.suggest_float("feature_fraction", 0.3, 0.9),
            "bagging_fraction": trial.suggest_float("bagging_fraction", 0.5, 1.0),
            "lambda_l2": trial.suggest_float("lambda_l2", 0.01, 30.0, log=True),
            "lambda_l1": trial.suggest_float("lambda_l1", 0.0, 5.0),
            "min_gain_to_split": trial.suggest_float("min_gain_to_split", 0.0, 0.01),
            "max_bin": trial.suggest_categorical("max_bin", [63, 127, 255]),
        }
        model = lgb.train(params, dtrain, num_boost_round=MAX_ROUNDS, valid_sets=[dvalid],
                          callbacks=[lgb.early_stopping(ES_ROUNDS, verbose=False)])
        pred = model.predict(X_val, num_iteration=model.best_iteration)
        err = pred - y_val
        trial.set_user_attr("rmse", rmse(y_val, pred))
        trial.set_user_attr("best_iteration", int(model.best_iteration))
        return float(np.sqrt(np.mean(err[np.abs(err) <= TRIM] ** 2)))

    return objective


def main() -> None:
    parser = argparse.ArgumentParser(description="Optuna для LightGBM")
    parser.add_argument("--trials", type=int, default=25)
    parser.add_argument("--n-masks", type=int, default=20)
    parser.add_argument("--val-seed", type=int, default=777)
    parser.add_argument("--out", type=str, default="tune_lgb")
    args = parser.parse_args()
    obs, grid, _ = load_all()
    X, meta = train_examples(obs, grid, args.val_seed, args.n_masks, 0)
    X_val, meta_val = val_examples(obs, grid, args.val_seed)
    y = np.clip(meta[TARGET].to_numpy(), *CLIP)
    es = holdout_groups(groups_of(meta), 0.1, RANDOM_SEED)
    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=RANDOM_SEED))
    study.optimize(objective_factory(X, y, es, X_val, meta_val[TARGET].to_numpy()), n_trials=args.trials)
    out_dir = ARTIFACTS_DIR / args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    trials = [{"value": t.value, **t.params, **t.user_attrs} for t in study.trials if t.value is not None]
    (out_dir / "trials.json").write_text(json.dumps({"best": study.best_params, "trials": trials},
                                                    ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"best_trim": study.best_value, "best": study.best_params,
                      "best_rmse": study.best_trial.user_attrs}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
