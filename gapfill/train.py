"""Обучение LightGBM на синтетических пропусках и валидация, имитирующая контрольные точки test.

Схема: валидационная маска (15 % известных точек train + test) играет роль контрольных точек, остальные
точки — контекст. Обучающие примеры создаются n_masks дополнительными масками 15 % контекста. Метрика
отчёта — RMSE, взвешенный под состав test (85 % новых полигонов, 30 % 2025 года), см. data.testlike_rmse.

Запуск: uv run python -m gapfill.train --n-masks 20 --seed 0 --out lgb_v2 [--sensor-stage] [--residual]
"""

from __future__ import annotations

import argparse
import json
import os

import lightgbm as lgb
import numpy as np
import pandas as pd

from gapfill.config import ARTIFACTS_DIR, RANDOM_SEED, SENSOR_CODE, TARGET
from gapfill.data import gap_score, load_all, rmse, testlike_rmse
from gapfill.dataset import groups_of, train_examples, val_examples

LGB_PARAMS = {
    "objective": "regression", "learning_rate": 0.03, "num_leaves": 63, "min_data_in_leaf": 40,
    "feature_fraction": 0.6, "bagging_fraction": 0.8, "bagging_freq": 1, "lambda_l2": 2.0,
    "verbose": -1, "num_threads": max(1, (os.cpu_count() or 4) - 1), "seed": RANDOM_SEED,
}
SENSOR_PARAMS = {"objective": "multiclass", "num_class": 3, "learning_rate": 0.05, "num_leaves": 31,
                 "min_data_in_leaf": 40, "feature_fraction": 0.7, "verbose": -1,
                 "num_threads": LGB_PARAMS["num_threads"], "seed": RANDOM_SEED}
MAX_ROUNDS = 8000
ES_ROUNDS = 200
ES_SHARE = 0.1
TRIM = 0.3          # |ошибка| выше — выброс target, считаем «усечённый» RMSE без них
N_FOLDS = 5


def holdout_groups(groups: np.ndarray, share: float, seed: int) -> np.ndarray:
    """Булева маска строк, чьи группы отложены (для ранней остановки)."""
    rng = np.random.default_rng(seed)
    uniq = np.unique(groups)
    chosen = set(rng.choice(uniq, size=int(len(uniq) * share), replace=False))
    return np.array([g in chosen for g in groups])


def fit_lgb(X: pd.DataFrame, y: np.ndarray, groups: np.ndarray, params: dict,
            rounds: int | None = None) -> lgb.Booster:
    """Обучение; без rounds — с ранней остановкой на 10 % примеров, отложенных по полигон-годам."""
    if rounds is not None:
        return lgb.train(params, lgb.Dataset(X, y), num_boost_round=rounds)
    es = holdout_groups(groups, ES_SHARE, RANDOM_SEED)
    dtrain = lgb.Dataset(X.loc[~es], y[~es])
    dvalid = lgb.Dataset(X.loc[es], y[es], reference=dtrain)
    return lgb.train(params, dtrain, num_boost_round=MAX_ROUNDS, valid_sets=[dvalid],
                     callbacks=[lgb.early_stopping(ES_ROUNDS, verbose=False)])


def add_sensor_probs(X: pd.DataFrame, meta: pd.DataFrame, X_other: list[pd.DataFrame]) -> tuple:
    """Стадия 1: вероятности сенсора. Для обучающих примеров — out-of-fold по полигон-годам, для остальных
    матриц — модель на всех примерах. Возвращает (X с p_*, [X_other с p_*], число итераций)."""
    groups = groups_of(meta)
    y = meta["sensor"].to_numpy()
    fold = pd.Series(groups).map({g: i % N_FOLDS for i, g in enumerate(np.unique(groups))}).to_numpy()
    oof = np.zeros((len(X), 3))
    rounds = []
    for f in range(N_FOLDS):
        tr, va = fold != f, fold == f
        dtrain = lgb.Dataset(X.loc[tr], y[tr])
        booster = lgb.train(SENSOR_PARAMS, dtrain, num_boost_round=3000,
                            valid_sets=[lgb.Dataset(X.loc[va], y[va], reference=dtrain)],
                            callbacks=[lgb.early_stopping(100, verbose=False)])
        oof[va] = booster.predict(X.loc[va], num_iteration=booster.best_iteration)
        rounds.append(booster.best_iteration)
    full = lgb.train(SENSOR_PARAMS, lgb.Dataset(X, y), num_boost_round=int(np.mean(rounds) * 1.1))
    cols = [f"p_{s}" for s in SENSOR_CODE]
    others = [Xo.assign(**dict(zip(cols, full.predict(Xo).T))) for Xo in X_other]
    return X.assign(**dict(zip(cols, oof.T))), others, full


def base_prediction(X: pd.DataFrame) -> np.ndarray:
    """Опорное предсказание для остаточного обучения: кривая по всем сенсорам с запасными вариантами."""
    base = X["curve_all_15"].fillna(X["curve_all_6"]).fillna(X["nb_interp_val"]).fillna(X["clim_mean"])
    return base.fillna(0.3).to_numpy()


def baselines(X: pd.DataFrame, meta: pd.DataFrame) -> dict[str, np.ndarray]:
    """Простые предсказания для сравнения (oracle-варианты знают истинный сенсор)."""
    sensor = meta["sensor"].to_numpy()
    ss = np.select([sensor == c for c in SENSOR_CODE.values()],
                   [X[f"ss_{n}_interp"].to_numpy() for n in SENSOR_CODE])
    return {"interp_any": X["nb_interp_val"].to_numpy(), "same_sensor_interp_oracle": ss}


def evaluate(pred: np.ndarray, meta: pd.DataFrame, X: pd.DataFrame) -> dict:
    """RMSE общий, взвешенный под test, усечённый, по сенсорам и стратам; baseline для сравнения."""
    y = meta[TARGET].to_numpy()
    err = pred - y
    kind, is_2025 = meta["poly_kind"].to_numpy(), meta["is_2025"].to_numpy()
    out = {"rmse": rmse(y, pred), "rmse_testlike": testlike_rmse(err, kind, is_2025),
           "rmse_trim": float(np.sqrt(np.mean(err[np.abs(err) <= TRIM] ** 2))), "n": int(len(y))}
    out["gap_score_testlike"] = gap_score(out["rmse_testlike"])
    for name, code in SENSOR_CODE.items():
        out[f"rmse_{name}"] = rmse(y[meta["sensor"].to_numpy() == code], pred[meta["sensor"].to_numpy() == code])
    for k, y25 in [("old", False), ("old", True), ("new_hist", False), ("new_hist", True), ("new_2025only", True)]:
        sel = (kind == k) & (is_2025 == y25)
        out[f"rmse_{k}_{'2025' if y25 else 'hist'}"] = rmse(y[sel], pred[sel]) if sel.any() else None
    for name, p in baselines(X, meta).items():
        ok = ~np.isnan(p)
        out[f"baseline_{name}"] = rmse(y[ok], p[ok])
    out["rule_sensor_acc"] = float((X["rule_sensor"].to_numpy() == meta["sensor"].to_numpy()).mean())
    if "p_s2" in X:
        p_argmax = X[[f"p_{s}" for s in SENSOR_CODE]].to_numpy().argmax(1)
        out["stage1_sensor_acc"] = float((p_argmax == meta["sensor"].to_numpy()).mean())
    return out


def importance(model: lgb.Booster, top: int = 40) -> dict[str, float]:
    imp = pd.Series(model.feature_importance("gain"), index=model.feature_name())
    imp = imp / imp.sum()
    return {k: round(float(v), 4) for k, v in imp.sort_values(ascending=False).head(top).items()}


def run(args: argparse.Namespace) -> dict:
    """Полный цикл валидации по аргументам командной строки; артефакты в artifacts/<out>."""
    obs, grid, _ = load_all()
    X, meta = train_examples(obs, grid, args.val_seed, args.n_masks, args.seed)
    X_val, meta_val = val_examples(obs, grid, args.val_seed)
    print(f"обучение {X.shape}, валидация {X_val.shape}", flush=True)
    if args.sensor_stage:
        X, (X_val,), _ = add_sensor_probs(X, meta, [X_val])
    params = LGB_PARAMS | {"learning_rate": args.lr, "num_leaves": args.leaves, "min_data_in_leaf": args.min_leaf,
                           "feature_fraction": args.ff}
    y = np.clip(meta[TARGET].to_numpy(), args.clip[0], args.clip[1])
    base, base_val = (base_prediction(X), base_prediction(X_val)) if args.residual else (0.0, 0.0)
    model = fit_lgb(X, y - base, groups_of(meta), params)
    pred = model.predict(X_val, num_iteration=model.best_iteration) + base_val
    result = evaluate(pred, meta_val, X_val) | {"best_iteration": int(model.best_iteration),
                                                "n_train": int(len(X)), "n_features": int(X.shape[1]),
                                                "args": vars(args)}
    out_dir = ARTIFACTS_DIR / args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_model(str(out_dir / "lgb_val.txt"))
    meta_val.assign(pred=pred).to_parquet(out_dir / "val_pred.parquet")
    (out_dir / "result.json").write_text(json.dumps(result | {"importance": importance(model)},
                                                    ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Валидация LightGBM для восстановления primary_ndvi")
    parser.add_argument("--n-masks", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--val-seed", type=int, default=777)
    parser.add_argument("--out", type=str, default="lgb_v2")
    parser.add_argument("--clip", type=float, nargs=2, default=(-3.0, 3.0), help="обрезка target при обучении")
    parser.add_argument("--residual", action="store_true", help="учить остаток к опорной кривой")
    parser.add_argument("--sensor-stage", action="store_true", help="стадия классификации сенсора")
    parser.add_argument("--lr", type=float, default=0.03)
    parser.add_argument("--leaves", type=int, default=63)
    parser.add_argument("--min-leaf", type=int, default=40)
    parser.add_argument("--ff", type=float, default=0.6)
    return parser.parse_args(argv)


if __name__ == "__main__":
    result = run(parse_args())
    print(json.dumps({k: v for k, v in result.items() if k != "args"}, ensure_ascii=False, indent=2))
