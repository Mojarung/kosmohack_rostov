"""Итоговое обучение на всех известных точках и предсказание контрольных точек test → submission.csv.

Запуск: uv run python -m gapfill.predict --n-masks 30 --rounds 6000 --clip -0.1 1.0 --seeds 0 1 2
Модель обучается на train + известных точках test (n_masks масок), контекст для контрольных точек —
все известные точки test и train. Несколько seeds LightGBM усредняются.
"""

from __future__ import annotations

import argparse
import json

import lightgbm as lgb
import numpy as np
import pandas as pd

from gapfill.config import ARTIFACTS_DIR, SUBMISSION_PATH, TARGET
from gapfill.data import load_all
from gapfill.dataset import train_examples
from gapfill.features import build_features, sensor_prior_features
from gapfill.train import LGB_PARAMS

N_GAPS = 3112
PRED_RANGE = (-0.2, 1.0)


def gap_features(obs: pd.DataFrame, grid: pd.DataFrame, gaps: pd.DataFrame) -> pd.DataFrame:
    """Признаки контрольных точек по контексту всех известных точек."""
    return sensor_prior_features(build_features(gaps, obs, grid)).reset_index(drop=True)


def fit_seeds(X: pd.DataFrame, y: np.ndarray, rounds: int, seeds: list[int], params: dict) -> list[lgb.Booster]:
    """Несколько моделей с разными seeds (бэггинг по случайности LightGBM)."""
    return [lgb.train(params | {"seed": s, "bagging_seed": s, "feature_fraction_seed": s}, lgb.Dataset(X, y),
                      num_boost_round=rounds) for s in seeds]


def write_submission(gaps: pd.DataFrame, pred: np.ndarray, path=SUBMISSION_PATH) -> pd.DataFrame:
    """submission.csv: anon_polygon_id, date, primary_ndvi_pred; проверка формата перед записью."""
    sub = pd.DataFrame({"anon_polygon_id": gaps["pid"].to_numpy(),
                        "date": gaps["date"].dt.strftime("%Y-%m-%d").to_numpy(),
                        "primary_ndvi_pred": np.clip(pred, *PRED_RANGE)})
    assert len(sub) == N_GAPS, f"ожидалось {N_GAPS} строк, получено {len(sub)}"
    assert not sub.duplicated(["anon_polygon_id", "date"]).any(), "дубликаты полигон+дата"
    assert sub["primary_ndvi_pred"].notna().all(), "есть NaN в предсказаниях"
    sub.to_csv(path, index=False, encoding="utf-8")
    return sub


def main() -> None:
    parser = argparse.ArgumentParser(description="Итоговое предсказание контрольных точек")
    parser.add_argument("--n-masks", type=int, default=30)
    parser.add_argument("--rounds", type=int, default=6000)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--clip", type=float, nargs=2, default=(-0.1, 1.0))
    parser.add_argument("--lr", type=float, default=LGB_PARAMS["learning_rate"])
    parser.add_argument("--leaves", type=int, default=LGB_PARAMS["num_leaves"])
    parser.add_argument("--min-leaf", type=int, default=LGB_PARAMS["min_data_in_leaf"])
    parser.add_argument("--ff", type=float, default=LGB_PARAMS["feature_fraction"])
    parser.add_argument("--out", type=str, default="final_lgb")
    args = parser.parse_args()
    obs, grid, gaps = load_all()
    X, meta = train_examples(obs, grid, None, args.n_masks, seed=0)
    X_gap = gap_features(obs, grid, gaps)
    print(f"обучение {X.shape}, контрольных точек {X_gap.shape}", flush=True)
    params = LGB_PARAMS | {"learning_rate": args.lr, "num_leaves": args.leaves, "min_data_in_leaf": args.min_leaf,
                           "feature_fraction": args.ff}
    y = np.clip(meta[TARGET].to_numpy(), args.clip[0], args.clip[1])
    models = fit_seeds(X, y, args.rounds, args.seeds, params)
    preds = np.column_stack([m.predict(X_gap) for m in models])
    out_dir = ARTIFACTS_DIR / args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    for m, s in zip(models, args.seeds):
        m.save_model(str(out_dir / f"lgb_seed{s}.txt"))
    gaps[["pid", "date"]].assign(**{f"pred_seed{s}": preds[:, i] for i, s in enumerate(args.seeds)}).to_parquet(
        out_dir / "gap_pred.parquet")
    sub = write_submission(gaps, preds.mean(1))
    (out_dir / "info.json").write_text(json.dumps({"args": vars(args), "n_train": int(len(X)),
                                                   "pred_mean": float(sub["primary_ndvi_pred"].mean())},
                                                  ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"submission.csv записан: {len(sub)} строк, среднее {sub['primary_ndvi_pred'].mean():.4f}")


if __name__ == "__main__":
    main()
