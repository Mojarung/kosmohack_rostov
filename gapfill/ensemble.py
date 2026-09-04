"""Смешивание валидационных предсказаний нескольких моделей (одна и та же валидационная маска).

Запуск: uv run python -m gapfill.ensemble lgb_v3_dew nn_v1 [--weights 0.7 0.3]
Без --weights веса подбираются по МНК с ограничением суммы 1 (неотрицательные).
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd
from scipy.optimize import nnls

from gapfill.config import ARTIFACTS_DIR, TARGET
from gapfill.data import gap_score, rmse, testlike_rmse

KEY = ["pid", "date"]


def load_preds(names: list[str]) -> pd.DataFrame:
    """Таблица валидационных точек с колонками pred_<имя> по каждой модели (объединение по полигон+дата)."""
    base = None
    for name in names:
        df = pd.read_parquet(ARTIFACTS_DIR / name / "val_pred.parquet")
        df = df.assign(date=pd.to_datetime(df["date"]))
        cols = KEY + [TARGET, "sensor", "poly_kind", "is_2025", "pred"]
        df = df[cols].rename(columns={"pred": f"pred_{name}"})
        base = df if base is None else base.merge(df.drop(columns=[TARGET, "sensor", "poly_kind", "is_2025"]), on=KEY)
    return base


def fit_weights(P: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Неотрицательные веса с суммой 1 (НМНК с нормировкой)."""
    w, _ = nnls(P, y)
    return w / w.sum() if w.sum() > 0 else np.full(P.shape[1], 1 / P.shape[1])


def report(df: pd.DataFrame, pred: np.ndarray, label: str) -> dict:
    y = df[TARGET].to_numpy()
    err = pred - y
    out = {"model": label, "rmse": rmse(y, pred),
           "rmse_testlike": testlike_rmse(err, df["poly_kind"].to_numpy(), df["is_2025"].to_numpy()),
           "rmse_trim": float(np.sqrt(np.mean(err[np.abs(err) <= 0.3] ** 2)))}
    out["gap_score_testlike"] = gap_score(out["rmse_testlike"])
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Смешивание валидационных предсказаний")
    parser.add_argument("names", nargs="+")
    parser.add_argument("--weights", type=float, nargs="*")
    args = parser.parse_args()
    df = load_preds(args.names)
    P = df[[f"pred_{n}" for n in args.names]].to_numpy()
    y = df[TARGET].to_numpy()
    rows = [report(df, P[:, i], n) for i, n in enumerate(args.names)]
    w = np.array(args.weights) if args.weights else fit_weights(P, y)
    rows.append(report(df, P @ w, "blend " + " ".join(f"{n}:{wi:.2f}" for n, wi in zip(args.names, w))))
    rows.append(report(df, P.mean(1), "mean"))
    print(json.dumps(rows, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
