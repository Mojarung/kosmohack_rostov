"""Смешивание предсказаний контрольных точек нескольких итоговых моделей → submission.csv.

Запуск: uv run python -m gapfill.make_submission final_lgb:0.6 final_nn:0.4
Каждый аргумент — <папка в artifacts>:<вес>; в папке ищутся gap_pred*.parquet (усредняются по seeds).
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

from gapfill.config import ARTIFACTS_DIR
from gapfill.data import load_all
from gapfill.predict import write_submission

KEY = ["pid", "date"]


def load_gap_preds(name: str) -> pd.Series:
    """Среднее предсказание по всем gap_pred*.parquet в папке модели, индекс — (pid, date)."""
    files = sorted((ARTIFACTS_DIR / name).glob("gap_pred*.parquet"))
    if not files:
        raise FileNotFoundError(f"нет gap_pred*.parquet в artifacts/{name}")
    frames = []
    for f in files:
        df = pd.read_parquet(f)
        df = df.assign(date=pd.to_datetime(df["date"]))
        cols = [c for c in df.columns if c.startswith("pred")]
        frames.append(df.set_index(KEY)[cols].mean(axis=1).rename(f.stem))
    return pd.concat(frames, axis=1).mean(axis=1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Смесь итоговых предсказаний → submission.csv")
    parser.add_argument("parts", nargs="+", help="<папка>:<вес>")
    args = parser.parse_args()
    names, weights = zip(*[(p.split(":")[0], float(p.split(":")[1])) for p in args.parts])
    weights = np.array(weights) / np.sum(weights)
    _, _, gaps = load_all()
    key = pd.MultiIndex.from_arrays([gaps["pid"], gaps["date"]])
    preds = np.column_stack([load_gap_preds(n).reindex(key).to_numpy() for n in names])
    assert not np.isnan(preds).any(), "не для всех контрольных точек есть предсказания"
    blend = preds @ weights
    sub = write_submission(gaps, blend)
    corr = np.corrcoef(preds.T).round(4).tolist() if len(names) > 1 else None
    print(json.dumps({"parts": dict(zip(names, weights.round(3).tolist())), "n": len(sub),
                      "mean": float(sub["primary_ndvi_pred"].mean()), "corr_between_models": corr},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
