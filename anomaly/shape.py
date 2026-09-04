"""Нетипичная форма сезона: Isolation Forest по отклонениям фенометрик от нормы полигона.

Дополняет детектор эпизодов случаями, где сезон странный, но Z не обязательно опускается ниже −1 надолго:
сдвиги пика и старта, укороченный сезон, малый интеграл, много артефактов. Работает по seasons.csv.

Запуск: uv run python -m anomaly.shape [--share 0.05]
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

from anomaly.config import REPORT_DIR
from gapfill.config import RANDOM_SEED

FEATURES = ["dev_peak_ratio", "dev_peak_shift_days", "dev_sos_shift_days", "dev_integral_ratio", "dev_decline_shift_days",
            "mean_z", "share_days_depressed", "ph_amplitude"]
LABELS = {"dev_peak_ratio": "пик {:.0%} нормы", "dev_peak_shift_days": "пик сдвинут на {:+.0f} дн.",
          "dev_sos_shift_days": "старт роста сдвинут на {:+.0f} дн.", "dev_integral_ratio": "суммарная вегетация {:.0%} нормы",
          "dev_decline_shift_days": "спад сдвинут на {:+.0f} дн.", "mean_z": "средний Z {:+.1f}",
          "share_days_depressed": "дней ниже нормы {:.0%}", "ph_amplitude": "амплитуда {:.2f}"}


def prepare(seasons: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    """Матрица признаков (NaN → медиана) и маска валидных сезонов с собственной историей."""
    valid = seasons["ph_valid"].fillna(False).astype(bool) & seasons["norm_source"].str.contains("истор")
    X = seasons.loc[valid, FEATURES].astype(float)
    X = X.fillna(X.median())
    return X, valid.to_numpy()


def shape_scores(seasons: pd.DataFrame, share: float) -> pd.DataFrame:
    """Оценка нетипичности сезона (выше — страннее) и флаг для верхней доли share."""
    X, valid = prepare(seasons)
    model = IsolationForest(n_estimators=400, contamination=share, random_state=RANDOM_SEED)
    model.fit(X)
    score = -model.score_samples(X)
    flag = model.predict(X) == -1
    zs = (X - X.median()) / X.std(ddof=0).replace(0, 1)
    out = seasons.loc[valid, ["pid", "year"]].copy()
    out["shape_score"] = score
    out["shape_flag"] = flag
    # направление: сезон слабее нормы (отрицательная аномалия формы) или сильнее
    negative = (X["dev_integral_ratio"] < 1.0) | (X["mean_z"] < 0)
    out["shape_direction"] = np.where(negative, "слабее нормы", "сильнее нормы")
    out["shape_reasons"] = [_describe(row, zrow) for (_, row), (_, zrow) in zip(X.iterrows(), zs.iterrows())]
    return out


def _describe(row: pd.Series, zrow: pd.Series, top: int = 3) -> str:
    """Три самых отклонившихся признака словами."""
    order = zrow.abs().sort_values(ascending=False).index[:top]
    return "; ".join(LABELS[c].format(row[c]) for c in order)


def main() -> None:
    parser = argparse.ArgumentParser(description="Нетипичная форма сезона")
    parser.add_argument("--share", type=float, default=0.05)
    args = parser.parse_args()
    seasons = pd.read_csv(REPORT_DIR / "seasons.csv")
    episodes = pd.read_csv(REPORT_DIR / "episodes.csv")
    scores = shape_scores(seasons, args.share)
    has_ep = set(zip(episodes["pid"], episodes["year"]))
    scores["has_episode"] = [(p, y) in has_ep for p, y in zip(scores["pid"], scores["year"])]
    scores.to_csv(REPORT_DIR / "seasons_shape.csv", index=False, encoding="utf-8")
    flagged = scores[scores["shape_flag"]]
    summary = {"n_scored": int(len(scores)), "n_flagged": int(len(flagged)),
               "flagged_without_episode": int((~flagged["has_episode"]).sum()),
               "flagged_negative": int((flagged["shape_direction"] == "слабее нормы").sum()),
               "flagged_by_year": flagged.groupby("year").size().to_dict()}
    print(json.dumps(summary, ensure_ascii=False, indent=2, default=str))
    for _, r in flagged[~flagged["has_episode"]].head(10).iterrows():
        print(f"  {r.pid} {r.year} ({r.shape_direction}): {r.shape_reasons}")


if __name__ == "__main__":
    main()
