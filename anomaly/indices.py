"""Второй индекс для интерпретации: NDWI (влажность растительности) внутри эпизода против нормы полигона.

NDWI сравнивается отдельно по сенсорам (S2 и Landsat читают его по-разному), норма — те же даты
(±8 дней по дню года) других лет того же полигона и сенсора.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from anomaly.config import NORM_DOY_WINDOW

NDWI_COLS = {0: "s2_ndwi", 1: "landsat_ndwi"}
MIN_POINTS = 2


def ndwi_anomaly(series: pd.DataFrame, year: int, start: str, end: str) -> dict:
    """Средняя аномалия NDWI в эпизоде (по сенсорам), число точек; пусто, если данных мало."""
    s, e = pd.Timestamp(start), pd.Timestamp(end)
    inside = series.loc[(series["date"] >= s) & (series["date"] <= e) & ~series["artifact"]]
    anomalies, n_total = [], 0
    for code, col in NDWI_COLS.items():
        pts = inside.loc[(inside["sensor"] == code) & inside[col].notna()]
        ref = series.loc[(series["sensor"] == code) & (series["year"] != year) & series[col].notna()]
        if len(pts) < MIN_POINTS or ref.empty:
            continue
        doy_ref, val_ref = ref["doy"].to_numpy(), ref[col].to_numpy(dtype=float)
        for doy, v in zip(pts["doy"].to_numpy(), pts[col].to_numpy(dtype=float)):
            near = np.abs(doy_ref - doy) <= NORM_DOY_WINDOW
            if near.sum() >= 3:
                anomalies.append(v - val_ref[near].mean())
                n_total += 1
    if n_total < MIN_POINTS:
        return {}
    return {"ndwi_anomaly": float(np.mean(anomalies)), "ndwi_n": int(n_total)}


def ndwi_note(facts: dict) -> str | None:
    """Формулировка для объяснения: заметное отклонение NDWI (порог 0.05)."""
    a = facts.get("ndwi_anomaly")
    if a is None or abs(a) < 0.05:
        return None
    direction = "ниже" if a < 0 else "выше"
    hint = "растительность суше обычного (водный стресс)" if a < 0 else "влажность выше обычного"
    return f"индекс влажности NDWI {direction} нормы на {abs(a):.2f} ({facts['ndwi_n']} набл.) — {hint}"
