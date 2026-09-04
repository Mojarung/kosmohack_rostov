"""Baseline-методы. Работают прямо на таблице признаков из ``features.build_features``.

Числа на train (15 % скрытых наблюдений, EDA):

===========================================  =====
Метод                                        RMSE
===========================================  =====
среднее двух соседей (baseline организаторов) 0.091
линейная интерполяция по времени              0.092
среднее с приведением к шкале S2 + сенсор     0.083
===========================================  =====
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def predict_mean2(feats: pd.DataFrame) -> np.ndarray:
    """Baseline организаторов: среднее двух ближайших соседей, без учёта сенсоров."""
    prev_raw = feats.prev1_val + feats.prev1_off
    next_raw = feats.next1_val + feats.next1_off
    return np.nanmean(np.stack([prev_raw.values, next_raw.values]), axis=0)


def predict_linear(feats: pd.DataFrame) -> np.ndarray:
    """Линейная интерполяция по времени в исходных (несогласованных) шкалах сенсоров."""
    prev_raw = (feats.prev1_val + feats.prev1_off).values
    next_raw = (feats.next1_val + feats.next1_off).values
    w = (feats.next1_dt / (feats.prev1_dt + feats.next1_dt)).values
    both = np.isfinite(prev_raw) & np.isfinite(next_raw)
    return np.where(both, prev_raw * w + next_raw * (1 - w),
                    np.nanmean(np.stack([prev_raw, next_raw]), axis=0))


def predict_sensor_aware(feats: pd.DataFrame) -> np.ndarray:
    """Соседей приводим к шкале S2, интерполируем, возвращаем в шкалу угаданного сенсора."""
    return feats.base_interp.values


def predict_smooth(feats: pd.DataFrame) -> np.ndarray:
    """Робастная сезонная кривая в шкале S2 + смещение угаданного сенсора."""
    out = feats.base_smooth.values.copy()
    fallback = feats.base_interp.values
    bad = ~np.isfinite(out)
    out[bad] = fallback[bad]
    return out


def predict_blend(feats: pd.DataFrame, w_smooth: float = 0.5) -> np.ndarray:
    """Смесь интерполяции и сглаживания: интерполяция точнее вблизи, кривая устойчивее."""
    a = predict_sensor_aware(feats)
    b = predict_smooth(feats)
    out = w_smooth * b + (1 - w_smooth) * a
    return np.where(np.isfinite(out), out, np.where(np.isfinite(a), a, b))


BASELINES = {
    "mean2": predict_mean2,
    "linear": predict_linear,
    "sensor_aware": predict_sensor_aware,
    "smooth": predict_smooth,
    "blend": predict_blend,
}
