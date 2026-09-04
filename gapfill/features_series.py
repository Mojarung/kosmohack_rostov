"""Признаки по временному ряду одного полигона: соседи, одно-сенсорные соседи, гладкие кривые, окно MODIS.

Контекст полигона — известные точки, отсортированные по дате (контрольные/замаскированные точки
из контекста исключены). Все функции принимают контекст и дни целевых точек, возвращают словарь
{имя признака: массив длиной n_targets}.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from gapfill.config import BANDWIDTHS, MAX_NEIGHBOR_DAYS, MODIS_WINDOW, N_NEIGHBORS, SENSOR_CODE, SENSOR_OFFSET
from gapfill.smooth import local_linear


@dataclass(frozen=True)
class PolygonContext:
    """Массивы известных наблюдений одного полигона, отсортированные по дню."""
    day: np.ndarray      # day_num, int
    year: np.ndarray
    doy: np.ndarray
    y: np.ndarray        # primary_ndvi
    sensor: np.ndarray   # код сенсора 0/1/2
    h: np.ndarray        # primary_ndvi, приведённый к шкале S2 глобальными смещениями
    evi: np.ndarray      # EVI сенсора-источника (NaN у MODIS-точек без EVI)
    ndwi: np.ndarray     # NDWI сенсора-источника (NaN у MODIS)


def harmonize(y: np.ndarray, sensor: np.ndarray) -> np.ndarray:
    """Приводит значения к шкале Sentinel-2 глобальными смещениями."""
    offsets = np.array([SENSOR_OFFSET[s] for s in SENSOR_CODE], dtype=float)
    return y - offsets[sensor]


def _take(arr: np.ndarray, idx: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Безопасная выборка по индексам: там, где valid=False, возвращает NaN."""
    safe = np.clip(idx, 0, max(len(arr) - 1, 0))
    out = arr[safe].astype(float) if len(arr) else np.full(len(idx), np.nan)
    return np.where(valid, out, np.nan)


def _neighbor_block(ctx: PolygonContext, t: np.ndarray, prefix: str, with_indices: bool) -> dict:
    """k ближайших точек контекста слева и справа от каждой целевой даты."""
    out = {}
    pos = np.searchsorted(ctx.day, t, side="left")
    for k in range(1, N_NEIGHBORS + 1):
        for side, idx in (("p", pos - k), ("n", pos + k - 1)):
            valid = (idx >= 0) & (idx < len(ctx.day))
            days = np.abs(_take(ctx.day, idx, valid) - t)
            valid = valid & (days <= MAX_NEIGHBOR_DAYS)
            name = f"{prefix}_{side}{k}"
            out[f"{name}_days"] = np.where(valid, days, np.nan)
            out[f"{name}_val"] = _take(ctx.y, idx, valid)
            out[f"{name}_h"] = _take(ctx.h, idx, valid)
            if with_indices:
                out[f"{name}_sensor"] = _take(ctx.sensor, idx, valid)
                out[f"{name}_evi"] = _take(ctx.evi, idx, valid)
                out[f"{name}_ndwi"] = _take(ctx.ndwi, idx, valid)
    return out


def neighbor_features(ctx: PolygonContext, t: np.ndarray) -> dict:
    """Соседи любого сенсора плюс интерполяция по времени между ближайшими."""
    out = _neighbor_block(ctx, t, "nb", with_indices=True)
    dp, dn = out["nb_p1_days"], out["nb_n1_days"]
    w = dp / (dp + dn)
    out["nb_interp_h"] = out["nb_p1_h"] + (out["nb_n1_h"] - out["nb_p1_h"]) * w
    out["nb_interp_val"] = out["nb_p1_val"] + (out["nb_n1_val"] - out["nb_p1_val"]) * w
    out["nb_min_days"] = np.fmin(dp, dn)
    return out


def same_sensor_features(ctx: PolygonContext, t: np.ndarray) -> dict:
    """Для каждого сенсора отдельно: ближайшие точки этого сенсора и интерполяция между ними."""
    out = {}
    for name, code in SENSOR_CODE.items():
        sel = ctx.sensor == code
        sub = PolygonContext(*(getattr(ctx, f)[sel] for f in ctx.__dataclass_fields__))
        block = _neighbor_block(sub, t, f"ss_{name}", with_indices=name != "modis")
        keep = {k: v for k, v in block.items() if "_p1_" in k or "_n1_" in k or "_p2_" in k or "_n2_" in k}
        dp, dn = keep[f"ss_{name}_p1_days"], keep[f"ss_{name}_n1_days"]
        vp, vn = keep[f"ss_{name}_p1_val"], keep[f"ss_{name}_n1_val"]
        interp = vp + (vn - vp) * dp / (dp + dn)
        keep[f"ss_{name}_interp"] = np.where(np.isnan(interp), np.where(np.isnan(vp), vn, vp), interp)
        keep[f"ss_{name}_n30"] = _count_in_window(sub.day, t, 30)
        out |= keep
    return out


def _count_in_window(days: np.ndarray, t: np.ndarray, half: int) -> np.ndarray:
    """Число точек контекста в окне ±half дней от каждой целевой даты."""
    return (np.searchsorted(days, t + half, "right") - np.searchsorted(days, t - half, "left")).astype(float)


def curve_features(ctx: PolygonContext, t: np.ndarray) -> dict:
    """Гладкие кривые: S2+Landsat (гармонизированные), только MODIS, все сенсоры."""
    out = {}
    groups = {"sl": ctx.sensor != SENSOR_CODE["modis"], "md": ctx.sensor == SENSOR_CODE["modis"],
              "all": np.ones(len(ctx.day), dtype=bool)}
    for gname, sel in groups.items():
        for bw in BANDWIDTHS:
            if gname == "md" and bw < 10:
                continue
            a, b, s = local_linear(ctx.day[sel], ctx.h[sel], t, bw)
            out[f"curve_{gname}_{int(bw)}"] = a
            out[f"slope_{gname}_{int(bw)}"] = b
            out[f"wsum_{gname}_{int(bw)}"] = s
    out["curve_sl_minus_md"] = out["curve_sl_15"] - out["curve_md_15"]
    return out


def modis_window_features(ctx: PolygonContext, t: np.ndarray) -> dict:
    """Среднее/максимум гармонизированных S2+Landsat в окне композита MODIS [t, t+15] и в зеркальном [t−15, t]."""
    sel = ctx.sensor != SENSOR_CODE["modis"]
    day, h = ctx.day[sel], ctx.h[sel]
    out = {}
    for name, lo, hi in (("fwd", 0, MODIS_WINDOW), ("bwd", -MODIS_WINDOW, 0)):
        i0 = np.searchsorted(day, t + lo, "left")
        i1 = np.searchsorted(day, t + hi, "right")
        csum = np.concatenate([[0.0], np.cumsum(h)])
        n = (i1 - i0).astype(float)
        out[f"mw_{name}_mean"] = np.where(n > 0, (csum[i1] - csum[i0]) / np.maximum(n, 1), np.nan)
        out[f"mw_{name}_n"] = n
        out[f"mw_{name}_max"] = np.array([h[a:b].max() if b > a else np.nan for a, b in zip(i0, i1)])
    return out


def density_features(ctx: PolygonContext, t: np.ndarray) -> dict:
    """Плотность контекста вокруг целевой даты."""
    return {"dens_15": _count_in_window(ctx.day, t, 15), "dens_30": _count_in_window(ctx.day, t, 30),
            "dens_60": _count_in_window(ctx.day, t, 60)}


def series_features(ctx: PolygonContext, t: np.ndarray) -> dict:
    """Все признаки по ряду полигона для целевых дней t."""
    return (neighbor_features(ctx, t) | same_sensor_features(ctx, t) | curve_features(ctx, t)
            | modis_window_features(ctx, t) | density_features(ctx, t))
