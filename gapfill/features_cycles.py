"""Признаки циклов съёмки: по ним модель угадывает, какой сенсор сформировал скрытую точку.

S2 ходит по 5-дневному циклу (остаток day_num % 5 фиксирован для орбиты), Landsat 8/9 — по 8-дневному
(каждый спутник по 16-дневному), MODIS MOD13Q1 живёт на фиксированной сетке дней года с шагом 16.
"""

from __future__ import annotations

import numpy as np

from gapfill.config import MODIS_GRID_DOY, SENSOR_CODE
from gapfill.features_series import PolygonContext

WINDOW = {"s2": 30, "landsat": 40}
PERIODS = {"s2": (5,), "landsat": (8, 16)}


def _nearest_same_residue(day_sub: np.ndarray, t: np.ndarray, period: int) -> tuple[np.ndarray, np.ndarray]:
    """Дни до ближайшей точки сенсора с тем же остатком по периоду: слева и справа (NaN, если нет)."""
    if len(day_sub) == 0:
        return np.full(len(t), np.nan), np.full(len(t), np.nan)
    d = day_sub[None, :] - t[:, None]
    same = (day_sub % period)[None, :] == (t % period)[:, None]
    prev = np.where(same & (d < 0), -d, np.inf).min(1)
    nxt = np.where(same & (d > 0), d, np.inf).min(1)
    return np.where(np.isinf(prev), np.nan, prev), np.where(np.isinf(nxt), np.nan, nxt)


def _sensor_cycle_block(ctx: PolygonContext, t: np.ndarray, year_t: np.ndarray, name: str) -> dict:
    """Счётчики наблюдений сенсора в том же году и в окне: всего и с тем же остатком по периоду."""
    sel = ctx.sensor == SENSOR_CODE[name]
    day, year = ctx.day[sel], ctx.year[sel]
    out = {}
    same_year = year[None, :] == year_t[:, None]
    in_win = np.abs(day[None, :] - t[:, None]) <= WINDOW[name]
    out[f"cy_{name}_n_year"] = same_year.sum(1).astype(float)
    out[f"cy_{name}_n_win"] = in_win.sum(1).astype(float)
    for p in PERIODS[name]:
        same_res = (day % p)[None, :] == (t % p)[:, None]
        n_year = (same_year & same_res).sum(1).astype(float)
        n_win = (in_win & same_res).sum(1).astype(float)
        out[f"cy_{name}_r{p}_n_year"] = n_year
        out[f"cy_{name}_r{p}_n_win"] = n_win
        out[f"cy_{name}_r{p}_share_year"] = n_year / np.maximum(out[f"cy_{name}_n_year"], 1)
        prev, nxt = _nearest_same_residue(day, t, p)
        out[f"cy_{name}_r{p}_prev"] = prev
        out[f"cy_{name}_r{p}_next"] = nxt
    return out


def cycle_features(ctx: PolygonContext, t: np.ndarray, year_t: np.ndarray, doy_t: np.ndarray) -> dict:
    """Все признаки циклов для целевых точек одного полигона."""
    out = {"res5": (t % 5).astype(float), "res8": (t % 8).astype(float), "res16": (t % 16).astype(float),
           "on_modis_grid": np.isin(doy_t, MODIS_GRID_DOY).astype(float)}
    for name in ("s2", "landsat"):
        out |= _sensor_cycle_block(ctx, t, year_t, name)
    md = ctx.sensor == SENSOR_CODE["modis"]
    day_md, year_md = ctx.day[md], ctx.year[md]
    out["cy_modis_n_year"] = (year_md[None, :] == year_t[:, None]).sum(1).astype(float)
    out["cy_modis_n_win"] = (np.abs(day_md[None, :] - t[:, None]) <= 40).sum(1).astype(float)
    prev, nxt = _nearest_same_residue(day_md, t, 16)
    out["cy_modis_r16_prev"] = prev
    out["cy_modis_r16_next"] = nxt
    return out


def rule_sensor(feats: dict) -> np.ndarray:
    """Правило из EDA: S2, если в году есть S2 с тем же остатком %5; иначе Landsat по %8; иначе MODIS по сетке."""
    return np.select(
        [feats["cy_s2_r5_n_year"] > 0, feats["cy_landsat_r8_n_year"] > 0, feats["on_modis_grid"] > 0],
        [SENSOR_CODE["s2"], SENSOR_CODE["landsat"], SENSOR_CODE["modis"]], default=-1,
    ).astype(float)
