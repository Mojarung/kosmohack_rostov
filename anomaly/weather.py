"""Погодный контекст эпизода: ERA5 за окно до и внутри эпизода против нормы этих же дней по другим годам."""

from __future__ import annotations

import numpy as np
import pandas as pd

from anomaly.config import DRY_DAY_MM, HOT_DAY_C, WEATHER_BEFORE_DAYS
from gapfill.config import WEATHER_COLS


def daily_weather(grid_poly: pd.DataFrame) -> pd.DataFrame | None:
    """Ежедневная погода полигона (замаскированные дни интерполируются до 3 дней); None, если погоды нет."""
    if grid_poly[WEATHER_COLS].notna().sum().sum() == 0:
        return None
    s = grid_poly.set_index("day_num")[WEATHER_COLS].sort_index()
    s = s.reindex(np.arange(s.index.min(), s.index.max() + 1)).interpolate(limit=3, limit_area="inside")
    dates = pd.Timestamp("2000-01-01") + pd.to_timedelta(s.index, unit="D")
    return s.assign(date=dates, doy=dates.dayofyear, year=dates.year).rename_axis("day_num").reset_index()


def regional_weather(grid: pd.DataFrame) -> pd.DataFrame | None:
    """Погода региона: медиана ERA5 по всем полигонам с погодой на каждый день (для полигонов без ERA5)."""
    w = grid.dropna(subset=WEATHER_COLS)
    if w.empty:
        return None
    med = w.groupby("day_num")[WEATHER_COLS].median().sort_index()
    med = med.reindex(np.arange(med.index.min(), med.index.max() + 1)).interpolate(limit=3, limit_area="inside")
    dates = pd.Timestamp("2000-01-01") + pd.to_timedelta(med.index, unit="D")
    return med.assign(date=dates, doy=dates.dayofyear, year=dates.year).rename_axis("day_num").reset_index()


def _window(w: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    return w.loc[(w["date"] >= start) & (w["date"] <= end)]


def window_stats(w: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> dict:
    """Сумма осадков, средняя температура, число сухих и жарких дней, максимальная сухая серия."""
    seg = _window(w, start, end)
    if seg.empty or seg["era5_precip_mm"].isna().all():
        return {}
    dry = (seg["era5_precip_mm"] < DRY_DAY_MM).to_numpy()
    runs, best, cur = [], 0, 0
    for d in dry:
        cur = cur + 1 if d else 0
        best = max(best, cur)
    return {"precip_mm": float(seg["era5_precip_mm"].sum()), "temp_c": float(seg["era5_temp_c"].mean()),
            "dry_days": int(dry.sum()), "hot_days": int((seg["era5_temp_c"] > HOT_DAY_C).sum()),
            "max_dry_spell": int(best), "n_days": int(len(seg))}


def window_norm(w: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> dict:
    """Норма тех же календарных дней по другим годам: средняя сумма осадков и средняя температура."""
    years = sorted(set(w["year"]) - {start.year})
    sums, temps = [], []
    for y in years:
        try:
            s, e = start.replace(year=y), end.replace(year=y)
        except ValueError:      # 29 февраля не бывает в сезоне, но на всякий случай
            continue
        seg = _window(w, s, e)
        if len(seg) >= 0.8 * (end - start).days + 1 and seg["era5_precip_mm"].notna().mean() > 0.8:
            sums.append(float(seg["era5_precip_mm"].sum()))
            temps.append(float(seg["era5_temp_c"].mean()))
    if len(sums) < 3:
        return {}
    return {"precip_norm_mm": float(np.mean(sums)), "temp_norm_c": float(np.mean(temps)), "norm_years": len(sums)}


def episode_weather(w: pd.DataFrame | None, start: str, end: str) -> dict:
    """Погодные факты эпизода: окно до начала (30 дней) и сам эпизод, оба против нормы."""
    if w is None:
        return {"available": False}
    s, e = pd.Timestamp(start), pd.Timestamp(end)
    out = {"available": True}
    for name, (a, b) in {"before": (s - pd.Timedelta(days=WEATHER_BEFORE_DAYS), s - pd.Timedelta(days=1)),
                         "during": (s, e), "combined": (s - pd.Timedelta(days=WEATHER_BEFORE_DAYS), e)}.items():
        stats, norm = window_stats(w, a, b), window_norm(w, a, b)
        if not stats:
            continue
        block = dict(stats)
        if norm:
            block["precip_deficit_pct"] = 100.0 * (1 - stats["precip_mm"] / max(norm["precip_norm_mm"], 1e-6))
            block["temp_anomaly_c"] = stats["temp_c"] - norm["temp_norm_c"]
            block |= norm
        out[name] = block
    return out
