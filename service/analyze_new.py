"""Анализ нового полигона (собранного сервисом) тем же пайплайном детекции, что и для данных кейса."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from anomaly.climatology import norm_for_year
from anomaly.detect import find_episodes, norm_phenology, phenology, phenology_deviation, z_series
from anomaly.indices import ndwi_anomaly, ndwi_note
from anomaly.report import classify, describe, severity_label
from anomaly.series import curves_by_year, harmonized_series
from anomaly.weather import daily_weather, episode_weather
from service.data import SENSOR_NAMES, _series_json

MIN_YEARS_FOR_NORM = 3


def _grid_from_weather(pid: str, weather: pd.DataFrame) -> pd.DataFrame:
    """Ежедневная сетка полигона с погодой в формате grid из gapfill.data."""
    w = weather.copy()
    return w.assign(pid=pid, day_num=(w["date"] - pd.Timestamp("2000-01-01")).dt.days.astype("int32"),
                    year=w["date"].dt.year, doy=w["date"].dt.dayofyear)


def analyze_new_polygon(pid: str, obs: pd.DataFrame, weather: pd.DataFrame) -> dict:
    """Кривые, нормы, Z, эпизоды с объяснениями и погода — JSON для интерфейса (как service.data.Store.polygon)."""
    series = harmonized_series(obs)
    curves = curves_by_year(series)
    if len(curves) < MIN_YEARS_FOR_NORM + 1:
        raise ValueError(f"собрано только {len(curves)} сезонов, для нормы нужно не меньше {MIN_YEARS_FOR_NORM + 1}")
    grid = _grid_from_weather(pid, weather)
    wx = daily_weather(grid)
    years, episodes = {}, []
    for year, curve in curves.items():
        norm, source = norm_for_year(curves, year, None)
        zs = z_series(curve, norm)
        pheno = phenology(curve)
        dev = phenology_deviation(pheno, norm_phenology(norm))
        obs_year = series.loc[series["year"] == year]
        years[int(year)] = {
            "norm_source": source,
            "observations": [{"date": d.strftime("%Y-%m-%d"), "value": float(v), "harmonized": float(h),
                              "sensor": SENSOR_NAMES[int(s)], "artifact": bool(a)}
                             for d, v, h, s, a in zip(obs_year["date"], obs_year["primary_ndvi"], obs_year["h"],
                                                      obs_year["sensor"], obs_year["artifact"])],
            "restored": [], "curve": _series_json(zs, "value"), "norm_mean": _series_json(zs, "norm_mean"),
            "norm_std": _series_json(zs, "norm_std"), "z": _series_json(zs, "z"),
        }
        if norm["norm_mean"].notna().sum() < 30:
            continue
        obs_days = obs_year.loc[~obs_year["artifact"], "day_num"].to_numpy()
        for ep in find_episodes(zs, obs_days):
            window = ((obs_year["date"] >= pd.Timestamp(ep["start"]) - pd.Timedelta(days=10))
                      & (obs_year["date"] <= pd.Timestamp(ep["end"]) + pd.Timedelta(days=10)))
            near = int((obs_year["artifact"] & window).sum())
            weather_facts = episode_weather(wx, ep["start"], ep["end"]) | {"source": "ERA5 (Open-Meteo) по центроиду поля"}
            cause, conf, reasons = classify(ep, dev, pheno, weather_facts, near, {"available": False})
            if (note := ndwi_note(ndwi_anomaly(series, year, ep["start"], ep["end"]))) is not None:
                reasons = reasons + [note]
            episodes.append({"pid": pid, "year": int(year), **ep, "severity": severity_label(ep), "cause": cause,
                             "confidence": conf, "norm_source": source, "weather_source": weather_facts["source"],
                             "artifacts_near": near, "weather": weather_facts, "reasons": " | ".join(reasons),
                             "text": describe(pid, int(year), ep, cause, conf, reasons, source)})
    weather_json = {}
    if wx is not None:
        for year, g in wx.groupby("year"):
            g = g.dropna(subset=["era5_temp_c"])
            weather_json[int(year)] = {"date": [d.strftime("%Y-%m-%d") for d in g["date"]],
                                       "temp": [round(float(v), 2) for v in g["era5_temp_c"]],
                                       "precip": [round(float(v), 2) for v in g["era5_precip_mm"]]}
    return {"pid": pid, "crop": "не задана", "kind": "новая территория (данные собраны сервисом)", "years": years,
            "episodes": episodes, "shape": [], "weather": weather_json}
