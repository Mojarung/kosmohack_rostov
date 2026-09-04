"""Мост между собранным рядом и моделями: восстановление пропусков + аномалии.

Одна функция ``analyze_series`` используется и для полигона, нарисованного на карте,
и для полигона из датасета. Разница только в том, откуда пришёл ряд.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ndvi.anomalies import detect_episodes, episodes_to_frame
from ndvi.climatology import Climatology
from ndvi.pipeline import Artifacts, predict_gaps
from ndvi.sensors import estimate_offsets

#: шаг регулярной сетки, на которой восстанавливается ряд (дней)
GRID_STEP_DAYS = 5


def build_daily_frame(series: pd.DataFrame, start: str, end: str,
                      polygon_id: str, crop_type: str) -> pd.DataFrame:
    """Достраивает наблюдения до сплошного календаря: погода есть каждый день, NDVI — нет."""
    idx = pd.date_range(start, end, freq="D")
    base = pd.DataFrame({"date": idx})
    df = base.merge(series, on="date", how="left")
    df["anon_polygon_id"] = polygon_id
    df["crop_type"] = crop_type
    df["year"] = df.date.dt.year.astype("int16")
    df["doy"] = df.date.dt.dayofyear.astype("int16")
    df["src"] = np.select(
        [df.s2_ndvi.notna(), df.landsat_ndvi.notna(), df.modis_ndvi.notna()],
        ["s2", "landsat", "modis"], default="none")
    for col in ("s2_evi", "s2_ndwi", "landsat_evi", "landsat_ndwi", "modis_evi"):
        if col not in df:
            df[col] = np.nan
    return df


def analyze_series(series: pd.DataFrame, artifacts: Artifacts, start: str, end: str,
                   polygon_id: str = "AOI-USER", crop_type: str = "не указана",
                   grid_step: int = GRID_STEP_DAYS, season_only: bool = True,
                   reference: pd.DataFrame | None = None) -> dict:
    """Ряд -> восстановленный ряд, коридор нормы, эпизоды аномалий, погода.

    ``season_only`` ограничивает восстановление вегетационным сезоном (апрель–октябрь):
    зимой NDVI по снегу шумит, а сельхозсмысла в нём нет.

    ``reference`` — наблюдения полей, помеченных как обучающие. Они не подменяют собственную
    норму поля, а работают откатом: у свежего полигона своей истории нет, и коридор нормы
    строится по эталонным полям той же культуры. Это ровно проблема «холодного старта»
    из данных соревнования, где часть тестовых полей представлена одним сезоном.
    """
    df = build_daily_frame(series, start, end, polygon_id, crop_type)
    observed = df[df.primary_ndvi.notna()]
    if len(observed) < 8:
        return {"ok": False, "error": "слишком мало наблюдений для анализа",
                "n_observations": int(len(observed))}

    # сетка восстановления: каждые grid_step дней там, где наблюдения нет
    grid = pd.date_range(start, end, freq=f"{grid_step}D")
    if season_only:
        grid = grid[(grid.dayofyear >= 90) & (grid.dayofyear <= 305)]
    have = set(observed.date)
    gap_dates = [d for d in grid if d not in have]

    offsets = estimate_offsets(observed)
    # каскад нормы: своя история полигона -> эталонные поля той же культуры -> все эталонные
    clim_source = observed
    n_reference = 0
    if reference is not None and len(reference):
        ref = reference[reference.primary_ndvi.notna()]
        ref = ref[ref.anon_polygon_id != polygon_id]
        if len(ref):
            clim_source = pd.concat([observed, ref], ignore_index=True)
            n_reference = int(ref.anon_polygon_id.nunique())
    clim = Climatology().fit(clim_source)

    filled = pd.DataFrame(columns=["date", "primary_ndvi_pred", "hidden_src"])
    if gap_dates:
        targets = df[df.date.isin(gap_dates)][["anon_polygon_id", "date", "year", "doy", "crop_type"]]
        feats = predict_gaps(df, targets, artifacts, clim=clim)
        filled = feats[["date", "primary_ndvi_pred", "hidden_src"]]

    # объединяем наблюдения и восстановленные точки в один ряд для детекции аномалий
    obs_part = observed[["date", "year", "doy", "primary_ndvi", "src",
                         "era5_temp_c", "era5_precip_mm"]].copy()
    obs_part["is_filled"] = False
    fill_part = filled.rename(columns={"primary_ndvi_pred": "primary_ndvi", "hidden_src": "src"}).copy()
    if len(fill_part):
        fill_part["year"] = pd.to_datetime(fill_part.date).dt.year.astype("int16")
        fill_part["doy"] = pd.to_datetime(fill_part.date).dt.dayofyear.astype("int16")
        fill_part = fill_part.merge(df[["date", "era5_temp_c", "era5_precip_mm"]], on="date", how="left")
        fill_part["is_filled"] = True
    full = pd.concat([obs_part, fill_part], ignore_index=True).sort_values("date")
    full["anon_polygon_id"] = polygon_id
    full["crop_type"] = crop_type

    enriched, episodes = detect_episodes(full, context=df, offsets=offsets)
    eps = episodes_to_frame(episodes)

    # погода помесячно — для нижней панели графика
    weather = df.dropna(subset=["era5_temp_c"])[["date", "era5_temp_c", "era5_precip_mm"]]

    e = enriched.sort_values("date")
    return {
        "ok": True,
        "polygon_id": polygon_id,
        "crop_type": crop_type,
        "period": {"start": start, "end": end},
        "n_observations": int(len(observed)),
        "n_filled": int(len(filled)),
        "reference_fields": n_reference,
        "sensor_offsets": {k: round(v, 4) for k, v in offsets.items()},
        "series": [
            {"date": d.strftime("%Y-%m-%d"), "ndvi": _f(v), "ndvi_s2": _f(vs),
             "sensor": s, "filled": bool(fl), "z": _f(z),
             "norm": _f(nm), "norm_lo": _f(nm - sd) if np.isfinite(nm) and np.isfinite(sd) else None,
             "norm_hi": _f(nm + sd) if np.isfinite(nm) and np.isfinite(sd) else None}
            for d, v, vs, s, fl, z, nm, sd in zip(
                e.date, e.primary_ndvi, e.ndvi_s2, e.src, e.is_filled, e.z, e.norm_mean, e.norm_std)
        ],
        "weather": [
            {"date": d.strftime("%Y-%m-%d"), "temp": _f(t), "precip": _f(p)}
            for d, t, p in zip(weather.date, weather.era5_temp_c, weather.era5_precip_mm)
        ],
        "episodes": eps.to_dict("records") if len(eps) else [],
    }


def _f(v):
    """None вместо NaN: JSON не умеет NaN."""
    try:
        v = float(v)
    except (TypeError, ValueError):
        return None
    return round(v, 4) if np.isfinite(v) else None
