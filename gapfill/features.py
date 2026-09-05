"""Сборка матрицы признаков для целевых точек по контексту известных наблюдений.

build_features(targets, context, grid): targets — строки, которые нужно восстановить (pid, day_num, year,
doy, crop); context — известные наблюдения (без целевых); grid — все строки для погоды ERA5.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from gapfill.config import SENSOR_CODE, TARGET
from gapfill.features_context import (
    PolygonWeights,
    climatology_features,
    day_effect_features,
    day_tables,
    loo_residuals,
    polygon_bias_features,
    weather_features,
)
from gapfill.features_cycles import cycle_features, rule_sensor
from gapfill.features_series import PolygonContext, harmonize, series_features

SOURCE_INDEX = {"evi": ["s2_evi", "landsat_evi", "modis_evi"], "ndwi": ["s2_ndwi", "landsat_ndwi", None]}
CTX_FIELDS = ["day_num", "year", "doy", TARGET, "sensor", "h", "evi", "ndwi"]


def _source_index(rows: pd.DataFrame, kind: str) -> np.ndarray:
    """Значение EVI/NDWI сенсора-источника primary_ndvi (NaN, если у сенсора нет такого индекса)."""
    sensor = rows["sensor"].to_numpy()
    out = np.full(len(rows), np.nan)
    for code, col in enumerate(SOURCE_INDEX[kind]):
        if col is not None:
            out = np.where(sensor == code, rows[col].to_numpy(dtype=float), out)
    return out


def polygon_context(rows: pd.DataFrame) -> PolygonContext:
    """Контекст полигона из аннотированных строк (rows уже отсортированы по дню)."""
    return PolygonContext(*(rows[c].to_numpy(dtype=float if c != "day_num" else np.int64) for c in CTX_FIELDS))


def annotate_context(context: pd.DataFrame) -> pd.DataFrame:
    """Добавляет к контексту h (шкала S2), evi/ndwi источника и LOO-остаток res по полигонам."""
    ctx = context.sort_values(["pid", "day_num"]).reset_index(drop=True)
    ctx = ctx.assign(h=harmonize(ctx[TARGET].to_numpy(), ctx["sensor"].to_numpy()),
                     evi=_source_index(ctx, "evi"), ndwi=_source_index(ctx, "ndwi"))
    res = np.full(len(ctx), np.nan)
    for idx in ctx.groupby("pid").indices.values():
        res[idx] = loo_residuals(polygon_context(ctx.iloc[idx]))
    return ctx.assign(res=res)


def _polygon_features(tg: pd.DataFrame, rows: pd.DataFrame, grid_poly: pd.DataFrame) -> pd.DataFrame:
    """Все «полигонные» признаки для целевых строк tg одного полигона."""
    t = tg["day_num"].to_numpy(dtype=np.int64)
    year_t = tg["year"].to_numpy(dtype=np.int64)
    doy_t = tg["doy"].to_numpy(dtype=np.int64)
    ctx = polygon_context(rows)
    feats = (series_features(ctx, t) | cycle_features(ctx, t, year_t, doy_t)
             | climatology_features(ctx, t, year_t, doy_t)
             | polygon_bias_features(rows, rows["res"].to_numpy(), year_t)
             | weather_features(grid_poly, t))
    feats["n_ctx_poly"] = np.full(len(t), float(len(rows)))
    feats["n_years_poly"] = np.full(len(t), float(rows["year"].nunique()))
    feats["poly_mean_h"] = np.full(len(t), float(rows["h"].mean()) if len(rows) else np.nan)
    feats["rule_sensor"] = rule_sensor(feats)
    return pd.DataFrame(feats, index=tg.index)


def build_features(targets: pd.DataFrame, context: pd.DataFrame, grid: pd.DataFrame) -> pd.DataFrame:
    """Матрица признаков для targets по контексту context; индекс совпадает с targets.index."""
    ctx_df = annotate_context(context)
    tables = day_tables(ctx_df)
    ctx_by = dict(iter(ctx_df.groupby("pid")))
    grid_by = dict(iter(grid.groupby("pid")))
    empty = ctx_df.iloc[0:0]
    pieces = [_polygon_features(tg, ctx_by.get(pid, empty), grid_by[pid])
              for pid, tg in targets.groupby("pid", sort=False)]
    X = pd.concat(pieces).loc[targets.index]
    day = targets["day_num"].to_numpy(dtype=np.int64)
    X = X.assign(**day_effect_features(day, targets["year"].to_numpy(), tables))
    X = X.assign(**PolygonWeights(ctx_df).features(targets["pid"].to_numpy(), day))
    return X.assign(doy=targets["doy"].to_numpy(dtype=float), year=targets["year"].to_numpy(dtype=float),
                    crop=targets["crop"].to_numpy(dtype=float))


def sensor_prior_features(X: pd.DataFrame) -> pd.DataFrame:
    """Оценки значения при гипотезе «сенсор = s»: интерполяция того же сенсора и кривая с обратным смещением."""
    from gapfill.config import SENSOR_OFFSET
    out = {}
    for name in SENSOR_CODE:
        curve = X["curve_sl_6"] if name != "modis" else X["mw_fwd_mean"].fillna(X["curve_sl_15"])
        out[f"hyp_{name}"] = curve + SENSOR_OFFSET[name] + X[f"pb_res_{name}"].fillna(0.0)
    return X.assign(**out)
