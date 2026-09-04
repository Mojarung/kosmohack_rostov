"""Работа с тремя сенсорами: смещения между ними и угадывание сенсора скрытой точки.

Из EDA: Landsat читает на +0.037 выше S2, MODIS на +0.083. Целевой ряд ``primary_ndvi``
склеен из трёх сенсоров, поэтому скрытая точка принадлежит конкретному сенсору, и
«среднее соседей» систематически промахивается, когда сенсоры соседей и гэпа разные.

Сенсор скрытой точки угадывается по расписанию съёмки: MODIS всегда на doy = 1 (mod 16),
Landsat с шагом 8 дней от других Landsat-дат, Sentinel-2 с шагом 5 дней. Простое правило
даёт около 81 % попаданий (MODIS и Landsat почти безошибочно).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ndvi.data import SENSORS

#: запасные значения смещений (из EDA на train), если оценить не на чем
DEFAULT_OFFSETS = {"s2": 0.0, "landsat": 0.037, "modis": 0.083, "none": 0.0}


def estimate_offsets(obs: pd.DataFrame) -> dict[str, float]:
    """Средние смещения сенсоров относительно S2 по парам наблюдений в один день."""
    off = {"s2": 0.0}
    pair = obs[obs.s2_ndvi.notna() & obs.landsat_ndvi.notna()]
    off["landsat"] = float((pair.landsat_ndvi - pair.s2_ndvi).mean()) if len(pair) > 30 else DEFAULT_OFFSETS["landsat"]
    pair = obs[obs.s2_ndvi.notna() & obs.modis_ndvi.notna()]
    off["modis"] = float((pair.modis_ndvi - pair.s2_ndvi).mean()) if len(pair) > 30 else DEFAULT_OFFSETS["modis"]
    off["none"] = 0.0
    return off


def estimate_offsets_by_polygon(obs: pd.DataFrame, min_pairs: int = 20,
                                global_off: dict[str, float] | None = None) -> pd.DataFrame:
    """Смещения по каждому полигону отдельно, с откатом на глобальные при нехватке пар."""
    g = global_off or estimate_offsets(obs)
    rows = []
    for pid, d in obs.groupby("anon_polygon_id"):
        rec = {"anon_polygon_id": pid, "s2": 0.0, "none": 0.0}
        for sensor in ("landsat", "modis"):
            pair = d[d.s2_ndvi.notna() & d[f"{sensor}_ndvi"].notna()]
            rec[sensor] = float((pair[f"{sensor}_ndvi"] - pair.s2_ndvi).mean()) if len(pair) >= min_pairs else g[sensor]
        rows.append(rec)
    return pd.DataFrame(rows).set_index("anon_polygon_id")


def infer_hidden_sensor(season: pd.DataFrame, date: pd.Timestamp) -> str:
    """Угадывает сенсор скрытой точки по расписанию съёмки внутри сезона полигона.

    ``season`` — видимые строки того же полигона и того же года (с сенсорными колонками).
    """
    doy = date.dayofyear
    ls_dates = season.loc[season.landsat_ndvi.notna(), "date"]
    s2_dates = season.loc[season.s2_ndvi.notna(), "date"]
    on_modis = doy % 16 == 1
    on_ls = len(ls_dates) > 0 and bool((((date - ls_dates).dt.days % 8) == 0).any())
    on_s2 = len(s2_dates) > 0 and bool((((date - s2_dates).dt.days % 5) == 0).any())
    if on_modis and not on_ls:
        return "modis"
    if on_s2 and not on_ls and not on_modis:
        return "s2"
    if on_ls and not on_s2:
        return "landsat"
    if on_ls and on_s2:
        return "landsat"  # неоднозначно: Landsat выигрывает, он точнее опознаётся
    if on_modis:
        return "modis"
    return "landsat"


def infer_hidden_sensors(visible: pd.DataFrame, targets: pd.DataFrame) -> pd.Series:
    """Векторная обёртка: угадывает сенсор для каждой строки ``targets``.

    Обе таблицы должны содержать ``anon_polygon_id``, ``date``, ``year`` и сенсорные колонки.
    """
    vis_by_key = {k: v for k, v in visible.groupby(["anon_polygon_id", "year"])}
    empty = visible.iloc[:0]
    out = [
        infer_hidden_sensor(vis_by_key.get((pid, yr), empty), dt)
        for pid, yr, dt in zip(targets.anon_polygon_id.values, targets.year.values, targets.date)
    ]
    return pd.Series(out, index=targets.index, name="hidden_src")


def to_s2_scale(values, sensor, offsets: dict[str, float]):
    """Приводит значения любого сенсора к шкале Sentinel-2."""
    off = pd.Series(sensor).map(offsets).fillna(0.0).values
    return np.asarray(values, dtype=float) - off


def from_s2_scale(values, sensor, offsets: dict[str, float]):
    """Обратный перевод из шкалы S2 в шкалу конкретного сенсора."""
    off = pd.Series(sensor).map(offsets).fillna(0.0).values
    return np.asarray(values, dtype=float) + off


__all__ = [
    "SENSORS", "DEFAULT_OFFSETS", "estimate_offsets", "estimate_offsets_by_polygon",
    "infer_hidden_sensor", "infer_hidden_sensors", "to_s2_scale", "from_s2_scale",
]
