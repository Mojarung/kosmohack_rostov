"""Ряд полигона: гармонизация сенсоров, отбраковка артефактов, гладкая ежедневная кривая сезона.

Все функции возвращают новые объекты и не изменяют входные DataFrame.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from anomaly.config import (ARTIFACT_ABS, ARTIFACT_CONFIRM_DAYS, ARTIFACT_CONFIRM_DIFF, ARTIFACT_MAD_K, CURVE_BW,
                            CURVE_EDGE_TOLERANCE_DAYS, SEASON_DOY)
from gapfill.config import SENSOR_CODE, SENSOR_OFFSET
from gapfill.smooth import local_linear

SHRINK_PAIRS = 10       # сжатие смещения полигона к глобальному (в парах наблюдений)
CURVE_RANGE = (-0.1, 1.0)  # физический диапазон NDVI для кривой
MIN_SLOPE_POINTS = 2.0     # наклон кривой оценивается только при ≥ 2 эффективных точках в окне ядра
EPOCH = pd.Timestamp("2000-01-01")


def polygon_offsets(rows: pd.DataFrame) -> dict[str, float]:
    """Смещения Landsat и MODIS относительно S2 для полигона по совместным наблюдениям в один день,
    сжатые к глобальным значениям из EDA при малом числе пар."""
    out = {"s2": 0.0}
    for name, col in (("landsat", "landsat_ndvi"), ("modis", "modis_ndvi")):
        pair = rows.dropna(subset=[col, "s2_ndvi"])
        n = len(pair)
        local = float((pair[col] - pair["s2_ndvi"]).mean()) if n else SENSOR_OFFSET[name]
        out[name] = (n * local + SHRINK_PAIRS * SENSOR_OFFSET[name]) / (n + SHRINK_PAIRS)
    return out


def _confirmed_by_neighbour(day: np.ndarray, h: np.ndarray, suspect: np.ndarray) -> np.ndarray:
    """Какие подозрительные точки подтверждены соседним наблюдением (значит, это не облако, а сдвиг уровня).

    Облако и тень дают одиночный провал: соседние снимки возвращаются на прежний уровень. Уборка, гибель посева
    или отрастание дают устойчивый сдвиг: следующий (или предыдущий) снимок стоит на новом уровне.
    Поэтому точка считается подтверждённой, если ближайшее к ней наблюдение **не из числа подозрительных**
    (до или после, не дальше ARTIFACT_CONFIRM_DAYS дней) отличается по гармонизированному NDVI не больше
    ARTIFACT_CONFIRM_DIFF. Подозрительные соседи при поиске пропускаются, поэтому два подряд облачных снимка
    не могут подтвердить друг друга (за ними стоит чистая точка на прежнем уровне), а два подряд снимка после
    уборки — могут: за ними стоит чистая точка уже на новом уровне.
    """
    clean = np.flatnonzero(~suspect)
    out = np.zeros(len(day), dtype=bool)
    for i in np.flatnonzero(suspect):
        pos = int(np.searchsorted(clean, i))
        for j in (clean[pos - 1] if pos > 0 else None, clean[pos] if pos < len(clean) else None):
            if j is None:
                continue
            if abs(day[i] - day[j]) <= ARTIFACT_CONFIRM_DAYS and abs(h[i] - h[j]) <= ARTIFACT_CONFIRM_DIFF:
                out[i] = True
                break
    return out


def harmonized_series(rows: pd.DataFrame) -> pd.DataFrame:
    """Известные наблюдения полигона в шкале S2 с LOO-остатком и флагом артефакта.

    Артефактом считается точка с большим LOO-остатком (|остаток| > max(ARTIFACT_ABS, 3·MAD)), которую
    **не подтверждают соседние наблюдения** (см. `_confirmed_by_neighbour`): резкая уборка тоже даёт большой
    остаток, потому что кривую в этот момент тянут вверх более ранние точки, но её подтверждает следующий снимок.

    rows — строки obs одного полигона (из gapfill.data.load_all) с колонкой sensor.
    """
    s = rows.sort_values("day_num").reset_index(drop=True)
    offsets = polygon_offsets(s)
    codes = {v: k for k, v in SENSOR_CODE.items()}
    off = np.array([offsets[codes[int(c)]] for c in s["sensor"]])
    h = s["primary_ndvi"].to_numpy(dtype=float) - off
    day = s["day_num"].to_numpy(dtype=float)
    fit, _, w = local_linear(day, h, day, CURVE_BW, loo=True)
    res = np.where(w >= 0.5, h - fit, np.nan)
    mad = float(np.nanmedian(np.abs(res - np.nanmedian(res)))) * 1.4826 if np.isfinite(res).any() else 0.05
    suspect = np.abs(np.nan_to_num(res)) > max(ARTIFACT_ABS, ARTIFACT_MAD_K * mad)
    artifact = suspect & ~_confirmed_by_neighbour(day, h, suspect)
    return s.assign(h=h, res=res, artifact=artifact, offset=off)


def season_days(year: int) -> pd.DataFrame:
    """Ежедневная сетка сезона (1 апреля — 30 октября) с day_num и днём года."""
    dates = pd.date_range(f"{year}-04-01", f"{year}-10-30", freq="D")
    return pd.DataFrame({"date": dates, "day_num": (dates - EPOCH).days, "doy": dates.dayofyear, "year": year})


def today_day_num() -> float:
    """Сегодняшний день в шкале day_num по дате UTC (граница «дальше данных нет и быть не может»)."""
    return float((pd.Timestamp.now("UTC").tz_localize(None).normalize() - EPOCH).days)


def _reliable_span(day: np.ndarray) -> tuple[float, float]:
    """Отрезок day_num, где кривая опирается на наблюдения: [первое − допуск; min(последнее + допуск, сегодня)].

    За пределами наблюдений ядро шириной 8 дней ещё около десяти дней даёт заметный вес и продолжает кривую по
    инерции — это экстраполяция, а не измерение, и из-за неё эпизод «заканчивался» уже в будущем. Для прошлых
    сезонов ограничение «сегодня» никогда не срабатывает (последнее наблюдение давно позади), поэтому результаты
    исторических сезонов не зависят от дня запуска — ограничение видно только в текущем, ещё не закрытом сезоне.
    """
    if day.size == 0:
        return np.inf, -np.inf
    return float(day.min()) - CURVE_EDGE_TOLERANCE_DAYS, min(float(day.max()) + CURVE_EDGE_TOLERANCE_DAYS,
                                                             today_day_num())


def daily_curve(series: pd.DataFrame, year: int, bw: float = CURVE_BW) -> pd.DataFrame:
    """Гладкая кривая сезона на ежедневной сетке по гармонизированным наблюдениям без артефактов.

    Вне надёжного отрезка (см. `_reliable_span`) кривая обнуляется: value = NaN, weight = 0, — чтобы Z-score,
    фенометрики, норма и эпизоды не опирались на экстраполяцию за последнее наблюдение или в будущее.

    Возвращает day_num, date, doy, value, slope, weight (сумма весов ядра — надёжность), n_obs сезона.
    """
    grid = season_days(year)
    sel = (series["year"] == year) & ~series["artifact"]
    day = series.loc[sel, "day_num"].to_numpy(dtype=float)
    h = series.loc[sel, "h"].to_numpy(dtype=float)
    grid_day = grid["day_num"].to_numpy(dtype=float)
    value, slope, weight = local_linear(day, h, grid_day, bw, min_points=MIN_SLOPE_POINTS)
    # на всякий случай удерживаем кривую в физическом диапазоне NDVI
    value = np.clip(value, CURVE_RANGE[0], CURVE_RANGE[1])
    lo, hi = _reliable_span(day)
    inside = (grid_day >= lo) & (grid_day <= hi)
    return grid.assign(value=np.where(inside, value, np.nan), slope=np.where(inside, slope, np.nan),
                       weight=np.where(inside, weight, 0.0), n_obs=int(sel.sum()))


def curves_by_year(series: pd.DataFrame) -> dict[int, pd.DataFrame]:
    """Кривые всех сезонов полигона, у которых есть хотя бы 5 наблюдений без артефактов."""
    out = {}
    for year, g in series.groupby("year"):
        if int((~g["artifact"]).sum()) >= 5:
            out[int(year)] = daily_curve(series, int(year))
    return out


def in_season(doy: np.ndarray) -> np.ndarray:
    return (doy >= SEASON_DOY[0]) & (doy <= SEASON_DOY[1])
