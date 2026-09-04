"""Климатические нормы: норма организаторов (сырые значения) и гармонизированная норма по кривым других лет.

Норма организаторов воспроизведена в exp-100: leave-one-year-out среднее сырых primary_ndvi в окне ±8 дней
по дню года. Она смешивает сенсоры, поэтому для детекции используется гармонизированная норма на кривых.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from anomaly.config import CROP_NORM_MIN_POLYGONS, NORM_DOY_WINDOW, STD_FLOOR
from anomaly.series import season_days


def organizer_norm(rows: pd.DataFrame, window: int = NORM_DOY_WINDOW) -> pd.DataFrame:
    """Норма организаторов для каждой строки полигона: mean/std сырых значений других лет в окне ±window по doy."""
    doy = rows["doy"].to_numpy()
    year = rows["year"].to_numpy()
    v = rows["primary_ndvi"].to_numpy(dtype=float)
    mask = (np.abs(doy[None, :] - doy[:, None]) <= window) & (year[None, :] != year[:, None])
    n = mask.sum(1)
    mean = np.where(n > 0, (mask * v[None, :]).sum(1) / np.maximum(n, 1), np.nan)
    sq = (mask * v[None, :] ** 2).sum(1) / np.maximum(n, 1)
    std = np.sqrt(np.maximum(sq - mean ** 2, 0.0))
    n_years = np.array([len(np.unique(year[m])) for m in mask])
    z = np.where(std > 0, (v - mean) / np.where(std > 0, std, 1.0), np.nan)
    return rows.assign(org_mean=mean, org_std=std, org_n_years=n_years, org_z=z)


def status_from_z(z: np.ndarray) -> np.ndarray:
    """Статус по порогам ТЗ."""
    return np.select([z < -2, z < -1], ["Критическая аномалия", "Угнетение биомассы"], "Штатное развитие")


def norm_curve(curves: dict[int, pd.DataFrame], year: int, min_weight: float = 0.8) -> pd.DataFrame:
    """Гармонизированная норма сезона year: среднее и std кривых других лет по каждому дню сезона.

    В среднее входят только дни, где кривая соответствующего года надёжна (weight >= min_weight).
    """
    grid = season_days(year)[["doy"]]
    vals = []
    for y, c in curves.items():
        if y == year:
            continue
        c2 = c.set_index("doy").reindex(grid["doy"])
        vals.append(np.where(c2["weight"].to_numpy() >= min_weight, c2["value"].to_numpy(), np.nan))
    if not vals:
        return grid.assign(norm_mean=np.nan, norm_std=np.nan, norm_n=0)
    m = np.vstack(vals)
    n = np.isfinite(m).sum(0)
    with np.errstate(all="ignore"):
        mean = np.nanmean(m, axis=0)
        std = np.nanstd(m, axis=0)
    return grid.assign(norm_mean=np.where(n > 0, mean, np.nan), norm_std=np.where(n > 1, np.maximum(std, STD_FLOOR), np.nan),
                       norm_n=n)


def crop_norms(curves_all: dict[str, dict[int, pd.DataFrame]], crop_of: dict[str, int]) -> dict[int, pd.DataFrame]:
    """Норма «по культуре» для полигонов без истории: среднее кривых всех полигонов культуры по дню года."""
    out = {}
    for crop in set(crop_of.values()):
        pids = [p for p, c in crop_of.items() if c == crop and p in curves_all]
        if len(pids) < CROP_NORM_MIN_POLYGONS:
            continue
        stacks = []
        for p in pids:
            for c in curves_all[p].values():
                c2 = c.set_index("doy")["value"].where(c.set_index("doy")["weight"] >= 0.8)
                stacks.append(c2.reindex(range(91, 305)).to_numpy())
        m = np.vstack(stacks)
        n = np.isfinite(m).sum(0)
        with np.errstate(all="ignore"):
            out[crop] = pd.DataFrame({"doy": range(91, 305), "norm_mean": np.where(n > 2, np.nanmean(m, 0), np.nan),
                                      "norm_std": np.where(n > 2, np.maximum(np.nanstd(m, 0), STD_FLOOR), np.nan), "norm_n": n})
    return out


def norm_for_year(curves: dict[int, pd.DataFrame], year: int, crop_norm: pd.DataFrame | None) -> tuple[pd.DataFrame, str]:
    """Норма для сезона: собственная (если есть ≥ 3 других лет), иначе по культуре. Возвращает (норма, источник)."""
    own = norm_curve(curves, year)
    if int((own["norm_n"] >= 3).sum()) >= 60:
        return own, "история полигона"
    if crop_norm is not None:
        grid = season_days(year)[["doy"]].merge(crop_norm, on="doy", how="left")
        return grid, "среднее по культуре (истории полигона нет)"
    return own, "нормы нет"
