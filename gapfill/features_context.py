"""Контекстные признаки: LOO-остатки, «шум дня» между полигонами, смещения сенсоров полигона,
климатология полигона и погода ERA5 с ежедневной сетки."""

from __future__ import annotations

import numpy as np
import pandas as pd

from gapfill.config import SENSOR_CODE, WEATHER_COLS
from gapfill.features_series import PolygonContext
from gapfill.smooth import local_linear

RES_BW = {"sl": 8.0, "md": 15.0}        # полосы ядра для остатков: S2+Landsat и MODIS
MIN_WEIGHT = 0.5                        # ниже — остаток недостоверен (почти нет соседей)
CLIM_DOY = 7                            # окно ±дней года для собственной климатологии
DAY_OFFSETS = {"d0": (0,), "pm1": (-1, 1), "pm3": (-3, -2, 2, 3)}
PAIRS = {"ls_s2": ("landsat_ndvi", "s2_ndvi"), "md_s2": ("modis_ndvi", "s2_ndvi"),
         "md_ls": ("modis_ndvi", "landsat_ndvi")}


def loo_residuals(ctx: PolygonContext) -> np.ndarray:
    """LOO-остаток каждой точки контекста относительно гладкой кривой своей группы сенсоров."""
    res = np.full(len(ctx.day), np.nan)
    md = ctx.sensor == SENSOR_CODE["modis"]
    for sel, bw in ((~md, RES_BW["sl"]), (md, RES_BW["md"])):
        if sel.sum() < 3:
            continue
        a, _, s = local_linear(ctx.day[sel], ctx.h[sel], ctx.day[sel], bw, loo=True)
        r = ctx.h[sel] - a
        r[s < MIN_WEIGHT] = np.nan
        res[sel] = r
    return res


def day_tables(context: pd.DataFrame) -> dict[str, pd.Series]:
    """Суммы и счётчики LOO-остатков и наблюдений по (день, сенсор) по всем полигонам."""
    g = context.groupby(["day_num", "sensor"])
    return {"sum": g["res"].sum(min_count=1), "cnt": g["res"].count(), "n_obs": g.size(),
            "active": context.groupby("year")["pid"].nunique()}


def _lookup(table: pd.Series, day: np.ndarray, code: int) -> np.ndarray:
    idx = pd.MultiIndex.from_arrays([day, np.full(len(day), code)])
    return table.reindex(idx).to_numpy(dtype=float)


def day_effect_features(day: np.ndarray, year: np.ndarray, tables: dict[str, pd.Series]) -> dict:
    """Средний остаток других полигонов в ту же дату (и рядом) по каждому сенсору; сколько полигонов снято."""
    out = {}
    active = tables["active"].reindex(year).to_numpy(dtype=float)
    for name, code in SENSOR_CODE.items():
        for oname, offsets in DAY_OFFSETS.items():
            s = np.nansum([_lookup(tables["sum"], day + o, code) for o in offsets], axis=0)
            c = np.nansum([_lookup(tables["cnt"], day + o, code) for o in offsets], axis=0)
            out[f"de_{name}_{oname}"] = np.where(c > 0, s / np.maximum(c, 1), np.nan)
            out[f"de_{name}_{oname}_n"] = c
        n_obs = np.nan_to_num(_lookup(tables["n_obs"], day, code))
        out[f"xp_{name}_n"] = n_obs
        out[f"xp_{name}_share"] = n_obs / np.maximum(active, 1)
    out["xp_active_year"] = active
    return out


def polygon_bias_features(rows: pd.DataFrame, res: np.ndarray, year_t: np.ndarray) -> dict:
    """Смещения сенсоров внутри полигона: по совместным наблюдениям в один день и по LOO-остаткам."""
    out = {}
    year = rows["year"].to_numpy()
    for name, (a, b) in PAIRS.items():
        diff = (rows[a] - rows[b]).to_numpy(dtype=float)
        out |= _mean_all_and_by_year(f"pb_{name}", diff, year, year_t)
    for name, code in SENSOR_CODE.items():
        r = np.where(rows["sensor"].to_numpy() == code, res, np.nan)
        out |= _mean_all_and_by_year(f"pb_res_{name}", r, year, year_t)
    return out


def _mean_all_and_by_year(prefix: str, values: np.ndarray, year: np.ndarray, year_t: np.ndarray) -> dict:
    """Среднее и счётчик по всем годам и отдельно по году целевой точки."""
    ok = ~np.isnan(values)
    n_all = float(ok.sum())
    mean_all = float(values[ok].mean()) if n_all else np.nan
    same = year[None, :] == year_t[:, None]
    n_year = (same & ok[None, :]).sum(1).astype(float)
    sum_year = np.where(same & ok[None, :], values[None, :], 0.0).sum(1)
    return {prefix: np.full(len(year_t), mean_all), f"{prefix}_n": np.full(len(year_t), n_all),
            f"{prefix}_y": np.where(n_year > 0, sum_year / np.maximum(n_year, 1), np.nan),
            f"{prefix}_y_n": n_year}


def _masked_mean(values: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Среднее, std и счётчик values по строкам маски (n_t × n_ctx)."""
    n = mask.sum(1).astype(float)
    v = np.where(mask, values[None, :], 0.0)
    mean = np.where(n > 0, v.sum(1) / np.maximum(n, 1), np.nan)
    var = np.where(n > 1, (v * v).sum(1) / np.maximum(n, 1) - mean ** 2, np.nan)
    return mean, np.sqrt(np.maximum(var, 0.0)), n


def climatology_features(ctx: PolygonContext, t: np.ndarray, year_t: np.ndarray, doy_t: np.ndarray) -> dict:
    """Собственная норма полигона по другим годам и аномалия текущего сезона относительно неё."""
    near = np.abs(ctx.doy[None, :] - doy_t[:, None]) <= CLIM_DOY
    other = ctx.year[None, :] != year_t[:, None]
    clim_mean, clim_std, clim_n = _masked_mean(ctx.h, near & other)
    # норма для самих точек контекста (по другим годам) → аномалия каждой точки
    near_cc = np.abs(ctx.doy[None, :] - ctx.doy[:, None]) <= CLIM_DOY
    other_cc = ctx.year[None, :] != ctx.year[:, None]
    clim_ctx, _, n_ctx = _masked_mean(ctx.h, near_cc & other_cc)
    anom = np.where(n_ctx >= 3, ctx.h - clim_ctx, np.nan)
    out = {"clim_mean": clim_mean, "clim_std": clim_std, "clim_n": clim_n}
    same_year = ctx.year[None, :] == year_t[:, None]
    for half in (30, 60):
        win = np.abs(ctx.day[None, :] - t[:, None]) <= half
        m, _, n = _masked_mean(np.nan_to_num(anom), win & same_year & ~np.isnan(anom)[None, :])
        out[f"anom_{half}"] = m
        out[f"anom_{half}_n"] = n
    out["clim_plus_anom"] = clim_mean + np.nan_to_num(out["anom_30"])
    return out


def weather_features(grid_poly: pd.DataFrame, t: np.ndarray) -> dict:
    """Погода ERA5 в день t и накопленно за 7/30 дней (интерполяция замаскированных строк до 3 дней)."""
    names = ["w_temp", "w_precip", "w_precip_pm1", "w_precip7", "w_precip30", "w_temp7", "w_temp30"]
    if grid_poly[WEATHER_COLS].notna().sum().sum() == 0:
        return {n: np.full(len(t), np.nan) for n in names}
    s = grid_poly.set_index("day_num")[WEATHER_COLS].sort_index()
    s = s.reindex(np.arange(s.index.min(), s.index.max() + 1)).interpolate(limit=3, limit_area="inside")
    temp, precip = s["era5_temp_c"], s["era5_precip_mm"]
    return {
        "w_temp": temp.reindex(t).to_numpy(), "w_precip": precip.reindex(t).to_numpy(),
        "w_precip_pm1": precip.rolling(3, center=True, min_periods=2).sum().reindex(t).to_numpy(),
        "w_precip7": precip.rolling(7, min_periods=4).sum().reindex(t).to_numpy(),
        "w_precip30": precip.rolling(30, min_periods=20).sum().reindex(t).to_numpy(),
        "w_temp7": temp.rolling(7, min_periods=4).mean().reindex(t).to_numpy(),
        "w_temp30": temp.rolling(30, min_periods=20).mean().reindex(t).to_numpy(),
    }


# --- «Шум дня», взвешенный по похожести полигонов -------------------------------------------------
# Координат нет, но парная корреляция LOO-остатков двух полигонов в одни и те же дни показывает,
# насколько они «соседи» по атмосфере и геометрии съёмки. Веса = усечённая корреляция со сжатием к среднему.
SHRINK_PAIRS = 30       # сила сжатия корреляции к глобальному среднему (в парах наблюдений)
MIN_PAIRS = 8           # ниже — пара полигонов считается неоценённой
LOW_RESIDUAL = -0.15    # порог «облачного» остатка для доли облачных полигонов дня


class PolygonWeights:
    """Матрица весов похожести полигонов и плотные таблицы остатков по (день, сенсор, полигон)."""

    def __init__(self, ctx: pd.DataFrame):
        ok = ctx["res"].notna()
        piv = ctx.loc[ok].pivot_table(index=["day_num", "sensor"], columns="pid", values="res", aggfunc="mean")
        self.pids = list(piv.columns)
        self.pos = {p: i for i, p in enumerate(self.pids)}
        present = piv.notna().to_numpy(dtype=float)
        n_pairs = present.T @ present
        corr = piv.corr(min_periods=MIN_PAIRS).to_numpy()
        off = ~np.eye(len(self.pids), dtype=bool)
        global_mean = float(np.nanmean(corr[off])) if np.isfinite(corr[off]).any() else 0.0
        corr = np.where(np.isnan(corr), global_mean, corr)
        shrunk = (n_pairs * corr + SHRINK_PAIRS * global_mean) / (n_pairs + SHRINK_PAIRS)
        w = np.clip(shrunk, 0.0, None)
        np.fill_diagonal(w, 0.0)
        self.w = w
        self.dmin = int(ctx["day_num"].min())
        n_days = int(ctx["day_num"].max()) - self.dmin + 1
        self.res = np.zeros((n_days, 3, len(self.pids)), dtype=float)
        self.mask = np.zeros((n_days, 3, len(self.pids)), dtype=float)
        rows = ctx.loc[ok]
        d = rows["day_num"].to_numpy() - self.dmin
        s = rows["sensor"].to_numpy()
        p = rows["pid"].map(self.pos).to_numpy()
        self.res[d, s, p] = rows["res"].to_numpy()
        self.mask[d, s, p] = 1.0

    def features(self, pid: np.ndarray, day: np.ndarray) -> dict:
        """Взвешенный остаток других полигонов в день day по каждому сенсору + доля «облачных» полигонов."""
        out = {}
        d = day - self.dmin
        inside = (d >= 0) & (d < self.res.shape[0])
        d = np.clip(d, 0, self.res.shape[0] - 1)
        p_idx = np.array([self.pos.get(p, -1) for p in pid])
        w = np.where(p_idx[:, None] >= 0, self.w[np.maximum(p_idx, 0)], 0.0)
        for name, code in SENSOR_CODE.items():
            r, m = self.res[d, code], self.mask[d, code]
            num, den = (r * m * w).sum(1), (m * w).sum(1)
            out[f"dew_{name}"] = np.where(inside & (den > 1e-6), num / np.maximum(den, 1e-6), np.nan)
            out[f"dew_{name}_wsum"] = np.where(inside, den, 0.0)
            cnt = m.sum(1)
            out[f"de_{name}_min"] = np.where(inside & (cnt > 0), np.where(m > 0, r, np.inf).min(1), np.nan)
            out[f"de_{name}_lowshare"] = np.where(inside & (cnt > 0), ((r < LOW_RESIDUAL) & (m > 0)).sum(1) / np.maximum(cnt, 1), np.nan)
        return out
