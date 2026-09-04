"""Данные для нейросети по ежедневной сетке сезона: один образец = полигон-год (213 дней × каналы).

Каналы дня: NDVI/EVI/NDWI каждого сенсора с масками наличия, погода ERA5, климатология полигона
по другим годам, «шум дня» других полигонов по сенсорам, позиция в сезоне, культура, флаг запроса.
Маскирование: замаскированные известные дни теряют все сенсорные каналы (как контрольные точки test)
и становятся целями. Маска глобальная (доля 15 % известных точек), как в табличном пайплайне.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from gapfill.config import INDEX_COLS, SENSOR_CODE, SENSOR_OFFSET, TARGET, WEATHER_COLS
from gapfill.features import annotate_context

SEASON_LEN = 213                     # 1 апреля … 30 октября
CLIM_HALF = 7                        # ±дней года для климатологии
N_CROPS = 4
TEMP_SCALE, PRECIP_SCALE = 30.0, 10.0


@dataclass
class SeasonTensors:
    """Массивы по образцам (полигон-год) на ежедневной сетке."""
    pid: np.ndarray            # (n,) идентификатор полигона
    year: np.ndarray           # (n,)
    day0: np.ndarray           # (n,) day_num 1 апреля
    idx: np.ndarray            # (n, 213) индекс в таблице obs или -1
    vals: np.ndarray           # (n, 213, 8) индексы сенсоров, 0 где нет
    present: np.ndarray        # (n, 213, 8) 1 где значение есть
    y: np.ndarray              # (n, 213) primary_ndvi (0, если нет)
    known: np.ndarray          # (n, 213) известен ли primary_ndvi
    sensor: np.ndarray         # (n, 213) код сенсора-источника или -1
    weather: np.ndarray        # (n, 213, 2) температура/осадки (масштабированные), 0 где нет
    weather_ok: np.ndarray     # (n, 213)
    crop: np.ndarray           # (n,)
    poly_index: np.ndarray     # (n,) номер полигона (для климатологии)
    doy0: np.ndarray           # (n,) doy 1 апреля


def _season_index(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Номер дня внутри сезона (0..212) и day_num 1 апреля того же года."""
    apr1 = pd.to_datetime(df["year"].astype(str) + "-04-01")
    day0 = (apr1 - pd.Timestamp("2000-01-01")).dt.days.to_numpy()
    return df["day_num"].to_numpy() - day0, day0


def build_tensors(obs: pd.DataFrame, grid: pd.DataFrame) -> SeasonTensors:
    """Собирает тензоры сезонов по всем строкам grid (наблюдения + погода) и таблице obs (известные точки)."""
    keys = grid[["pid", "year"]].drop_duplicates().sort_values(["pid", "year"]).reset_index(drop=True)
    key_to_i = {(p, y): i for i, (p, y) in enumerate(zip(keys["pid"], keys["year"]))}
    n = len(keys)
    t = SeasonTensors(
        pid=keys["pid"].to_numpy(), year=keys["year"].to_numpy().astype(int),
        day0=np.zeros(n, dtype=np.int64), idx=np.full((n, SEASON_LEN), -1, dtype=np.int64),
        vals=np.zeros((n, SEASON_LEN, len(INDEX_COLS)), np.float32),
        present=np.zeros((n, SEASON_LEN, len(INDEX_COLS)), np.float32),
        y=np.zeros((n, SEASON_LEN), np.float32), known=np.zeros((n, SEASON_LEN), bool),
        sensor=np.full((n, SEASON_LEN), -1, np.int8), weather=np.zeros((n, SEASON_LEN, 2), np.float32),
        weather_ok=np.zeros((n, SEASON_LEN), np.float32), crop=np.zeros(n, np.int64),
        poly_index=pd.factorize(keys["pid"])[0], doy0=np.zeros(n, np.int64),
    )
    pos, day0 = _season_index(grid)
    ok = (pos >= 0) & (pos < SEASON_LEN)
    gi = np.array([key_to_i[(p, y)] for p, y in zip(grid["pid"], grid["year"])])
    t.day0[gi] = day0
    t.doy0[gi] = grid["doy"].to_numpy() - pos
    t.crop[gi] = grid["crop"].to_numpy()
    w = grid[WEATHER_COLS].to_numpy(dtype=np.float32)
    w_ok = ~np.isnan(w).any(1)
    sel = ok & w_ok
    t.weather[gi[sel], pos[sel]] = w[sel] / np.array([TEMP_SCALE, PRECIP_SCALE], np.float32)
    t.weather_ok[gi[sel], pos[sel]] = 1.0
    opos, _ = _season_index(obs)
    oi = np.array([key_to_i[(p, y)] for p, y in zip(obs["pid"], obs["year"])])
    v = obs[INDEX_COLS].to_numpy(dtype=np.float32)
    t.vals[oi, opos] = np.nan_to_num(v)
    t.present[oi, opos] = (~np.isnan(v)).astype(np.float32)
    t.y[oi, opos] = obs[TARGET].to_numpy(dtype=np.float32)
    t.known[oi, opos] = True
    t.sensor[oi, opos] = obs["sensor"].to_numpy()
    t.idx[oi, opos] = np.arange(len(obs))
    return t


def loo_residual_array(obs: pd.DataFrame, context_mask: np.ndarray) -> np.ndarray:
    """LOO-остаток каждой точки obs относительно кривой своего полигона по контексту (NaN вне контекста)."""
    ctx = obs.loc[context_mask].assign(_row=np.flatnonzero(context_mask))
    ann = annotate_context(ctx)
    res = np.full(len(obs), np.nan, np.float32)
    res[ann["_row"].to_numpy()] = ann["res"].to_numpy(dtype=np.float32)
    return res


class EpochInputs:
    """Каналы, зависящие от маски эпохи: климатология полигона и «шум дня» других полигонов."""

    def __init__(self, t: SeasonTensors, obs: pd.DataFrame, res: np.ndarray):
        self.t = t
        self.day_num = obs["day_num"].to_numpy()
        self.obs_sensor = obs["sensor"].to_numpy()
        pid_to_poly = dict(zip(t.pid, t.poly_index))
        self.obs_poly = obs["pid"].map(pid_to_poly).to_numpy()
        self.res = np.nan_to_num(res)
        self.res_ok = ~np.isnan(res)
        self.dmin = int(self.day_num.min())
        self.n_days = int(self.day_num.max() - self.dmin + 1)
        self.n_poly = int(t.poly_index.max() + 1)
        offsets = np.array([SENSOR_OFFSET[s] for s in SENSOR_CODE], np.float32)
        self.h = np.where(t.known, t.y - offsets[np.maximum(t.sensor, 0)], 0.0).astype(np.float32)

    def day_effects(self, visible: np.ndarray) -> np.ndarray:
        """(n_samples, 213, 6): средний остаток и число других полигонов по сенсорам в каждый день."""
        use = visible & self.res_ok
        d = self.day_num[use] - self.dmin
        s = self.obs_sensor[use]
        p = self.obs_poly[use]
        r = self.res[use]
        total_sum = np.zeros((self.n_days, 3), np.float32)
        total_cnt = np.zeros((self.n_days, 3), np.float32)
        np.add.at(total_sum, (d, s), r)
        np.add.at(total_cnt, (d, s), 1.0)
        own_sum = np.zeros((self.n_poly, self.n_days, 3), np.float32)
        own_cnt = np.zeros((self.n_poly, self.n_days, 3), np.float32)
        np.add.at(own_sum, (p, d, s), r)
        np.add.at(own_cnt, (p, d, s), 1.0)
        t = self.t
        rows = t.day0[:, None] + np.arange(SEASON_LEN)[None, :] - self.dmin
        rows = np.clip(rows, 0, self.n_days - 1)
        sm = total_sum[rows] - own_sum[t.poly_index[:, None], rows]
        cn = total_cnt[rows] - own_cnt[t.poly_index[:, None], rows]
        mean = np.where(cn > 0, sm / np.maximum(cn, 1), 0.0)
        return np.concatenate([mean, np.minimum(cn, 20) / 20.0], axis=2).astype(np.float32)

    def climatology(self, visible_days: np.ndarray) -> np.ndarray:
        """(n_samples, 213, 2): норма полигона по другим годам (±7 дней) и её наличие."""
        t = self.t
        v = np.where(visible_days, self.h, 0.0)
        c = visible_days.astype(np.float32)
        kernel = np.ones(2 * CLIM_HALF + 1, np.float32)
        v_s = np.apply_along_axis(lambda a: np.convolve(a, kernel, "same"), 1, v)
        c_s = np.apply_along_axis(lambda a: np.convolve(a, kernel, "same"), 1, c)
        poly_v = np.zeros((self.n_poly, SEASON_LEN), np.float32)
        poly_c = np.zeros((self.n_poly, SEASON_LEN), np.float32)
        np.add.at(poly_v, t.poly_index, v_s)
        np.add.at(poly_c, t.poly_index, c_s)
        other_v = poly_v[t.poly_index] - v_s
        other_c = poly_c[t.poly_index] - c_s
        mean = np.where(other_c > 0, other_v / np.maximum(other_c, 1e-6), 0.0)
        return np.stack([mean, np.minimum(other_c, 30) / 30.0], axis=2).astype(np.float32)


def assemble(t: SeasonTensors, ep: EpochInputs, visible_obs: np.ndarray, qry_days: np.ndarray) -> dict:
    """Собирает входные каналы и цели для заданной видимости точек obs и матрицы дней-запросов.

    visible_obs — какие точки obs видны (контекст), длина len(obs); qry_days — (n, 213) дни-цели.
    """
    idx_ok = t.idx >= 0
    vis_days = np.zeros(t.known.shape, bool)
    vis_days[idx_ok] = visible_obs[t.idx[idx_ok]]
    pres = t.present * vis_days[:, :, None]
    doy = (t.doy0[:, None] + np.arange(SEASON_LEN)[None, :]) / 365.0
    crop = np.zeros((len(t.pid), SEASON_LEN, N_CROPS), np.float32)
    crop[np.arange(len(t.pid)), :, np.clip(t.crop, 0, N_CROPS - 1)] = 1.0
    channels = np.concatenate([
        t.vals * pres, pres, t.weather * t.weather_ok[:, :, None], t.weather_ok[:, :, None],
        ep.climatology(vis_days), ep.day_effects(visible_obs),
        np.sin(2 * np.pi * doy)[:, :, None], np.cos(2 * np.pi * doy)[:, :, None],
        crop, qry_days[:, :, None].astype(np.float32),
    ], axis=2).astype(np.float32)
    return {"x": channels, "y": t.y, "query": qry_days}


N_CHANNELS = len(INDEX_COLS) * 2 + 3 + 2 + 6 + 2 + N_CROPS + 1
