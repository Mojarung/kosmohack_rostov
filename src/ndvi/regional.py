"""Региональный контекст: что творилось в этот день на других полях.

Облачный фронт, дымка, снег накрывают не одно поле, а всю область сразу. Поэтому если
в дату гэпа соседние поля просели относительно своих собственных кривых сезона, скрытая
точка нашего поля почти наверняка просела тоже. Это единственный способ подобраться
к облачным выбросам — 4 % точек, которые дают 41 % ошибки и по собственным соседям поля
не предсказуемы.

Координат у нас нет, но соседство восстанавливается тремя способами, от грубого к точному:

1. **один виток съёмки** — совпадающие даты наблюдений (полоса шириной в сотни километров);
2. **одна ячейка ERA5** — у полей в одной ячейке реанализа буквально одинаковый суточный ряд
   температуры (73 % тестовых полигонов имеют такого близнеца); ячейка — это ~10 км;
3. **близнец по динамике NDVI** — поля с сильно коррелированной многолетней траекторией
   (та же культура, тот же севооборот, те же соседи по ландшафту).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ndvi.data import NDVI_MAX, NDVI_MIN
from ndvi.sensors import SAME_ORBIT_JACCARD, AcquisitionCalendar, SensorHarmonizer
from ndvi.smoothing import robust_local_linear

SMOOTH_BW = 20.0
DROP_THRESHOLD = -0.08      # остаток ниже этого — «поле просело» (типичный облачный провал)
MAX_WINDOW_DAYS = 2         # если в сам день никто не снимался, смотрим ±1..2 дня

SAME_CELL_CORR = 0.9999     # корреляция суточной температуры, при которой поля в одной ячейке ERA5
TWIN_MIN_OVERLAP = 60       # общих дат наблюдений, чтобы верить корреляции траекторий NDVI

EMPTY = {
    "reg_n": 0.0, "reg_n_close": 0.0, "reg_n_same": 0.0, "reg_n_cell": 0.0, "reg_window_dt": np.nan,
    "reg_resid_med": np.nan, "reg_resid_close": np.nan, "reg_resid_same": np.nan,
    "reg_resid_cell": np.nan, "reg_resid_twin": np.nan,
    "reg_resid_wmean": np.nan, "reg_resid_min": np.nan, "reg_share_drop": np.nan,
    "reg_level_close": np.nan, "reg_level_med": np.nan, "reg_level_cell": np.nan, "reg_level_twin": np.nan,
    "twin_best_corr": np.nan,
}


class RegionalContext:
    """Остатки «наблюдение минус собственная кривая» по всем полям, разложенные по дням."""

    def __init__(self, calendar: AcquisitionCalendar, bandwidth: float = SMOOTH_BW):
        self.calendar = calendar
        self.bandwidth = bandwidth
        self._by_day: dict = {}
        n = len(calendar.pids)
        self._cell = np.zeros((n, n), dtype=bool)        # одна ячейка ERA5
        self._twin = np.zeros((n, n), dtype=np.float32)  # корреляция траекторий NDVI

    def _fit_neighbourhoods(self, obs: pd.DataFrame, context: pd.DataFrame | None) -> None:
        """Матрицы соседства: одна ячейка ERA5 и близнецы по динамике NDVI."""
        idx = self.calendar._index
        n = len(self.calendar.pids)
        src = context if context is not None else obs
        w = src.dropna(subset=["era5_temp_c"])
        if len(w):
            piv = w.pivot_table(index="date", columns="anon_polygon_id", values="era5_temp_c")
            corr = piv.corr(min_periods=100).to_numpy()
            cols = list(piv.columns)
            for i, a in enumerate(cols):
                ia = idx.get(a, -1)
                if ia < 0:
                    continue
                for j in range(i + 1, len(cols)):
                    ib = idx.get(cols[j], -1)
                    if ib >= 0 and np.isfinite(corr[i, j]) and corr[i, j] > SAME_CELL_CORR:
                        self._cell[ia, ib] = self._cell[ib, ia] = True
        piv = obs.pivot_table(index="date", columns="anon_polygon_id", values="_val_s2")
        corr = piv.corr(min_periods=TWIN_MIN_OVERLAP).to_numpy()
        cols = list(piv.columns)
        for i, a in enumerate(cols):
            ia = idx.get(a, -1)
            if ia < 0:
                continue
            for j, b in enumerate(cols):
                ib = idx.get(b, -1)
                if ib >= 0 and ib != ia and np.isfinite(corr[i, j]):
                    self._twin[ia, ib] = corr[i, j]

    def fit(self, observed: pd.DataFrame, harm: SensorHarmonizer,
            context: pd.DataFrame | None = None) -> "RegionalContext":
        obs = observed[observed.src != "none"].copy()
        obs["_days"] = (obs.date - pd.Timestamp("2010-01-01")).dt.days.astype(int)
        obs["_val_s2"] = harm.to_s2(obs.primary_ndvi.clip(NDVI_MIN, NDVI_MAX).to_numpy(),
                                    obs.src.to_numpy(), obs.year.to_numpy(), obs.anon_polygon_id.to_numpy())
        obs["_resid"] = np.nan

        # остаток каждой точки относительно кривой сезона своего поля, без самой точки
        for (_, _), idx in obs.groupby(["anon_polygon_id", "year"]).indices.items():
            if idx.size < 4:
                continue
            sub = obs.iloc[idx]
            x = sub["_days"].to_numpy(float)
            y = sub["_val_s2"].to_numpy(float)
            fit = robust_local_linear(x, y, x, self.bandwidth, leave_one_out=True)
            obs.iloc[idx, obs.columns.get_loc("_resid")] = y - fit

        self._fit_neighbourhoods(obs, context)
        obs = obs[obs["_resid"].notna()]
        pid_index = self.calendar._index
        for day, grp in obs.groupby("date"):
            self._by_day[day] = (
                np.array([pid_index.get(p, -1) for p in grp.anon_polygon_id], dtype=np.int32),
                grp["_resid"].to_numpy(float),
                grp["_val_s2"].to_numpy(float),
                grp.src.to_numpy(),
            )
        return self

    def features(self, pid: str, date, hidden_src: str) -> dict:
        own = self.calendar._index.get(pid, -1)
        key = "landsat" if hidden_src == "landsat" else "s2"
        sim_row = self.calendar._sim.get(key)
        for dt in range(MAX_WINDOW_DAYS + 1):
            days = [date] if dt == 0 else [date - pd.Timedelta(days=dt), date + pd.Timedelta(days=dt)]
            parts = [self._by_day[d] for d in days if d in self._by_day]
            if not parts:
                continue
            idx = np.concatenate([p[0] for p in parts])
            resid = np.concatenate([p[1] for p in parts])
            val = np.concatenate([p[2] for p in parts])
            src = np.concatenate([p[3] for p in parts])
            keep = idx != own
            if not keep.any():
                continue
            idx, resid, val, src = idx[keep], resid[keep], val[keep], src[keep]
            if own >= 0 and sim_row is not None:
                sim = np.where(idx >= 0, sim_row[own, np.maximum(idx, 0)], 0.0)
            else:
                sim = np.zeros(idx.size)
            close = sim > SAME_ORBIT_JACCARD
            same = src == hidden_src
            valid = idx >= 0
            cell = np.zeros(idx.size, dtype=bool)
            twin = np.zeros(idx.size)
            if own >= 0:
                cell[valid] = self._cell[own, idx[valid]]
                twin[valid] = self._twin[own, idx[valid]]
            # близнецы по динамике: вес растёт с корреляцией, ниже 0.6 не считаем
            tw = np.clip(twin - 0.6, 0, None) ** 2
            w = sim + 0.05  # чужие витки тоже что-то значат, но заметно меньше
            out = {
                "reg_n": float(idx.size),
                "reg_n_close": float(close.sum()),
                "reg_n_same": float(same.sum()),
                "reg_n_cell": float(cell.sum()),
                "reg_window_dt": float(dt),
                "reg_resid_med": float(np.median(resid)),
                "reg_resid_close": float(np.median(resid[close])) if close.any() else np.nan,
                "reg_resid_same": float(np.median(resid[same])) if same.any() else np.nan,
                "reg_resid_cell": float(np.median(resid[cell])) if cell.any() else np.nan,
                "reg_resid_twin": float((tw * resid).sum() / tw.sum()) if tw.sum() > 0 else np.nan,
                "reg_resid_wmean": float((w * resid).sum() / w.sum()),
                "reg_resid_min": float(resid.min()),
                "reg_share_drop": float((resid < DROP_THRESHOLD).mean()),
                "reg_level_close": float(np.median(val[close])) if close.any() else np.nan,
                "reg_level_med": float(np.median(val)),
                "reg_level_cell": float(np.median(val[cell])) if cell.any() else np.nan,
                "reg_level_twin": float((tw * val).sum() / tw.sum()) if tw.sum() > 0 else np.nan,
                "twin_best_corr": float(twin.max()) if idx.size else np.nan,
            }
            return out
        return dict(EMPTY)
