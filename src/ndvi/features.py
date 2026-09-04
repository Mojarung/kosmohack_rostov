"""Сборка признаков для точки-гэпа.

Вход всегда один и тот же: «видимая» часть данных (train, либо test без скрытых точек)
плюс таблица целевых строк. Никакой информации из самой скрытой точки не используется —
это гарантирует, что валидация честная, а inference устроен ровно так же.

Состав признаков:
* соседи слева/справа (до 3 с каждой стороны): значение в шкале S2, лаг в днях, сенсор;
* угаданный сенсор скрытой точки и его смещение (главный рычаг по метрике), плюс сырые
  меры календаря съёмки — сколько соседних полей снимал каждый сенсор в этот день;
* робастно сглаженная кривая сезона в шкале S2 (устойчива к облачным провалам);
* собственная климатическая норма полигона (без текущего года) и отклонение соседей от неё;
* погода ERA5 около даты и накопленные осадки/жара за 30 и 60 дней;
* региональный контекст дня: насколько просели относительно своих кривых другие поля,
  снятые в тот же день (облачный фронт накрывает всех сразу);
* контекст: культура, doy, длина истории полигона, плотность ряда, длина цепочки гэпов.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ndvi.climatology import Climatology
from ndvi.data import NDVI_MAX, NDVI_MIN
from ndvi.regional import RegionalContext
from ndvi.sensors import DEFAULT_OFFSETS, AcquisitionCalendar, SensorHarmonizer, infer_hidden_sensors
from ndvi.smoothing import robust_local_linear

K_NEIGHBORS = 3
SMOOTH_BW = 20.0
SMOOTH_BW_WIDE = 45.0


def _neighbor_block(obs_days, obs_val_s2, obs_src_off, t_days, k=K_NEIGHBORS):
    """Возвращает (значения S2, лаги, смещения сенсоров) для k соседей слева и справа."""
    n = obs_days.size
    pos = np.searchsorted(obs_days, t_days)
    prev_v = np.full((t_days.size, k), np.nan)
    prev_d = np.full((t_days.size, k), np.nan)
    prev_o = np.full((t_days.size, k), np.nan)
    next_v = np.full((t_days.size, k), np.nan)
    next_d = np.full((t_days.size, k), np.nan)
    next_o = np.full((t_days.size, k), np.nan)
    for j in range(k):
        ip = pos - 1 - j
        ok = ip >= 0
        if ok.any():
            idx = ip[ok]
            prev_v[ok, j] = obs_val_s2[idx]
            prev_d[ok, j] = t_days[ok] - obs_days[idx]
            prev_o[ok, j] = obs_src_off[idx]
        inx = pos + j
        ok = inx < n
        if ok.any():
            idx = inx[ok]
            next_v[ok, j] = obs_val_s2[idx]
            next_d[ok, j] = obs_days[idx] - t_days[ok]
            next_o[ok, j] = obs_src_off[idx]
    return (prev_v, prev_d, prev_o), (next_v, next_d, next_o)


def _weather_block(w_days, w_temp, w_precip, t_days):
    """Погода на дату гэпа и накопленные показатели за предшествующие окна."""
    out = {}
    if w_days.size == 0:
        for key in ("era5_temp_c", "era5_precip_mm", "precip_30d", "precip_60_30d",
                    "temp_15d", "hot_days_30d", "dry_days_30d"):
            out[key] = np.full(t_days.size, np.nan)
        return out
    out["era5_temp_c"] = np.interp(t_days, w_days, w_temp, left=np.nan, right=np.nan)
    out["era5_precip_mm"] = np.interp(t_days, w_days, w_precip, left=np.nan, right=np.nan)
    d = t_days[:, None] - w_days[None, :]
    win30 = (d > 0) & (d <= 30)
    win60 = (d > 30) & (d <= 60)
    win15 = (d > 0) & (d <= 15)
    with np.errstate(invalid="ignore"):
        out["precip_30d"] = (win30 * np.nan_to_num(w_precip)[None, :]).sum(axis=1)
        out["precip_60_30d"] = (win60 * np.nan_to_num(w_precip)[None, :]).sum(axis=1)
        cnt15 = win15.sum(axis=1)
        out["temp_15d"] = np.where(cnt15 > 0, (win15 * np.nan_to_num(w_temp)[None, :]).sum(axis=1) / np.maximum(cnt15, 1), np.nan)
        out["hot_days_30d"] = (win30 & (w_temp[None, :] > 30)).sum(axis=1).astype(float)
        out["dry_days_30d"] = (win30 & (w_precip[None, :] < 0.1)).sum(axis=1).astype(float)
    # окна без данных помечаем как пропуск, а не как ноль осадков
    has30 = win30.any(axis=1)
    for key in ("precip_30d", "hot_days_30d", "dry_days_30d"):
        out[key] = np.where(has30, out[key], np.nan)
    out["precip_60_30d"] = np.where(win60.any(axis=1), out["precip_60_30d"], np.nan)
    return out


def build_features(context: pd.DataFrame, targets: pd.DataFrame,
                   offsets: dict[str, float] | None = None,
                   clim: Climatology | None = None,
                   exclude_current_year_in_clim: bool = True,
                   calendar: AcquisitionCalendar | None = None) -> pd.DataFrame:
    """Собирает таблицу признаков для строк ``targets``.

    ``context`` — все строки, доступные модели (в строках-гэпах динамические колонки уже NaN).
    """
    offsets = offsets or DEFAULT_OFFSETS
    observed = context[context.primary_ndvi.notna()]
    if calendar is None:
        calendar = AcquisitionCalendar().fit(observed)
    # линейная гармонизация сенсоров с поправками по году и полигону
    harm = SensorHarmonizer().fit(observed)

    obs = observed.copy()
    obs["_days"] = (obs.date - pd.Timestamp("2010-01-01")).dt.days.astype(int)
    # значение соседа клиппим: выброс -0.47 в июле не должен утаскивать интерполяцию
    obs["_val_s2"] = harm.to_s2(obs.primary_ndvi.clip(NDVI_MIN, NDVI_MAX).to_numpy(),
                                obs.src.to_numpy(), obs.year.to_numpy(), obs.anon_polygon_id.to_numpy())
    obs["_off"] = obs.primary_ndvi.clip(NDVI_MIN, NDVI_MAX) - obs["_val_s2"]

    # норму считаем в единой шкале S2 — иначе она склеена из сенсоров с разными смещениями
    clim = Climatology().fit(obs.assign(primary_ndvi=obs["_val_s2"]))

    tg = targets.copy()
    tg["_days"] = (tg.date - pd.Timestamp("2010-01-01")).dt.days.astype(int)
    tg["hidden_src"] = infer_hidden_sensors(observed, tg, calendar)
    # смещение угаданного сенсора на типичном уровне — как признак; точный пересчёт ниже
    tg["hidden_off"] = tg.hidden_src.map(harm.offsets()).fillna(0.0)

    weather = context[context.era5_temp_c.notna()].copy()
    weather["_days"] = (weather.date - pd.Timestamp("2010-01-01")).dt.days.astype(int)

    obs_by_pid = {k: v.sort_values("_days") for k, v in obs.groupby("anon_polygon_id")}
    w_by_pid = {k: v.sort_values("_days") for k, v in weather.groupby("anon_polygon_id")}
    rows_per_season = context.groupby(["anon_polygon_id", "year"]).size()

    parts = []
    for pid, tg_p in tg.groupby("anon_polygon_id"):
        o = obs_by_pid.get(pid)
        t_days = tg_p["_days"].to_numpy()
        n_t = t_days.size
        f = pd.DataFrame(index=tg_p.index)

        if o is None or len(o) == 0:
            od, ov, oo = np.array([]), np.array([]), np.array([])
        else:
            od = o["_days"].to_numpy()
            ov = o["_val_s2"].to_numpy()
            oo = o["_off"].to_numpy()

        (pv, pd_, po), (nv, nd, no) = _neighbor_block(od, ov, oo, t_days)
        for j in range(K_NEIGHBORS):
            f[f"prev{j+1}_val"] = pv[:, j]
            f[f"prev{j+1}_dt"] = pd_[:, j]
            f[f"prev{j+1}_off"] = po[:, j]
            f[f"next{j+1}_val"] = nv[:, j]
            f[f"next{j+1}_dt"] = nd[:, j]
            f[f"next{j+1}_off"] = no[:, j]

        # линейная интерполяция в шкале S2 по ближайшим соседям
        w = nd[:, 0] / (pd_[:, 0] + nd[:, 0])
        both = np.isfinite(pv[:, 0]) & np.isfinite(nv[:, 0])
        lin = np.where(both, pv[:, 0] * w + nv[:, 0] * (1 - w),
                       np.nanmean(np.stack([pv[:, 0], nv[:, 0]]), axis=0))
        f["interp_s2"] = lin
        f["mean2_s2"] = np.nanmean(np.stack([pv[:, 0], nv[:, 0]]), axis=0)
        f["mean6_s2"] = np.nanmean(np.concatenate([pv, nv], axis=1), axis=1)
        f["median6_s2"] = np.nanmedian(np.concatenate([pv, nv], axis=1), axis=1)
        f["neighbor_spread"] = np.nanstd(np.concatenate([pv, nv], axis=1), axis=1)
        f["slope_local"] = (nv[:, 0] - pv[:, 0]) / np.maximum(pd_[:, 0] + nd[:, 0], 1)
        f["dt_min"] = np.nanmin(np.stack([pd_[:, 0], nd[:, 0]]), axis=0)
        f["dt_sum"] = pd_[:, 0] + nd[:, 0]
        f["one_sided"] = (~both).astype(float)

        # робастная кривая сезона в шкале S2 (по своему полигон-сезону)
        sm = np.full(n_t, np.nan)
        sm_wide = np.full(n_t, np.nan)
        if od.size:
            years_t = tg_p.year.to_numpy()
            o_years = o.year.to_numpy()
            for yr in np.unique(years_t):
                mt = years_t == yr
                mo = o_years == yr
                if mo.sum() >= 3:
                    sm[mt] = robust_local_linear(od[mo], ov[mo], t_days[mt], SMOOTH_BW)
                    sm_wide[mt] = robust_local_linear(od[mo], ov[mo], t_days[mt], SMOOTH_BW_WIDE)
        f["smooth_s2"] = sm
        f["smooth_wide_s2"] = sm_wide
        f["smooth_minus_interp"] = sm - lin

        # погода
        wp = w_by_pid.get(pid)
        if wp is None or len(wp) == 0:
            wb = _weather_block(np.array([]), np.array([]), np.array([]), t_days)
        else:
            wb = _weather_block(wp["_days"].to_numpy(), wp.era5_temp_c.to_numpy(float),
                                wp.era5_precip_mm.to_numpy(float), t_days)
        for k_, v_ in wb.items():
            f[k_] = v_

        # длина цепочки гэпов рядом: соседний гэп мешает опереться на соседа
        dd = np.abs(t_days[:, None] - t_days[None, :])
        f["gaps_near"] = ((dd > 0) & (dd <= 3)).sum(axis=1).astype(float)

        f["n_obs_polygon"] = float(od.size)
        f["n_years_polygon"] = float(o.year.nunique()) if o is not None and len(o) else 0.0
        season_rows = rows_per_season.reindex(
            pd.MultiIndex.from_arrays([tg_p.anon_polygon_id, tg_p.year])).to_numpy(float)
        f["rows_in_season"] = season_rows
        parts.append(f)

    feats = pd.concat(parts).reindex(tg.index)

    # климатическая норма (уже в шкале S2) и отклонения от неё
    cl = clim.lookup_frame(tg, exclude_current_year=exclude_current_year_in_clim)
    feats = feats.join(cl)
    feats["clim_mean_s2"] = feats.clim_mean
    feats["interp_minus_clim"] = feats.interp_s2 - feats.clim_mean_s2
    feats["smooth_minus_clim"] = feats.smooth_s2 - feats.clim_mean_s2

    # региональный контекст дня: остатки других полей относительно их собственных кривых
    regional = RegionalContext(calendar).fit(observed, harm, context=context)
    reg = pd.DataFrame([regional.features(pid, dt, hs)
                        for pid, dt, hs in zip(tg.anon_polygon_id.values, tg.date, tg.hidden_src.values)],
                       index=tg.index)
    feats = feats.join(reg)
    # то же в шкале нашего поля: региональный остаток, приложенный к нашей опоре
    # каскад: ячейка ERA5 → близнецы → один виток → все; первый доступный уровень
    feats["interp_plus_reg"] = feats.interp_s2 + (feats.reg_resid_cell.fillna(feats.reg_resid_twin)
                                                  .fillna(feats.reg_resid_close)
                                                  .fillna(feats.reg_resid_wmean).fillna(0.0))

    # календарь съёмки: сколько соседних полей снимал каждый сенсор в этот день.
    # Модель видит не только вывод «какой сенсор», но и насколько он уверенный.
    cal_rows = [calendar.scores(pid, dt) for pid, dt in zip(tg.anon_polygon_id.values, tg.date)]
    cal = pd.DataFrame(cal_rows, index=tg.index)
    feats = feats.join(cal)
    close_total = cal[[f"close_{s}" for s in ("s2", "landsat", "modis")]].sum(axis=1)
    feats["cal_confidence"] = np.where(
        close_total > 0,
        cal[[f"close_{s}" for s in ("s2", "landsat", "modis")]].max(axis=1) / close_total.replace(0, np.nan),
        np.nan)

    # контекст точки
    feats["doy"] = tg.doy.astype(float).values
    feats["year"] = tg.year.astype(float).values
    feats["hidden_off"] = tg.hidden_off.values
    for s in ("s2", "landsat", "modis"):
        feats[f"hidden_is_{s}"] = (tg.hidden_src == s).astype(float).values
    feats["crop_type"] = tg.crop_type.values

    # базовые прогнозы, от которых модель учит поправку: из шкалы S2 в шкалу угаданного сенсора
    src, yrs, pids = tg.hidden_src.to_numpy(), tg.year.to_numpy(), tg.anon_polygon_id.to_numpy()
    for name, col in (("base_interp", "interp_s2"), ("base_smooth", "smooth_s2"),
                      ("base_mean2", "mean2_s2"), ("base_interp_reg", "interp_plus_reg")):
        feats[name] = harm.from_s2(feats[col].to_numpy(), src, yrs, pids)

    meta = tg[["anon_polygon_id", "date", "hidden_src"]]
    return meta.join(feats)


NUMERIC_FEATURES = None  # заполняется в models/gbm.py по факту сборки
