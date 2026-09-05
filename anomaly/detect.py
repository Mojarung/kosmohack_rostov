"""Детекция негативных аномальных периодов: Z-score гладкой кривой относительно нормы, эпизоды, фенометрики.

Эпизод — непрерывный период дней сезона, где Z < −1 и кривая надёжна; короткие (< 7 дней) и не подтверждённые
наблюдениями эпизоды отбрасываются: одиночные провалы — это облака и смена сенсора, а не угнетение.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from anomaly.config import (
    EPISODE_MERGE_GAP,
    EPISODE_MIN_DAYS,
    EPISODE_MIN_OBS,
    MAIN_SEASON_DOY,
    MIN_CURVE_WEIGHT,
    PHASES,
    PROLONGED_DAYS,
    PROLONGED_MEAN_Z,
    STRONG_Z,
    SUSTAINED_DAYS,
    SUSTAINED_MEAN_Z,
    Z_CRITICAL,
    Z_DEPRESSION,
)


def z_series(curve: pd.DataFrame, norm: pd.DataFrame) -> pd.DataFrame:
    """Ежедневный Z-score кривой сезона относительно нормы (NaN там, где кривая или норма ненадёжны)."""
    df = curve.merge(norm, on="doy", how="left")
    ok = (df["weight"] >= MIN_CURVE_WEIGHT) & df["norm_mean"].notna() & df["norm_std"].notna()
    z = np.where(ok, (df["value"] - df["norm_mean"]) / df["norm_std"], np.nan)
    return df.assign(z=z, deficit=np.where(ok, df["value"] - df["norm_mean"], np.nan))


def _runs(flag: np.ndarray) -> list[tuple[int, int]]:
    """Индексы начала и конца (включительно) непрерывных участков True."""
    runs, start = [], None
    for i, f in enumerate(flag):
        if f and start is None:
            start = i
        if not f and start is not None:
            runs.append((start, i - 1))
            start = None
    if start is not None:
        runs.append((start, len(flag) - 1))
    return runs


def _merge_runs(runs: list[tuple[int, int]], gap: int) -> list[tuple[int, int]]:
    """Сливает участки, разделённые не более чем gap дней."""
    merged = []
    for a, b in runs:
        if merged and a - merged[-1][1] - 1 <= gap:
            merged[-1] = (merged[-1][0], b)
        else:
            merged.append((a, b))
    return merged


def phase_of(doy: int) -> str:
    """Фаза сезона по дню года."""
    for name, (lo, hi) in PHASES.items():
        if lo <= doy <= hi:
            return name
    return "вне сезона"


def is_episode(days: int, mean_z: float, min_z: float, n_obs: int, strict: bool = True) -> bool:
    """Критерий ТЗ: устойчивое (длинное и в среднем глубокое) и/или сильное отклонение, подтверждённое наблюдениями."""
    if days < EPISODE_MIN_DAYS or n_obs < EPISODE_MIN_OBS:
        return False
    if not strict:
        return True
    sustained = days >= SUSTAINED_DAYS and mean_z <= SUSTAINED_MEAN_Z
    strong = min_z <= STRONG_Z
    prolonged = days >= PROLONGED_DAYS and mean_z <= PROLONGED_MEAN_Z   # долгое мягкое угнетение целого сезона
    return sustained or strong or prolonged


def find_episodes(zs: pd.DataFrame, obs_days: np.ndarray, strict: bool = True) -> list[dict]:
    """Эпизоды угнетения по ежедневному Z-score; obs_days — day_num наблюдений сезона (без артефактов)."""
    flag = (zs["z"] < Z_DEPRESSION).to_numpy()
    episodes = []
    for a, b in _merge_runs(_runs(flag), EPISODE_MERGE_GAP):
        seg = zs.iloc[a:b + 1]
        n_obs = int(((obs_days >= seg["day_num"].iloc[0]) & (obs_days <= seg["day_num"].iloc[-1])).sum())
        if not is_episode(len(seg), float(seg["z"].mean()), float(seg["z"].min()), n_obs, strict):
            continue
        i_min = int(seg["z"].idxmin())
        episodes.append({
            "start": seg["date"].iloc[0].date().isoformat(), "end": seg["date"].iloc[-1].date().isoformat(),
            "days": int(len(seg)), "n_obs": n_obs, "min_z": float(seg["z"].min()), "mean_z": float(seg["z"].mean()),
            "critical_days": int((seg["z"] < Z_CRITICAL).sum()), "max_deficit": float(seg["deficit"].min()),
            "mean_deficit": float(seg["deficit"].mean()), "worst_date": zs.loc[i_min, "date"].date().isoformat(),
            "ndvi_at_worst": float(zs.loc[i_min, "value"]), "norm_at_worst": float(zs.loc[i_min, "norm_mean"]),
            "phase": phase_of(int(seg["doy"].iloc[0])), "phase_end": phase_of(int(seg["doy"].iloc[-1])),
            "start_doy": int(seg["doy"].iloc[0]), "end_doy": int(seg["doy"].iloc[-1]),
        })
    return episodes


def phenology(curve: pd.DataFrame) -> dict:
    """Фенометрики сезона по надёжной части кривой: пик, его дата, старт роста, спад, амплитуда, интеграл."""
    ok = (curve["weight"] >= MIN_CURVE_WEIGHT) & curve["value"].notna()
    if ok.sum() < 30:
        return {"valid": False}
    c = curve.loc[ok]
    v = c["value"].to_numpy()
    doy = c["doy"].to_numpy()
    main = doy <= MAIN_SEASON_DOY[1]          # пик ищем в апреле–августе: осенью всходит следующая культура
    if main.sum() < 20:
        return {"valid": False}
    i_peak = int(np.flatnonzero(main)[np.argmax(v[main])])
    peak, base = float(v[i_peak]), float(np.nanpercentile(v, 5))   # база — 5-й перцентиль, устойчив к краевым провалам
    amp = peak - base
    thr = base + 0.2 * amp
    above = doy[(v >= thr) & main]
    sos = int(above.min()) if len(above) else int(doy[main][0])
    after = doy[(doy > doy[i_peak]) & (v <= peak - 0.5 * amp)]
    half_down = int(after.min()) if len(after) else None
    return {"valid": True, "peak": peak, "peak_doy": int(doy[i_peak]), "base": base, "amplitude": amp,
            "sos_doy": sos, "half_decline_doy": half_down, "integral": float(np.trapezoid(v, doy)),
            "covered_days": int(ok.sum())}


def phenology_deviation(pheno: dict, norm_pheno: dict) -> dict:
    """Отклонения фенометрик сезона от нормы (норма — фенометрики кривой нормы)."""
    if not pheno.get("valid") or not norm_pheno.get("valid"):
        return {}
    out = {"peak_ratio": pheno["peak"] / max(norm_pheno["peak"], 1e-6),
           "peak_shift_days": pheno["peak_doy"] - norm_pheno["peak_doy"],
           "sos_shift_days": pheno["sos_doy"] - norm_pheno["sos_doy"],
           "integral_ratio": pheno["integral"] / max(norm_pheno["integral"], 1e-6)}
    if pheno.get("half_decline_doy") and norm_pheno.get("half_decline_doy"):
        out["decline_shift_days"] = pheno["half_decline_doy"] - norm_pheno["half_decline_doy"]
    return out


def norm_phenology(norm: pd.DataFrame) -> dict:
    """Фенометрики кривой нормы (как у обычной кривой, weight = 1 там, где норма есть)."""
    c = norm.assign(value=norm["norm_mean"], weight=norm["norm_mean"].notna().astype(float))
    return phenology(c)
