"""Протокол валидации, повторяющий то, как построен приватный тест.

Организаторы скрыли 15 % наблюдений и затёрли в этих строках все динамические колонки.
Мы делаем то же самое на train. Дополнительно:

* **group-split по полигонам** — в test 85 % точек на полигонах, которых нет в train;
* **hold-out сезона** (по умолчанию 2024) — 30 % точек test в сезоне 2025, которого нет в train;
* разрезы отчёта: по сенсору скрытой точки, по расстоянию до соседа, по длине истории
  полигона, по годам. Общий RMSE прячет то, что MODIS-точки в полтора раза хуже остальных.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ndvi.data import DYNAMIC_COLS

DEFAULT_SEED = 42
HOLDOUT_YEAR = 2024


@dataclass
class MaskedSplit:
    """Результат маскирования: видимые данные и правда по скрытым точкам."""

    context: pd.DataFrame   # данные, доступные модели (гэпы затёрты)
    targets: pd.DataFrame   # строки-гэпы (только статические колонки)
    truth: pd.DataFrame     # anon_polygon_id, date, primary_ndvi, src_true


def make_masked(train: pd.DataFrame, frac: float = 0.15, seed: int = DEFAULT_SEED) -> MaskedSplit:
    """Скрывает долю ``frac`` наблюдений так же, как это сделано в приватном тесте."""
    rng = np.random.default_rng(seed)
    obs_idx = train.index[train.primary_ndvi.notna()].to_numpy()
    hide = rng.choice(obs_idx, size=int(frac * obs_idx.size), replace=False)
    hide.sort()

    truth = train.loc[hide, ["anon_polygon_id", "date", "primary_ndvi", "src"]].rename(
        columns={"src": "src_true"}).reset_index(drop=True)

    ctx = train.copy()
    cols = [c for c in DYNAMIC_COLS if c in ctx.columns]
    ctx.loc[hide, cols] = np.nan
    ctx.loc[hide, "src"] = "none"
    if "status" in ctx.columns:
        ctx.loc[hide, "status"] = np.nan
    ctx["is_gap"] = False
    ctx.loc[hide, "is_gap"] = True

    targets = ctx.loc[hide, ["anon_polygon_id", "date", "year", "doy", "crop_type"]].copy()
    return MaskedSplit(context=ctx, targets=targets, truth=truth)


def rmse(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, float)
    y_pred = np.asarray(y_pred, float)
    m = np.isfinite(y_true) & np.isfinite(y_pred)
    if m.sum() == 0:
        return float("nan")
    return float(np.sqrt(np.mean((y_true[m] - y_pred[m]) ** 2)))


def gap_score(rmse_value: float) -> float:
    """GapScore организаторов: 30 баллов при RMSE 0, 0 баллов при RMSE >= 0.10."""
    return float(max(0.0, 30.0 * (1.0 - rmse_value / 0.10)))


def segment_report(df: pd.DataFrame, pred_cols: list[str], target_col: str = "y_true") -> pd.DataFrame:
    """Таблица RMSE по разрезам. ``df`` должен содержать признаки, правду и предсказания."""
    segs: dict[str, pd.Series] = {"всего": pd.Series(True, index=df.index)}
    if "src_true" in df:
        for s in ("s2", "landsat", "modis"):
            segs[f"сенсор {s}"] = df.src_true == s
    if "dt_min" in df:
        segs["сосед <=1 дн"] = df.dt_min <= 1
        segs["сосед 2-5 дн"] = (df.dt_min > 1) & (df.dt_min <= 5)
        segs["сосед >5 дн"] = df.dt_min > 5
    if "one_sided" in df:
        segs["край сезона"] = df.one_sided > 0
    if "n_years_polygon" in df:
        segs["история >=5 лет"] = df.n_years_polygon >= 5
        segs["история <5 лет"] = df.n_years_polygon < 5
    if "year" in df:
        segs[f"сезон {HOLDOUT_YEAR}"] = df.year == HOLDOUT_YEAR

    rows = []
    for name, mask in segs.items():
        sub = df[mask]
        if len(sub) == 0:
            continue
        rec = {"сегмент": name, "n": len(sub)}
        for c in pred_cols:
            rec[c] = round(rmse(sub[target_col], sub[c]), 4)
        rows.append(rec)
    return pd.DataFrame(rows)


def group_folds(groups: pd.Series, n_splits: int = 5, seed: int = DEFAULT_SEED):
    """Разбиение по полигонам: полигон целиком либо в обучении, либо в валидации."""
    uniq = np.array(sorted(groups.unique()))
    rng = np.random.default_rng(seed)
    rng.shuffle(uniq)
    buckets = np.array_split(uniq, n_splits)
    g = groups.to_numpy()
    for b in buckets:
        val = np.isin(g, b)
        yield ~val, val


def apply_cold_start(context: pd.DataFrame, targets: pd.DataFrame, frac: float,
                     seed: int = DEFAULT_SEED):
    """Эмулирует «холодный старт»: у части полигонов остаётся ровно один сезон без истории.

    В test есть полигоны, представленные только сезоном 2025 — у них нет ни собственной
    климатической нормы, ни прошлых лет. Чтобы измерить качество на этом сегменте, у
    выбранных полигонов оставляем один случайный сезон, а гэпы вне него выбрасываем.

    Возвращает ``(context, targets, cold_pids)``.
    """
    if frac <= 0:
        return context, targets, set()
    rng = np.random.default_rng(seed + 1)
    pids = np.array(sorted(context.anon_polygon_id.unique()))
    cold = set(rng.choice(pids, size=max(1, int(frac * pids.size)), replace=False))

    gap_years = targets.groupby("anon_polygon_id").year.agg(lambda s: sorted(set(s)))
    keep_year = {}
    for pid in list(cold):
        yrs = gap_years.get(pid)
        if not yrs:
            cold.discard(pid)
            continue
        keep_year[pid] = int(rng.choice(yrs))

    ctx_pid = context.anon_polygon_id.to_numpy()
    ctx_year = context.year.to_numpy()
    drop = np.zeros(len(context), dtype=bool)
    for pid, yr in keep_year.items():
        drop |= (ctx_pid == pid) & (ctx_year != yr)
    ctx = context[~drop].copy()

    t_pid = targets.anon_polygon_id.to_numpy()
    t_year = targets.year.to_numpy()
    t_drop = np.zeros(len(targets), dtype=bool)
    for pid, yr in keep_year.items():
        t_drop |= (t_pid == pid) & (t_year != yr)
    return ctx, targets[~t_drop].copy(), cold
