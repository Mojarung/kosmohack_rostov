"""Анализ контрольных точек тестового набора (is_synthetic_gap = True).

Для каждой контрольной строки считаем расстояние до ближайших известных
наблюдений primary_ndvi того же полигона (внутри сезона и по полигону в целом),
плотность известных соседей в окне и сенсор-источник ближайшей точки.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from eda.config import CROP_ORDER, SENSOR_NDVI, TARGET
from eda.plotting import save_fig, setup_style

EPOCH = pd.Timestamp("2000-01-01")
DIST_CLIP = 60  # дней; хвост распределения на графиках обрезаем
WINDOWS = (15, 30)  # окна для подсчёта плотности известных соседей, дней
QUANTILES = (0.25, 0.5, 0.75, 0.9)
SEASON = ["anon_polygon_id", "cal_year"]
POLYGON = ["anon_polygon_id"]


def sensor_of(df: pd.DataFrame) -> pd.Series:
    """Сенсор-источник primary_ndvi по правилу приоритета s2 > landsat > modis."""
    out = pd.Series("нет", index=df.index, dtype="str")
    # Идём от низшего приоритета к высшему: s2 записывается последним и побеждает
    for col in reversed(SENSOR_NDVI):
        out = out.mask(df[col].notna(), col.split("_")[0])
    return out


def compute_neighbors(frame: pd.DataFrame, is_target, is_known, by: list[str]) -> pd.DataFrame:
    """Ближайшие известные точки до/после для целевых строк внутри групп by.

    Возвращает DataFrame с индексом целевых строк и колонками
    date_prev/val_prev/sensor_prev, date_next/val_next/sensor_next,
    days_prev/days_next/min_dist (NaN, если соседа нет).
    """
    is_target = np.asarray(is_target, dtype=bool)
    is_known = np.asarray(is_known, dtype=bool)
    known = frame.loc[is_known, by + ["date", TARGET]]
    known = known.assign(sensor=sensor_of(frame.loc[is_known])).sort_values("date")
    targets = frame.loc[is_target, by + ["date"]].rename_axis("row_id").reset_index().sort_values("date")
    prev = known.rename(columns={"date": "date_prev", TARGET: "val_prev", "sensor": "sensor_prev"})
    nxt = known.rename(columns={"date": "date_next", TARGET: "val_next", "sensor": "sensor_next"})
    merged = pd.merge_asof(targets, prev, left_on="date", right_on="date_prev", by=by,
                           direction="backward", allow_exact_matches=False)
    merged = pd.merge_asof(merged, nxt, left_on="date", right_on="date_next", by=by,
                           direction="forward", allow_exact_matches=False)
    merged = merged.set_index("row_id").sort_index()
    days_prev = (merged["date"] - merged["date_prev"]).dt.days
    days_next = (merged["date_next"] - merged["date"]).dt.days
    return merged.drop(columns=by + ["date"]).assign(
        days_prev=days_prev, days_next=days_next, min_dist=np.fmin(days_prev, days_next),
    )


def count_known_in_window(frame: pd.DataFrame, is_target, is_known, by: list[str], window: int) -> pd.Series:
    """Число известных точек в окне ±window дней вокруг целевых строк (в группах by)."""
    is_target = np.asarray(is_target, dtype=bool)
    is_known = np.asarray(is_known, dtype=bool)
    days = (frame["date"] - EPOCH).dt.days
    keys = [frame[c] for c in by]
    known_days = {k: np.sort(v.to_numpy()) for k, v in days[is_known].groupby([k[is_known] for k in keys])}
    pieces = []
    for k, tgt in days[is_target].groupby([k[is_target] for k in keys]):
        kd = known_days.get(k, np.empty(0))
        t = tgt.to_numpy()
        n = np.searchsorted(kd, t + window, "right") - np.searchsorted(kd, t - window, "left")
        pieces.append(pd.Series(n, index=tgt.index))
    return pd.concat(pieces).reindex(frame.index[is_target])


def build_gap_table(test: pd.DataFrame) -> pd.DataFrame:
    """Таблица контрольных точек с признаками соседей (в сезоне и по полигону) и плотностью."""
    is_gap = test["is_gap"].to_numpy()
    is_known = test["is_known"].to_numpy()
    season = compute_neighbors(test, is_gap, is_known, SEASON)
    poly = compute_neighbors(test, is_gap, is_known, POLYGON)[["days_prev", "days_next", "min_dist"]]
    counts = {f"n_known_{w}": count_known_in_window(test, is_gap, is_known, POLYGON, w) for w in WINDOWS}
    base = test.loc[is_gap, ["anon_polygon_id", "date", "crop_type", "cal_year", "cal_doy"]]
    tbl = base.join(season).join(poly.add_suffix("_poly")).assign(**counts)
    nearer_prev = tbl["days_prev"].fillna(np.inf) <= tbl["days_next"].fillna(np.inf)
    nearest = tbl["sensor_prev"].where(nearer_prev, tbl["sensor_next"]).fillna("нет")
    return tbl.assign(sensor_nearest=nearest)


def year_stats(test: pd.DataFrame) -> pd.DataFrame:
    """По годам: число контрольных точек, доля замаскированных среди исходно известных, покрытие."""
    g = test.groupby("cal_year", observed=True)
    n_gap, n_known, n_all = g["is_gap"].sum(), g["is_known"].sum(), g.size()
    return pd.DataFrame({
        "n_gap": n_gap,
        "masked_share": n_gap / (n_gap + n_known),
        "known_share": n_known / n_all,
    })


def plot_distances(tbl: pd.DataFrame) -> None:
    """Гистограммы и ECDF расстояний до соседей внутри сезона."""
    cols = [
        ("days_prev", "до предыдущего известного"),
        ("days_next", "до следующего известного"),
        ("min_dist", "до ближайшего известного"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(13, 7))
    for (col, title), ax_h, ax_e in zip(cols, axes[0], axes[1]):
        s = tbl[col].dropna().clip(upper=DIST_CLIP)
        sns.histplot(s, bins=np.arange(0.5, DIST_CLIP + 1.5, 1), ax=ax_h, color="#2b6cb0")
        ax_h.set(title=f"Дней {title}", xlabel="дней (хвост обрезан на 60)", ylabel="контрольных точек")
        sns.ecdfplot(s, ax=ax_e, color="#2b6cb0")
        ax_e.set(xlabel="дней", ylabel="доля точек ≤ x")
        for q in (3, 7, 15):
            ax_e.axvline(q, ls=":", color="gray", lw=0.8)
    fig.suptitle("Расстояние от контрольных точек test до ближайших известных наблюдений (внутри сезона)")
    save_fig(fig, "gaps_neighbor_distances")


def plot_by_year(ys: pd.DataFrame) -> None:
    """Контрольные точки по годам и доля замаскированных/известных строк."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.5))
    bars = ax1.bar(ys.index.astype(str), ys["n_gap"], color="#c8423f")
    ax1.bar_label(bars, fontsize=7)
    ax1.set(title="Контрольные точки test по годам", xlabel="год", ylabel="контрольных точек")
    ax1.tick_params(axis="x", rotation=45)
    ax2.plot(ys.index, ys["masked_share"], marker="o", color="#c8423f",
             label="доля замаскированных среди исходно известных")
    ax2.plot(ys.index, ys["known_share"], marker="s", color="#3a9d5d",
             label="доля дней с известным primary_ndvi")
    ax2.set(title="Доли по годам", xlabel="год", ylabel="доля", ylim=(0, 1))
    ax2.legend(loc="upper left")
    save_fig(fig, "gaps_by_year")


def plot_boxes(tbl: pd.DataFrame) -> None:
    """Распределение расстояния до ближайшего соседа по культурам и годам."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4.5), width_ratios=[1, 2])
    sns.boxplot(data=tbl, x="crop_type", y="min_dist", order=CROP_ORDER, ax=ax1,
                showfliers=False, color="#9ac1e8")
    ax1.set(title="По культурам", xlabel="", ylabel="дней до ближайшего известного")
    ax1.tick_params(axis="x", rotation=20)
    sns.boxplot(data=tbl, x="cal_year", y="min_dist", ax=ax2, showfliers=False, color="#9ac1e8")
    ax2.set(title="По годам", xlabel="год", ylabel="")
    ax2.tick_params(axis="x", rotation=45)
    fig.suptitle("Расстояние до ближайшего известного наблюдения в сезоне (без выбросов)")
    save_fig(fig, "gaps_min_dist_boxes")


def plot_doy(tbl: pd.DataFrame, test: pd.DataFrame) -> None:
    """Сезонное положение контрольных точек против известных наблюдений."""
    known = test.loc[test["is_known"], ["cal_doy"]].assign(kind="известные наблюдения")
    gaps = tbl[["cal_doy"]].assign(kind="контрольные точки")
    data = pd.concat([known, gaps], ignore_index=True)
    fig, ax = plt.subplots(figsize=(10, 4))
    sns.histplot(data=data, x="cal_doy", hue="kind", stat="density", common_norm=False,
                 element="step", bins=range(85, 312, 7), ax=ax,
                 palette={"известные наблюдения": "#3a9d5d", "контрольные точки": "#c8423f"})
    ax.set(title="Распределение по дню года: контрольные точки vs известные наблюдения (test)",
           xlabel="день года", ylabel="плотность")
    save_fig(fig, "gaps_doy_distribution")


def plot_presence(tbl: pd.DataFrame) -> None:
    """Наличие соседей в сезоне и сенсор ближайшей известной точки."""
    has_prev, has_next = tbl["days_prev"].notna(), tbl["days_next"].notna()
    presence = pd.Series({
        "оба соседа": int((has_prev & has_next).sum()),
        "нет слева (начало сезона)": int((~has_prev & has_next).sum()),
        "нет справа (конец сезона)": int((has_prev & ~has_next).sum()),
        "нет соседей в сезоне": int((~has_prev & ~has_next).sum()),
    })
    sensors = tbl["sensor_nearest"].value_counts()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    bars = ax1.bar(presence.index, presence.to_numpy(), color=["#3a9d5d", "#e8a33d", "#e8a33d", "#c8423f"])
    ax1.bar_label(bars)
    ax1.set(title="Наличие известных соседей в сезоне", ylabel="контрольных точек")
    ax1.tick_params(axis="x", rotation=15)
    bars = ax2.bar(sensors.index, sensors.to_numpy(), color="#805ad5")
    ax2.bar_label(bars)
    ax2.set(title="Сенсор-источник ближайшей известной точки", ylabel="контрольных точек")
    save_fig(fig, "gaps_presence_and_sensor")


def _quantiles(s: pd.Series) -> dict[str, float]:
    """Квантили ряда в виде словаря q25/q50/q75/q90."""
    return {f"q{int(p * 100)}": float(v) for p, v in s.quantile(list(QUANTILES)).items()}


def summarize(tbl: pd.DataFrame, ys: pd.DataFrame, train: pd.DataFrame) -> dict:
    """Ключевые числа для отчёта."""
    new_polys = set(tbl["anon_polygon_id"]) - set(train["anon_polygon_id"])
    is_new = tbl["anon_polygon_id"].isin(new_polys)
    md = tbl["min_dist"]
    return {
        "n_gaps": int(len(tbl)),
        "n_gaps_new_polygons": int(is_new.sum()),
        "n_gaps_old_polygons": int((~is_new).sum()),
        "n_new_polygons": len(new_polys),
        "days_prev": _quantiles(tbl["days_prev"].dropna()),
        "days_next": _quantiles(tbl["days_next"].dropna()),
        "min_dist": _quantiles(md.dropna()),
        "share_min_dist_le_3": float((md <= 3).mean()),
        "share_min_dist_le_7": float((md <= 7).mean()),
        "share_min_dist_le_15": float((md <= 15).mean()),
        "share_no_prev_in_season": float(tbl["days_prev"].isna().mean()),
        "share_no_next_in_season": float(tbl["days_next"].isna().mean()),
        "share_no_neighbors_in_season": float(md.isna().mean()),
        "share_no_prev_in_polygon": float(tbl["days_prev_poly"].isna().mean()),
        "share_no_next_in_polygon": float(tbl["days_next_poly"].isna().mean()),
        "n_known_15_median": float(tbl["n_known_15"].median()),
        "n_known_30_median": float(tbl["n_known_30"].median()),
        "share_n_known_15_zero": float((tbl["n_known_15"] == 0).mean()),
        "sensor_nearest": {str(k): int(v) for k, v in tbl["sensor_nearest"].value_counts().items()},
        "gaps_by_year": {str(k): int(v) for k, v in ys["n_gap"].items()},
        "masked_share_by_year": {str(k): round(float(v), 3) for k, v in ys["masked_share"].items()},
        "min_dist_median_by_crop": {str(k): float(v) for k, v in
                                    tbl.groupby("crop_type", observed=True)["min_dist"].median().items()},
    }


def run(train: pd.DataFrame, test: pd.DataFrame) -> dict:
    """Строит графики по контрольным точкам test и возвращает сводку чисел."""
    setup_style()
    tbl = build_gap_table(test)
    ys = year_stats(test)
    plot_distances(tbl)
    plot_by_year(ys)
    plot_boxes(tbl)
    plot_doy(tbl, test)
    plot_presence(tbl)
    return summarize(tbl, ys, train)
