"""Структура primary_ndvi: какой сенсор его формирует, согласованность сенсоров
между собой, систематические смещения и отрицательные выбросы."""

from __future__ import annotations

import itertools

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from eda.config import SENSOR_COLORS, SENSOR_NDVI, TARGET
from eda.plotting import save_fig

SOURCES = ["s2", "landsat", "modis"]
PAIRS = list(itertools.combinations(SENSOR_NDVI, 2))


def with_source(df: pd.DataFrame) -> pd.DataFrame:
    """Источник primary_ndvi по правилу приоритета s2 > landsat > modis."""
    conds = [df[c].notna() for c in SENSOR_NDVI]
    src = np.select(conds, SOURCES, default="none")
    return df.assign(source=pd.Categorical(src, categories=SOURCES + ["none"]))


def _observed(train: pd.DataFrame, test: pd.DataFrame) -> pd.DataFrame:
    """Известные строки train и test с источником target."""
    both = pd.concat([train, test.loc[~test["is_gap"]]], ignore_index=True)
    return with_source(both.loc[both["is_known"]])


def check_priority_rule(known: pd.DataFrame) -> dict:
    """Доля строк, где primary_ndvi совпадает со значением сенсора-источника."""
    src_value = np.select([known["source"] == s for s in SOURCES],
                          [known[c] for c in SENSOR_NDVI], default=np.nan)
    match = np.isclose(known[TARGET].to_numpy(), src_value, atol=1e-9)
    return {"priority_rule_match_share": round(float(match.mean()), 4),
            "known_without_any_sensor": int((known["source"] == "none").sum())}


def plot_source_share_by_year(known: pd.DataFrame) -> dict:
    share = pd.crosstab(known["cal_year"], known["source"], normalize="index")[SOURCES]
    fig, ax = plt.subplots(figsize=(10, 4))
    share.plot.bar(stacked=True, ax=ax, width=0.8, color=[SENSOR_COLORS[c] for c in SENSOR_NDVI])
    ax.set(title="Источник primary_ndvi по годам (доля известных точек)", xlabel="Год", ylabel="Доля")
    ax.legend(title="Сенсор", loc="upper left")
    save_fig(fig, "sensors_source_share_by_year")
    overall = known["source"].value_counts(normalize=True)
    return {"source_share_overall": {s: round(float(overall.get(s, 0)), 3) for s in SOURCES}}


def _pair_stats(df: pd.DataFrame, a: str, b: str) -> dict:
    pair = df.dropna(subset=[a, b])
    diff = pair[b] - pair[a]
    return {"n": int(len(pair)), "bias": round(float(diff.mean()), 4),
            "rmse": round(float(np.sqrt((diff ** 2).mean())), 4),
            "r": round(float(pair[a].corr(pair[b])), 4)}


def plot_sensor_agreement(both: pd.DataFrame) -> dict:
    """Попарные диаграммы рассеяния сенсоров в дни совместных наблюдений."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.4))
    stats = {}
    for ax, (a, b) in zip(axes, PAIRS):
        pair = both.dropna(subset=[a, b])
        st = _pair_stats(both, a, b)
        stats[f"{a}__{b}"] = st
        ax.hexbin(pair[a], pair[b], gridsize=45, cmap="Blues", mincnt=1, extent=(-0.1, 1, -0.1, 1))
        ax.plot([-0.1, 1], [-0.1, 1], color="#c53030", lw=1, ls="--")
        ax.set(title=f"{a} vs {b}\nn={st['n']}, bias={st['bias']:+.3f}, RMSE={st['rmse']:.3f}, r={st['r']:.3f}",
               xlabel=a, ylabel=b)
    save_fig(fig, "sensors_agreement")
    return {"pair_stats": stats}


def plot_bias_by_month(both: pd.DataFrame) -> None:
    """Смещение Landsat и MODIS относительно Sentinel-2 по месяцам."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
    for ax, other in zip(axes, ["landsat_ndvi", "modis_ndvi"]):
        pair = both.dropna(subset=["s2_ndvi", other]).assign(month=lambda d: d["date"].dt.month)
        pair = pair.assign(diff=pair[other] - pair["s2_ndvi"])
        pair.boxplot(column="diff", by="month", ax=ax, showfliers=False, grid=False)
        ax.axhline(0, color="#c53030", lw=1, ls="--")
        ax.set(title=f"{other} − s2_ndvi", xlabel="Месяц", ylabel="Разница NDVI")
    fig.suptitle("Систематическое смещение сенсоров относительно Sentinel-2 (без выбросов)")
    save_fig(fig, "sensors_bias_by_month")


def plot_index_relations(both: pd.DataFrame) -> None:
    """Связь NDVI с EVI и NDWI по Sentinel-2."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for ax, other in zip(axes, ["s2_evi", "s2_ndwi"]):
        pair = both.dropna(subset=["s2_ndvi", other])
        ax.hexbin(pair["s2_ndvi"], pair[other], gridsize=45, cmap="Greens", mincnt=1)
        ax.set(title=f"s2_ndvi vs {other} (r={pair['s2_ndvi'].corr(pair[other]):.3f})",
               xlabel="s2_ndvi", ylabel=other)
    save_fig(fig, "sensors_index_relations")


def plot_negative_outliers(train: pd.DataFrame, known: pd.DataFrame) -> dict:
    """Контекст ±30 дней вокруг четырёх самых отрицательных значений primary_ndvi."""
    neg = known.loc[known[TARGET] < 0].sort_values(TARGET)
    worst = neg.head(4)
    fig, axes = plt.subplots(1, 4, figsize=(15, 3.6), sharey=True)
    for ax, (_, row) in zip(axes, worst.iterrows()):
        win = train.loc[(train["anon_polygon_id"] == row["anon_polygon_id"]) & train["is_known"]
                        & (train["date"] - row["date"]).dt.days.abs().le(30)]
        ax.plot(win["date"], win[TARGET], marker="o", ms=4, color="#4a5568")
        ax.scatter([row["date"]], [row[TARGET]], color="#c53030", s=60, zorder=3)
        ax.set(title=f"{row['anon_polygon_id']} {row['date']:%Y-%m-%d}\nисточник: {row['source']}")
        ax.tick_params(axis="x", labelrotation=45, labelsize=7)
    axes[0].set_ylabel("primary_ndvi")
    fig.suptitle("Отрицательные выбросы primary_ndvi и их окружение (train)")
    save_fig(fig, "sensors_negative_outliers")
    return {"negative_count": int(len(neg)),
            "negative_by_source": {str(k): int(v) for k, v in neg["source"].value_counts().items() if v > 0},
            "below_minus_0_2": int((neg[TARGET] < -0.2).sum())}


def run(train: pd.DataFrame, test: pd.DataFrame) -> dict:
    """Все графики по сенсорам и ключевые числа."""
    known = _observed(train, test)
    both = pd.concat([train, test.loc[~test["is_gap"]]], ignore_index=True)
    out = check_priority_rule(known)
    out |= plot_source_share_by_year(known)
    out |= plot_sensor_agreement(both)
    plot_bias_by_month(both)
    plot_index_relations(both)
    out |= plot_negative_outliers(train, with_source(train.loc[train["is_known"]]))
    return out
