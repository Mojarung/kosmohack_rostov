"""Сезонность: форма кривой NDVI по культурам и годам, климатическая норма,
примеры временных рядов отдельных полигонов."""

from __future__ import annotations

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from eda.config import CROP_ORDER, DOY_MIN, STATUS_COLORS, TARGET
from eda.load import known_series
from eda.plotting import save_fig

# Начала месяцев в неделях сезона (0 = 1 апреля)
MONTH_WEEKS = {"апр": 0, "май": 4.3, "июн": 8.7, "июл": 13.0, "авг": 17.4, "сен": 21.9, "окт": 26.1}


def _with_week(df: pd.DataFrame) -> pd.DataFrame:
    """Номер недели сезона — бин для усреднения по дню года."""
    return df.assign(week=((df["cal_doy"] - DOY_MIN) // 7).astype("int32"))


def _week_axis(ax: plt.Axes) -> None:
    ax.set_xticks(list(MONTH_WEEKS.values()), list(MONTH_WEEKS.keys()))
    ax.set_xlabel("Месяц")


def plot_curve_by_crop(train: pd.DataFrame) -> dict:
    k = _with_week(known_series(train))
    fig, ax = plt.subplots(figsize=(10, 4.5))
    sns.lineplot(data=k, x="week", y=TARGET, hue="crop_type", hue_order=CROP_ORDER,
                 estimator="median", errorbar=("pi", 50), ax=ax)
    _week_axis(ax)
    ax.set(title="Сезонная кривая primary_ndvi по культурам (медиана, полоса — межквартильный размах)",
           ylabel="primary_ndvi")
    ax.legend(title="Культура")
    save_fig(fig, "seasonality_curve_by_crop")

    weekly = k.groupby(["crop_type", "week"], observed=True)[TARGET].median().unstack("week")
    peak_week = weekly.idxmax(axis=1)
    return {
        "peak_doy_by_crop": {str(c): int(DOY_MIN + w * 7) for c, w in peak_week.items()},
        "peak_ndvi_by_crop": {str(c): round(float(v), 3) for c, v in weekly.max(axis=1).items()},
        "season_median_by_crop": {str(c): round(float(v), 3)
                                  for c, v in k.groupby("crop_type", observed=True)[TARGET].median().items()},
    }


def plot_curve_by_year(train: pd.DataFrame) -> dict:
    k = _with_week(known_series(train))
    weekly = k.groupby(["cal_year", "week"])[TARGET].median().unstack("cal_year")
    cmap = matplotlib.colormaps["viridis"]
    years = weekly.columns.to_list()
    fig, ax = plt.subplots(figsize=(10, 4.5))
    for i, year in enumerate(years):
        ax.plot(weekly.index, weekly[year], color=cmap(i / max(len(years) - 1, 1)), lw=1.4, label=str(year))
    _week_axis(ax)
    ax.set(title="Медианная сезонная кривая primary_ndvi по годам (все культуры)", ylabel="primary_ndvi")
    ax.legend(ncol=3, fontsize=7)
    save_fig(fig, "seasonality_curve_by_year")
    season_median = k.groupby("cal_year")[TARGET].median()
    return {"season_median_by_year": {int(y): round(float(v), 3) for y, v in season_median.items()}}


def _pick_examples(train: pd.DataFrame) -> list[tuple[str, int]]:
    """Полигон-годы для иллюстрации: самый аномальный и самый «штатный» с плотным покрытием."""
    k = train.loc[train["is_known"] & (train["n_reference_years"] > 0)]
    stats = k.groupby(["anon_polygon_id", "cal_year"]).agg(
        n=(TARGET, "size"), crit=("status", lambda s: (s == "Критическая аномалия").mean()),
        depr=("status", lambda s: (s != "Штатное развитие").mean()))
    dense = stats.loc[stats["n"] >= 40]
    worst = dense["depr"].idxmax()
    calm = dense.loc[dense["depr"] == 0, "n"].idxmax()
    return [worst, calm]


def plot_climatology_examples(train: pd.DataFrame) -> None:
    """Наблюдения против климатической нормы (mean ± std) для двух полигон-лет."""
    examples = _pick_examples(train)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.2), sharey=True)
    for ax, (poly, year) in zip(axes, examples):
        d = train.loc[(train["anon_polygon_id"] == poly) & (train["cal_year"] == year)]
        clim = d.dropna(subset=["ndvi_climatology_mean"])
        ax.fill_between(clim["cal_doy"], clim["ndvi_climatology_mean"] - clim["ndvi_climatology_std"],
                        clim["ndvi_climatology_mean"] + clim["ndvi_climatology_std"],
                        color="#cbd5e0", alpha=0.6, label="норма ± std")
        ax.plot(clim["cal_doy"], clim["ndvi_climatology_mean"], color="#4a5568", lw=1.2, label="климатическая норма")
        known = d.loc[d["is_known"]]
        colors = known["status"].astype(str).map(STATUS_COLORS).fillna("#000000")
        ax.scatter(known["cal_doy"], known[TARGET], c=colors, s=18, zorder=3, label="наблюдения (цвет = статус)")
        ax.set(title=f"{poly}, {year} ({d['crop_type'].iloc[0]})", xlabel="День года")
        ax.legend(fontsize=8)
    axes[0].set_ylabel("primary_ndvi")
    save_fig(fig, "seasonality_climatology_examples")


def plot_polygon_series(train: pd.DataFrame) -> None:
    """Полные ряды трёх полигонов разных культур за 2019–2024, цвет точки = статус."""
    polys = (train.drop_duplicates("anon_polygon_id").groupby("crop_type", observed=True)["anon_polygon_id"]
             .first().reindex(CROP_ORDER).dropna().iloc[:3])
    fig, axes = plt.subplots(len(polys), 1, figsize=(14, 2.6 * len(polys)), sharex=True)
    for ax, (crop, poly) in zip(np.atleast_1d(axes), polys.items()):
        d = train.loc[(train["anon_polygon_id"] == poly) & train["is_known"] & (train["cal_year"] >= 2019)]
        ax.plot(d["date"], d[TARGET], color="#a0aec0", lw=0.8, zorder=1)
        colors = d["status"].astype(str).map(STATUS_COLORS).fillna("#000000")
        ax.scatter(d["date"], d[TARGET], c=colors, s=12, zorder=2)
        ax.set(title=f"{poly} — {crop}", ylabel="primary_ndvi")
    handles = [plt.Line2D([], [], marker="o", ls="", color=c, label=s) for s, c in STATUS_COLORS.items()]
    np.atleast_1d(axes)[0].legend(handles=handles, loc="upper right", fontsize=8)
    save_fig(fig, "seasonality_polygon_series")


def plot_distribution(train: pd.DataFrame) -> dict:
    k = known_series(train)
    fig, ax = plt.subplots(figsize=(9, 4))
    sns.histplot(data=k, x=TARGET, hue="crop_type", hue_order=CROP_ORDER, element="step",
                 stat="density", common_norm=False, binwidth=0.02, binrange=(-0.2, 1.0), ax=ax)
    ax.set(title="Распределение primary_ndvi по культурам", xlabel="primary_ndvi")
    save_fig(fig, "seasonality_ndvi_distribution")
    q = k[TARGET].quantile([0.01, 0.05, 0.5, 0.95, 0.99])
    return {"ndvi_quantiles": {f"p{int(p * 100)}": round(float(v), 3) for p, v in q.items()}}


def run(train: pd.DataFrame, test: pd.DataFrame) -> dict:
    """Все графики сезонности и ключевые числа."""
    out = plot_curve_by_crop(train)
    out |= plot_curve_by_year(train)
    plot_climatology_examples(train)
    plot_polygon_series(train)
    out |= plot_distribution(train)
    known = train.loc[train["is_known"]]
    out["share_known_with_climatology"] = round(float((known["n_reference_years"] > 0).mean()), 3)
    return out
