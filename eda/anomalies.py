"""Аномалии: статусы и Z-score по годам и культурам, связь с погодой ERA5,
эпизоды устойчивого угнетения."""

from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from eda.config import CROP_ORDER, STATUS_COLORS, STATUS_ORDER, TARGET, WEATHER, Z_CRITICAL, Z_DEPRESSION
from eda.load import known_series
from eda.plotting import save_fig
from eda.sensors import SOURCES, with_source

# Окно накопления погодных признаков перед наблюдением, дней
WEATHER_WINDOW = 30
# Минимум строк за сезон, чтобы считать сетку полигон-года ежедневной (полный сезон = 213 дней)
FULL_GRID_MIN_ROWS = 200


def _stacked_status(share: pd.DataFrame, ax: plt.Axes, title: str, xlabel: str) -> None:
    share[STATUS_ORDER].plot.bar(stacked=True, ax=ax, width=0.8,
                                 color=[STATUS_COLORS[s] for s in STATUS_ORDER])
    ax.set(title=title, xlabel=xlabel, ylabel="Доля известных точек", ylim=(0, 1))
    ax.legend(loc="lower left", fontsize=8)


def plot_status_shares(train: pd.DataFrame) -> dict:
    k = known_series(train)
    by_year = pd.crosstab(k["cal_year"], k["status"], normalize="index")
    by_crop = pd.crosstab(k["crop_type"], k["status"], normalize="index").reindex(CROP_ORDER)
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.2), width_ratios=[2.2, 1])
    _stacked_status(by_year, axes[0], "Статусы по годам (train)", "Год")
    _stacked_status(by_crop, axes[1], "Статусы по культурам", "Культура")
    axes[1].tick_params(axis="x", labelrotation=20)
    save_fig(fig, "anomalies_status_shares")
    anomalous = 1 - by_year["Штатное развитие"]
    return {
        "status_share_overall": {s: round(float(v), 4) for s, v in k["status"].value_counts(normalize=True).items()},
        "anomalous_share_by_year": {int(y): round(float(v), 3) for y, v in anomalous.items()},
        "worst_years": [int(y) for y in anomalous.sort_values(ascending=False).index[:3]],
        "critical_share_by_crop": {str(c): round(float(v), 3) for c, v in by_crop["Критическая аномалия"].items()},
    }


def plot_zscore(train: pd.DataFrame) -> dict:
    k = known_series(train).dropna(subset=["ndvi_zscore"])
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.2))
    sns.histplot(data=k, x="ndvi_zscore", hue="crop_type", hue_order=CROP_ORDER, element="step",
                 stat="density", common_norm=False, binwidth=0.2, binrange=(-6, 4), ax=axes[0])
    for z, c in [(Z_DEPRESSION, STATUS_COLORS["Угнетение биомассы"]), (Z_CRITICAL, STATUS_COLORS["Критическая аномалия"])]:
        axes[0].axvline(z, color=c, ls="--", lw=1.2)
    axes[0].set(title="Распределение ndvi_zscore по культурам (пороги −1 и −2)", xlabel="Z-score")

    heat = k.assign(month=k["date"].dt.month).pivot_table(index="cal_year", columns="month",
                                                          values="ndvi_zscore", aggfunc="mean")
    sns.heatmap(heat, cmap="RdYlGn", center=0, vmin=-2, vmax=1, annot=True, fmt=".1f",
                annot_kws={"size": 7}, ax=axes[1], cbar_kws={"label": "средний Z-score"})
    axes[1].set(title="Средний Z-score: год × месяц", xlabel="Месяц", ylabel="Год")
    save_fig(fig, "anomalies_zscore")
    q = k["ndvi_zscore"].quantile([0.01, 0.05, 0.25, 0.5, 0.75, 0.95])
    return {"zscore_quantiles": {f"p{int(p * 100)}": round(float(v), 3) for p, v in q.items()},
            "zscore_min": round(float(k["ndvi_zscore"].min()), 2)}


def _rolling_time(group: pd.DataFrame, col: str, how: str) -> pd.Series:
    """Скользящее окно по календарному времени, а не по числу строк.

    Часть полигонов не имеет ежедневной сетки (строки только в дни наблюдений),
    поэтому окно по строкам там дало бы окно длиной в сезон.
    """
    window = group.set_index("date")[col].rolling(f"{WEATHER_WINDOW}D", min_periods=5)
    values = window.mean() if how == "mean" else window.sum()
    return pd.Series(values.to_numpy(), index=group.index)


def weather_anomalies(df: pd.DataFrame) -> pd.DataFrame:
    """Скользящие 30-дневные погодные признаки и их отклонение от нормы полигона по дню года.

    Норма — среднее по всем годам того же полигона для того же дня года. Флаг full_grid
    отмечает полигон-годы с полной ежедневной сеткой: только там сумма осадков корректна.
    """
    d = df.sort_values(["anon_polygon_id", "date"])
    by_py = d.groupby(["anon_polygon_id", "cal_year"], group_keys=False)
    roll = d.assign(
        temp30=by_py.apply(lambda g: _rolling_time(g, "era5_temp_c", "mean"), include_groups=False),
        precip30=by_py.apply(lambda g: _rolling_time(g, "era5_precip_mm", "sum"), include_groups=False),
        full_grid=by_py["date"].transform("size") >= FULL_GRID_MIN_ROWS,
    )
    by_pd = roll.groupby(["anon_polygon_id", "cal_doy"])
    return roll.assign(
        temp30_anom=roll["temp30"] - by_pd["temp30"].transform("mean"),
        precip30_anom=roll["precip30"] - by_pd["precip30"].transform("mean"),
    )


def plot_weather_vs_zscore(train: pd.DataFrame) -> dict:
    w = weather_anomalies(train)
    k = w.loc[w["is_known"] & w["full_grid"]].dropna(subset=["ndvi_zscore", "temp30_anom", "precip30_anom"])
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    for ax, col, label in zip(axes, ["precip30_anom", "temp30_anom"],
                              ["Аномалия осадков за 30 дней, мм", "Аномалия температуры за 30 дней, °C"]):
        sns.regplot(data=k.sample(min(len(k), 6000), random_state=0), x=col, y="ndvi_zscore",
                    scatter_kws={"s": 6, "alpha": 0.25}, line_kws={"color": "#c53030"}, ax=ax)
        r = k[col].corr(k["ndvi_zscore"])
        ax.set(title=f"Z-score против {label.lower()} (r = {r:+.3f})", xlabel=label, ylabel="ndvi_zscore",
               ylim=(-6, 4))
    fig.suptitle("Связь отклонения NDVI от нормы с погодой перед наблюдением (train)")
    save_fig(fig, "anomalies_weather_vs_zscore")
    crit = k.loc[k["status"] == "Критическая аномалия"]
    return {
        "corr_precip30_anom_zscore": round(float(k["precip30_anom"].corr(k["ndvi_zscore"])), 3),
        "corr_temp30_anom_zscore": round(float(k["temp30_anom"].corr(k["ndvi_zscore"])), 3),
        "critical_mean_precip30_anom": round(float(crit["precip30_anom"].mean()), 1),
        "critical_mean_temp30_anom": round(float(crit["temp30_anom"].mean()), 2),
    }


def episodes(train: pd.DataFrame) -> pd.DataFrame:
    """Эпизоды угнетения: подряд идущие известные точки с Z < −1 в одном полигоне и сезоне."""
    k = known_series(train).dropna(subset=["ndvi_zscore"])
    flagged = k["ndvi_zscore"] < Z_DEPRESSION
    group_change = (k["anon_polygon_id"] != k["anon_polygon_id"].shift()) | (k["cal_year"] != k["cal_year"].shift())
    new_episode = flagged & (~flagged.shift(fill_value=False) | group_change)
    ep_id = new_episode.cumsum().where(flagged)
    return (k.assign(ep_id=ep_id).dropna(subset=["ep_id"])
            .groupby("ep_id").agg(polygon=("anon_polygon_id", "first"), year=("cal_year", "first"),
                                  crop=("crop_type", "first"), start=("date", "min"), end=("date", "max"),
                                  n_points=("ndvi_zscore", "size"), min_z=("ndvi_zscore", "min"))
            .assign(days=lambda e: (e["end"] - e["start"]).dt.days + 1))


def plot_episode_example(train: pd.DataFrame, ep: pd.Series, ax: plt.Axes) -> None:
    d = train.loc[(train["anon_polygon_id"] == ep["polygon"]) & (train["cal_year"] == ep["year"])]
    clim = d.dropna(subset=["ndvi_climatology_mean"])
    ax.fill_between(clim["date"], clim["ndvi_climatology_mean"] - clim["ndvi_climatology_std"],
                    clim["ndvi_climatology_mean"] + clim["ndvi_climatology_std"], color="#cbd5e0", alpha=0.6)
    ax.plot(clim["date"], clim["ndvi_climatology_mean"], color="#4a5568", lw=1)
    known = d.loc[d["is_known"]]
    ax.scatter(known["date"], known[TARGET], c=known["status"].astype(str).map(STATUS_COLORS).fillna("#000"), s=16, zorder=3)
    ax.axvspan(ep["start"], ep["end"], color="#c53030", alpha=0.12)
    ax.set(title=f"{ep['polygon']}, {ep['year']} ({ep['crop']}): эпизод {ep['days']} дн., {ep['n_points']} точек, min Z={ep['min_z']:.1f}",
           ylabel="primary_ndvi")
    ax2 = ax.twinx()
    ax2.bar(d["date"], d["era5_precip_mm"], color="#3182ce", alpha=0.35, width=1)
    ax2.set_ylabel("Осадки ERA5, мм/день", color="#3182ce")


def plot_episodes(train: pd.DataFrame) -> dict:
    ep = episodes(train)
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.8))
    sns.histplot(ep["n_points"], binwidth=1, binrange=(0.5, 20.5), ax=axes[0])
    axes[0].set(title="Длина эпизодов угнетения в точках", xlabel="Известных точек подряд с Z < −1")
    sns.histplot(ep["days"], binwidth=7, binrange=(0, 140), ax=axes[1])
    axes[1].set(title="Длина эпизодов в днях", xlabel="Дней от первой до последней точки")
    save_fig(fig, "anomalies_episode_lengths")

    top = ep.sort_values(["n_points", "days"], ascending=False).head(2)
    fig, axes = plt.subplots(2, 1, figsize=(13, 7.5))
    for ax, (_, row) in zip(axes, top.iterrows()):
        plot_episode_example(train, row, ax)
    save_fig(fig, "anomalies_episode_examples")
    return {
        "n_episodes": int(len(ep)),
        "n_flagged_points": int(ep["n_points"].sum()),
        "isolated_points_share": round(float((ep["n_points"] == 1).sum() / ep["n_points"].sum()), 3),
        "episode_single_point_share": round(float((ep["n_points"] == 1).mean()), 3),
        "episode_ge3_points_share": round(float((ep["n_points"] >= 3).mean()), 3),
        "episode_days_median": float(ep["days"].median()),
        "episodes_by_year": {int(y): int(v) for y, v in ep["year"].value_counts().sort_index().items()},
        "longest_episodes": [
            {"polygon": r["polygon"], "year": int(r["year"]), "crop": str(r["crop"]), "days": int(r["days"]),
             "n_points": int(r["n_points"]), "min_z": round(float(r["min_z"]), 2)} for _, r in top.iterrows()],
    }


def plot_zscore_by_sensor(train: pd.DataFrame) -> dict:
    """Z-score и доля аномалий в зависимости от сенсора-источника primary_ndvi."""
    k = with_source(known_series(train)).dropna(subset=["ndvi_zscore"])
    k = k.assign(source=k["source"].astype(str))
    stats = k.groupby("source").agg(mean_z=("ndvi_zscore", "mean"), n=("ndvi_zscore", "size"),
                                    anomalous=("ndvi_zscore", lambda s: float((s < Z_DEPRESSION).mean()))).reindex(SOURCES)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    sns.boxplot(data=k, x="source", y="ndvi_zscore", order=SOURCES, showfliers=False, ax=axes[0],
                palette={"s2": "#2b6cb0", "landsat": "#805ad5", "modis": "#dd6b20"}, hue="source", legend=False)
    axes[0].axhline(Z_DEPRESSION, ls="--", color=STATUS_COLORS["Угнетение биомассы"], lw=1)
    axes[0].set(title="Z-score по сенсору-источнику (без выбросов)", xlabel="Сенсор", ylabel="ndvi_zscore")
    bars = axes[1].bar(stats.index, stats["anomalous"], color=["#2b6cb0", "#805ad5", "#dd6b20"])
    axes[1].bar_label(bars, fmt="%.1%%")
    axes[1].set(title="Доля точек с Z < −1 по сенсору", xlabel="Сенсор", ylabel="Доля", ylim=(0, 0.35))
    save_fig(fig, "anomalies_zscore_by_sensor")
    return {"zscore_mean_by_sensor": {s: round(float(v), 3) for s, v in stats["mean_z"].items()},
            "anomalous_share_by_sensor": {s: round(float(v), 3) for s, v in stats["anomalous"].items()}}


def plot_yearly_weather(train: pd.DataFrame) -> dict:
    """Годовые суммы осадков и средняя температура против среднего Z-score (полигоны с ежедневной сеткой)."""
    full = train.groupby(["anon_polygon_id", "cal_year"])["date"].transform("size") >= FULL_GRID_MIN_ROWS
    d = train.loc[full]
    per_poly = d.groupby(["anon_polygon_id", "cal_year"]).agg(precip=("era5_precip_mm", "sum"), temp=("era5_temp_c", "mean"))
    yearly = per_poly.groupby("cal_year").mean().join(d.loc[d["is_known"]].groupby("cal_year")["ndvi_zscore"].mean())
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, col, label in zip(axes, ["precip", "temp"], ["Сумма осадков за сезон, мм", "Средняя температура сезона, °C"]):
        ax.scatter(yearly[col], yearly["ndvi_zscore"], color="#2b6cb0", s=40)
        for year, row in yearly.iterrows():
            ax.annotate(str(year), (row[col], row["ndvi_zscore"]), fontsize=8, xytext=(3, 3), textcoords="offset points")
        ax.set(title=f"{label}: r = {yearly[col].corr(yearly['ndvi_zscore']):+.2f}", xlabel=label, ylabel="Средний Z-score года")
    fig.suptitle("Погода сезона и отклонение NDVI от нормы по годам (train)")
    save_fig(fig, "anomalies_yearly_weather")
    return {"corr_yearly_precip_zscore": round(float(yearly["precip"].corr(yearly["ndvi_zscore"])), 3),
            "corr_yearly_temp_zscore": round(float(yearly["temp"].corr(yearly["ndvi_zscore"])), 3),
            "yearly_precip_mm": {int(y): round(float(v), 0) for y, v in yearly["precip"].items()}}


def plot_weather_seasonal(train: pd.DataFrame) -> None:
    """Температура и осадки по годам: медиана по полигонам, скользящее окно 15 дней."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    cmap = plt.get_cmap("coolwarm")
    years = sorted(train["cal_year"].unique())
    for ax, col, label in zip(axes, WEATHER, ["Температура, °C", "Осадки, мм/день"]):
        daily = train.groupby(["cal_year", "cal_doy"])[col].median().unstack("cal_year")
        smooth = daily.rolling(15, center=True, min_periods=5).mean()
        for i, y in enumerate(years):
            ax.plot(smooth.index, smooth[y], color=cmap(i / (len(years) - 1)), lw=1, label=str(y))
        ax.set(title=f"{label} (ERA5, медиана по полигонам, окно 15 дней)", xlabel="День года", ylabel=label)
    axes[1].legend(ncol=3, fontsize=7)
    save_fig(fig, "anomalies_weather_seasonal")


def run(train: pd.DataFrame, test: pd.DataFrame) -> dict:
    """Все графики по аномалиям и ключевые числа."""
    out = plot_status_shares(train)
    out |= plot_zscore(train)
    out |= plot_weather_vs_zscore(train)
    out |= plot_yearly_weather(train)
    out |= plot_zscore_by_sensor(train)
    out |= plot_episodes(train)
    plot_weather_seasonal(train)
    return out
