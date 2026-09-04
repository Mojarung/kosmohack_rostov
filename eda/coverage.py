"""Покрытие наблюдениями: доля известных значений, разрывы между наблюдениями,
доступность сенсоров по годам и по фазе сезона."""

from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from eda.config import SENSOR_COLORS, SENSOR_NDVI, TARGET
from eda.load import known_series
from eda.plotting import save_fig

# Границы «ранних» и «поздних» лет: с 2017 года покрытие заметно выше
EARLY_YEARS = range(2010, 2017)
# Полный сезон 1 апреля – 30 октября = 213 строк ежедневной сетки
SEASON_DAYS = 213


def _known_share_by_year(train: pd.DataFrame, test: pd.DataFrame) -> pd.DataFrame:
    """Доля дней сезона с известным target по годам: train, test и test с учётом контрольных точек."""
    tr = train.groupby("cal_year")["is_known"].mean().rename("train")
    te = test.groupby("cal_year")["is_known"].mean().rename("test (известные)")
    te_full = (
        test.assign(k=test["is_known"] | test["is_gap"])
        .groupby("cal_year")["k"].mean()
        .rename("test (известные + контрольные)")
    )
    return pd.concat([tr, te, te_full], axis=1)


def plot_known_share_by_year(train: pd.DataFrame, test: pd.DataFrame) -> pd.DataFrame:
    share = _known_share_by_year(train, test)
    fig, ax = plt.subplots(figsize=(10, 4))
    share.plot.bar(ax=ax, width=0.8, color=["#2b6cb0", "#a0aec0", "#dd6b20"])
    ax.set(title="Доля дней сезона с известным primary_ndvi", xlabel="Год", ylabel="Доля дней", ylim=(0, 0.6))
    ax.legend(loc="upper left")
    save_fig(fig, "coverage_known_share_by_year")
    return share


def _all_observed(train: pd.DataFrame, test: pd.DataFrame) -> pd.DataFrame:
    """Объединение train и незамаскированных строк test — для статистики по сенсорам."""
    return pd.concat([train, test.loc[~test["is_gap"]]], ignore_index=True)


def plot_sensor_availability(train: pd.DataFrame, test: pd.DataFrame) -> pd.DataFrame:
    both = _all_observed(train, test)
    avail = both[SENSOR_NDVI].notna().groupby(both["cal_year"]).mean()
    fig, ax = plt.subplots(figsize=(10, 4))
    for col in SENSOR_NDVI:
        ax.plot(avail.index, avail[col], marker="o", label=col, color=SENSOR_COLORS[col])
    ax.set(title="Доля дней сезона с наблюдением по каждому сенсору (train + test)",
           xlabel="Год", ylabel="Доля дней", ylim=(0, 0.45))
    ax.legend()
    save_fig(fig, "coverage_sensor_availability_by_year")
    return avail


def plot_polygon_year_heatmap(train: pd.DataFrame, test: pd.DataFrame) -> None:
    """Тепловая карта «полигон × год»: доля дней с известным target."""
    def pivot(df: pd.DataFrame) -> pd.DataFrame:
        return df.pivot_table(index="anon_polygon_id", columns="cal_year", values="is_known",
                              aggfunc="mean", observed=True)

    fig, axes = plt.subplots(1, 2, figsize=(15, 11), width_ratios=[1, 1.15])
    kws = {"cmap": "YlGn", "vmin": 0, "vmax": 0.6, "yticklabels": True, "linewidths": 0.2}
    sns.heatmap(pivot(train), ax=axes[0], cbar=False, **kws)
    sns.heatmap(pivot(test), ax=axes[1], cbar_kws={"label": "доля дней с известным primary_ndvi"}, **kws)
    axes[0].set(title="train: 39 полигонов", xlabel="Год", ylabel="Полигон")
    axes[1].set(title="test: 78 полигонов (контрольные точки замаскированы)", xlabel="Год", ylabel="")
    for ax in axes:
        ax.tick_params(axis="y", labelsize=6)
    save_fig(fig, "coverage_polygon_year_heatmap")


def natural_gaps(df: pd.DataFrame) -> pd.DataFrame:
    """Разрывы (в днях) между соседними известными наблюдениями внутри одного полигона и сезона."""
    k = known_series(df)
    by_poly = k.groupby("anon_polygon_id")
    same_season = by_poly["cal_year"].diff().eq(0)
    gap_days = by_poly["date"].diff().dt.days
    return k.assign(gap_days=gap_days).loc[same_season]


def plot_natural_gaps(train: pd.DataFrame) -> dict:
    gaps = natural_gaps(train)
    period = gaps["cal_year"].isin(EARLY_YEARS).map({True: "2010–2016", False: "2017–2024"})
    gaps = gaps.assign(period=period)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    sns.histplot(data=gaps, x="gap_days", hue="period", binwidth=1, binrange=(0, 60),
                 element="step", stat="probability", common_norm=False, ax=axes[0])
    axes[0].set(title="Разрывы между соседними известными наблюдениями (train)",
                xlabel="Дней между наблюдениями", ylabel="Доля")
    sns.ecdfplot(data=gaps, x="gap_days", hue="period", ax=axes[1])
    axes[1].set(title="Накопленное распределение разрывов", xlabel="Дней между наблюдениями",
                ylabel="Доля разрывов ≤ x", xlim=(0, 60))
    save_fig(fig, "coverage_natural_gap_lengths")

    by_year = gaps.groupby("cal_year")["gap_days"].median()
    return {
        "gap_days_median_by_year": {int(y): float(v) for y, v in by_year.items()},
        "gap_days_median_early": float(gaps.loc[gaps["period"] == "2010–2016", "gap_days"].median()),
        "gap_days_median_late": float(gaps.loc[gaps["period"] == "2017–2024", "gap_days"].median()),
        "gap_days_share_gt15": float((gaps["gap_days"] > 15).mean()),
        "gap_days_share_gt30": float((gaps["gap_days"] > 30).mean()),
    }


def plot_coverage_by_doy(train: pd.DataFrame, test: pd.DataFrame) -> None:
    """Доля известных значений и доступность сенсоров по дню года (скользящее окно 7 дней)."""
    both = _all_observed(train, test)
    cols = {"primary_ndvi": "#2d3748", **{c: SENSOR_COLORS[c] for c in SENSOR_NDVI}}
    by_doy = both[list(cols)].notna().groupby(both["cal_doy"]).mean().rolling(7, center=True).mean()
    fig, ax = plt.subplots(figsize=(10, 4))
    for col, color in cols.items():
        ax.plot(by_doy.index, by_doy[col], label=col, color=color, lw=2 if col == TARGET else 1.2)
    _month_axis(ax)
    ax.set(title="Доля дней с наблюдением по фазе сезона (train + test, сглаживание 7 дней)",
           ylabel="Доля дней", ylim=(0, 0.5))
    ax.legend()
    save_fig(fig, "coverage_by_doy")


def _month_axis(ax: plt.Axes) -> None:
    """Подписи оси X по началам месяцев вместо дня года."""
    ticks = [91, 121, 152, 182, 213, 244, 274, 305]
    ax.set_xticks(ticks, ["апр", "май", "июн", "июл", "авг", "сен", "окт", "ноя"])
    ax.set_xlabel("Месяц")


def grid_regime(df: pd.DataFrame) -> dict:
    """Режим сетки: полигон-годы с полной ежедневной сеткой (213 строк) и без неё.

    У части полигонов строки есть только в дни наблюдений, поэтому их «доля известных»
    на тепловой карте завышена, а погода ERA5 в остальные дни отсутствует.
    """
    rows = df.groupby(["anon_polygon_id", "cal_year"]).size()
    partial = rows < SEASON_DAYS
    polygons = rows.loc[partial].index.get_level_values(0).unique().sort_values()
    return {
        "polygon_years_total": int(len(rows)),
        "polygon_years_partial_grid": int(partial.sum()),
        "polygons_partial_grid": polygons.to_list(),
        "partial_grid_rows_median": float(rows.loc[partial].median()) if partial.any() else None,
    }


def era5_sharing(df: pd.DataFrame) -> dict:
    """Сколько различных рядов ERA5 приходится на полигоны: соседние поля сидят в одной ячейке реанализа."""
    wide = df.dropna(subset=["era5_temp_c"]).pivot_table(index="date", columns="anon_polygon_id", values="era5_temp_c")
    signature = wide.round(6).T.apply(lambda r: hash(tuple(r.fillna(-999).to_numpy())), axis=1)
    groups = [sorted(g.index.to_list()) for _, g in signature.groupby(signature) if len(g) > 1]
    return {"polygons_with_era5": int(wide.shape[1]), "distinct_era5_series": int(signature.nunique()),
            "shared_era5_groups": groups}


def run(train: pd.DataFrame, test: pd.DataFrame) -> dict:
    """Строит все графики покрытия и возвращает ключевые числа."""
    share = plot_known_share_by_year(train, test)
    avail = plot_sensor_availability(train, test)
    plot_polygon_year_heatmap(train, test)
    plot_coverage_by_doy(train, test)
    gaps = plot_natural_gaps(train)
    return {
        "grid_train": grid_regime(train),
        "grid_test": grid_regime(test),
        "era5_train": era5_sharing(train),
        "era5_test": era5_sharing(test),
        "known_share_train_by_year": {int(y): round(float(v), 3) for y, v in share["train"].dropna().items()},
        "known_share_test_by_year": {int(y): round(float(v), 3) for y, v in share["test (известные)"].dropna().items()},
        "sensor_availability_overall": {c: round(float(v), 3) for c, v in avail.mean().items()},
        "sensor_availability_2024": {c: round(float(v), 3) for c, v in avail.loc[2024].items()},
        **gaps,
    }
