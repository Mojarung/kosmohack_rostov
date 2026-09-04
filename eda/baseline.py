"""Синтетические пропуски на train и базовые методы восстановления primary_ndvi.

Маскируем часть известных точек train, восстанавливаем их простыми методами
и считаем RMSE / GapScore — это отправная точка для сравнения моделей.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import ks_2samp

from eda.config import CROP_ORDER, GAP_SCORE_MAX, RANDOM_SEED, RMSE_THRESHOLD, TARGET
from eda.gaps import compute_neighbors, sensor_of
from eda.plotting import save_fig, setup_style

N_SYNTHETIC = 3000
DOY_WINDOW = 7  # окно ±дней doy для собственной климатологии
DIST_BINS = [0, 3, 7, 15, 30, np.inf]
DIST_LABELS = ["1–3", "4–7", "8–15", "16–30", ">30"]
DIST_CLIP = 60
POLYGON = ["anon_polygon_id"]
GAP_COLS = [TARGET, "anon_polygon_id", "date", "crop_type", "cal_year", "cal_doy",
            "ndvi_climatology_mean", "n_reference_years"]


def gap_score(rmse: float) -> float:
    """GapScore из ТЗ: 30 баллов при RMSE = 0, ноль при RMSE >= 0.10."""
    return round(GAP_SCORE_MAX * max(0.0, 1 - rmse / RMSE_THRESHOLD), 2)


def sample_synthetic_gaps(train: pd.DataFrame, n: int = N_SYNTHETIC, seed: int = RANDOM_SEED) -> pd.Index:
    """Выбирает n известных точек так, чтобы пары полигон-год были представлены примерно поровну."""
    known = train.loc[train["is_known"]]
    group_size = known.groupby(["anon_polygon_id", "cal_year"], observed=True)[TARGET].transform("size")
    return known.sample(n=n, weights=1.0 / group_size, random_state=seed).index


def build_gap_table(train: pd.DataFrame, gap_index: pd.Index) -> pd.DataFrame:
    """Таблица замаскированных точек с истинным target и признаками соседей.

    Соседи ищутся только среди оставшихся известных точек того же полигона.
    """
    is_target = train.index.isin(gap_index)
    is_known = train["is_known"].to_numpy() & ~is_target
    neighbors = compute_neighbors(train, is_target, is_known, POLYGON)
    return train.loc[gap_index, GAP_COLS].join(neighbors)


def predict_nearest_mean(g: pd.DataFrame) -> pd.Series:
    """Baseline из ТЗ: среднее ближайших известных значений до и после."""
    return g[["val_prev", "val_next"]].mean(axis=1)


def predict_nearest_one(g: pd.DataFrame) -> pd.Series:
    """Значение ближайшей по времени известной точки."""
    use_prev = g["days_prev"].fillna(np.inf) <= g["days_next"].fillna(np.inf)
    return g["val_prev"].where(use_prev, g["val_next"])


def predict_linear_time(g: pd.DataFrame) -> pd.Series:
    """Линейная интерполяция по времени между соседями; без пары — nearest_mean."""
    w = g["days_prev"] / (g["days_prev"] + g["days_next"])
    interp = g["val_prev"] + (g["val_next"] - g["val_prev"]) * w
    return interp.fillna(predict_nearest_mean(g))


def predict_climatology(g: pd.DataFrame) -> pd.Series:
    """Климатическая норма из той же строки (если есть), иначе nearest_mean."""
    clim = g["ndvi_climatology_mean"].where(g["n_reference_years"] > 0)
    return clim.fillna(predict_nearest_mean(g))


def own_climatology(known: pd.DataFrame, g: pd.DataFrame, window: int = DOY_WINDOW) -> pd.Series:
    """Среднее известных значений того же полигона в окне ±window дней doy по другим годам."""
    known_by_poly = {k: v for k, v in known.groupby("anon_polygon_id", observed=True)}
    pieces = []
    for poly, gg in g.groupby("anon_polygon_id", observed=True):
        kk = known_by_poly.get(poly)
        if kk is None:
            pieces.append(pd.Series(np.nan, index=gg.index))
            continue
        near = np.abs(gg["cal_doy"].to_numpy()[:, None] - kk["cal_doy"].to_numpy()[None, :]) <= window
        other_year = gg["cal_year"].to_numpy()[:, None] != kk["cal_year"].to_numpy()[None, :]
        mask = near & other_year
        total = np.where(mask, kk[TARGET].to_numpy()[None, :], 0.0).sum(axis=1)
        count = mask.sum(axis=1)
        pieces.append(pd.Series(np.where(count > 0, total / np.maximum(count, 1), np.nan), index=gg.index))
    return pd.concat(pieces).reindex(g.index)


def predict_doy_polygon_mean(g: pd.DataFrame, known: pd.DataFrame) -> pd.Series:
    """Собственная климатология полигона по doy; без данных — nearest_mean."""
    return own_climatology(known, g).fillna(predict_nearest_mean(g))


def predict_all(g: pd.DataFrame, known: pd.DataFrame) -> pd.DataFrame:
    """Предсказания всех методов (столбцы) для замаскированных точек (строки)."""
    preds = pd.DataFrame({
        "nearest_mean": predict_nearest_mean(g),
        "linear_time": predict_linear_time(g),
        "nearest_one": predict_nearest_one(g),
        "climatology": predict_climatology(g),
        "doy_polygon_mean": predict_doy_polygon_mean(g, known),
    })
    return preds.assign(blend=preds[["linear_time", "climatology"]].mean(axis=1))


def rmse(y: pd.Series, p: pd.Series) -> float:
    """Корень из средней квадратичной ошибки (NaN игнорируются)."""
    return float(np.sqrt(np.nanmean((y - p) ** 2)))


def grouped_rmse(y: pd.Series, preds: pd.DataFrame, group: pd.Series) -> pd.DataFrame:
    """RMSE каждого метода в разрезе группы: строки — группы, столбцы — методы."""
    sq = preds.sub(y, axis=0) ** 2
    return sq.groupby(group, observed=True).mean().pow(0.5)


def method_metrics(y: pd.Series, preds: pd.DataFrame) -> pd.DataFrame:
    """RMSE, MAE и GapScore по методам, отсортировано по RMSE."""
    rows = {
        m: {"rmse": rmse(y, preds[m]), "mae": float(np.nanmean(np.abs(y - preds[m]))),
            "n_nan": int(preds[m].isna().sum())}
        for m in preds.columns
    }
    df = pd.DataFrame(rows).T.sort_values("rmse")
    return df.assign(gap_score=df["rmse"].map(gap_score))


def plot_method_rmse(metrics: pd.DataFrame) -> None:
    """RMSE по методам с порогом метрики 0.10."""
    fig, ax = plt.subplots(figsize=(9, 4.2))
    colors = ["#3a9d5d" if v < RMSE_THRESHOLD else "#c8423f" for v in metrics["rmse"]]
    bars = ax.bar(metrics.index, metrics["rmse"], color=colors)
    ax.bar_label(bars, labels=[f"{r:.4f}\nGapScore {g}" for r, g in zip(metrics["rmse"], metrics["gap_score"])],
                 fontsize=8)
    ax.axhline(RMSE_THRESHOLD, ls="--", color="gray", label="порог метрики RMSE = 0.10")
    ax.set(title="Baseline-методы на синтетических пропусках train", ylabel="RMSE", xlabel="")
    ax.set_ylim(0, max(RMSE_THRESHOLD, metrics["rmse"].max()) * 1.35)
    ax.legend(loc="upper left")
    save_fig(fig, "baseline_rmse_by_method")


def plot_rmse_by_distance(by_dist: pd.DataFrame, counts: pd.Series, methods: list[str]) -> None:
    """RMSE лучших методов в зависимости от расстояния до ближайшего соседа."""
    fig, ax = plt.subplots(figsize=(9, 4.2))
    for m in methods:
        ax.plot(by_dist.index.astype(str), by_dist[m], marker="o", label=m)
    ax.axhline(RMSE_THRESHOLD, ls="--", color="gray", lw=0.9)
    labels = [f"{b}\n(n={counts.get(b, 0)})" for b in by_dist.index.astype(str)]
    ax.set_xticks(range(len(labels)), labels)
    ax.set(title="RMSE по расстоянию до ближайшего известного соседа (дней)", ylabel="RMSE", xlabel="")
    ax.legend()
    save_fig(fig, "baseline_rmse_by_distance")


def plot_scatter(y: pd.Series, p: pd.Series, name: str) -> None:
    """Истина против предсказания лучшего метода."""
    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    ax.scatter(y, p, s=6, alpha=0.3, color="#2b6cb0")
    lo, hi = float(min(y.min(), p.min())), float(max(y.max(), p.max()))
    ax.plot([lo, hi], [lo, hi], color="gray", lw=1, ls="--")
    ax.set(title=f"{name}: истина vs предсказание", xlabel="primary_ndvi (истина)", ylabel="предсказание")
    save_fig(fig, "baseline_scatter_best")


def plot_error_hist(err: pd.Series) -> None:
    """Распределение ошибок baseline из ТЗ."""
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.histplot(err.dropna(), bins=80, ax=ax, color="#dd6b20")
    for v in (-RMSE_THRESHOLD, RMSE_THRESHOLD):
        ax.axvline(v, ls="--", color="gray", lw=0.9)
    ax.set(title="nearest_mean: ошибка предсказания (пред − истина)", xlabel="ошибка", ylabel="точек")
    save_fig(fig, "baseline_error_hist_nearest_mean")


def plot_dist_compare(syn: pd.Series, real: pd.Series, ks: float) -> None:
    """Расстояние до ближайшего соседа: синтетика на train против контрольных точек test."""
    data = pd.concat([
        pd.DataFrame({"min_dist": syn.clip(upper=DIST_CLIP), "набор": "синтетические пропуски (train)"}),
        pd.DataFrame({"min_dist": real.clip(upper=DIST_CLIP), "набор": "контрольные точки (test)"}),
    ], ignore_index=True)
    fig, ax = plt.subplots(figsize=(9, 4))
    sns.histplot(data=data, x="min_dist", hue="набор", stat="density", common_norm=False,
                 element="step", bins=np.arange(0.5, DIST_CLIP + 1.5, 2), ax=ax,
                 palette={"синтетические пропуски (train)": "#2b6cb0", "контрольные точки (test)": "#c8423f"})
    ax.set(title=f"Дней до ближайшего известного соседа (по полигону), KS = {ks:.3f}",
           xlabel="дней (хвост обрезан на 60)", ylabel="плотность")
    save_fig(fig, "baseline_min_dist_synthetic_vs_test")


def climatology_consistency(train: pd.DataFrame) -> dict:
    """Насколько ndvi_climatology_mean постоянна для пары полигон-doy между годами."""
    known = train.loc[train["is_known"] & (train["n_reference_years"] > 0)]
    per_key = known.groupby(["anon_polygon_id", "cal_doy"], observed=True)["ndvi_climatology_mean"]
    nunique, size = per_key.nunique(), per_key.size()
    multi = size > 1
    return {
        "clim_keys_with_multiple_years": int(multi.sum()),
        "clim_share_varying_between_years": float((nunique[multi] > 1).mean()) if multi.any() else 0.0,
    }


def sensor_consistency(train: pd.DataFrame, g: pd.DataFrame, preds: pd.DataFrame) -> dict:
    """RMSE nearest_mean в зависимости от того, совпадает ли сенсор соседей с сенсором скрытой точки.

    Сенсор скрытой точки известен на train (правило приоритета s2 > landsat > modis).
    """
    truth = sensor_of(train.loc[g.index])
    same_prev = g["sensor_prev"].eq(truth) | g["sensor_prev"].isna()
    same_next = g["sensor_next"].eq(truth) | g["sensor_next"].isna()
    both_same = same_prev & same_next
    none_same = ~same_prev & ~same_next
    err = preds["nearest_mean"] - g[TARGET]
    grp = pd.Series(np.select([both_same, none_same], ["оба соседа того же сенсора", "оба соседа другого сенсора"],
                              default="один сосед другого сенсора"), index=g.index)
    per_group = (err ** 2).groupby(grp).mean().pow(0.5)
    counts = grp.value_counts()
    return {
        "rmse_nearest_mean_by_sensor_match": _to_dict(per_group),
        "n_by_sensor_match": {str(k): int(v) for k, v in counts.items()},
        "truth_sensor_share": _to_dict(truth.value_counts(normalize=True), 3),
    }


def _to_dict(s: pd.Series, digits: int = 4) -> dict:
    """Series -> JSON-совместимый словарь с округлением."""
    return {str(k): round(float(v), digits) for k, v in s.items()}


def run(train: pd.DataFrame, test: pd.DataFrame) -> dict:
    """Синтетические пропуски, baseline-методы, графики и сводка чисел."""
    setup_style()
    gap_index = sample_synthetic_gaps(train)
    g = build_gap_table(train, gap_index)
    known = train.loc[train["is_known"] & ~train.index.isin(gap_index)]
    preds = predict_all(g, known)
    y = g[TARGET]
    metrics = method_metrics(y, preds)
    best = str(metrics.index[0])

    dist_bin = pd.cut(g["min_dist"], DIST_BINS, labels=DIST_LABELS)
    by_dist = grouped_rmse(y, preds, dist_bin)
    by_crop = grouped_rmse(y, preds, g["crop_type"])
    by_year = grouped_rmse(y, preds, g["cal_year"])
    test_dist = compute_neighbors(test, test["is_gap"].to_numpy(), test["is_known"].to_numpy(), POLYGON)["min_dist"]
    ks = ks_2samp(g["min_dist"].dropna(), test_dist.dropna())

    plot_method_rmse(metrics)
    plot_rmse_by_distance(by_dist, dist_bin.value_counts(), list(metrics.index[:4]))
    plot_scatter(y, preds[best], best)
    plot_error_hist(preds["nearest_mean"] - y)
    plot_dist_compare(g["min_dist"], test_dist, ks.statistic)

    return {
        "n_synthetic": int(len(g)),
        "masked_share_of_known": float(len(g) / train["is_known"].sum()),
        "best_method": best,
        "metrics": {m: _to_dict(row) for m, row in metrics.iterrows()},
        "rmse_by_distance": {m: _to_dict(by_dist[m]) for m in by_dist.columns},
        "n_by_distance": {str(k): int(v) for k, v in dist_bin.value_counts().sort_index().items()},
        "rmse_by_crop_best": _to_dict(by_crop.loc[CROP_ORDER, best]),
        "rmse_by_crop_nearest_mean": _to_dict(by_crop.loc[CROP_ORDER, "nearest_mean"]),
        "rmse_by_year_best": _to_dict(by_year[best]),
        "rmse_by_year_nearest_mean": _to_dict(by_year["nearest_mean"]),
        "min_dist_median_synthetic": float(g["min_dist"].median()),
        "min_dist_median_test": float(test_dist.median()),
        "ks_min_dist_statistic": float(ks.statistic),
        "ks_min_dist_pvalue": float(ks.pvalue),
        **climatology_consistency(train),
        **sensor_consistency(train, g, preds),
    }
