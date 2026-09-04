"""Циклы съёмки сенсоров: можно ли по дате угадать, какой сенсор сформировал primary_ndvi.

MODIS MOD13Q1 — 16-дневные композиты с фиксированной сеткой дат, Sentinel-2 — 5-дневный
цикл (две орбиты дают дополнительные шаги 2/3 дня), Landsat 8/9 — 8-дневный цикл (соседние
витки дают шаг 1 день). В контрольных точках test сенсоры замаскированы, но дата известна,
поэтому источник можно восстановить по циклам того же полигона-года.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from eda.config import SENSOR_COLORS, SENSOR_NDVI
from eda.plotting import save_fig
from eda.sensors import SOURCES, with_source

EPOCH = pd.Timestamp("2000-01-01")
PERIODS = {"s2_ndvi": 5, "landsat_ndvi": 8, "modis_ndvi": 16}
KEYS = ["anon_polygon_id", "cal_year"]


def with_day_num(df: pd.DataFrame) -> pd.DataFrame:
    """Порядковый номер дня — для остатков по периоду цикла."""
    return df.assign(day_num=(df["date"] - EPOCH).dt.days.astype("int64"))


def _observed(train: pd.DataFrame, test: pd.DataFrame) -> pd.DataFrame:
    return with_day_num(pd.concat([train, test.loc[~test["is_gap"]]], ignore_index=True))


def revisit_diffs(both: pd.DataFrame, col: str) -> pd.Series:
    """Шаги в днях между соседними наблюдениями одного сенсора внутри полигона-сезона."""
    d = both.loc[both[col].notna()].sort_values(["anon_polygon_id", "date"])
    same_season = d.groupby("anon_polygon_id")["cal_year"].diff().eq(0)
    return d.groupby("anon_polygon_id")["date"].diff().dt.days.loc[same_season]


def plot_revisit_diffs(both: pd.DataFrame) -> dict:
    fig, axes = plt.subplots(1, 3, figsize=(14, 3.8))
    out = {}
    for ax, col in zip(axes, SENSOR_NDVI):
        diffs = revisit_diffs(both, col)
        share = diffs.value_counts(normalize=True).sort_index()
        share.loc[share.index <= 20].plot.bar(ax=ax, color=SENSOR_COLORS[col], width=0.8)
        ax.set(title=f"{col}: шаг между наблюдениями", xlabel="Дней", ylabel="Доля")
        ax.tick_params(axis="x", labelrotation=0)
        out[col] = {int(k): round(float(v), 3) for k, v in share.head(6).items()}
    fig.suptitle("Циклы повторной съёмки сенсоров (train + test)")
    save_fig(fig, "cycles_revisit_diffs")
    return {"revisit_diff_shares": out}


def modis_grid(both: pd.DataFrame) -> list[int]:
    """Дни года, в которые вообще встречаются наблюдения MODIS."""
    return sorted(int(v) for v in both.loc[both["modis_ndvi"].notna(), "cal_doy"].unique())


def _cycle_counts(both: pd.DataFrame, col: str) -> pd.DataFrame:
    """Число наблюдений сенсора в каждом полигоне-году для каждого остатка дня по периоду."""
    p = PERIODS[col]
    d = both.loc[both[col].notna()].assign(res=lambda x: x["day_num"] % p)
    return d.groupby(KEYS + ["res"]).size().rename(f"cnt_{col}").reset_index()


def _attach_counts(points: pd.DataFrame, counts: pd.DataFrame, col: str) -> pd.Series:
    p = PERIODS[col]
    keyed = points.assign(res=points["day_num"] % p)
    merged = keyed[KEYS + ["res"]].merge(counts, on=KEYS + ["res"], how="left")
    return merged[f"cnt_{col}"].fillna(0).to_numpy()


def predict_source(points: pd.DataFrame, reference: pd.DataFrame, leave_one_out: bool) -> np.ndarray:
    """Правило: s2, если остаток даты встречается у S2 того же полигона-года; иначе landsat;
    иначе modis, если день года лежит на сетке MODIS; иначе 'unknown'.

    При leave_one_out из счётчиков вычитается вклад самой точки (оценка на train).
    """
    cnt_s2 = _attach_counts(points, _cycle_counts(reference, "s2_ndvi"), "s2_ndvi")
    cnt_ls = _attach_counts(points, _cycle_counts(reference, "landsat_ndvi"), "landsat_ndvi")
    if leave_one_out:
        cnt_s2 = cnt_s2 - points["s2_ndvi"].notna().to_numpy()
        cnt_ls = cnt_ls - points["landsat_ndvi"].notna().to_numpy()
    on_grid = points["cal_doy"].isin(modis_grid(reference)).to_numpy()
    return np.select([cnt_s2 > 0, cnt_ls > 0, on_grid], SOURCES, default="unknown")


def evaluate_rule(train: pd.DataFrame, both: pd.DataFrame) -> dict:
    """Точность правила на известных точках train (leave-one-out по счётчикам)."""
    known = with_source(with_day_num(train.loc[train["is_known"]]))
    pred = predict_source(known, both, leave_one_out=True)
    truth = known["source"].astype(str).to_numpy()
    matrix = pd.crosstab(pd.Series(truth, name="истинный"), pd.Series(pred, name="предсказанный"))
    acc = float((pred == truth).mean())
    acc_by_year = (pd.Series(pred == truth).groupby(known["cal_year"].to_numpy()).mean())

    fig, ax = plt.subplots(figsize=(6, 4.5))
    sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues", ax=ax, cbar=False)
    ax.set_title(f"Источник primary_ndvi по дате: точность правила {acc:.1%} (train, LOO)")
    save_fig(fig, "cycles_source_rule_confusion")
    return {
        "rule_accuracy_train": round(acc, 4),
        "rule_accuracy_by_year": {int(y): round(float(v), 3) for y, v in acc_by_year.items()},
        "rule_confusion": {str(t): {str(p): int(v) for p, v in row.items()} for t, row in matrix.iterrows()},
    }


def predict_gaps(test: pd.DataFrame, both: pd.DataFrame) -> dict:
    """Предсказанный источник для контрольных точек test и сравнение с известными точками test."""
    gaps = with_day_num(test.loc[test["is_gap"]])
    pred = pd.Series(predict_source(gaps, both, leave_one_out=False), name="source")
    known = with_source(test.loc[test["is_known"]])["source"].astype(str)
    share = pd.concat([pred.value_counts(normalize=True).rename("контрольные (предсказано)"),
                       known.value_counts(normalize=True).rename("известные (факт)")], axis=1).fillna(0)
    fig, ax = plt.subplots(figsize=(7, 3.8))
    share.plot.bar(ax=ax, width=0.75, color=["#dd6b20", "#a0aec0"])
    ax.set(title="Источник primary_ndvi в test: контрольные точки против известных", ylabel="Доля")
    ax.tick_params(axis="x", labelrotation=0)
    save_fig(fig, "cycles_gap_source_prediction")
    return {"gap_source_predicted": {k: int(v) for k, v in pred.value_counts().items()},
            "gap_share_on_modis_grid": round(float(gaps["cal_doy"].isin(modis_grid(both)).mean()), 3)}


def run(train: pd.DataFrame, test: pd.DataFrame) -> dict:
    """Графики по циклам съёмки и оценка правила определения сенсора по дате."""
    both = _observed(train, test)
    out = plot_revisit_diffs(both)
    out["modis_grid_doy"] = modis_grid(both)
    out |= evaluate_rule(train, both)
    out |= predict_gaps(test, both)
    return out
