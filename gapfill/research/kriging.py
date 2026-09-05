"""Условное восстановление общего шума сенсора по ковариациям полей."""

import numpy as np
import pandas as pd

from gapfill.config import SENSOR_CODE
from gapfill.research.features import sensor_rows
from gapfill.smooth import local_linear


def covariance(values):
    """Попарная ковариация при пропусках со сжатием и положительным спектром."""
    present = np.isfinite(values).astype(float)
    x = np.nan_to_num(values)
    count = present.T @ present
    mean = x.T @ present / np.maximum(count, 1)
    cov = x.T @ x / np.maximum(count, 1) - mean * mean.T
    variance = np.maximum(np.diag(cov), .0002)
    cov *= count / (count + 50.)
    np.fill_diagonal(cov, variance)
    eig, vectors = np.linalg.eigh(cov)
    cov = (vectors * np.maximum(eig, np.median(variance) * .02)) @ vectors.T
    return cov


def _sensor_features(targets, context, name):
    """Все поля одного дня предсказываются за одно решение небольшой системы."""
    rows = sensor_rows(context, name)
    pieces = []
    for _, own in rows.groupby("pid"):
        curve, _, weight = local_linear(own.day_num.to_numpy(), own.value.to_numpy(), own.day_num.to_numpy(),
                                         15. if name == "modis" else 8., loo=True, min_points=2.)
        pieces.append(own.assign(res=np.where(weight > .5, (own.value - curve).clip(-.3, .3), np.nan)))
    pivot = pd.concat(pieces).pivot(index="day_num", columns="pid", values="res")
    values = pivot.to_numpy()
    cov = covariance(values)
    mean = np.nanmean(values, axis=0)
    positions = {p: i for i, p in enumerate(pivot.columns)}
    row_of = {d: i for i, d in enumerate(pivot.index)}
    out = {f"kg_{name}_{suffix}": np.full(len(targets), np.nan)
           for suffix in ("n", "pred10", "sd10", "pred40", "sd40")}
    for day, indices in targets.groupby("day_num", sort=False).indices.items():
        if day not in row_of:
            continue
        r = values[row_of[day]]
        available = np.isfinite(r)
        columns = np.array([positions.get(p, -1) for p in targets.iloc[indices].pid])
        out[f"kg_{name}_n"][indices] = available.sum()
        known = columns >= 0
        indices, columns = indices[known], columns[known]
        if not available.any() or not len(columns):
            continue
        delta = r[available] - mean[available]
        for shrink in (.1, .4):
            matrix = (1 - shrink) * cov + shrink * np.diag(np.diag(cov))
            cross = matrix[np.ix_(columns, available)]
            conditional = np.linalg.solve(matrix[np.ix_(available, available)], cross.T)
            pred = mean[columns] + conditional.T @ delta
            uncertainty = np.maximum(np.diag(matrix)[columns] - (cross * conditional.T).sum(1), 0.)
            out[f"kg_{name}_pred{int(100 * shrink)}"][indices] = pred
            out[f"kg_{name}_sd{int(100 * shrink)}"][indices] = np.sqrt(uncertainty)
    return pd.DataFrame(out, index=targets.index)


def kriging_features(targets, context):
    """Признаки учитывают корреляции доноров между собой, а не только с целевым полем."""
    return pd.concat([_sensor_features(targets, context, name) for name in SENSOR_CODE], axis=1)
