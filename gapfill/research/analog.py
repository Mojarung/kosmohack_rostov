"""Перенос измерения соседнего поля с локальной поправкой на различие сезонных кривых."""

import numpy as np
import pandas as pd

from gapfill.config import SENSOR_CODE
from gapfill.research.features import _pair_statistics, sensor_rows
from gapfill.smooth import local_linear


def _one_sensor(targets, rows, name):
    """Отличие полей оценивается по ближайшим видимым наблюдениям текущего сезона."""
    piv = rows.pivot(index="day_num", columns="pid", values="value")
    _, _, score = _pair_statistics(piv.to_numpy())
    pos = {p: i for i, p in enumerate(piv.columns)}
    groups = {p: r for p, r in rows.groupby("pid")}
    query_values = piv.reindex(targets.day_num).to_numpy()
    pieces = []
    for pid, indices in targets.groupby("pid", sort=False).indices.items():
        if pid not in pos:
            continue
        own = groups[pid]
        day, y = own.day_num.to_numpy(), own.value.to_numpy()
        tq = targets.iloc[indices].day_num.to_numpy()
        donors = np.argsort(-score[pos[pid]])[:10]
        preds, weights = {15: [], 35: []}, []
        for j in donors:
            donor = groups[piv.columns[j]]
            d, v = donor.day_num.to_numpy(), donor.value.to_numpy()
            k = np.searchsorted(d, day)
            nearest = np.minimum(np.abs(day - d[np.maximum(k - 1, 0)]),
                                 np.abs(day - d[np.minimum(k, len(d) - 1)]))
            good = nearest <= 12
            delta = y[good] - np.interp(day[good], d, v)
            for bw in preds:
                correction, _, count = local_linear(day[good], delta, tq, float(bw), min_points=2.)
                pred = query_values[indices, j] + correction
                preds[bw].append(np.where(count > .5, pred, np.nan))
            weights.append(score[pos[pid], j])
        block = {}
        for bw, values in preds.items():
            matrix = np.asarray(values).T
            available = np.isfinite(matrix)
            w = np.where(available, np.asarray(weights)[None, :], 0.)
            for n in (1, 3, 10):
                order = np.argsort(-w, axis=1)[:, :n]
                wt = np.take_along_axis(w, order, axis=1)
                pt = np.take_along_axis(np.nan_to_num(matrix), order, axis=1)
                den = wt.sum(1)
                block[f"an_{name}_bw{bw}_k{n}"] = np.where(den > 1e-9, (wt * pt).sum(1) / np.maximum(den, 1e-9), np.nan)
                block[f"an_{name}_bw{bw}_k{n}_w"] = den
        pieces.append(pd.DataFrame(block, index=targets.index[indices]))
    return pd.concat(pieces).reindex(targets.index)


def analog_features(targets, context):
    """Новые признаки не требуют координат и не используют скрытые целевые значения."""
    blocks = [_one_sensor(targets, sensor_rows(context, name), name) for name in SENSOR_CODE]
    out = pd.concat(blocks, axis=1)
    out["an_polygon_id"] = targets.pid.str.removeprefix("AOI-").astype(int).to_numpy()
    return out
