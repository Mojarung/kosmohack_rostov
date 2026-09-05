"""Дополнительные признаки: все сенсоры и перенос измерений между похожими полями."""

import numpy as np
import pandas as pd

from gapfill.config import SENSOR_CODE, SENSOR_OFFSET
from gapfill.features_series import PolygonContext, _neighbor_block
from gapfill.smooth import local_linear


def sensor_rows(context, name):
    """Извлекает измерения сенсора, в том числе не выбранного главным в исходной строке."""
    col = f"{name}_ndvi"
    rows = context.loc[context[col].notna()].copy()
    rows["value"] = rows[col].clip(-.1, 1.)
    return rows.sort_values(["pid", "day_num"])


def _curve_features(rows, times, prefix):
    """Кривые по отдельному сенсору и устойчивые характеристики окна."""
    day, value = rows.day_num.to_numpy(), rows.value.to_numpy()
    out = {}
    for bw in (6., 12., 24.):
        a, b, w = local_linear(day, value, times, bw, min_points=2.)
        out[f"{prefix}_curve{int(bw)}"] = a
        out[f"{prefix}_slope{int(bw)}"] = b
        out[f"{prefix}_weight{int(bw)}"] = w
    for half in (16, 40):
        lo = np.searchsorted(day, times - half)
        hi = np.searchsorted(day, times + half, side="right")
        out[f"{prefix}_median{half}"] = np.array([
            np.median(value[a:b]) if b > a else np.nan for a, b in zip(lo, hi)])
    return out


def full_sensor_features(targets, context):
    """Соседи и кривые всех доступных сенсоров без использования целевой строки."""
    pieces = []
    for name, code in SENSOR_CODE.items():
        rows = sensor_rows(context, name)
        grouped = dict(iter(rows.groupby("pid")))
        parts = []
        for pid, tg in targets.groupby("pid", sort=False):
            own = grouped.get(pid, rows.iloc[:0])
            day = own.day_num.to_numpy()
            val = own.value.to_numpy()
            evi = own[f"{name}_evi"].to_numpy()
            ndwi = own.get(f"{name}_ndwi", pd.Series(np.nan, index=own.index)).to_numpy()
            ctx = PolygonContext(day, own.year.to_numpy(), own.doy.to_numpy(), val,
                                 np.full(len(own), code), val - SENSOR_OFFSET[name], evi, ndwi)
            prefix = f"raw_{name}"
            block = _neighbor_block(ctx, tg.day_num.to_numpy(), prefix, with_indices=True)
            block = {k: v for k, v in block.items() if not k.endswith(("_sensor", "_h")) and "3_" not in k}
            dp, dn = block[f"{prefix}_p1_days"], block[f"{prefix}_n1_days"]
            vp, vn = block[f"{prefix}_p1_val"], block[f"{prefix}_n1_val"]
            inter = vp + (vn - vp) * dp / np.maximum(dp + dn, 1)
            block[f"{prefix}_interp"] = np.where(np.isnan(inter), np.fmin(vp, vn), inter)
            block.update(_curve_features(own, tg.day_num.to_numpy(), prefix))
            parts.append(pd.DataFrame(block, index=tg.index))
        pieces.append(pd.concat(parts).loc[targets.index])
    return pd.concat(pieces, axis=1)


def _pair_statistics(values):
    """Сжатые парные регрессии по совместным видимым измерениям."""
    present = np.isfinite(values).astype(float)
    x = np.nan_to_num(values)
    n = present.T @ present
    sm = x.T @ present
    mean = sm / np.maximum(n, 1)
    sq = (x * x).T @ present
    var = np.maximum(sq / np.maximum(n, 1) - mean ** 2, 0.)
    cov = x.T @ x / np.maximum(n, 1) - mean * mean.T
    corr = cov / np.sqrt(np.maximum(var * var.T, 1e-8))
    beta = np.clip(cov / (var.T + .002), 0., 2.)
    score = np.clip(corr, 0., 1.) ** 4 * n / (n + 40.)
    score[n < 12] = 0.
    np.fill_diagonal(score, 0.)
    return mean, beta, score


def _sensor_spatial(targets, rows, name):
    """Перенос абсолютного значения и остатка от кривой по ближайшим полям."""
    piv = rows.pivot(index="day_num", columns="pid", values="value")
    pids = list(piv.columns)
    pid_pos = {p: i for i, p in enumerate(pids)}
    values = piv.to_numpy()
    means, betas, scores = _pair_statistics(values)
    res_rows = []
    for _, own in rows.groupby("pid"):
        curve, _, count = local_linear(own.day_num.to_numpy(), own.value.to_numpy(),
                                       own.day_num.to_numpy(), 15. if name == "modis" else 8., loo=True,
                                       min_points=2.)
        res_rows.append(own.assign(res=np.where(count >= .5, (own.value - curve).clip(-.3, .3), np.nan)))
    res_piv = pd.concat(res_rows).pivot(index="day_num", columns="pid", values="res").reindex_like(piv)
    _, res_betas, res_scores = _pair_statistics(res_piv.to_numpy())
    v_query = piv.reindex(targets.day_num).to_numpy()
    r_query = res_piv.reindex(targets.day_num).to_numpy()
    out = {}
    for pid, indices in targets.groupby("pid", sort=False).indices.items():
        if pid not in pid_pos:
            continue
        p = pid_pos[pid]
        v, r = v_query[indices], r_query[indices]
        calibrated = means[p] + betas[p] * (v - means[:, p])
        for kind, query, weight in (("value", calibrated, scores[p]),
                                    ("res", r * res_betas[p], res_scores[p])):
            valid_weight = np.where(np.isfinite(query), weight[None, :], 0.)
            order = np.argsort(-valid_weight, axis=1)
            for k in (1, 3, 8, 78):
                take = order[:, :min(k, len(pids))]
                w = np.take_along_axis(valid_weight, take, axis=1)
                y = np.take_along_axis(np.nan_to_num(query), take, axis=1)
                den = w.sum(1)
                pred = np.where(den > 0, (w * y).sum(1) / np.maximum(den, 1e-9), np.nan)
                for suffix, a in (("pred", pred), ("weight", den)):
                    col = f"sp_{name}_{kind}_k{k}_{suffix}"
                    out.setdefault(col, np.full(len(targets), np.nan))[indices] = a
    return pd.DataFrame(out, index=targets.index)


def spatial_features(targets, context):
    """Все пространственные признаки; сами целевые значения не попадают в матрицы."""
    return pd.concat([_sensor_spatial(targets, sensor_rows(context, name), name)
                      for name in SENSOR_CODE], axis=1)
