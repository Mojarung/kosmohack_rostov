"""Робастное сглаживание сезонной кривой NDVI.

Ряд NDVI испорчен облаками и тенями: одиночные провалы вниз на 0.3-0.5 — обычное дело.
Обычная интерполяция по соседям на таких точках ломается. Локальная линейная регрессия
с ядром tricube и итерациями Тьюки (biweight) даёт кривую, которая игнорирует такие
выбросы, но следует за реальной динамикой сезона.
"""

from __future__ import annotations

import numpy as np


def _tricube(u: np.ndarray) -> np.ndarray:
    u = np.clip(np.abs(u), 0.0, 1.0)
    return (1.0 - u ** 3) ** 3


def _fit_at(x: np.ndarray, y: np.ndarray, targets: np.ndarray,
            robust_w: np.ndarray, bandwidth: float, min_points: int,
            leave_one_out: bool = False) -> np.ndarray:
    """Локальная линейная регрессия во всех точках ``targets`` сразу (матрица весов).

    ``leave_one_out`` — когда ``targets`` это сами ``x``: точка не участвует в собственной
    оценке. Так остаток «наблюдение минус кривая» честно показывает, насколько день выбивается.
    """
    d = np.abs(targets[:, None] - x[None, :])                 # (n_targets, n_x)
    w = _tricube(d / bandwidth) * robust_w[None, :]
    if leave_one_out and targets.shape[0] == x.shape[0]:
        np.fill_diagonal(w, 0.0)
    empty = w.sum(axis=1) <= 1e-9
    if empty.any():
        # окно пустое — берём min_points ближайших наблюдений с единичным весом
        k = min(min_points, x.size)
        idx = np.argsort(d[empty], axis=1)[:, :k]
        w_e = np.zeros((idx.shape[0], x.size))
        np.put_along_axis(w_e, idx, 1.0, axis=1)
        w[empty] = w_e
    sw = w.sum(axis=1)
    mx = (w * x[None, :]).sum(axis=1) / sw
    my = (w * y[None, :]).sum(axis=1) / sw
    dx = x[None, :] - mx[:, None]
    sxx = (w * dx ** 2).sum(axis=1)
    sxy = (w * dx * (y[None, :] - my[:, None])).sum(axis=1)
    slope = np.where(sxx > 1e-9, sxy / np.where(sxx > 1e-9, sxx, 1.0), 0.0)
    return my + slope * (targets - mx)


def robust_local_linear(x, y, x0, bandwidth: float = 20.0, iters: int = 2,
                        min_points: int = 3, leave_one_out: bool = False) -> np.ndarray:
    """Оценивает значение ряда в точках ``x0`` локальной линейной регрессией.

    ``x`` — дни (число), ``y`` — значения в ОДНОЙ шкале, ``bandwidth`` — полуширина окна в днях,
    ``iters`` — число робастных итераций (после каждой большие остатки получают меньший вес),
    ``leave_one_out`` — при ``x0 is x`` исключать точку из собственной оценки.
    """
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    x0 = np.atleast_1d(np.asarray(x0, float))
    if x.size == 0:
        return np.full(x0.shape, np.nan)
    if x.size < min_points:
        return np.full(x0.shape, float(np.nanmean(y)))

    robust_w = np.ones_like(y)
    for _ in range(iters):
        fit = _fit_at(x, y, x, robust_w, bandwidth, min_points)
        resid = y - fit
        s = float(np.median(np.abs(resid)))
        s = max(s, 1e-6)
        u = np.clip(resid / (6.0 * s), -1, 1)
        robust_w = (1.0 - u ** 2) ** 2
    return _fit_at(x, y, x0, robust_w, bandwidth, min_points, leave_one_out=leave_one_out)
