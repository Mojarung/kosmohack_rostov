"""Локальная линейная регрессия с гауссовым ядром по времени.

Используется как «гладкая кривая» сезона: оценка значения и наклона в произвольный день
по нерегулярным наблюдениям. Поддерживает leave-one-out (точка исключается из своей оценки),
что нужно для честных остатков контекстных наблюдений.
"""

from __future__ import annotations

import numpy as np

from gapfill.config import KERNEL_CUTOFF

CHUNK = 512
EPS = 1e-9


def _weights(x_ctx: np.ndarray, x_eval: np.ndarray, bw: float) -> np.ndarray:
    """Матрица весов ядра (n_eval × n_ctx); за пределами cutoff полос вес ноль."""
    d = x_ctx[None, :] - x_eval[:, None]
    w = np.exp(-0.5 * (d / bw) ** 2)
    w[np.abs(d) > KERNEL_CUTOFF * bw] = 0.0
    return w


def _solve(w: np.ndarray, d: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Взвешенная линейная подгонка y ≈ a + b·d по строкам матрицы весов; возвращает (a, b, сумма весов)."""
    s0 = w.sum(1)
    s1 = (w * d).sum(1)
    s2 = (w * d * d).sum(1)
    t0 = (w * y[None, :]).sum(1)
    t1 = (w * d * y[None, :]).sum(1)
    det = s0 * s2 - s1 * s1
    ok = det > EPS
    a = np.where(ok, (s2 * t0 - s1 * t1) / np.where(ok, det, 1.0), t0 / np.maximum(s0, EPS))
    b = np.where(ok, (s0 * t1 - s1 * t0) / np.where(ok, det, 1.0), 0.0)
    empty = s0 < 1e-6
    a[empty] = np.nan
    b[empty] = np.nan
    return a, b, s0


def local_linear(x_ctx: np.ndarray, y_ctx: np.ndarray, x_eval: np.ndarray, bw: float,
                 loo: bool = False) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Оценка (значение, наклон, сумма весов) в точках x_eval по контексту (x_ctx, y_ctx).

    При loo=True x_eval должен совпадать с x_ctx: каждая точка исключается из собственной оценки.
    """
    x_ctx = np.asarray(x_ctx, dtype=float)
    y_ctx = np.asarray(y_ctx, dtype=float)
    x_eval = np.asarray(x_eval, dtype=float)
    n = len(x_eval)
    a = np.full(n, np.nan)
    b = np.full(n, np.nan)
    s = np.zeros(n)
    if len(x_ctx) == 0 or n == 0:
        return a, b, s
    for start in range(0, n, CHUNK):
        sl = slice(start, min(start + CHUNK, n))
        w = _weights(x_ctx, x_eval[sl], bw)
        if loo:
            rows = np.arange(sl.start, sl.stop)
            w[rows - sl.start, rows] = 0.0
        d = x_ctx[None, :] - x_eval[sl, None]
        a[sl], b[sl], s[sl] = _solve(w, d, y_ctx)
    return a, b, s
