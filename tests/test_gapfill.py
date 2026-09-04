"""Тесты ядра пайплайна восстановления primary_ndvi (без обучения моделей)."""

import numpy as np
import pandas as pd
import pytest

from gapfill.data import gap_score, make_mask, rmse
from gapfill.data import testlike_rmse as weighted_rmse
from gapfill.smooth import local_linear


def test_local_linear_recovers_line():
    """На точной прямой локальная линейная регрессия возвращает её значения и наклон."""
    x = np.arange(0, 60, 3, dtype=float)
    y = 0.1 + 0.02 * x
    a, b, s = local_linear(x, y, np.array([10.0, 31.0]), bw=8.0)
    assert np.allclose(a, 0.1 + 0.02 * np.array([10.0, 31.0]), atol=1e-6)
    assert np.allclose(b, 0.02, atol=1e-6)
    assert (s > 0).all()


def test_local_linear_loo_excludes_self():
    """При loo=True выброс не влияет на собственную оценку: остаток равен величине выброса."""
    x = np.arange(0, 40, 4, dtype=float)
    y = np.full(len(x), 0.5)
    y[5] = 0.9
    a, _, _ = local_linear(x, y, x, bw=6.0, loo=True)
    assert abs(a[5] - 0.5) < 1e-6
    assert (np.abs(y[5] - a[5]) - 0.4) < 1e-6


def test_local_linear_empty_context_gives_nan():
    a, b, s = local_linear(np.array([]), np.array([]), np.array([1.0, 2.0]), bw=5.0)
    assert np.isnan(a).all() and np.isnan(b).all() and (s == 0).all()


def test_make_mask_share_and_determinism():
    obs = pd.DataFrame({"x": np.arange(1000)})
    m1, m2 = make_mask(obs, share=0.15, seed=3), make_mask(obs, share=0.15, seed=3)
    assert m1.sum() == 150 and np.array_equal(m1, m2)
    assert not np.array_equal(m1, make_mask(obs, share=0.15, seed=4))


def test_gap_score_matches_task_table():
    assert gap_score(0.10) == 0 and gap_score(0.08) == 6 and gap_score(0.05) == 15
    assert gap_score(0.02) == 24 and gap_score(0.0) == 30


def test_testlike_rmse_weights_strata():
    """Страта, отсутствующая в test, не влияет на взвешенный RMSE."""
    err = np.array([0.1, 0.1, 5.0])
    kind = np.array(["new_hist", "new_hist", "old"])
    is_2025 = np.array([False, True, False])   # old/hist в test не встречается
    assert weighted_rmse(err, kind, is_2025) == pytest.approx(0.1)
    assert rmse(np.zeros(3), err) > 1
