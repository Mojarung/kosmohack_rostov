"""Тесты ядра пайплайна восстановления primary_ndvi (без обучения моделей)."""

import numpy as np
import pandas as pd
import pytest

from gapfill.data import data_tag, gap_score, load_all, make_mask, rmse
from gapfill.data import testlike_rmse as weighted_rmse
from gapfill.smooth import local_linear

COLUMNS = ["anon_polygon_id", "date", "s2_ndvi", "s2_evi", "s2_ndwi", "landsat_ndvi", "landsat_evi", "landsat_ndwi",
           "modis_ndvi", "modis_evi", "era5_temp_c", "era5_precip_mm", "year", "primary_ndvi", "doy",
           "n_reference_years", "is_synthetic_gap", "crop_type"]


def _rows(pid: str, dates: list[str], values: list[float | None], gap: list[bool]) -> pd.DataFrame:
    """Мини-датасет в формате организаторов: известная точка — значение S2, контрольная — всё пусто."""
    rows = []
    for d, v, g in zip(dates, values, gap):
        row = dict.fromkeys(COLUMNS)
        row.update({"anon_polygon_id": pid, "date": d, "is_synthetic_gap": g, "crop_type": "зерновые"})
        if v is not None and not g:
            row.update({"s2_ndvi": v, "primary_ndvi": v, "year": int(d[:4])})
        rows.append(row)
    return pd.DataFrame(rows, columns=COLUMNS)


def test_load_all_extra_file_gives_context_but_not_gaps(tmp_path):
    """Известные точки дополнительного файла попадают в obs, его контрольные точки не предсказываются,
    при совпадении полигон+дата приоритет у основного test."""
    train = _rows("AOI-A", ["2020-05-01", "2020-05-06"], [0.5, 0.6], [False, False])
    test = _rows("AOI-B", ["2020-05-01", "2020-05-06", "2020-05-11"], [0.4, None, 0.45], [False, True, False])
    extra = _rows("AOI-B", ["2020-05-11", "2020-05-16", "2020-05-21"], [0.9, None, 0.5], [False, True, False])
    paths = [tmp_path / f"{n}.csv" for n in ("train", "test", "extra")]
    for p, df in zip(paths, (train, test, extra)):
        df.to_csv(p, index=False, encoding="utf-8")
    obs, grid, gaps = load_all(*paths[:2], extra_paths=[paths[2]])
    assert len(gaps) == 1 and gaps["pid"].iloc[0] == "AOI-B" and gaps["split"].iloc[0] == "test"
    assert sorted(obs["split"].unique()) == ["extra", "test", "train"]
    assert obs.loc[obs["date"] == "2020-05-11", "primary_ndvi"].iloc[0] == pytest.approx(0.45)   # test важнее extra
    assert int(grid["is_gap"].sum()) == 2 and len(grid) == 7
    obs_no_extra, _, _ = load_all(*paths[:2], extra_paths=[])
    assert len(obs_no_extra) == 4 and data_tag(obs_no_extra) != data_tag(obs)


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
