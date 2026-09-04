"""Тесты ядра детекции аномалий на синтетических рядах."""

import numpy as np
import pandas as pd
import pytest

from anomaly.climatology import organizer_norm, status_from_z
from anomaly.detect import _merge_runs, _runs, find_episodes, is_episode, phase_of, z_series
from anomaly.series import daily_curve, harmonized_series, season_days
from anomaly.weather import window_stats


def _synthetic_polygon(years=(2018, 2019, 2020), drop_year=2020, n_per_year=40, seed=0) -> pd.DataFrame:
    """Полигон с колоколообразным сезоном; в drop_year после 1 июня NDVI падает на 0.3."""
    rng = np.random.default_rng(seed)
    rows = []
    for year in years:
        days = season_days(year)
        idx = np.sort(rng.choice(len(days), size=n_per_year, replace=False))
        for i in idx:
            d = days.iloc[i]
            doy = int(d["doy"])
            base = 0.2 + 0.6 * np.exp(-((doy - 140) / 35) ** 2)
            if year == drop_year and doy > 152:
                base -= 0.3
            rows.append({"pid": "T-1", "date": d["date"], "day_num": int(d["day_num"]), "year": year, "doy": doy,
                         "primary_ndvi": base + rng.normal(0, 0.02), "sensor": 0, "s2_ndvi": base, "landsat_ndvi": np.nan,
                         "modis_ndvi": np.nan, "s2_ndwi": np.nan, "landsat_ndwi": np.nan})
    return pd.DataFrame(rows)


def test_runs_and_merge():
    flag = np.array([0, 1, 1, 0, 0, 1, 0, 0, 0, 1, 1], dtype=bool)
    assert _runs(flag) == [(1, 2), (5, 5), (9, 10)]
    assert _merge_runs(_runs(flag), gap=2) == [(1, 5), (9, 10)]


def test_is_episode_criteria():
    assert is_episode(days=20, mean_z=-1.3, min_z=-1.4, n_obs=3)          # устойчивый
    assert is_episode(days=8, mean_z=-1.1, min_z=-1.8, n_obs=2)           # сильный
    assert not is_episode(days=8, mean_z=-1.1, min_z=-1.2, n_obs=2)       # ни то ни другое
    assert not is_episode(days=20, mean_z=-1.5, min_z=-2.0, n_obs=1)      # мало наблюдений
    assert is_episode(days=8, mean_z=-1.1, min_z=-1.2, n_obs=2, strict=False)


def test_phase_of():
    assert phase_of(100).startswith("весна")
    assert phase_of(150) == "пик вегетации"
    assert phase_of(400) == "вне сезона"


def test_status_from_z():
    assert list(status_from_z(np.array([0.0, -1.5, -2.5]))) == ["Штатное развитие", "Угнетение биомассы", "Критическая аномалия"]


def test_organizer_norm_excludes_current_year():
    rows = _synthetic_polygon()
    org = organizer_norm(rows)
    # норма считается по другим годам: у 2020 сравниваются только 2018 и 2019
    assert (org.loc[org["year"] == 2020, "org_n_years"] <= 2).all()
    assert org["org_mean"].notna().mean() > 0.9


def test_detects_drop_episode_only_in_drop_year():
    rows = _synthetic_polygon()
    series = harmonized_series(rows)
    curves = {y: daily_curve(series, y) for y in (2018, 2019, 2020)}
    from anomaly.climatology import norm_curve
    for year in (2019, 2020):
        norm = norm_curve(curves, year, min_weight=0.5)
        zs = z_series(curves[year], norm)
        obs_days = series.loc[series["year"] == year, "day_num"].to_numpy()
        eps = find_episodes(zs, obs_days)
        if year == 2020:
            assert eps and eps[0]["min_z"] < -1.5 and eps[0]["start_doy"] >= 140
        else:
            assert not eps


def test_window_stats_counts_dry_and_hot_days():
    dates = pd.date_range("2020-06-01", periods=10)
    w = pd.DataFrame({"date": dates, "era5_precip_mm": [0, 0, 0, 5, 0, 0, 0, 0, 2, 0],
                      "era5_temp_c": [25, 28, 29, 20, 30, 30, 26, 27.5, 24, 22]})
    st = window_stats(w, dates[0], dates[-1])
    assert st["n_days"] == 10 and st["precip_mm"] == pytest.approx(7)
    assert st["dry_days"] == 8 and st["max_dry_spell"] == 4 and st["hot_days"] == 5
