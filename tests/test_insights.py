"""Реальные наблюдения, сезонный ориентир и общая площадь, доступная на двух снимках."""

import numpy as np
import pytest

from service.imagery import compare, normalized_difference
from service.insights import deviation_trend


def season(before, after):
    dates = ["2025-06-03", "2025-06-10", "2025-06-17", "2025-06-24"]
    return {"observations": [{"date": d, "harmonized": .6 + z * .1, "sensor": "Sentinel-2", "artifact": False}
                              for d, z in zip(dates, [before, before, after, after])],
            "norm_mean": [{"date": d, "value": .6} for d in dates],
            "norm_std": [{"date": d, "value": .1} for d in dates]}


@pytest.mark.parametrize("before,after,status", [(-1.5, -3, "worsening"), (-3, -1.5, "easing"),
                                                (-1.5, -.5, "recovered"), (-2, -2.1, "stable"),
                                                (.5, 0, "within")])
def test_observed_deviation_direction(before, after, status):
    result = deviation_trend(season(before, after))
    assert result["status"] == status
    assert result["before_count"] == result["after_count"] == 2
    assert result["delta_z"] == pytest.approx(after - before)


def test_seasonal_ndvi_decline_is_not_a_worsening_if_norm_declines():
    data = season(-2, -2)
    for i in (2, 3):
        data["observations"][i]["harmonized"] -= .2
        data["norm_mean"][i]["value"] -= .2
    assert deviation_trend(data)["status"] == "stable"


def test_duplicate_sensors_and_reconstructions_cannot_fill_missing_dates():
    data = season(-2, -3)
    data["observations"][0]["artifact"] = True
    data["observations"].append(data["observations"][1] | {"sensor": "Landsat"})
    data["curve"] = data["restored"] = [{"date": "2025-06-05", "value": .1}]
    result = deviation_trend(data)
    assert not result["available"] and result["before_count"] == 1


def test_no_history_and_long_observation_gap_are_unknown():
    data = season(-2, -3)
    data["norm_mean"] = []
    assert not deviation_trend(data)["available"]
    data = season(-2, -3)
    data["observations"].append({"date": "2025-08-20", "harmonized": .4, "sensor": "Sentinel-2"})
    assert not deviation_trend(data)["available"]
    assert not deviation_trend({})["available"]


def test_area_denominator_excludes_clouds_on_either_image():
    previous = np.array([[.7, .7, np.nan], [.7, .7, .7]])
    current = np.array([[.4, .6, .2], [np.nan, .7, .8]])
    delta, stats = compare(current, previous, 6)
    assert np.isnan(delta[0, 2]) and np.isnan(delta[1, 0])
    assert stats["area_ha"] == .16
    assert stats["drop_area_ha"] == .08  # includes an exact 0.1 decrease
    assert stats["drop_share"] == .5
    assert stats["clear_share"] == pytest.approx(4 / 6, abs=.0001)
    _, missing = compare(current * np.nan, previous, 6)
    assert missing["drop_share"] is None


def test_ndmi_uses_swir_and_masks_outside_and_clouds():
    nir = np.full((2, 2), .6)
    swir = np.full((2, 2), .2)
    result = normalized_difference(nir, swir, np.array([[True, False], [True, True]]),
                                   np.array([[True, True], [False, True]]))
    assert result[0, 0] == pytest.approx(.5)
    assert np.isnan(result[0, 1]) and np.isnan(result[1, 0])
