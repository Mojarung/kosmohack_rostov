"""Проверка погодных расчётов на известных числах и неполной истории."""

import numpy as np
import pandas as pd
import pytest

from service.weather_metrics import (
    cumulative_gdd,
    daily_gdd,
    longest_dry_spell,
    mean_temperature_heat,
    prepare_weather,
    water_balance,
    weather_context,
)


def records(start="2010-01-01", end="2015-12-31"):
    return [{"date": str(day.date()), "era5_precip_mm": 1, "et0_mm": 3, "era5_temp_c": 18,
             "temp_min_c": 12, "temp_max_c": 24} for day in pd.date_range(start, end)]


def test_water_balance_requires_30_actual_calendar_days():
    data = records("2025-01-01", "2025-03-31")
    data.pop(35)
    frame = prepare_weather(data)
    balance = water_balance(frame)
    assert np.isnan(balance.iloc[28])
    assert balance.iloc[29] == -60
    assert balance.iloc[35:65].isna().all()
    assert balance.iloc[65] == -60


def test_crop_methods_and_temperature_units():
    frame = prepare_weather([{"date": "2025-04-01", "temp_min_c": 5, "temp_max_c": 35}])
    assert daily_gdd(frame, "corn").iloc[0] == 10
    assert daily_gdd(frame, "sunflower").iloc[0] == pytest.approx(20 - 20 / 3)
    assert daily_gdd(frame, "custom", 10).iloc[0] == 10
    cold = prepare_weather([{"date": "2025-04-01", "temp_min_c": -10, "temp_max_c": 0}])
    assert daily_gdd(cold, "corn").iloc[0] == 0


def test_gdd_stops_after_missing_temperature_and_does_not_restart():
    data = records("2025-04-01", "2025-04-10")
    data[2]["temp_min_c"] = None
    heat = cumulative_gdd(prepare_weather(data), "2025-04-01", "custom", 10)
    assert heat.iloc[:2].tolist() == [8, 16]
    assert heat.iloc[2:].isna().all()


def test_gdd_requires_observations_from_actual_start():
    heat = cumulative_gdd(prepare_weather(records("2025-04-02", "2025-04-10")), "2025-04-01", "corn")
    assert heat.isna().all()


def test_baseline_excludes_selected_and_future_years():
    data = records()
    for item in data:
        if item["date"] >= "2014-01-01":
            item["era5_precip_mm"] = 10
            item["temp_max_c"] = 40
    ctx = weather_context(data, 2014, {"profile": "custom", "base": 10, "sowing_date": "2014-04-01"})
    index = ctx["water"]["date"].index("2014-04-01")
    assert ctx["water"]["value"][index] == 210
    assert ctx["water"]["mean"][index] == -60
    assert ctx["water"]["history_years"] == [2010, 2011, 2012, 2013]
    assert ctx["heat"]["value"][index] == 16
    assert ctx["heat"]["mean"][index] == 8
    assert ctx["heat"]["value"][index + 1] == 32


def test_calendar_windows_cross_new_year_and_leap_day():
    frame = prepare_weather(records("2023-12-01", "2024-04-01"))
    balance = water_balance(frame)
    assert balance.loc["2024-01-10"] == -60
    assert balance.loc["2024-03-01"] == -60


def test_missing_history_and_unknown_crop_have_explicit_unavailability():
    ctx = weather_context(records("2025-03-01", "2025-10-30"), 2025)
    assert ctx["water"]["available"]
    assert all(value is None for value in ctx["water"]["mean"])
    assert not ctx["heat"]["available"]
    empty = weather_context([], 2025)
    assert not empty["water"]["available"]
    assert not empty["heat"]["available"]


def test_anonymous_dataset_does_not_invent_et0_or_extremes():
    data = [{"date": "2025-04-01", "era5_temp_c": 20, "era5_precip_mm": 0}]
    ctx = weather_context(data, 2025, {"profile": "corn", "sowing_date": "2025-04-01"})
    assert ctx["raw"]["temp"][ctx["raw"]["date"].index("2025-04-01")] == 20
    assert not ctx["water"]["available"]
    assert not ctx["heat"]["available"]


def test_basic_charts_work_with_only_existing_temperature_and_rain():
    data = [{"date": str(day.date()), "era5_temp_c": 18, "era5_precip_mm": 2}
            for year in range(2021, 2027) for day in pd.date_range(f"{year}-04-01", f"{year}-10-30")]
    for row in data:
        if row["date"] >= "2025-01-01":
            row.update(era5_temp_c=20, era5_precip_mm=3)
    ctx = weather_context(data, 2025)
    i = ctx["rain"]["date"].index("2025-04-30")
    assert ctx["rain"]["available"] and ctx["thermal"]["available"]
    assert ctx["rain"]["value"][i] == 90
    assert ctx["rain"]["mean"][i] == 60
    assert ctx["thermal"]["value"][i] == 300
    assert ctx["thermal"]["mean"][i] == 240
    assert ctx["thermal"]["history_years"] == [2021, 2022, 2023, 2024]
    assert not ctx["water"]["available"]


def test_basic_heat_and_rain_keep_missing_calendar_days_unknown():
    data = [{"date": str(day.date()), "era5_temp_c": 20, "era5_precip_mm": 2}
            for day in pd.date_range("2025-04-01", "2025-06-30")]
    data[0]["era5_temp_c"] = 8
    data.pop(30)
    frame = prepare_weather(data)
    thermal = mean_temperature_heat(frame, 2025)
    assert thermal.iloc[0] == 0
    assert thermal.iloc[29] == 290
    assert thermal.iloc[30:].isna().all()
    rain = weather_context(data, 2025)["rain"]
    assert rain["value"][rain["date"].index("2025-04-29")] is None
    assert rain["value"][rain["date"].index("2025-04-30")] == 60
    assert rain["value"][rain["date"].index("2025-05-30")] is None
    assert rain["value"][rain["date"].index("2025-05-31")] == 60


def test_dry_period_uses_strict_threshold_and_breaks_on_missing_day():
    values = [0, 0.9, 1, 0, None, 0, 0.2, 0, 2]
    frame = prepare_weather([{"date": str(day.date()), "era5_precip_mm": rain}
                             for day, rain in zip(pd.date_range("2025-04-01", periods=len(values)), values)])
    dry = longest_dry_spell(frame, 2025)
    assert dry == {"available": True, "days": 3, "start": "2025-04-06", "end": "2025-04-08", "complete": False}
    assert not longest_dry_spell(prepare_weather([]), 2025)["available"]
