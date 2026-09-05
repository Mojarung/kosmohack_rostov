"""Водный баланс и градусо-дни: прозрачные расчёты без заполнения пропусков нулями."""

from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd

PROFILES = {
    "corn": {"label": "Кукуруза", "base": 10.0, "cap": 30.0, "clip": True,
             "source": "https://blog-crop-news.extension.umn.edu/2026/05/frost-concerns-after-early-planting.html"},
    "sunflower": {"label": "Подсолнечник", "base": 20 / 3, "cap": None, "clip": False,
                  "source": "https://www.ndsu.edu/agriculture/extension/publications/sunflower-production-guide"},
    "custom": {"label": "Свой порог (уточнить у агронома)", "base": None, "cap": None, "clip": False,
               "source": "https://extension.umn.edu/agriculture/crop-production/forages/using-growing-degree-days-to-plan-early-season-alfalfa-harvests"},
}
WEATHER_COLUMNS = ["era5_temp_c", "era5_precip_mm", "temp_min_c", "temp_max_c", "et0_mm"]


def prepare_weather(records: list[dict]) -> pd.DataFrame:
    """Разворачивает ежедневную сетку: отсутствующий день остаётся NaN."""
    frame = pd.DataFrame(records)
    if frame.empty:
        return pd.DataFrame(columns=WEATHER_COLUMNS, index=pd.DatetimeIndex([], name="date"))
    frame["date"] = pd.to_datetime(frame["date"]).dt.tz_localize(None).dt.normalize()
    frame = frame.drop_duplicates("date").set_index("date").sort_index()
    frame = frame.reindex(pd.date_range(frame.index.min(), frame.index.max(), freq="D"))
    for col in WEATHER_COLUMNS:
        frame[col] = pd.to_numeric(frame.get(col, np.nan), errors="coerce")
        frame[col] = frame[col].where(np.isfinite(frame[col]))
    for col in ("era5_precip_mm", "et0_mm"):
        frame[col] = frame[col].where(frame[col] >= 0)
    invalid = frame["temp_min_c"] > frame["temp_max_c"]
    frame.loc[invalid, ["temp_min_c", "temp_max_c"]] = np.nan
    return frame[WEATHER_COLUMNS]


def water_balance(frame: pd.DataFrame) -> pd.Series:
    """Баланс ровно 30 последовательных полных суток, мм; пропуск делает окно неизвестным."""
    return (frame["era5_precip_mm"] - frame["et0_mm"]).rolling(30, min_periods=30).sum()


def daily_gdd(frame: pd.DataFrame, profile: str, base: float | None = None) -> pd.Series:
    """Суточные °C·дни; для кукурузы обе температуры ограничены диапазоном 10–30 °C."""
    spec = PROFILES[profile]
    threshold = float(base) if profile == "custom" else spec["base"]
    low, high = frame["temp_min_c"], frame["temp_max_c"]
    if spec["clip"]:
        low, high = low.clip(threshold, spec["cap"]), high.clip(threshold, spec["cap"])
    return ((low + high) / 2 - threshold).clip(lower=0)


def cumulative_gdd(frame: pd.DataFrame, start: str, profile: str, base: float | None = None) -> pd.Series:
    """Не продолжает накопление после неизвестного дня; включает дату начала в сумму."""
    beginning = pd.Timestamp(start)
    end = pd.Timestamp(beginning.year, 12, 31)
    daily = daily_gdd(frame, profile, base).reindex(pd.date_range(beginning, end))
    return daily.cumsum(skipna=False)


def _calendar_values(series: pd.Series, dates: pd.DatetimeIndex, year: int) -> np.ndarray:
    """Сопоставляет календарные месяц/день; 29 февраля отсутствует в невисокосном году."""
    values = []
    for day in dates:
        try:
            values.append(series.get(day.replace(year=year), np.nan))
        except ValueError:
            values.append(np.nan)
    return np.array(values, dtype=float)


def _comparison(current: pd.Series, dates: pd.DatetimeIndex, historical: dict[int, pd.Series]) -> dict:
    """Ориентир по прошлым годам, минимум три значения на дату; без будущих сезонов."""
    matrix = pd.DataFrame({year: _calendar_values(values, dates, year) for year, values in historical.items()}, index=dates)
    count = matrix.count(axis=1)
    mean = matrix.mean(axis=1).where(count >= 3)
    lower = matrix.quantile(0.1, axis=1).where(count >= 3) if len(matrix.columns) else mean
    upper = matrix.quantile(0.9, axis=1).where(count >= 3) if len(matrix.columns) else mean
    return {"date": dates.strftime("%Y-%m-%d").tolist(), "value": _numbers(current.reindex(dates)),
            "mean": _numbers(mean), "low": _numbers(lower), "high": _numbers(upper),
            "n_years": count.astype(int).tolist(),
            "history_years": [year for year in historical if np.isfinite(matrix[year]).any()]}


def _numbers(series) -> list:
    """Конечные числа или null для JSON и разрывов на графике."""
    return [round(float(value), 3) if pd.notna(value) and np.isfinite(value) else None for value in series]


def mean_temperature_heat(frame: pd.DataFrame, year: int) -> pd.Series:
    """Сезонная сумма max(Tср − 10, 0) от 1 апреля; не профиль конкретной культуры."""
    dates = pd.date_range(f"{year}-04-01", f"{year}-10-30")
    daily = (frame["era5_temp_c"].reindex(dates) - 10).clip(lower=0)
    return daily.cumsum(skipna=False)


def longest_dry_spell(frame: pd.DataFrame, year: int) -> dict:
    """Максимум подряд дней с осадками < 1 мм; неизвестный день разрывает серию."""
    rain = frame["era5_precip_mm"].reindex(pd.date_range(f"{year}-04-01", f"{year}-10-30"))
    best = {"available": bool(rain.notna().any()), "days": 0, "start": None, "end": None,
            "complete": bool(rain.notna().all())}
    run, start = 0, None
    for day, value in rain.items():
        if pd.notna(value) and value < 1:
            if not run:
                start = day.date().isoformat()
            run += 1
            if run > best["days"]:
                best.update(days=run, start=start, end=day.date().isoformat())
        else:
            run = 0
    return best


def basic_weather_metrics(frame, dates, year, historical_years) -> dict:
    """Доступные в исходном датасете показатели: осадки и тепло, без подстановки ET₀/Tmin/Tmax."""
    rain_sum = frame["era5_precip_mm"].rolling(30, min_periods=30).sum()
    rain = _comparison(rain_sum, dates, dict.fromkeys(historical_years, rain_sum))
    thermal = _comparison(mean_temperature_heat(frame, year), dates,
                          {y: mean_temperature_heat(frame, y) for y in historical_years})
    for metric in (rain, thermal):
        metric["available"] = any(value is not None for value in metric["value"])
    thermal.update(base=10, start=f"{year}-04-01", method="daily_mean")
    return {"rain": rain, "thermal": thermal, "dry_spell": longest_dry_spell(frame, year)}


def weather_context(records: list[dict], year: int, settings: dict | None = None, source: str = "") -> dict:
    """Оба погодных графика для сезона; неподтверждённая культура не выбирается автоматически."""
    frame = prepare_weather(records)
    dates = pd.date_range(f"{year}-03-01", f"{year}-10-30")
    historical_years = range(max(1940, year - 30), year)
    balance = water_balance(frame)
    water = _comparison(balance, dates, dict.fromkeys(historical_years, balance))
    water["available"] = any(v is not None for v in water["value"])
    water["reason"] = "Нужны осадки и ET₀ за 30 полных дней. В этом отчёте их недостаточно."
    settings = settings or {}
    heat = _heat_context(frame, dates, historical_years, settings)
    raw = frame.reindex(dates)
    return {**basic_weather_metrics(frame, dates, year, historical_years),
            "water": water, "heat": heat, "source": source, "settings": settings,
            "raw": {"date": dates.strftime("%Y-%m-%d").tolist(),
                    "temp": _numbers(raw["era5_temp_c"]), "precip": _numbers(raw["era5_precip_mm"])}}


def _heat_context(frame, dates, historical_years, settings) -> dict:
    """Сравнивает накопления от одной календарной даты, а не неизвестных исторических дат посева."""
    profile, start = settings.get("profile"), settings.get("sowing_date")
    if profile not in PROFILES or not start:
        return {"available": False, "reason": "Укажите культуру и дату посева / начала вегетации для этого сезона."}
    base = settings.get("base")
    current = cumulative_gdd(frame, start, profile, base)
    history = {}
    for year in historical_years:
        try:
            beginning = dt.date.fromisoformat(start).replace(year=year)
        except ValueError:
            continue
        history[year] = cumulative_gdd(frame, beginning.isoformat(), profile, base)
    heat = _comparison(current, dates, history)
    heat["available"] = any(v is not None for v in heat["value"])
    heat["reason"] = "Нет полной ежедневной Tmin/Tmax от даты начала. Пропуски не заменяются нулём."
    heat["profile"] = PROFILES[profile] | {"base": base if profile == "custom" else PROFILES[profile]["base"]}
    heat["start"] = start
    return heat
