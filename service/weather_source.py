"""Полная суточная история ERA5 с файловым кэшем для погодных показателей поля."""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import logging
import urllib.parse
import urllib.request
from pathlib import Path

import pandas as pd

URL = "https://archive-api.open-meteo.com/v1/archive"
CACHE = Path(__file__).resolve().parents[1] / "artifacts" / "weather"
COLUMNS = {"temperature_2m_mean": "era5_temp_c", "temperature_2m_min": "temp_min_c",
           "temperature_2m_max": "temp_max_c", "precipitation_sum": "era5_precip_mm",
           "et0_fao_evapotranspiration": "et0_mm"}
SOURCE = "ERA5 · Open-Meteo · ячейка погоды у центра поля · UTC"


def _fetch(lat: float, lon: float, start: dt.date, end: dt.date) -> dict:
    """Кэширует завершённые интервалы; недоступный ответ не подменяется пустой погодой."""
    params = {"latitude": round(lat, 5), "longitude": round(lon, 5), "start_date": start.isoformat(),
              "end_date": end.isoformat(), "daily": ",".join(COLUMNS), "models": "era5", "timezone": "UTC"}
    query = urllib.parse.urlencode(params)
    path = CACHE / (hashlib.sha256(query.encode()).hexdigest() + ".json")
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    request = urllib.request.Request(URL + "?" + query, headers={"User-Agent": "kosmohack-field-report/1.0"})
    with urllib.request.urlopen(request, timeout=90) as response:
        payload = json.load(response)
    daily = payload.get("daily", {})
    if not daily.get("time") or not all(key in daily for key in COLUMNS):
        raise ValueError("ERA5 не вернула запрошенные суточные переменные")
    CACHE.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(daily, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    return daily


def collect_weather(lat: float, lon: float, years: range) -> pd.DataFrame:
    """Собирает полные годы и до 30 лет предыстории, сохраняя успешные блоки при частичном отказе."""
    start_year = max(1940, years[0] - 30)
    end = min(dt.date(years[-1], 12, 31), dt.datetime.now(dt.UTC).date() - dt.timedelta(days=7))
    frames, failures = [], []
    for first in range(start_year, years[-1] + 1, 10):
        start = dt.date(first, 1, 1)
        last = min(dt.date(first + 9, 12, 31), end)
        if start > last:
            continue
        try:
            daily = _fetch(lat, lon, start, last)
            frame = pd.DataFrame({"date": pd.to_datetime(daily["time"]),
                                  **{out: daily[key] for key, out in COLUMNS.items()}})
            frames.append(frame)
        except Exception as exc:
            logging.getLogger(__name__).exception("Не загружена погода %s — %s", start, last)
            failures.append(f"{start} — {last}: {type(exc).__name__}")
    if not frames:
        raise ValueError("Не удалось загрузить ERA5: " + "; ".join(failures))
    result = pd.concat(frames, ignore_index=True).sort_values("date").drop_duplicates("date")
    result.attrs["warnings"] = failures
    return result


def weather_records(frame: pd.DataFrame) -> list[dict]:
    """JSON-совместимые исходные значения, включая настоящие пропуски."""
    return json.loads(frame.to_json(orient="records", date_format="iso"))
