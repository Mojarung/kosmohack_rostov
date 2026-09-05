"""Поиск места через Nominatim; координаты обрабатываются без внешнего запроса."""

import json
import math
import os
import re
import sqlite3
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

from fastapi import APIRouter, HTTPException, Query

router = APIRouter()
CACHE_PATH = Path(os.environ.get("GEOCODING_CACHE", str(
    Path(__file__).resolve().parents[1] / "artifacts/service/geocoding.sqlite")))
COORDINATES = re.compile(r"^\s*([+-]?\d+(?:\.\d+)?)\s*[,;\s]\s*([+-]?\d+(?:\.\d+)?)\s*$")


def coordinate_result(query: str) -> list[dict] | None:
    """Ввод: широта, долгота; GeoJSON и камера используют долготу, широту."""
    match = COORDINATES.fullmatch(query)
    if not match:
        return None
    lat, lon = map(float, match.groups())
    if not (-90 <= lat <= 90 and -180 <= lon <= 180):
        raise HTTPException(400, "Широта должна быть от −90 до 90, долгота — от −180 до 180")
    return [{"id": f"coord:{lat},{lon}", "label": f"{lat:g}, {lon:g}",
             "center": [lon, lat], "bbox": None, "address": {}, "kind": "coordinates"}]


def cached_or_reserve(url: str, query: str) -> tuple[str, list[dict] | None]:
    """Общий SQLite-кэш и лимит между процессами: максимум один запрос за 1.1 секунды."""
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    key = json.dumps([url, query.casefold()], ensure_ascii=False)
    now = time.time()
    with sqlite3.connect(CACHE_PATH, timeout=5) as db:
        db.execute("CREATE TABLE IF NOT EXISTS cache (key TEXT PRIMARY KEY, at REAL, body TEXT)")
        db.execute("CREATE TABLE IF NOT EXISTS throttle (id INTEGER PRIMARY KEY, at REAL)")
        db.execute("BEGIN IMMEDIATE")
        row = db.execute("SELECT body FROM cache WHERE key = ? AND at > ?", (key, now - 86400)).fetchone()
        if row:
            return key, json.loads(row[0])
        last = db.execute("SELECT at FROM throttle WHERE id = 1").fetchone()
        if last and now - last[0] < 1.1:
            raise HTTPException(429, "Подождите секунду и повторите поиск", headers={"Retry-After": "2"})
        db.execute("INSERT OR REPLACE INTO throttle VALUES (1, ?)", (now,))
    return key, None


def normalize_place(item: dict) -> dict | None:
    """Отбрасывает повреждённые ответы и приводит рамку к порядку запад,юг,восток,север."""
    try:
        lat, lon = float(item["lat"]), float(item["lon"])
        if not (-90 <= lat <= 90 and -180 <= lon <= 180):
            return None
        south, north, west, east = map(float, item["boundingbox"])
        bbox = [west, south, east, north]
        if not all(math.isfinite(v) for v in bbox) or not (-90 <= south <= lat <= north <= 90
                and -180 <= west <= lon <= east <= 180):
            return None
        label = str(item["display_name"]).strip()
        if not label:
            return None
        address = item.get("address", {})
        return {"id": f"{item['osm_type']}:{item['osm_id']}", "label": label,
                "center": [lon, lat], "bbox": bbox,
                "address": {k: v for k, v in address.items() if isinstance(v, str)},
                "kind": str(item.get("type", "place"))}
    except (KeyError, TypeError, ValueError, AttributeError):
        return None


def search_places(query: str) -> list[dict]:
    """Ручной поиск с тайм-аутом, идентификацией приложения и кэшем на сутки."""
    query = " ".join(query.split())
    if len(query) < 2:
        raise HTTPException(400, "Введите хотя бы два символа")
    coordinates = coordinate_result(query)
    if coordinates is not None:
        return coordinates
    url = os.environ.get("NOMINATIM_URL", "https://nominatim.openstreetmap.org/search")
    key, cached = cached_or_reserve(url, query)
    if cached is not None:
        return cached
    params = urllib.parse.urlencode({"q": query, "format": "jsonv2", "addressdetails": 1,
                                    "limit": 6, "accept-language": "ru"})
    req = urllib.request.Request(f"{url}?{params}", headers={
        "User-Agent": os.environ.get("GEOCODING_USER_AGENT", "kosmohack-ndvi-monitor/1.0"),
        "Accept": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=10) as response:
            payload = json.loads(response.read(1_000_000))
        if not isinstance(payload, list):
            raise ValueError("Ожидался список мест")
        results = [place for item in payload[:6] if isinstance(item, dict)
                   and (place := normalize_place(item)) is not None]
    except (urllib.error.URLError, OSError, ValueError) as exc:
        raise HTTPException(502, "Поиск адресов временно недоступен. Попробуйте ещё раз или введите координаты.") from exc
    with sqlite3.connect(CACHE_PATH, timeout=5) as db:
        db.execute("INSERT OR REPLACE INTO cache VALUES (?, ?, ?)", (key, time.time(), json.dumps(results)))
        db.execute("DELETE FROM cache WHERE key NOT IN (SELECT key FROM cache ORDER BY at DESC LIMIT 512)")
    return results


@router.get("/api/places")
def places(q: str = Query(min_length=2, max_length=200)) -> list[dict]:
    """Улица, адрес, населённый пункт, область, объект или координаты: широта, долгота."""
    return search_places(q)
