"""Поиск контуров OpenStreetMap без зависимостей спутникового сборщика."""

import json
import logging
import math
import time
import urllib.error
import urllib.parse
import urllib.request
from functools import lru_cache

OVERPASS_URLS = ("https://maps.mail.ru/osm/tools/overpass/api/interpreter",
                "https://overpass.private.coffee/api/interpreter",
                "https://overpass-api.de/api/interpreter")
log = logging.getLogger(__name__)


def parse_bbox(value: str) -> tuple[float, float, float, float]:
    """Проверяет координаты и размер рамки карты до сетевого запроса."""
    try:
        s, w, n, e = (float(v) for v in value.split(","))
    except ValueError as exc:
        raise ValueError("рамка карты должна содержать четыре координаты: юг,запад,север,восток") from exc
    if not all(math.isfinite(v) for v in (s, w, n, e)) or not (-90 <= s < n <= 90 and -180 <= w < e <= 180):
        raise ValueError("некорректные границы области на карте")
    if (n - s) * (e - w) > 0.25:
        raise ValueError("приблизьте карту: область слишком велика для запроса контуров")
    return s, w, n, e


def osm_fields(bbox: tuple[float, float, float, float]) -> list[dict]:
    """Повторный поиск в той же рамке использует кэш до пяти минут."""
    return _cached_fields(bbox, int(time.monotonic() // 300))


def _query_overpass(query: str) -> dict:
    """Переключается на другой публичный сервер при тайм-ауте или отказе источника."""
    error = None
    for url in OVERPASS_URLS:
        req = urllib.request.Request(url, data=urllib.parse.urlencode({"data": query}).encode(),
                                     headers={"User-Agent": "kosmohack-ndvi/0.1"})
        try:
            with urllib.request.urlopen(req, timeout=35) as resp:
                payload = json.load(resp)
            if payload.get("remark"):
                raise ValueError(payload["remark"])
            return payload
        except (urllib.error.URLError, OSError, ValueError) as exc:
            error = exc
            log.warning("Сервер OSM %s не ответил: %s", url, exc)
    raise RuntimeError("серверы поиска полей временно недоступны; повторите запрос позже") from error


@lru_cache(maxsize=64)
def _cached_fields(bbox: tuple[float, float, float, float], bucket: int) -> list[dict]:
    """Получает замкнутые контуры с/х полей в рамке (юг, запад, север, восток)."""
    s, w, n, e = bbox
    query = f'[out:json][timeout:25];(way["landuse"="farmland"]({s},{w},{n},{e}););out geom 200;'
    payload = _query_overpass(query)
    out = []
    for el in payload.get("elements", []):
        pts = [[p["lon"], p["lat"]] for p in el.get("geometry", [])]
        if len(pts) >= 4 and pts[0] == pts[-1]:
            out.append({"id": el["id"], "name": el.get("tags", {}).get("name", f"поле OSM {el['id']}"),
                        "geometry": {"type": "Polygon", "coordinates": [pts]}})
    return out
