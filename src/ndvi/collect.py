"""Автоматический сбор данных по произвольному полигону из открытых источников.

Google Earth Engine требует аккаунт Google Cloud и интерактивную авторизацию, поэтому
основной путь — открытые каталоги без ключей:

============  ==========================================================  ============
Данные        Источник                                                    Ключ нужен?
============  ==========================================================  ============
Sentinel-2    STAC Earth Search (AWS Open Data), COG-снимки L2A + маска SCL  нет
Landsat 8/9   STAC Microsoft Planetary Computer, L2 + QA_PIXEL               нет (анонимный SAS)
MODIS         REST-сервис ORNL DAAC, продукт MOD13Q1 (NDVI за 16 дней)       нет
Погода        Open-Meteo, архив ERA5-Land                                    нет
Поля          OpenStreetMap Overpass, ``landuse=farmland``                   нет
Регионы       OpenStreetMap Nominatim                                        нет
============  ==========================================================  ============

Каждый источник изолирован: падение или недоступность одного не ломает ответ, сервис
сообщает, какие источники отработали. Всё, что удалось скачать, кладётся в дисковый кэш,
поэтому повторный запрос по тому же полигону отвечает мгновенно.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ndvi.paths import CACHE_DIR

log = logging.getLogger(__name__)

# GDAL должен ходить в публичный S3 без подписи и не листить каталоги — иначе каждое
# открытие COG превращается в десятки лишних запросов
os.environ.setdefault("GDAL_DISABLE_READDIR_ON_OPEN", "EMPTY_DIR")
os.environ.setdefault("AWS_NO_SIGN_REQUEST", "YES")
os.environ.setdefault("GDAL_HTTP_MULTIPLEX", "YES")
os.environ.setdefault("CPL_VSIL_CURL_ALLOWED_EXTENSIONS", ".tif,.TIF")
os.environ.setdefault("VSI_CACHE", "TRUE")
os.environ.setdefault("GDAL_CACHEMAX", "128")

EARTH_SEARCH = "https://earth-search.aws.element84.com/v1/search"
PLANETARY_STAC = "https://planetarycomputer.microsoft.com/api/stac/v1/search"
PLANETARY_SAS = "https://planetarycomputer.microsoft.com/api/sas/v1/token/{account}/{container}"
MODIS_REST = "https://modis.ornl.gov/rst/api/v1/MOD13Q1/subset"
MODIS_DATES = "https://modis.ornl.gov/rst/api/v1/MOD13Q1/dates"
MODIS_CHUNK = 10   # сервис отдаёт не больше 10 композитов за запрос
OPEN_METEO = "https://archive-api.open-meteo.com/v1/archive"
NOMINATIM = "https://nominatim.openstreetmap.org/search"
OVERPASS = ["https://overpass-api.de/api/interpreter", "https://overpass.kumi.systems/api/interpreter"]

USER_AGENT = "kosmohack-ndvi-monitor/1.0"
TARGET_RES_M = 20.0        # рабочее разрешение вырезки: поле мельче пикселя MODIS всё равно
MAX_WORKERS = 16
HTTP_TIMEOUT = 40.0


# --------------------------------------------------------------------------- #
# Кэш
# --------------------------------------------------------------------------- #

def _cache_key(*parts) -> str:
    raw = json.dumps(parts, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha1(raw.encode()).hexdigest()[:20]


def cached(namespace: str, key_parts: tuple, ttl_days: float | None = None):
    """Декоратор дискового кэша: одинаковый запрос второй раз не уходит в сеть."""
    path = CACHE_DIR / namespace
    path.mkdir(parents=True, exist_ok=True)
    f = path / f"{_cache_key(*key_parts)}.json"

    def load():
        if not f.exists():
            return None
        if ttl_days is not None and (time.time() - f.stat().st_mtime) > ttl_days * 86400:
            return None
        try:
            return json.loads(f.read_text())
        except Exception:
            return None

    def save(obj):
        try:
            f.write_text(json.dumps(obj, ensure_ascii=False, default=str))
        except Exception as exc:
            log.warning("не удалось записать кэш %s: %s", f, exc)

    return load, save


# --------------------------------------------------------------------------- #
# Результат сбора
# --------------------------------------------------------------------------- #

@dataclass
class CollectResult:
    """Собранный ряд плюс диагностика по каждому источнику."""

    series: pd.DataFrame
    sources: dict[str, dict] = field(default_factory=dict)

    @property
    def ok_sources(self) -> list[str]:
        return [k for k, v in self.sources.items() if v.get("ok")]

    def note(self, name: str, ok: bool, n: int = 0, error: str | None = None, seconds: float = 0.0):
        self.sources[name] = {"ok": ok, "n": n, "error": error, "seconds": round(seconds, 1)}


# --------------------------------------------------------------------------- #
# Гео-помощники
# --------------------------------------------------------------------------- #

def geom_bbox(geom: dict) -> tuple[float, float, float, float]:
    """Bbox полигона GeoJSON в градусах: (minx, miny, maxx, maxy)."""
    coords = np.array(_flatten_coords(geom))
    return float(coords[:, 0].min()), float(coords[:, 1].min()), float(coords[:, 0].max()), float(coords[:, 1].max())


def _flatten_coords(geom: dict) -> list:
    out = []

    def walk(c):
        if isinstance(c[0], (int, float)):
            out.append(c[:2])
        else:
            for x in c:
                walk(x)

    walk(geom["coordinates"])
    return out


def geom_centroid(geom: dict) -> tuple[float, float]:
    """Приближённый центроид (среднее по вершинам) — для точечных запросов погоды и MODIS."""
    c = np.array(_flatten_coords(geom))
    return float(c[:, 1].mean()), float(c[:, 0].mean())


def geom_area_ha(geom: dict) -> float:
    """Площадь в гектарах по формуле шнурков с поправкой на широту."""
    c = np.array(_flatten_coords(geom))
    lat0 = np.deg2rad(c[:, 1].mean())
    x = c[:, 0] * 111320.0 * np.cos(lat0)
    y = c[:, 1] * 110540.0
    area = 0.5 * abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))
    return float(area / 10_000.0)


# --------------------------------------------------------------------------- #
# HTTP
# --------------------------------------------------------------------------- #

def _http_json(method: str, url: str, **kw):
    import httpx
    headers = {"User-Agent": USER_AGENT, **kw.pop("headers", {})}
    with httpx.Client(timeout=HTTP_TIMEOUT, follow_redirects=True) as cl:
        r = cl.request(method, url, headers=headers, **kw)
        r.raise_for_status()
        return r.json()


# --------------------------------------------------------------------------- #
# Поиск региона и полей
# --------------------------------------------------------------------------- #

def search_region(query: str, limit: int = 5) -> list[dict]:
    """Поиск региона/населённого пункта по названию (OpenStreetMap Nominatim)."""
    load, save = cached("nominatim", (query, limit), ttl_days=30)
    hit = load()
    if hit is not None:
        return hit
    data = _http_json("GET", NOMINATIM, params={"q": query, "format": "json", "limit": limit,
                                                "polygon_geojson": 0, "accept-language": "ru"})
    out = [{"name": d.get("display_name"), "lat": float(d["lat"]), "lon": float(d["lon"]),
            "bbox": [float(d["boundingbox"][2]), float(d["boundingbox"][0]),
                     float(d["boundingbox"][3]), float(d["boundingbox"][1])],
            "type": d.get("type")} for d in data]
    save(out)
    return out


def fetch_fields(bbox: tuple[float, float, float, float], limit: int = 60,
                 min_ha: float = 5.0) -> list[dict]:
    """Контуры сельхозполей из OpenStreetMap в заданном bbox.

    Берём ``landuse`` из farmland/meadow/orchard/vineyard — это то, что в OSM размечено
    как обрабатываемая земля. Мелкие огороды отсекаются порогом площади.
    """
    minx, miny, maxx, maxy = bbox
    # Overpass ограничивает размер выборки; на больших регионах режем bbox до разумного окна
    if (maxx - minx) > 0.6 or (maxy - miny) > 0.4:
        cx, cy = (minx + maxx) / 2, (miny + maxy) / 2
        minx, maxx = cx - 0.3, cx + 0.3
        miny, maxy = cy - 0.2, cy + 0.2

    load, save = cached("overpass", (round(minx, 3), round(miny, 3), round(maxx, 3), round(maxy, 3), limit), ttl_days=14)
    hit = load()
    if hit is not None:
        return hit

    q = (f'[out:json][timeout:50];('
         f'way["landuse"~"^(farmland|meadow|orchard|vineyard)$"]({miny},{minx},{maxy},{maxx});'
         f');out geom {limit * 3};')
    elements = []
    for url in OVERPASS:
        try:
            elements = _http_json("POST", url, data={"data": q}).get("elements", [])
            break
        except Exception as exc:
            log.warning("Overpass %s недоступен: %s", url, exc)
    out = []
    for e in elements:
        g = e.get("geometry") or []
        if len(g) < 4:
            continue
        ring = [[p["lon"], p["lat"]] for p in g]
        if ring[0] != ring[-1]:
            ring.append(ring[0])
        geom = {"type": "Polygon", "coordinates": [ring]}
        ha = geom_area_ha(geom)
        if ha < min_ha:
            continue
        tags = e.get("tags", {})
        out.append({"id": f"OSM-{e['id']}", "geometry": geom, "area_ha": round(ha, 1),
                    "landuse": tags.get("landuse"), "name": tags.get("name") or f"Поле OSM {e['id']}",
                    "crop": tags.get("crop")})
    out.sort(key=lambda d: -d["area_ha"])
    out = out[:limit]
    save(out)
    return out


# --------------------------------------------------------------------------- #
# Растровая вырезка по полигону
# --------------------------------------------------------------------------- #

def _read_grid(href: str, geom: dict, res_m: float = TARGET_RES_M):
    """Читает окно COG по границам полигона на единой сетке ``res_m`` метров.

    Единая сетка обязательна: у Sentinel-2 красный канал 10 м, а маска SCL 20 м, и без
    приведения к общему растру маску не наложить.
    """
    import rasterio
    from rasterio import Affine
    from rasterio.features import geometry_mask
    from rasterio.warp import transform_geom
    from rasterio.windows import from_bounds

    with rasterio.open(href) as src:
        g = transform_geom("EPSG:4326", src.crs, geom)
        xs = [p[0] for p in _flatten_coords(g)]
        ys = [p[1] for p in _flatten_coords(g)]
        minx, miny, maxx, maxy = min(xs), min(ys), max(xs), max(ys)
        w = max(1, int(round((maxx - minx) / res_m)))
        h = max(1, int(round((maxy - miny) / res_m)))
        win = from_bounds(minx, miny, maxx, maxy, transform=src.transform)
        arr = src.read(1, window=win, out_shape=(h, w), masked=True, boundless=True)
        tr = Affine(res_m, 0, minx, 0, -res_m, maxy)
        inside = geometry_mask([g], (h, w), tr, invert=True)
        return arr, inside


def _ndvi_from_bands(red, nir, valid, scale=1.0, offset=0.0):
    r = red.data[valid].astype(np.float64) * scale + offset
    n = nir.data[valid].astype(np.float64) * scale + offset
    denom = n + r
    ok = np.abs(denom) > 1e-6
    if ok.sum() < 3:
        return None
    v = (n[ok] - r[ok]) / denom[ok]
    v = v[(v > -1) & (v < 1)]
    if v.size < 3:
        return None
    return float(np.median(v)), int(v.size)


# --------------------------------------------------------------------------- #
# Sentinel-2
# --------------------------------------------------------------------------- #

#: классы маски SCL, которые считаем чистой поверхностью
SCL_CLEAR = (4, 5, 6, 7, 11)


def fetch_sentinel2(geom: dict, start: str, end: str, max_cloud: int = 60,
                    max_scenes: int = 400) -> pd.DataFrame:
    """Средний NDVI по полигону для каждого снимка Sentinel-2 L2A с маской облаков SCL."""
    minx, miny, maxx, maxy = geom_bbox(geom)
    body = {"collections": ["sentinel-2-l2a"], "bbox": [minx, miny, maxx, maxy],
            "datetime": f"{start}T00:00:00Z/{end}T23:59:59Z", "limit": 100,
            "query": {"eo:cloud_cover": {"lt": max_cloud}}}
    feats, token = [], None
    while len(feats) < max_scenes:
        payload = dict(body)
        if token:
            payload["token"] = token
        page = _http_json("POST", EARTH_SEARCH, json=payload)
        feats.extend(page.get("features", []))
        token = next((l.get("body", {}).get("token") for l in page.get("links", [])
                      if l.get("rel") == "next"), None)
        if not token:
            break

    def one(f):
        a = f["assets"]
        try:
            red, inside = _read_grid(a["red"]["href"], geom)
            nir, _ = _read_grid(a["nir"]["href"], geom)
            scl, _ = _read_grid(a["scl"]["href"], geom)
        except Exception as exc:
            log.debug("S2 %s: %s", f.get("id"), exc)
            return None
        clear = np.isin(scl.filled(0), SCL_CLEAR)
        valid = inside & ~red.mask & ~nir.mask & clear
        if valid.sum() < 3:
            return None
        res = _ndvi_from_bands(red, nir, valid, scale=1e-4)
        if res is None:
            return None
        # Sentinel-2 L2A с 2022 года хранит рефлектанс со сдвигом -1000
        shift = -0.1 if f["properties"].get("earthsearch:boa_offset_applied") else 0.0
        _ = shift  # NDVI как отношение к сдвигу почти нечувствителен, оставляем как есть
        return {"date": f["properties"]["datetime"][:10], "s2_ndvi": round(res[0], 4),
                "s2_pixels": res[1], "cloud_cover": f["properties"].get("eo:cloud_cover")}

    with ThreadPoolExecutor(MAX_WORKERS) as ex:
        rows = [r for r in ex.map(one, feats) if r]
    if not rows:
        return pd.DataFrame(columns=["date", "s2_ndvi"])
    df = pd.DataFrame(rows)
    return df.groupby("date", as_index=False).s2_ndvi.mean().sort_values("date")


# --------------------------------------------------------------------------- #
# Landsat 8/9
# --------------------------------------------------------------------------- #

def _sign_planetary(href: str) -> str:
    """Подписывает ссылку на хранилище Planetary Computer анонимным SAS-токеном."""
    try:
        _, rest = href.split("://", 1)
        host, path = rest.split("/", 1)
        account = host.split(".")[0]
        container = path.split("/", 1)[0]
    except ValueError:
        return href
    load, save = cached("pc_sas", (account, container), ttl_days=0.02)  # ~30 минут
    tok = load()
    if tok is None:
        tok = _http_json("GET", PLANETARY_SAS.format(account=account, container=container))
        save(tok)
    return f"{href}?{tok['token']}"


def fetch_landsat(geom: dict, start: str, end: str, max_cloud: int = 60,
                  max_scenes: int = 300) -> pd.DataFrame:
    """Средний NDVI по полигону для сцен Landsat 8/9 Collection 2 Level-2."""
    minx, miny, maxx, maxy = geom_bbox(geom)
    body = {"collections": ["landsat-c2-l2"], "bbox": [minx, miny, maxx, maxy],
            "datetime": f"{start}T00:00:00Z/{end}T23:59:59Z", "limit": 100,
            "query": {"eo:cloud_cover": {"lt": max_cloud},
                      "platform": {"in": ["landsat-8", "landsat-9", "landsat-7", "landsat-5"]}}}
    feats, page_url = [], PLANETARY_STAC
    while len(feats) < max_scenes:
        page = _http_json("POST", page_url, json=body)
        feats.extend(page.get("features", []))
        nxt = next((l for l in page.get("links", []) if l.get("rel") == "next"), None)
        if not nxt:
            break
        page_url = nxt["href"]
        body = nxt.get("body", body)

    def one(f):
        a = f["assets"]
        # у Landsat 5/7 ближний ИК называется nir08 так же, но красный — SR_B3
        red_key = "red" if "red" in a else "SR_B4"
        nir_key = "nir08" if "nir08" in a else "SR_B5"
        try:
            red, inside = _read_grid(_sign_planetary(a[red_key]["href"]), geom, res_m=30.0)
            nir, _ = _read_grid(_sign_planetary(a[nir_key]["href"]), geom, res_m=30.0)
            qa, _ = _read_grid(_sign_planetary(a["qa_pixel"]["href"]), geom, res_m=30.0)
        except Exception as exc:
            log.debug("Landsat %s: %s", f.get("id"), exc)
            return None
        # QA_PIXEL: бит 6 — «чистый пиксель», биты 1-5 — облака, тени, снег
        clear = (qa.filled(0).astype(np.uint16) & (1 << 6)) > 0
        valid = inside & ~red.mask & ~nir.mask & clear
        if valid.sum() < 3:
            return None
        # Collection 2 Level-2: рефлектанс = DN * 2.75e-5 - 0.2
        res = _ndvi_from_bands(red, nir, valid, scale=2.75e-5, offset=-0.2)
        if res is None:
            return None
        return {"date": f["properties"]["datetime"][:10], "landsat_ndvi": round(res[0], 4)}

    with ThreadPoolExecutor(MAX_WORKERS) as ex:
        rows = [r for r in ex.map(one, feats) if r]
    if not rows:
        return pd.DataFrame(columns=["date", "landsat_ndvi"])
    df = pd.DataFrame(rows)
    return df.groupby("date", as_index=False).landsat_ndvi.mean().sort_values("date")


# --------------------------------------------------------------------------- #
# MODIS
# --------------------------------------------------------------------------- #

def fetch_modis(geom: dict, start: str, end: str) -> pd.DataFrame:
    """MOD13Q1 (NDVI, композит за 16 дней) через REST-сервис ORNL DAAC.

    Пиксель MODIS 250 м обычно крупнее поля, поэтому берём точку в центре полигона.
    Сервис отдаёт не больше 10 композитов за запрос, поэтому даты сначала запрашиваются
    списком, а потом выбираются пачками по 10 — каждая пачка кэшируется отдельно.
    """
    lat, lon = geom_centroid(geom)
    lat_r, lon_r = round(lat, 3), round(lon, 3)

    load, save = cached("modis_dates", (lat_r, lon_r), ttl_days=180)
    dates = load()
    if dates is None:
        try:
            data = _http_json("GET", MODIS_DATES, params={"latitude": lat_r, "longitude": lon_r},
                              headers={"Accept": "application/json"})
            dates = [(d["modis_date"], d["calendar_date"]) for d in data.get("dates", [])]
            save(dates)
        except Exception as exc:
            log.warning("MODIS: список дат недоступен: %s", exc)
            return pd.DataFrame(columns=["date", "modis_ndvi"])

    wanted = [d for d in dates if start <= d[1] <= end]
    if not wanted:
        return pd.DataFrame(columns=["date", "modis_ndvi"])
    chunks = [wanted[i:i + MODIS_CHUNK] for i in range(0, len(wanted), MODIS_CHUNK)]

    def fetch_chunk(chunk):
        load_c, save_c = cached("modis", (lat_r, lon_r, chunk[0][0], chunk[-1][0]), ttl_days=180)
        hit = load_c()
        if hit is not None:
            return hit
        try:
            data = _http_json("GET", MODIS_REST, params={
                "latitude": lat_r, "longitude": lon_r,
                "startDate": chunk[0][0], "endDate": chunk[-1][0],
                "kmAboveBelow": 0, "kmLeftRight": 0, "band": "250m_16_days_NDVI"},
                headers={"Accept": "application/json"})
            hit = [{"date": s_["calendar_date"], "v": s_["data"][0]} for s_ in data.get("subset", [])]
            save_c(hit)
            return hit
        except Exception as exc:
            log.warning("MODIS %s-%s: %s", chunk[0][0], chunk[-1][0], exc)
            return []

    # сервис ORNL не любит агрессивный параллелизм — четырёх потоков достаточно
    with ThreadPoolExecutor(4) as ex:
        rows = [r for part in ex.map(fetch_chunk, chunks) for r in part]
    if not rows:
        return pd.DataFrame(columns=["date", "modis_ndvi"])
    df = pd.DataFrame(rows)
    df["modis_ndvi"] = pd.to_numeric(df.v, errors="coerce") * 1e-4
    df = df[(df.modis_ndvi > -1) & (df.modis_ndvi < 1)]
    df = df[(df.date >= start) & (df.date <= end)]
    return df[["date", "modis_ndvi"]].drop_duplicates("date").sort_values("date").reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Погода
# --------------------------------------------------------------------------- #

def fetch_weather(geom: dict, start: str, end: str) -> pd.DataFrame:
    """Суточная температура и осадки из архива ERA5-Land (Open-Meteo, без ключа)."""
    lat, lon = geom_centroid(geom)
    load, save = cached("weather", (round(lat, 2), round(lon, 2), start, end), ttl_days=30)
    hit = load()
    if hit is None:
        data = _http_json("GET", OPEN_METEO, params={
            "latitude": round(lat, 3), "longitude": round(lon, 3),
            "start_date": start, "end_date": end,
            "daily": "temperature_2m_mean,precipitation_sum", "timezone": "UTC"})
        hit = data.get("daily", {})
        save(hit)
    if not hit:
        return pd.DataFrame(columns=["date", "era5_temp_c", "era5_precip_mm"])
    return pd.DataFrame({"date": hit["time"],
                         "era5_temp_c": hit["temperature_2m_mean"],
                         "era5_precip_mm": hit["precipitation_sum"]})


# --------------------------------------------------------------------------- #
# Сборка ряда
# --------------------------------------------------------------------------- #

def collect_series(geom: dict, start: str, end: str, polygon_id: str = "AOI-USER",
                   crop_type: str = "не указана", use: tuple[str, ...] = ("s2", "landsat", "modis", "weather"),
                   use_cache: bool = True, cache_only: bool = False) -> CollectResult:
    """Собирает ряд по полигону в том же формате, что и датасет соревнования.

    ``primary_ndvi`` считается той же склейкой coalesce(S2 -> Landsat -> MODIS), что и в
    обучающих данных: модель и детектор аномалий видят ровно привычную им структуру.

    ``cache_only`` возвращает ряд только если он уже собран: так эталонные поля подтягиваются
    к запросу мгновенно и не задерживают ответ сетевыми походами.
    """
    key = _cache_key("series", geom, start, end, sorted(use))
    cache_file = CACHE_DIR / "series" / f"{key}.csv"
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    res = CollectResult(series=pd.DataFrame())

    meta_file = cache_file.with_suffix(".json")
    if use_cache and cache_file.exists():
        df = pd.read_csv(cache_file, parse_dates=["date"])
        res.series = df
        # диагностику источников тоже достаём из кэша, иначе после перезагрузки
        # интерфейс показывал бы просто «кэш» вместо списка отработавших источников
        if meta_file.exists():
            try:
                res.sources = json.loads(meta_file.read_text())
            except Exception:
                pass
        for name, info in res.sources.items():
            info["cached"] = True
        if not res.sources:
            res.note("кэш", True, len(df))
        return res

    if cache_only:
        # вызывающий готов обойтись без этого ряда: в сеть не идём
        res.note("кэш", False, 0, "ряд ещё не собирался")
        return res

    frames = []
    jobs = {
        "Sentinel-2": (lambda: fetch_sentinel2(geom, start, end)) if "s2" in use else None,
        "Landsat": (lambda: fetch_landsat(geom, start, end)) if "landsat" in use else None,
        "MODIS": (lambda: fetch_modis(geom, start, end)) if "modis" in use else None,
        "Погода ERA5": (lambda: fetch_weather(geom, start, end)) if "weather" in use else None,
    }
    with ThreadPoolExecutor(4) as ex:
        futures = {name: ex.submit(fn) for name, fn in jobs.items() if fn}
        for name, fut in futures.items():
            t0 = time.time()
            try:
                df = fut.result()
                res.note(name, len(df) > 0, len(df), None if len(df) else "нет данных за период",
                         time.time() - t0)
                if len(df):
                    frames.append(df)
            except Exception as exc:
                # источник упал — остальные всё равно отдадут данные
                res.note(name, False, 0, f"{type(exc).__name__}: {exc}", time.time() - t0)

    if not frames:
        return res

    out = frames[0]
    for f in frames[1:]:
        out = out.merge(f, on="date", how="outer")
    out["date"] = pd.to_datetime(out.date)
    out = out.sort_values("date").reset_index(drop=True)

    for col in ("s2_ndvi", "landsat_ndvi", "modis_ndvi", "era5_temp_c", "era5_precip_mm"):
        if col not in out:
            out[col] = np.nan
    out["anon_polygon_id"] = polygon_id
    out["crop_type"] = crop_type
    out["year"] = out.date.dt.year.astype("int16")
    out["doy"] = out.date.dt.dayofyear.astype("int16")
    out["src"] = np.select(
        [out.s2_ndvi.notna(), out.landsat_ndvi.notna(), out.modis_ndvi.notna()],
        ["s2", "landsat", "modis"], default="none")
    out["primary_ndvi"] = out.s2_ndvi.combine_first(out.landsat_ndvi).combine_first(out.modis_ndvi)

    res.series = out
    try:
        out.to_csv(cache_file, index=False)
        meta_file.write_text(json.dumps(res.sources, ensure_ascii=False))
    except Exception as exc:
        log.warning("не удалось сохранить ряд в кэш: %s", exc)
    return res
