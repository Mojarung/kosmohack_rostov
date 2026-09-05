"""Автоматический сбор данных для произвольного полигона: Sentinel-2, Landsat, MODIS (STAC) и ERA5 (Open-Meteo).

Источники (без ключей и регистрации):
- Sentinel-2 L2A: Earth Search v1 (Element84), коллекция sentinel-2-l2a, маска облаков по SCL;
- Landsat 8/9 C2 L2: Microsoft Planetary Computer, коллекция landsat-c2-l2, маска по qa_pixel;
- MODIS MOD13Q1 v061: Planetary Computer, коллекция modis-13Q1-061 (16-дневный композит, датируется началом окна);
- ERA5 (среднесуточная температура, осадки): Open-Meteo Historical Weather API по центроиду полигона.
Из них собирается таблица наблюдений в формате данных кейса (primary_ndvi = S2 → Landsat → MODIS) и ежедневная
погода, после чего работает тот же пайплайн детекции (anomaly/).
"""

from __future__ import annotations

import logging
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import planetary_computer
import pystac_client
from odc.stac import load as stac_load
from rasterio.features import geometry_mask
from shapely.geometry import shape

from service.field_store import field_id
from service.weather_source import SOURCE, collect_weather, weather_records

log = logging.getLogger("service.collect")
EARTH_SEARCH = "https://earth-search.aws.element84.com/v1"
PLANETARY = "https://planetarycomputer.microsoft.com/api/stac/v1"
OPEN_METEO = "https://archive-api.open-meteo.com/v1/archive"
SEASON = ("04-01", "10-30")
MIN_VALID_SHARE = 0.6          # доля чистых пикселей в полигоне, ниже — сцена отбрасывается
MIN_PIXELS = 6
S2_CLEAR_SCL = (4, 5, 6)       # растительность, открытая почва, вода
LANDSAT_BAD_BITS = (1, 2, 3, 4, 5)   # dilated cloud, cirrus, cloud, shadow, snow


def _season_range(year: int) -> str:
    return f"{year}-{SEASON[0]}/{year}-{SEASON[1]}"


def utm_crs(geom) -> str:
    """Зона UTM по центроиду полигона (Ростовская область — 37N, но сервис не привязан к региону)."""
    c = geom.centroid
    zone = int((c.x + 180) // 6) + 1
    return f"EPSG:{32600 + zone if c.y >= 0 else 32700 + zone}"


def _load_parallel(items, bands: list[str], geom, resolution: int, dtype=None):
    """Загрузка всех сцен по рамке полигона одним вызовом: dask читает COG параллельно, результат — в памяти."""
    # fail_on_error=False: одна битая сцена (недоступный файл в каталоге) не должна ронять весь год
    lazy = stac_load(items, bands=bands, geopolygon=geom.__geo_interface__, resolution=resolution,
                     chunks={"time": 4}, groupby="solar_day", crs=utm_crs(geom), dtype=dtype, fail_on_error=False)
    return lazy.compute()


def _by_years(fn, years: range, max_workers: int = 4, label: str = "") -> pd.DataFrame:
    """Сезоны собираются параллельно (каждый год — отдельный запрос к каталогу и своя загрузка)."""
    def timed(year):
        t0 = time.time()
        try:
            frame = fn(year)
        except Exception as exc:
            log.exception("%s %s: год не загружен", label, year)
            frame = pd.DataFrame()
            frame.attrs["failure"] = f"{label} {year}: {type(exc).__name__}"
        log.info("%s %s: %d сцен за %.0f с", label, year, len(frame), time.time() - t0)
        return frame
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        frames = list(pool.map(timed, years))
    failures = [f.attrs["failure"] for f in frames if "failure" in f.attrs]
    frames = [f for f in frames if len(f)]
    result = pd.concat(frames, ignore_index=True).sort_values("date").reset_index(drop=True) if frames else pd.DataFrame()
    result.attrs["warnings"] = failures
    return result


def _polygon_mask(data, geom) -> np.ndarray:
    """Булева маска пикселей внутри полигона в системе координат загруженного массива."""
    from pyproj import Transformer
    crs = data.odc.crs
    transformer = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
    rings = [[transformer.transform(x, y) for x, y in ring.coords] for ring in [geom.exterior, *geom.interiors]]
    return ~geometry_mask([{"type": "Polygon", "coordinates": rings}], out_shape=(data.sizes["y"], data.sizes["x"]),
                          transform=data.odc.geobox.transform, invert=False)


def _mean_index(num, den, clear, inside) -> tuple[float, float, float]:
    """Средний NDVI по чистым пикселям полигона; возвращает (ndvi, доля чистых, число пикселей)."""
    valid = clear & inside & np.isfinite(num) & np.isfinite(den) & ((num + den) != 0)
    n_inside = int(inside.sum())
    if n_inside < MIN_PIXELS or valid.sum() / max(n_inside, 1) < MIN_VALID_SHARE:
        return np.nan, float(valid.sum() / max(n_inside, 1)), n_inside
    ndvi = (num[valid] - den[valid]) / (num[valid] + den[valid])
    return float(np.clip(ndvi.mean(), -1, 1)), float(valid.sum() / n_inside), n_inside


def _s2_year(geom, year: int) -> pd.DataFrame:
    client = pystac_client.Client.open(EARTH_SEARCH)
    items = list(client.search(collections=["sentinel-2-l2a"], intersects=geom.__geo_interface__,
                               datetime=_season_range(year), query={"eo:cloud_cover": {"lt": 80}}).items())
    if not items:
        return pd.DataFrame()
    items = public_s2_items(items)
    if not items:
        return pd.DataFrame()
    data = _load_parallel(items, ["red", "nir", "scl"], geom, resolution=20)
    inside = _polygon_mask(data, geom)
    rows = []
    for t in data.time.values:
        frame = data.sel(time=t)
        ndvi, share, _ = _mean_index(frame["nir"].values.astype(float), frame["red"].values.astype(float),
                                     np.isin(frame["scl"].values, S2_CLEAR_SCL), inside)
        if np.isfinite(ndvi):
            rows.append({"date": pd.Timestamp(t).normalize(), "s2_ndvi": ndvi, "s2_clear_share": share})
    return pd.DataFrame(rows)


def public_s2_items(items):
    """Исключает JP2-дубликаты: их псевдонимы перекрывают публичные HTTPS COG и ведут в платный S3."""
    result = []
    for item in items:
        if not all(key in item.assets and item.assets[key].href.startswith("https://") for key in ("red", "nir", "scl")):
            continue
        public = item.clone()
        for key in list(public.assets):
            if key.endswith("-jp2"):
                del public.assets[key]
        result.append(public)
    return result


def collect_s2(geom, years: range) -> pd.DataFrame:
    """Sentinel-2 L2A: NDVI по чистым пикселям (SCL) на каждую сцену."""
    return _by_years(lambda y: _s2_year(geom, y), years, label="S2")


def _landsat_year(geom, year: int) -> pd.DataFrame:
    client = pystac_client.Client.open(PLANETARY, modifier=planetary_computer.sign_inplace)
    items = list(client.search(collections=["landsat-c2-l2"], intersects=geom.__geo_interface__,
                               datetime=_season_range(year), query={"eo:cloud_cover": {"lt": 80},
                                                                    "platform": {"in": ["landsat-8", "landsat-9"]}}).items())
    if not items:
        return pd.DataFrame()
    data = _load_parallel(items, ["red", "nir08", "qa_pixel"], geom, resolution=30)
    inside = _polygon_mask(data, geom)
    rows = []
    for t in data.time.values:
        frame = data.sel(time=t)
        qa = frame["qa_pixel"].values.astype(np.int64)
        bad = np.zeros(qa.shape, dtype=bool)
        for bit in LANDSAT_BAD_BITS:
            bad |= (qa >> bit) & 1 == 1
        red = frame["red"].values.astype(float) * 0.0000275 - 0.2
        nir = frame["nir08"].values.astype(float) * 0.0000275 - 0.2
        ndvi, share, _ = _mean_index(nir, red, ~bad & (qa != 0), inside)
        if np.isfinite(ndvi):
            rows.append({"date": pd.Timestamp(t).normalize(), "landsat_ndvi": ndvi, "landsat_clear_share": share})
    return pd.DataFrame(rows)


def collect_landsat(geom, years: range) -> pd.DataFrame:
    """Landsat 8/9 C2 L2 (Planetary Computer): NDVI по пикселям без облаков/теней по qa_pixel."""
    return _by_years(lambda y: _landsat_year(geom, y), years, label="Landsat")


def _modis_year(geom, year: int) -> pd.DataFrame:
    client = pystac_client.Client.open(PLANETARY, modifier=planetary_computer.sign_inplace)
    items = list(client.search(collections=["modis-13Q1-061"], intersects=geom.__geo_interface__,
                               datetime=_season_range(year)).items())
    if not items:
        return pd.DataFrame()
    # надёжность хранится как int8 с nodata 255 — читаем в int16, иначе odc-stac падает на переполнении
    data = _load_parallel(items, ["250m_16_days_NDVI", "250m_16_days_pixel_reliability"], geom, resolution=250, dtype="int16")
    inside = _polygon_mask(data, geom)
    rows = []
    for t in data.time.values:
        frame = data.sel(time=t)
        ndvi = frame["250m_16_days_NDVI"].values.astype(float) * 0.0001
        good = np.isin(frame["250m_16_days_pixel_reliability"].values, (0, 1)) & inside & (ndvi > -1) & (ndvi < 1)
        if good.sum() >= 1:
            rows.append({"date": pd.Timestamp(t).normalize(), "modis_ndvi": float(np.clip(ndvi[good].mean(), -1, 1))})
    return pd.DataFrame(rows)


def collect_modis(geom, years: range) -> pd.DataFrame:
    """MODIS MOD13Q1 (Planetary Computer): готовый NDVI композита с фильтром надёжности."""
    return _by_years(lambda y: _modis_year(geom, y), years, label="MODIS")


def assemble_observations(pid: str, s2: pd.DataFrame, ls: pd.DataFrame, md: pd.DataFrame) -> pd.DataFrame:
    """Таблица наблюдений в формате кейса: одна строка на дату, primary_ndvi по приоритету S2 → Landsat → MODIS."""
    frames = [f.set_index("date") for f in (s2, ls, md) if len(f)]
    if not frames:
        raise ValueError("ни один спутниковый источник не вернул чистых наблюдений")
    df = pd.concat(frames, axis=1).sort_index().reset_index()
    for col in ("s2_ndvi", "landsat_ndvi", "modis_ndvi", "s2_evi", "s2_ndwi", "landsat_evi", "landsat_ndwi", "modis_evi"):
        if col not in df:
            df[col] = np.nan
    df["primary_ndvi"] = df["s2_ndvi"].fillna(df["landsat_ndvi"]).fillna(df["modis_ndvi"])
    df["sensor"] = np.select([df["s2_ndvi"].notna(), df["landsat_ndvi"].notna()], [0, 1], 2).astype("int8")
    df = df.assign(pid=pid, day_num=(df["date"] - pd.Timestamp("2000-01-01")).dt.days.astype("int32"),
                   year=df["date"].dt.year.astype("int16"), doy=df["date"].dt.dayofyear.astype("int16"), crop=-1)
    # композиты MODIS, начинающиеся в марте, попадают в выборку по пересечению дат — оставляем только сезон
    in_season = (df["doy"] >= 91) & (df["doy"] <= 304)
    return df.loc[in_season].dropna(subset=["primary_ndvi"]).reset_index(drop=True)


def analyze_geometry(geometry: dict, name: str, start_year: int, end_year: int) -> dict:
    """Полный цикл для нового полигона: сбор → наблюдения → погода → детекция → JSON для интерфейса."""
    from service.analyze_new import analyze_new_polygon
    geom = shape(geometry)
    years = range(start_year, end_year + 1)
    frames, notes, warnings = {}, [], []
    # каждый источник независим: если один недоступен, ряд строится по остальным, а пользователь видит пометку
    for label, fn in (("S2", collect_s2), ("Landsat", collect_landsat), ("MODIS", collect_modis)):
        try:
            frames[label] = fn(geom, years)
            warnings.extend(frames[label].attrs.get("warnings", []))
            notes.append(f"{label} {len(frames[label])} сцен")
        except Exception as exc:    # сетевые ошибки, недоступный каталог, пустой ответ
            log.exception("Источник %s недоступен", label)
            frames[label] = pd.DataFrame()
            notes.append(f"{label} недоступен ({type(exc).__name__})")
            warnings.append(f"{label} недоступен: {type(exc).__name__}")
    pid = field_id(geometry)
    obs = assemble_observations(pid, frames["S2"], frames["Landsat"], frames["MODIS"])
    centroid = geom.centroid
    try:
        weather = collect_weather(centroid.y, centroid.x, years).assign(pid=pid)
        warnings.extend(weather.attrs.get("warnings", []))
        notes.append(f"ERA5 {len(weather)} дней")
    except Exception as exc:
        log.exception("Метеоданные недоступны")
        weather = pd.DataFrame(columns=["date", "era5_temp_c", "era5_precip_mm", "pid"])
        notes.append(f"ERA5 недоступен ({type(exc).__name__})")
        warnings.append("Погода ERA5 не загрузилась")
    recent_weather = weather.loc[weather["date"].dt.year >= start_year] if len(weather) else weather
    result = analyze_new_polygon(pid, obs, recent_weather)
    result["_weather"] = weather_records(weather)
    result["weather_source"] = SOURCE if len(weather) else "Погода недоступна"
    result["name"] = name
    result["warnings"] = warnings
    result["collected"] = ", ".join(notes)
    result["geometry"] = geometry
    return result
