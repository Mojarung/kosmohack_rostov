"""Пиксельные карты Sentinel-2 C1, независимые от сглаженного ряда детектора."""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import logging
import uuid
from pathlib import Path

import numpy as np
import pandas as pd

from service.field_store import _write

ROOT = Path(__file__).resolve().parents[1] / "artifacts" / "field-imagery"
COLLECTION = "sentinel-2-c1-l2a"
RESOLUTION = 20
DROP_THRESHOLD = -0.1  # визуальный фильтр изменения, не агрономический диагноз
PALETTES = {
    "ndvi": ([ -1, 0, .2, .5, .8, 1], ["#9b8872", "#c2ad85", "#e7d596", "#91b967", "#337750", "#174d38"]),
    "ndmi": ([-1, -.4, 0, .3, .6, 1], ["#a98261", "#d8b88e", "#e6e6c9", "#8ac6b4", "#398c98", "#21536f"]),
    "change": ([-.4, -.1, 0, .1, .4], ["#ac524b", "#d99171", "#f5f2e5", "#85b795", "#28765b"]),
}


def directory(pid: str, year: int) -> Path:
    return ROOT / hashlib.sha256(pid.encode()).hexdigest() / "v1" / str(year)


def read(pid: str, year: int) -> dict | None:
    path = directory(pid, year) / "manifest.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def image_path(pid: str, year: int, date: str, index: str) -> Path | None:
    manifest = read(pid, year)
    if not manifest or index not in PALETTES or not any(s["date"] == date for s in manifest["scenes"]):
        return None
    path = directory(pid, year) / manifest["generation"] / f"{date}-{index}.png"
    return path if path.is_file() else None


def normalized_difference(a, b, clear, inside):
    """Невалидные/нефизические отражения и облака остаются прозрачными."""
    valid = inside & clear & np.isfinite(a) & np.isfinite(b) & (a >= 0) & (b >= 0) & ((a + b) > 0)
    out = np.full(a.shape, np.nan, dtype="float32")
    np.divide(a - b, a + b, out=out, where=valid)
    return out


def scene_stats(values, total: int) -> dict:
    valid = np.isfinite(values)
    n = int(valid.sum())
    return {"mean": round(float(values[valid].mean()), 4) if n else None,
            "clear_share": round(n / max(total, 1), 4), "area_ha": round(n * RESOLUTION**2 / 10000, 2)}


def compare(current, previous, total: int) -> tuple[np.ndarray, dict]:
    """Общая маска: площадь облаков не считается ни падением, ни восстановлением."""
    delta = current - previous
    stats = scene_stats(delta, total)
    valid = np.isfinite(delta)
    count = int(valid.sum())
    # Отражения хранятся в float32: учитываем погрешность на самой границе −0,1.
    lower = int((delta[valid] <= DROP_THRESHOLD + 1e-7).sum())
    stats.update(drop_area_ha=round(lower * RESOLUTION**2 / 10000, 2),
                 drop_share=round(lower / count, 4) if count else None, threshold=DROP_THRESHOLD)
    return delta, stats


def rgba(values: np.ndarray, index: str) -> np.ndarray:
    positions, hexes = PALETTES[index]
    colors = np.array([[int(h[j:j + 2], 16) for j in (1, 3, 5)] for h in hexes])
    safe = np.nan_to_num(values, nan=0)
    result = np.zeros((*values.shape, 4), dtype="uint8")
    for channel in range(3):
        result[..., channel] = np.interp(safe, positions, colors[:, channel]).astype("uint8")
    result[..., 3] = np.where(np.isfinite(values), 235, 0)
    return result


def _save_map(values, index, path, geobox, target):
    from matplotlib.image import imsave
    from rasterio.enums import Resampling
    from rasterio.warp import reproject

    transform, width, height = target
    destination = np.full((height, width), np.nan, dtype="float32")
    reproject(values, destination, src_transform=geobox.transform, src_crs=str(geobox.crs),
              dst_transform=transform, dst_crs="EPSG:4326", src_nodata=np.nan, dst_nodata=np.nan,
              resampling=Resampling.nearest)
    imsave(path, rgba(destination, index))


def collect(pid: str, geometry: dict, year: int) -> dict:
    """Один сезон, ограниченные партии; полный результат публикуется атомарно."""
    if (cached := read(pid, year)) is not None:
        return cached
    import pystac_client
    import rasterio
    from odc.stac import configure_rio, load
    from rasterio.transform import array_bounds
    from rasterio.warp import calculate_default_transform
    from shapely.geometry import shape

    from service.collect import EARTH_SEARCH, MIN_PIXELS, MIN_VALID_SHARE, S2_CLEAR_SCL, _polygon_mask, public_s2_items, utm_crs

    logging.basicConfig(level=logging.INFO)
    logging.getLogger("rasterio.session").setLevel(logging.WARNING)
    log = logging.getLogger(__name__)
    configure_rio(cloud_defaults=True, GDAL_HTTP_TIMEOUT=30, GDAL_HTTP_CONNECTTIMEOUT=15,
                  GDAL_HTTP_MAX_RETRY=1, GDAL_HTTP_RETRY_DELAY=1)
    geom = shape(geometry)
    end = min(dt.date(year, 10, 30), dt.datetime.now(dt.UTC).date())
    if end < dt.date(year, 4, 1):
        raise ValueError("Сезон ещё не начался")
    client = pystac_client.Client.open(EARTH_SEARCH, timeout=60)
    items = public_s2_items(list(client.search(collections=[COLLECTION], intersects=geometry,
        datetime=f"{year}-04-01/{end.isoformat()}", query={"eo:cloud_cover": {"lt": 80}}).items()))
    bands = ["red", "nir", "swir16", "scl"]
    items = [item for item in items if all(key in item.assets for key in bands)]
    if not items:
        raise ValueError("Sentinel-2 не вернул снимки этого сезона")
    log.info("Карта %s %s: %s сцен в каталоге", pid, year, len(items))
    # C1 хранит исходные DN. Применяем scale/offset из STAC; старую коллекцию с уже
    # изменёнными DN не смешиваем с C1. Разные калибровки читаются отдельными партиями.
    groups = {}
    for item in items:
        coefficients = tuple((item.assets[b].extra_fields["raster:bands"][0].get("scale", 1),
                              item.assets[b].extra_fields["raster:bands"][0].get("offset", 0)) for b in bands[:-1])
        groups.setdefault(coefficients, []).append(item)
    generation = uuid.uuid4().hex
    folder = directory(pid, year) / generation
    folder.mkdir(parents=True, exist_ok=True)
    scenes, arrays, geobox, inside = {}, {}, None, None
    with rasterio.Env(GDAL_HTTP_TIMEOUT="45", GDAL_HTTP_MAX_RETRY="2", GDAL_HTTP_RETRY_DELAY="1"):
        for coefficients, group in groups.items():
            days = {}
            for item in group:
                days.setdefault(item.datetime.date(), []).append(item)
            dates = sorted(days)
            for first in range(0, len(dates), 8):
                batch = [item for day in dates[first:first + 8] for item in days[day]]
                location = {"geobox": geobox} if geobox is not None else {
                    "geopolygon": geometry, "crs": utm_crs(geom), "resolution": RESOLUTION}
                lazy = load(batch, bands=bands, groupby="solar_day", chunks={"time": 2},
                            resampling="nearest", fail_on_error=False, **location)
                if lazy.sizes["y"] * lazy.sizes["x"] > 250_000:
                    raise ValueError("Контур слишком большой для карты 20 м. Выберите отдельное поле.")
                # GDAL в нескольких потоках под Windows может взаимно блокировать open().
                # Изолированный процесс и последовательные чтения сохраняют отзывчивость API.
                data = lazy.compute(scheduler="synchronous")
                log.info("Карта %s: загружены даты %s — %s", pid, dates[first], dates[min(first + 7, len(dates) - 1)])
                if geobox is None:
                    geobox, inside = data.odc.geobox, _polygon_mask(data, geom)
                total = int(inside.sum())
                if total < MIN_PIXELS:
                    raise ValueError("Поле слишком мало для шести пикселей Sentinel-2 по 20 м")
                for time in data.time.values:
                    frame = data.sel(time=time)
                    clear = np.isin(frame["scl"].values, S2_CLEAR_SCL)
                    reflectance = []
                    for band, (scale, offset) in zip(bands[:-1], coefficients):
                        raw = frame[band].values.astype("float32")
                        reflectance.append(np.where(raw > 0, raw * scale + offset, np.nan))
                    red, nir, swir = reflectance
                    ndvi = normalized_difference(nir, red, clear, inside)
                    if np.isfinite(ndvi).sum() / total < MIN_VALID_SHARE:
                        continue
                    ndmi = normalized_difference(nir, swir, clear, inside)
                    day = pd.Timestamp(time).date().isoformat()
                    # Если день попал в две группы калибровки, берём наиболее полную сцену.
                    if day in arrays and np.isfinite(arrays[day][0]).sum() >= np.isfinite(ndvi).sum():
                        continue
                    arrays[day] = (ndvi, ndmi)
                    scenes[day] = {"date": day, "ndvi": scene_stats(ndvi, total), "ndmi": scene_stats(ndmi, total)}
    if not scenes:
        raise ValueError("В этом сезоне нет снимков с чистым покрытием поля от 60%")
    transform, width, height = calculate_default_transform(str(geobox.crs), "EPSG:4326", geobox.width, geobox.height,
                                                           *geobox.boundingbox)
    west, south, east, north = array_bounds(height, width, transform)
    target = (transform, width, height)
    previous, previous_date = None, None
    for day in sorted(scenes):
        ndvi, ndmi = arrays[day]
        # Сохраняем численные значения для воспроизводимости и проверки карты.
        np.savez_compressed(folder / f"{day}.npz", ndvi=ndvi, ndmi=ndmi)
        for index, values in (("ndvi", ndvi), ("ndmi", ndmi)):
            _save_map(values, index, folder / f"{day}-{index}.png", geobox, target)
        if previous is not None:
            delta, stats = compare(ndvi, previous, int(inside.sum()))
            if stats["clear_share"] >= MIN_VALID_SHARE:
                scenes[day]["change"] = stats | {"previous_date": previous_date}
                _save_map(delta, "change", folder / f"{day}-change.png", geobox, target)
        previous, previous_date = ndvi, day
    manifest = {"status": "ready", "pid": pid, "year": year, "generation": generation,
        "source": "Sentinel-2 L2A C1 · Earth Search · 20 м", "geometry": geometry,
        "bounds": [[south, west], [north, east]], "area_ha": round(int(inside.sum()) * .04, 2),
        "scenes": [scenes[day] for day in sorted(scenes)], "collected_at": dt.datetime.now(dt.UTC).isoformat()}
    _write(directory(pid, year) / "manifest.json", manifest)
    return manifest


def ndmi_context(pid: str, year: int) -> dict:
    """Наблюдения без интерполяции; исторические месячные ориентиры по загруженным годам."""
    current = read(pid, year)
    points = current["scenes"] if current else []
    points = [s for s in points if s["ndmi"]["clear_share"] >= .6 and s["ndmi"]["mean"] is not None]
    previous = {y: read(pid, y) for y in range(max(2017, year - 10), year)}
    mean, low, high, count = [], [], [], []
    for point in points:
        month = point["date"][5:7]
        values = []
        for old in previous.values():
            samples = [s["ndmi"]["mean"] for s in old["scenes"] if s["date"][5:7] == month
                       and s["ndmi"]["clear_share"] >= .6 and s["ndmi"]["mean"] is not None] if old else []
            if samples:
                values.append(float(np.mean(samples)))
        count.append(len(values))
        mean.append(round(float(np.mean(values)), 3) if len(values) >= 3 else None)
        low.append(round(float(np.quantile(values, .1)), 3) if len(values) >= 3 else None)
        high.append(round(float(np.quantile(values, .9)), 3) if len(values) >= 3 else None)
    return {"available": bool(points), "date": [s["date"] for s in points],
            "value": [s["ndmi"]["mean"] for s in points], "mean": mean, "low": low, "high": high,
            "n_years": count, "history_years": [y for y, value in previous.items() if value]}
