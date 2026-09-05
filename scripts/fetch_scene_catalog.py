"""Каталог сцен Sentinel-2 и Landsat над Ростовской областью: дата, тайл, облачность.

Внешний признак для восстановления пропусков: если в день D все снимки Sentinel-2 над областью
были на 80 % в облаках, скрытое наблюдение этого дня почти наверняка загрязнено. Метаданные
берутся из открытых STAC-каталогов без ключей: Earth Search (S2 L2A) и Planetary Computer
(Landsat C2 L2). Дополнительно каталог — это настоящее расписание съёмки региона.
"""

import argparse
import time

import _bootstrap  # noqa: F401
import pandas as pd

from ndvi.collect import EARTH_SEARCH, PLANETARY_STAC, _http_json
from ndvi.paths import ARTIFACTS_DIR

# Ростовская область с запасом
BBOX = [38.0, 45.6, 44.6, 50.2]

ap = argparse.ArgumentParser()
ap.add_argument("--start", default="2010-01-01")
ap.add_argument("--end", default="2025-12-31")
ap.add_argument("--merge", action="store_true",
                help="дописать в существующий каталог только годы/сенсоры, которых в нём нет")
args = ap.parse_args()

existing = pd.DataFrame()
have = set()
if args.merge and (ARTIFACTS_DIR / "scene_catalog.csv").exists():
    existing = pd.read_csv(ARTIFACTS_DIR / "scene_catalog.csv")
    have = set(zip(existing.sensor, pd.to_datetime(existing.date).dt.year))


def paged(url, body, page_size=200, max_items=200000, sleep=0.15, retries=5):
    items, token, page_url = [], None, url
    while len(items) < max_items:
        payload = dict(body, limit=page_size)
        if token:
            payload["token"] = token
        page = None
        for attempt in range(retries):
            try:
                page = _http_json("POST", page_url, json=payload)
                break
            except Exception as exc:  # STAC-каталоги иногда отвечают 502 — ждём и повторяем
                if attempt == retries - 1:
                    raise
                time.sleep(3 * (attempt + 1))
        feats = page.get("features", [])
        items.extend(feats)
        nxt = next((l for l in page.get("links", []) if l.get("rel") == "next"), None)
        if not nxt or not feats:
            break
        if nxt.get("body", {}).get("token"):
            token = nxt["body"]["token"]
        elif nxt.get("href") and nxt["href"] != page_url:
            page_url = nxt["href"]
            payload = nxt.get("body", payload)
        else:
            break
        time.sleep(sleep)
    return items


def centroid(feat):
    bb = feat.get("bbox")
    if bb and len(bb) >= 4:
        return (bb[1] + bb[3]) / 2, (bb[0] + bb[2]) / 2
    return None, None


rows, t0 = existing.to_dict("records") if len(existing) else [], time.time()
for year in range(int(args.start[:4]), int(args.end[:4]) + 1):
    y0, y1 = f"{year}-01-01T00:00:00Z", f"{year}-12-31T23:59:59Z"
    if year >= 2015 and ("s2", year) not in have:
        try:
            s2 = paged(EARTH_SEARCH, {"collections": ["sentinel-2-l2a"], "bbox": BBOX, "datetime": f"{y0}/{y1}"})
            for f in s2:
                p = f["properties"]
                lat, lon = centroid(f)
                rows.append({"sensor": "s2", "date": p["datetime"][:10], "tile": p.get("grid:code") or p.get("s2:mgrs_tile"),
                             "orbit": p.get("sat:relative_orbit"), "platform": p.get("platform"),
                             "cloud": p.get("eo:cloud_cover"), "lat": lat, "lon": lon,
                             "nodata": p.get("s2:nodata_pixel_percentage")})
            print(f"{year}: S2 {len(s2)} сцен", flush=True)
        except Exception as exc:
            print(f"{year}: S2 ошибка {exc}", flush=True)
    if ("landsat", year) in have:
        continue
    try:
        ls = paged(PLANETARY_STAC, {"collections": ["landsat-c2-l2"], "bbox": BBOX, "datetime": f"{y0}/{y1}",
                                    "query": {"platform": {"in": ["landsat-5", "landsat-7", "landsat-8", "landsat-9"]}}})
        for f in ls:
            p = f["properties"]
            lat, lon = centroid(f)
            rows.append({"sensor": "landsat", "date": p["datetime"][:10],
                         "tile": f"{p.get('landsat:wrs_path')}/{p.get('landsat:wrs_row')}",
                         "orbit": p.get("landsat:wrs_path"), "platform": p.get("platform"),
                         "cloud": p.get("eo:cloud_cover"), "lat": lat, "lon": lon,
                         "nodata": p.get("landsat:cloud_cover_land")})
        print(f"{year}: Landsat {len(ls)} сцен", flush=True)
    except Exception as exc:
        print(f"{year}: Landsat ошибка {exc}", flush=True)
    pd.DataFrame(rows).to_csv(ARTIFACTS_DIR / "scene_catalog.csv", index=False)

df = pd.DataFrame(rows)
print(f"\nитого {len(df)} сцен: {df.groupby('sensor').size().to_dict()}, {time.time() - t0:.0f} с")
print("облачность по сенсорам (медиана):", df.groupby("sensor").cloud.median().round(1).to_dict())
