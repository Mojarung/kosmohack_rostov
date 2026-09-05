"""Расширение train реальными полями: контуры из OpenStreetMap + ряды с наших источников.

ТЗ прямо рекомендует «использовать контуры полей из OSM и собирать дополнительные ряды
Sentinel-2, Landsat, MODIS + ERA5». Скрипт берёт крупнейшие поля в нескольких районах
Ростовской области, собирает по ним ряды и складывает в data/extra/osm_fields.csv в формате
датасета соревнования. Дальше scripts/experiment_extra.py проверяет, помогают ли эти примеры
на честной валидации по данным организаторов.
"""

import argparse
import time

import _bootstrap  # noqa: F401
import pandas as pd

from ndvi.collect import collect_series, fetch_fields
from ndvi.paths import DATA_DIR

# районы Ростовской области с разной агроклиматикой: юг, центр, север, восток
DISTRICTS = {
    "Зерноград": (40.15, 46.75, 40.60, 46.98),
    "Сальск": (41.30, 46.35, 41.75, 46.60),
    "Миллерово": (40.20, 48.80, 40.65, 49.05),
    "Морозовск": (41.60, 48.20, 42.10, 48.45),
    "Азов": (39.20, 46.95, 39.60, 47.15),
    "Целина": (41.00, 46.45, 41.40, 46.65),
}

ap = argparse.ArgumentParser()
ap.add_argument("--per-district", type=int, default=5)
ap.add_argument("--start", default="2021-01-01")
ap.add_argument("--end", default="2025-11-01")
ap.add_argument("--min-ha", type=float, default=60.0)
args = ap.parse_args()

out_dir = DATA_DIR / "extra"
out_dir.mkdir(exist_ok=True)
frames, t0 = [], time.time()
for name, bbox in DISTRICTS.items():
    try:
        fields = fetch_fields(bbox, limit=args.per_district * 3, min_ha=args.min_ha)[: args.per_district]
    except Exception as exc:
        print(f"{name}: OSM недоступен ({exc})", flush=True)
        continue
    print(f"{name}: {len(fields)} полей", flush=True)
    for f in fields:
        t = time.time()
        pid = f"OSM-{name}-{f['id'].split('-')[-1]}"
        res = collect_series(f["geometry"], args.start, args.end, polygon_id=pid,
                             crop_type=f.get("crop") or "не указана")
        ok = ", ".join(res.ok_sources)
        if res.series.empty:
            print(f"  {pid}: пусто ({res.sources})", flush=True)
            continue
        s = res.series.copy()
        s["district"] = name
        s["area_ha"] = f["area_ha"]
        frames.append(s)
        print(f"  {pid}: {int(s.primary_ndvi.notna().sum())} набл., {f['area_ha']:.0f} га, "
              f"{ok}, {time.time() - t:.0f} с", flush=True)
        pd.concat(frames, ignore_index=True).to_csv(out_dir / "osm_fields.csv", index=False)

df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
print(f"\nитого: {df.anon_polygon_id.nunique() if len(df) else 0} полей, "
      f"{int(df.primary_ndvi.notna().sum()) if len(df) else 0} наблюдений, {time.time() - t0:.0f} с")
