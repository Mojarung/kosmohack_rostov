"""Проверка адаптивности: тот же пайплайн на полях вне Ростовской области.

Ни модель, ни детектор аномалий не знают, где находится поле: норма считается из
собственной истории полигона, смещения сенсоров оцениваются по его же наблюдениям.
Скрипт прогоняет полный цикл на нескольких удалённых регионах и печатает сводку.
"""

import time

import _bootstrap  # noqa: F401

from ndvi.collect import collect_series, geom_area_ha
from ndvi.pipeline import Artifacts
from ndvi.service.analysis import analyze_series


def square(lat, lon, d=0.012):
    return {"type": "Polygon", "coordinates": [[[lon - d, lat - d], [lon + d, lat - d],
                                                [lon + d, lat + d], [lon - d, lat + d],
                                                [lon - d, lat - d]]]}


REGIONS = [
    ("Ростовская область (базовый регион)", square(47.45, 40.31)),
    ("Краснодарский край", square(45.60, 39.30)),
    ("Алтайский край, 3000 км восточнее", square(52.80, 82.30)),
    ("Канзас, США — другое полушарие агротехники", square(38.60, -98.30)),
]

START, END = "2021-01-01", "2025-11-01"
art = Artifacts.load()

for name, geom in REGIONS:
    t0 = time.time()
    res = collect_series(geom, START, END)
    if res.series.empty:
        print(f"{name}: данных нет ({res.sources})")
        continue
    out = analyze_series(res.series, art, START, END, polygon_id=name)
    if not out["ok"]:
        print(f"{name}: {out['error']}")
        continue
    stress = [e for e in out["episodes"] if e["kind"] == "стресс"]
    ok = [k for k, v in res.sources.items() if v.get("ok")]
    print(f"{name}: площадь {geom_area_ha(geom):.0f} га, наблюдений {out['n_observations']}, "
          f"восстановлено {out['n_filled']}, эпизодов угнетения {len(stress)}, "
          f"источники {', '.join(ok)}, {time.time() - t0:.0f} с")
    if stress:
        print("   " + stress[0]["explanation"][:190])
