"""Оценка положения полигонов датасета по погодному отпечатку.

Результат — artifacts/polygon_locations.json, который использует карта веб-сервиса.
Точность порядка 40 км: это зона, а не контур поля (подробности в src/ndvi/geolocate.py).
"""

import time

import _bootstrap  # noqa: F401
import pandas as pd

from ndvi.data import load_test, load_train
from ndvi.geolocate import DEFAULT_WINDOW, build_grid, fetch_grid, locate_polygons, save

t0 = time.time()
tr, te = load_train(), load_test()
train_ids = set(tr.anon_polygon_id)
test_ids = set(te.anon_polygon_id)
roles = {pid: ("both" if pid in train_ids else "predict") for pid in test_ids}
for pid in train_ids - test_ids:
    roles[pid] = "train"
print(f"полигонов: обучение+предсказание {sum(v == 'both' for v in roles.values())}, "
      f"только предсказание {sum(v == 'predict' for v in roles.values())}, "
      f"только обучение {sum(v == 'train' for v in roles.values())}")

grid = fetch_grid(build_grid(), *DEFAULT_WINDOW)
print(f"узлов сетки с данными: {len(grid)}")

# сезон 2025 есть у всех тестовых полигонов, поэтому ищем по нему
loc = locate_polygons(te, grid, DEFAULT_WINDOW)
payload = save(loc, roles)
print(f"\nлокализовано {len(loc)} из {len(test_ids)} полигонов "
      f"(у остальных ERA5 в данных отсутствует)")
print(f"медианная корреляция осадков: {loc.corr_p.median():.3f}, "
      f"медианный RMSE температуры: {loc.rmse_t.median():.2f} °C")
print(f"разброс: широта {loc.lat.min():.1f}–{loc.lat.max():.1f}, "
      f"долгота {loc.lon.min():.1f}–{loc.lon.max():.1f}")
print(f"готово за {time.time() - t0:.0f} с")
