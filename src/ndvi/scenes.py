"""Внешний признак: облачность спутниковых сцен над областью в день наблюдения.

Каталоги STAC (Earth Search для Sentinel-2, Planetary Computer для Landsat) отдают для каждой
сцены долю облаков ``eo:cloud_cover``. Если в день D все сцены над Ростовской областью были
на 80 % в облаках, скрытое наблюдение этого дня почти наверняка загрязнено — а это ровно те
выбросы, которые по соседям поля не угадать.

Два уровня: региональный (все сцены над областью в день D) и ближайший тайл к оценённому
положению полигона (``artifacts/polygon_locations.json``, точность ±60 км — для тайла 110 км
это приемлемо). Каталог строится скриптом ``scripts/fetch_scene_catalog.py``.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from ndvi.paths import ARTIFACTS_DIR

CATALOG = ARTIFACTS_DIR / "scene_catalog.csv"
LOCATIONS = ARTIFACTS_DIR / "polygon_locations.json"

EMPTY = {
    "sc_s2_cloud": np.nan, "sc_s2_cloud_min": np.nan, "sc_s2_n": 0.0, "sc_s2_near_cloud": np.nan,
    "sc_ls_cloud": np.nan, "sc_ls_cloud_min": np.nan, "sc_ls_n": 0.0, "sc_ls_near_cloud": np.nan,
    "sc_s2_cloud_pm1": np.nan, "sc_ls_cloud_pm1": np.nan,
}


def _haversine(lat1, lon1, lat2, lon2):
    p = np.pi / 180
    a = (np.sin((lat2 - lat1) * p / 2) ** 2
         + np.cos(lat1 * p) * np.cos(lat2 * p) * np.sin((lon2 - lon1) * p / 2) ** 2)
    return 2 * 6371 * np.arcsin(np.sqrt(a))


class SceneCatalog:
    """Облачность сцен по дням и тайлам плюс оценённые положения полигонов."""

    def __init__(self, catalog_path=CATALOG, locations_path=LOCATIONS):
        self.available = catalog_path.exists()
        self._day: dict = {}
        self._tile: dict = {}
        self._tiles: dict = {}
        self._loc: dict = {}
        if not self.available:
            return
        df = pd.read_csv(catalog_path, parse_dates=["date"])
        df = df[df.cloud.notna()]
        for (sensor, day), g in df.groupby(["sensor", "date"]):
            self._day[(sensor, day)] = (float(g.cloud.mean()), float(g.cloud.min()), float(len(g)))
        for (sensor, day, tile), g in df.groupby(["sensor", "date", "tile"]):
            self._tile[(sensor, day, str(tile))] = float(g.cloud.mean())
        for sensor, g in df.groupby("sensor"):
            t = g.groupby("tile")[["lat", "lon"]].mean().dropna()
            self._tiles[sensor] = (t.index.astype(str).to_numpy(), t.lat.to_numpy(), t.lon.to_numpy())
        if locations_path.exists():
            try:
                for it in json.loads(locations_path.read_text()).get("items", []):
                    self._loc[it["anon_polygon_id"]] = (it["lat"], it["lon"])
            except Exception:
                pass

    def _nearest_tiles(self, sensor: str, pid: str, k: int = 2):
        loc = self._loc.get(pid)
        tiles = self._tiles.get(sensor)
        if loc is None or tiles is None:
            return []
        d = _haversine(loc[0], loc[1], tiles[1], tiles[2])
        order = np.argsort(d)[:k]
        return [tiles[0][i] for i in order if d[i] < 150]

    def features(self, pid: str, date) -> dict:
        if not self.available:
            return dict(EMPTY)
        out = dict(EMPTY)
        day = pd.Timestamp(date).normalize()
        for sensor, key in (("s2", "sc_s2"), ("landsat", "sc_ls")):
            rec = self._day.get((sensor, day))
            if rec:
                out[f"{key}_cloud"], out[f"{key}_cloud_min"], out[f"{key}_n"] = rec
            # соседние дни: сцены S2 над востоком области могут датироваться следующим днём по UTC
            pm = [self._day.get((sensor, day + pd.Timedelta(days=dd))) for dd in (-1, 1)]
            pm = [r[0] for r in pm if r]
            if pm:
                out[f"{key}_cloud_pm1"] = float(np.mean(pm))
            near = [self._tile.get((sensor, day, t)) for t in self._nearest_tiles(sensor, pid)]
            near = [v for v in near if v is not None]
            if near:
                out[f"{key}_near_cloud"] = float(np.mean(near))
        return out
