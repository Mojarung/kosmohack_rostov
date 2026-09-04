"""FastAPI-приложение сервиса мониторинга NDVI.

Экраны: карта с поиском региона, готовыми контурами полей из OSM и рисованием своего
полигона; результат анализа с восстановленным рядом, коридором нормы, лентой аномалий и
погодой; демо-режим на полигонах датасета (координат у них нет, карта недоступна).

Запуск: ``.venv/bin/python -m uvicorn ndvi.service.app:app --reload --port 8000``
"""

from __future__ import annotations

import json
import logging
import uuid
from datetime import date

import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from ndvi import collect
from ndvi.data import load_test, load_train
from ndvi.paths import ARTIFACTS_DIR, ROOT
from ndvi.pipeline import Artifacts
from ndvi.service import store
from ndvi.service.analysis import analyze_series

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
log = logging.getLogger("ndvi.service")

STATIC_DIR = ROOT / "web" / "static"
app = FastAPI(title="NDVI-мониторинг сельхозполей", version="1.0")

_state: dict = {}


def artifacts() -> Artifacts:
    """Модель загружается один раз при первом обращении."""
    if "artifacts" not in _state:
        _state["artifacts"] = Artifacts.load()
    return _state["artifacts"]


def dataset() -> pd.DataFrame:
    """Данные соревнования для демо-режима: train + открытая часть test."""
    if "dataset" not in _state:
        tr = load_train()
        te = load_test()
        te = te[~te.get("is_synthetic_gap", pd.Series(False, index=te.index)).fillna(False)]
        _state["dataset"] = pd.concat([tr, te], ignore_index=True).sort_values(
            ["anon_polygon_id", "date"]).reset_index(drop=True)
    return _state["dataset"]


# --------------------------------------------------------------------------- #
# Схемы запросов
# --------------------------------------------------------------------------- #

class AnalyzeRequest(BaseModel):
    geometry: dict = Field(..., description="полигон в формате GeoJSON")
    start: str = Field(default_factory=lambda: f"{date.today().year - 4}-01-01")
    end: str = Field(default_factory=lambda: date.today().isoformat())
    crop_type: str = "не указана"
    polygon_id: str | None = None
    name: str | None = None
    sources: list[str] = ["s2", "landsat", "modis", "weather"]
    use_cache: bool = True


class PolygonRequest(BaseModel):
    name: str
    geometry: dict
    crop_type: str = "не указана"
    source: str = "draw"
    id: str | None = None
    role: str = "predict"      # train | predict | both


class RoleRequest(BaseModel):
    role: str


# --------------------------------------------------------------------------- #
# Служебное
# --------------------------------------------------------------------------- #

@app.get("/api/health")
def health():
    """Готовность сервиса и доступность внешних источников (быстрая проверка)."""
    checks = {}
    try:
        collect.search_region("Ростов-на-Дону", limit=1)
        checks["Nominatim"] = True
    except Exception as exc:
        checks["Nominatim"] = f"{type(exc).__name__}"
    try:
        collect.fetch_weather({"type": "Point", "coordinates": [39.7, 47.2]},
                              "2024-06-01", "2024-06-03")
        checks["Open-Meteo (ERA5)"] = True
    except Exception as exc:
        checks["Open-Meteo (ERA5)"] = f"{type(exc).__name__}"
    model_ok = True
    try:
        artifacts()
    except Exception as exc:
        model_ok = f"{type(exc).__name__}: модель не обучена, запустите scripts/train.py"
    return {"ok": True, "model": model_ok, "sources": checks}


# --------------------------------------------------------------------------- #
# Карта: регионы и поля
# --------------------------------------------------------------------------- #

@app.get("/api/regions")
def regions(q: str = Query(..., min_length=2), limit: int = 5):
    """Поиск региона по названию."""
    try:
        return {"ok": True, "items": collect.search_region(q, limit)}
    except Exception as exc:
        raise HTTPException(502, f"поиск региона недоступен: {exc}")


@app.get("/api/fields")
def fields(minx: float, miny: float, maxx: float, maxy: float, limit: int = 60):
    """Готовые контуры сельхозполей из OpenStreetMap в видимой части карты."""
    try:
        items = collect.fetch_fields((minx, miny, maxx, maxy), limit=limit)
        return {"ok": True, "items": items, "n": len(items)}
    except Exception as exc:
        # карта не должна падать из-за перегруженного Overpass
        log.warning("Overpass недоступен: %s", exc)
        return {"ok": False, "items": [], "error": str(exc)}


# --------------------------------------------------------------------------- #
# Сохранённые полигоны
# --------------------------------------------------------------------------- #

@app.get("/api/polygons")
def polygons():
    return {"ok": True, "items": store.list_polygons()}


@app.post("/api/polygons")
def add_polygon(req: PolygonRequest):
    pid = req.id or f"AOI-{uuid.uuid4().hex[:8]}"
    area = collect.geom_area_ha(req.geometry)
    return {"ok": True, "item": store.save_polygon(pid, req.name, req.geometry,
                                                   req.crop_type, req.source, area, req.role)}


@app.patch("/api/polygons/{pid}/role")
def change_role(pid: str, req: RoleRequest):
    """Переводит полигон между обучающим и предсказываемым набором."""
    try:
        item = store.set_role(pid, req.role)
    except ValueError as exc:
        raise HTTPException(400, str(exc))
    if item is None:
        raise HTTPException(404, f"полигон {pid} не найден")
    return {"ok": True, "item": item}


@app.delete("/api/polygons/{pid}")
def remove_polygon(pid: str):
    return {"ok": store.delete_polygon(pid)}


@app.get("/api/dataset/locations")
def dataset_locations():
    """Оценённое по погоде положение полигонов датасета с ролью каждого.

    Координат в данных нет; положение восстановлено сопоставлением суточных рядов ERA5
    с архивом Open-Meteo и имеет точность порядка 40 км — это зона, а не контур поля.
    """
    f = ARTIFACTS_DIR / "polygon_locations.json"
    if not f.exists():
        return {"ok": False, "items": [],
                "error": "положение не рассчитано, запустите scripts/locate_polygons.py"}
    return {"ok": True, **json.loads(f.read_text())}


# --------------------------------------------------------------------------- #
# Анализ
# --------------------------------------------------------------------------- #

@app.post("/api/analyze")
def analyze(req: AnalyzeRequest):
    """Полный цикл: автосбор данных по полигону -> восстановление пропусков -> аномалии."""
    pid = req.polygon_id or f"AOI-{uuid.uuid4().hex[:8]}"
    result = collect.collect_series(req.geometry, req.start, req.end, polygon_id=pid,
                                    crop_type=req.crop_type, use=tuple(req.sources),
                                    use_cache=req.use_cache)
    if result.series.empty:
        return {"ok": False, "error": "ни один источник не вернул данные",
                "sources": result.sources}
    out = analyze_series(result.series, artifacts(), req.start, req.end,
                         polygon_id=pid, crop_type=req.crop_type)
    out["sources"] = result.sources
    out["area_ha"] = round(collect.geom_area_ha(req.geometry), 1)
    out["name"] = req.name or pid
    _state[f"last:{pid}"] = out
    return out


# --------------------------------------------------------------------------- #
# Демо-режим на данных соревнования
# --------------------------------------------------------------------------- #

@app.get("/api/demo/polygons")
def demo_polygons():
    """Список полигонов датасета. Координат в данных нет, поэтому карта здесь не работает."""
    d = dataset()
    train_ids = set(load_train().anon_polygon_id)
    test_ids = set(load_test().anon_polygon_id)
    g = d[d.primary_ndvi.notna()].groupby("anon_polygon_id")
    items = []
    for pid, sub in g:
        role = ("both" if pid in train_ids and pid in test_ids
                else "train" if pid in train_ids else "predict")
        items.append({"id": pid, "crop_type": str(sub.crop_type.iloc[0]),
                      "n_observations": int(len(sub)),
                      "years": [int(sub.year.min()), int(sub.year.max())],
                      "role": role, "role_ru": store.ROLE_RU[role]})
    return {"ok": True, "items": sorted(items, key=lambda x: x["id"]),
            "note": ("координат полей в датасете нет; их примерное положение оценено "
                     "по погодному отпечатку и показано на карте зонами")}


@app.get("/api/demo/{pid}")
def demo_polygon(pid: str, start: str | None = None, end: str | None = None):
    """Анализ полигона датасета тем же пайплайном, что и произвольного поля с карты."""
    d = dataset()
    sub = d[d.anon_polygon_id == pid]
    if sub.empty:
        raise HTTPException(404, f"полигон {pid} не найден")
    start = start or str(sub.date.min().date())
    end = end or str(sub.date.max().date())
    series = sub[["date", "s2_ndvi", "landsat_ndvi", "modis_ndvi",
                  "era5_temp_c", "era5_precip_mm", "primary_ndvi"]].copy()
    out = analyze_series(series, artifacts(), start, end, polygon_id=pid,
                         crop_type=str(sub.crop_type.iloc[0]))
    out["sources"] = {"датасет соревнования": {"ok": True, "n": int(len(sub))}}
    out["name"] = pid
    return out


# --------------------------------------------------------------------------- #
# Экспорт
# --------------------------------------------------------------------------- #

@app.get("/api/export/{pid}.csv", response_class=PlainTextResponse)
def export_csv(pid: str):
    """Выгрузка последнего результата анализа полигона в CSV."""
    out = _state.get(f"last:{pid}")
    if not out:
        raise HTTPException(404, "нет сохранённого результата: сначала выполните анализ")
    df = pd.DataFrame(out["series"])
    return PlainTextResponse(df.to_csv(index=False),
                             headers={"Content-Disposition": f'attachment; filename="{pid}.csv"'})


# --------------------------------------------------------------------------- #
# Фронтенд
# --------------------------------------------------------------------------- #

if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/")
def index():
    f = STATIC_DIR / "index.html"
    if not f.exists():
        return {"ok": True, "note": "фронтенд не собран, API доступен по /docs"}
    return FileResponse(f)
