"""Веб-сервис мониторинга вегетации: API + интерфейс.

Запуск: uv run uvicorn service.app:app --host 127.0.0.1 --port 8000
Интерфейс: http://127.0.0.1:8000/ — собранное приложение из web/dist (React + MUI X Charts + AntV L7).
Если сборки нет (не запускали npm run build), отдаётся простой резервный интерфейс service/static/index.html;
он же всегда доступен по адресу /legacy. Полигоны кейса анонимны и показываются списком, новая территория
рисуется на карте, данные для неё собираются автоматически (см. service.collect).
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from service import polygons as user_polygons
from service.data import Store
from service.meta import build_meta

STATIC_DIR = Path(__file__).resolve().parent / "static"
WEB_DIST = Path(__file__).resolve().parents[1] / "web" / "dist"
WEB_INDEX = WEB_DIST / "index.html"

app = FastAPI(title="NDVI-мониторинг полей", version="1.0")

# Режим разработки: интерфейс поднимается отдельно на 5173 и ходит в этот API.
app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=r"http://(localhost|127\.0\.0\.1):\d+",
    allow_methods=["*"],
    allow_headers=["*"],
)

_store: Store | None = None


def store() -> Store:
    global _store
    if _store is None:
        _store = Store()
    return _store


class AnalyzeRequest(BaseModel):
    """Произвольный полигон пользователя (GeoJSON Polygon в WGS84) и период анализа."""
    geometry: dict = Field(description="GeoJSON Polygon")
    name: str = "новое поле"
    start_year: int = 2019
    end_year: int = 2025


@app.get("/")
def index() -> FileResponse:
    """Главная страница: собранное приложение, иначе резервный интерфейс."""
    return FileResponse(WEB_INDEX if WEB_INDEX.exists() else STATIC_DIR / "index.html")


@app.get("/legacy")
def legacy_index() -> FileResponse:
    """Резервный интерфейс на одном HTML-файле: работает без сборки фронтенда."""
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/polygons")
def polygons() -> list[dict]:
    """Список полигонов кейса со сводкой по эпизодам."""
    return store().polygons()


@app.get("/api/polygon/{pid}")
def polygon(pid: str) -> dict:
    """Ряды, кривые, нормы, Z, эпизоды и погода одного полигона."""
    if pid not in set(store().obs["pid"]):
        raise HTTPException(404, f"полигон {pid} не найден")
    return store().polygon(pid)


@app.get("/api/episodes")
def episodes(year: int | None = None, cause: str | None = None, severity: str | None = None) -> list[dict]:
    """Все эпизоды с фильтрами по году, причине и тяжести."""
    df = store().episodes
    if len(df) == 0:
        return []
    if year is not None:
        df = df[df["year"] == year]
    if cause:
        df = df[df["cause"] == cause]
    if severity:
        df = df[df["severity"] == severity]
    cols = ["pid", "year", "start", "end", "days", "min_z", "severity", "cause", "confidence", "text"]
    return df[cols].to_dict(orient="records")


@app.get("/api/summary")
def summary() -> dict:
    """Сводка: эпизоды по причинам, годам и тяжести."""
    df = store().episodes
    if len(df) == 0:
        return {}
    return {"by_cause": df["cause"].value_counts().to_dict(), "by_year": df.groupby("year").size().to_dict(),
            "by_severity": df["severity"].value_counts().to_dict(), "n_polygons": int(df["pid"].nunique())}


@app.get("/api/meta")
def meta() -> dict:
    """Сводка о решении: метрики обеих задач, состав данных, источники (для экрана «Как это работает»)."""
    return build_meta(n_gaps=int(len(store().gaps)))


@app.get("/api/fields")
def fields(bbox: str) -> list[dict]:
    """Готовые контуры полей OpenStreetMap в рамке карты: bbox = юг,запад,север,восток."""
    try:
        from service.collect import osm_fields
        s, w, n, e = (float(v) for v in bbox.split(","))
        if (n - s) * (e - w) > 0.25:
            raise HTTPException(400, "приблизьте карту: область слишком велика для запроса контуров")
        return osm_fields((s, w, n, e))
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(502, f"Overpass API недоступен: {exc}") from exc


_pool: ProcessPoolExecutor | None = None


def _run_collect(geometry: dict, name: str, start_year: int, end_year: int) -> dict:
    from service.collect import analyze_geometry
    return analyze_geometry(geometry, name, start_year, end_year)


@app.post("/api/analyze")
def analyze(req: AnalyzeRequest) -> JSONResponse:
    """Новая территория: сбор данных из открытых источников и анализ тем же пайплайном.

    Сбор идёт в отдельном процессе: GDAL/rasterio и dask внутри пула потоков сервера подвисают.
    """
    global _pool
    try:
        import service.collect  # noqa: F401  проверка, что группа geo установлена
    except ImportError as exc:
        raise HTTPException(501, f"сбор данных недоступен: {exc}") from exc
    if _pool is None:
        _pool = ProcessPoolExecutor(max_workers=1)
    try:
        result = _pool.submit(_run_collect, req.geometry, req.name, req.start_year, req.end_year).result(timeout=1200)
    except Exception as exc:        # ошибки внешних API отдаём пользователю понятным текстом
        raise HTTPException(502, f"не удалось собрать данные: {type(exc).__name__}: {exc}") from exc
    entry = user_polygons.save(result, req.name)      # поле попадает в набор пользователя
    return JSONResponse(result | {"uid": entry["uid"], "name": entry["name"]})


@app.get("/api/user-polygons")
def list_user_polygons() -> list[dict]:
    """Набор полигонов пользователя: имя, геометрия, сводка по эпизодам и состояние последнего сезона."""
    return user_polygons.list_saved()


@app.get("/api/user-polygons/{uid}")
def get_user_polygon(uid: str) -> dict:
    """Сохранённый результат анализа (без повторного сбора данных)."""
    result = user_polygons.load(uid)
    if result is None:
        raise HTTPException(404, "полигон не найден")
    return result


@app.delete("/api/user-polygons/{uid}")
def delete_user_polygon(uid: str) -> dict:
    """Удаление полигона из набора."""
    if not user_polygons.delete(uid):
        raise HTTPException(404, "полигон не найден")
    return {"deleted": uid}


app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

if WEB_INDEX.exists():
    # Файлы сборки (assets/*.js, иконки) и клиентская маршрутизация: любой неизвестный путь,
    # кроме /api и /static, отдаёт index.html, чтобы работали прямые ссылки вида /field/AOI-0005.
    app.mount("/assets", StaticFiles(directory=str(WEB_DIST / "assets")), name="assets")

    @app.get("/{path:path}", include_in_schema=False)
    def spa(path: str) -> FileResponse:
        # Отдаём только файлы внутри web/dist: путь приходит от клиента, поэтому проверяем,
        # что после разрешения он не вышел за пределы папки сборки (защита от «../»).
        candidate = (WEB_DIST / path).resolve()
        if path and candidate.is_file() and candidate.is_relative_to(WEB_DIST.resolve()):
            return FileResponse(candidate)
        return FileResponse(WEB_INDEX)
