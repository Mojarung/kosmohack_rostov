"""Веб-сервис мониторинга вегетации: API + интерфейс.

Запуск: uv run uvicorn service.app:app --host 127.0.0.1 --port 8000
либо uv run --locked --group service python -m service — тот же сервер плюс проверка зависимостей
автосбора при старте (готовность — GET /api/health).
Интерфейс: http://127.0.0.1:8000/ — собранное приложение из web/dist (React + MUI X Charts + AntV L7).
Если сборки нет (не запускали npm run build), отдаётся простой резервный интерфейс service/static/index.html;
он же всегда доступен по адресу /legacy. Полигоны кейса анонимны и показываются списком, новая территория
рисуется на карте, данные для неё собираются автоматически (см. service.collect).
"""

from __future__ import annotations

import datetime as dt
import logging
from concurrent.futures import ProcessPoolExecutor
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, model_validator

from service import field_store
from service import polygons as user_polygons
from service.data import Store
from service.meta import build_meta
from service.osm import osm_fields, parse_bbox
from service.progress import JOB_RE, read_progress
from service.reporting import legacy_weather_records
from service.runtime import check_runtime, plotly_bundle
from service.weather_metrics import PROFILES, weather_context

STATIC_DIR = Path(__file__).resolve().parent / "static"
WEB_DIST = Path(__file__).resolve().parents[1] / "web" / "dist"
WEB_INDEX = WEB_DIST / "index.html"

_store: Store | None = None
_pool: ProcessPoolExecutor | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """До открытия порта проверяет сборщик; при завершении освобождает пул процессов."""
    global _pool
    check_runtime()
    try:
        yield
    finally:
        if _pool is not None:
            _pool.shutdown(wait=False, cancel_futures=True)
            _pool = None


app = FastAPI(title="NDVI-мониторинг полей", version="1.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=r"http://(localhost|127\.0\.0\.1):\d+",
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
def health() -> dict:
    """Готовность API после успешной проверки зависимостей сборщика."""
    return {"status": "ok", "collection": True}


def store() -> Store:
    global _store
    if _store is None:
        _store = Store()
    return _store


class AnalyzeRequest(BaseModel):
    """Произвольный полигон пользователя (GeoJSON Polygon в WGS84) и период анализа."""
    geometry: dict = Field(description="GeoJSON Polygon")
    name: str = Field(default="новое поле", min_length=1, max_length=120)
    start_year: int = Field(default=2019, ge=1980)
    end_year: int = Field(default=2025, le=dt.datetime.now(dt.UTC).year)
    job: str | None = Field(default=None, pattern=JOB_RE.pattern,
                            description="идентификатор задания для опроса GET /api/analyze/progress/{job}")

    @model_validator(mode="after")
    def valid_area(self):
        """Отсекает неверную геометрию и период до дорогих внешних запросов."""
        from shapely.geometry import shape
        if not 3 <= self.end_year - self.start_year <= 20:
            raise ValueError("Для исторической нормы нужны от 4 до 21 сезона в правильном порядке")
        try:
            geom = shape(self.geometry)
            west, south, east, north = geom.bounds
            valid = (geom.geom_type == "Polygon" and geom.is_valid and 0 < geom.area <= 0.25
                     and -180 <= west < east <= 180 and -90 <= south < north <= 90)
        except Exception as exc:
            raise ValueError("Нужен корректный замкнутый контур поля GeoJSON Polygon") from exc
        if not valid:
            raise ValueError("Нужен корректный контур поля, площадь рамки не более 0.25 квадратных градусов")
        self.name = self.name.strip() or "новое поле"
        return self


class AgroRequest(BaseModel):
    """Параметры конкретного сезона и заметка аналитика."""
    year: int = Field(ge=1980, le=dt.datetime.now(dt.UTC).year)
    profile: str | None = None
    sowing_date: dt.date | None = None
    base: float = Field(default=10, ge=-5, le=20, allow_inf_nan=False)
    review_status: str = "Не разобрано"
    comment: str = Field(default="", max_length=3000)

    @model_validator(mode="after")
    def valid_settings(self):
        """Профиль ярового сезона не применяется к неизвестной культуре или дате другого года."""
        if self.profile is not None and self.profile not in PROFILES:
            raise ValueError("Выберите поддерживаемый профиль культуры")
        if self.sowing_date and self.sowing_date.year != self.year:
            raise ValueError("Дата начала должна относиться к выбранному году; озимым нужен отдельный профиль")
        if self.review_status not in {"Не разобрано", "На проверке", "Разобрано"}:
            raise ValueError("Неверный статус проверки")
        return self


@app.get("/")
def index() -> FileResponse:
    """Главная страница: собранное приложение, иначе резервный интерфейс."""
    return FileResponse(WEB_INDEX if WEB_INDEX.exists() else STATIC_DIR / "index.html",
                        headers={"Cache-Control": "no-store"})


@app.get("/legacy")
def legacy_index() -> FileResponse:
    """Резервный интерфейс на одном HTML-файле: работает без сборки фронтенда."""
    return FileResponse(STATIC_DIR / "index.html", headers={"Cache-Control": "no-store"})


@app.get("/vendor/plotly.min.js")
def plotly_javascript() -> FileResponse:
    """Отдаёт локальную библиотеку графиков из зафиксированной зависимости (нужна резервному интерфейсу)."""
    return FileResponse(plotly_bundle(), media_type="application/javascript",
                        headers={"Cache-Control": "public, max-age=86400"})


@app.get("/api/polygons")
def polygons() -> list[dict]:
    """Список полигонов кейса со сводкой по эпизодам."""
    return store().polygons()


@app.get("/api/polygon/{pid}")
def polygon(pid: str) -> dict:
    """Ряды, кривые, нормы, Z, эпизоды и погода одного полигона."""
    saved = field_store.read_report(pid) or user_polygons.find_report(pid)
    if saved is not None:
        return field_store.public_report(saved)
    if pid not in set(store().obs["pid"]):
        raise HTTPException(404, f"полигон {pid} не найден")
    from service.insights import with_insights
    return with_insights(store().polygon(pid))


@app.get("/api/saved-fields")
def saved_fields() -> list[dict]:
    """Сохранённые географические поля пользователя."""
    return field_store.list_fields()


@app.get("/api/agro-profiles")
def agro_profiles() -> dict:
    return PROFILES


def _agro_context(pid: str, year: int) -> dict:
    report = field_store.read_report(pid) or user_polygons.find_report(pid) or polygon(pid)
    if str(year) not in {str(y) for y in report["years"]}:
        raise HTTPException(404, "Сезон отсутствует в этом отчёте")
    records = report.get("_weather", legacy_weather_records(report))
    return weather_context(records, year, field_store.read_settings(pid, year), report.get("weather_source", ""))


@app.get("/api/polygon/{pid}/agro")
def agro(pid: str, year: int) -> dict:
    return _agro_context(pid, year)


@app.post("/api/polygon/{pid}/agro")
def configure_agro(pid: str, req: AgroRequest) -> dict:
    report = polygon(pid)
    if str(req.year) not in {str(y) for y in report["years"]}:
        raise HTTPException(404, "Сезон отсутствует в этом отчёте")
    field_store.save_settings(pid, req.year, req.model_dump(mode="json", exclude={"year"}))
    return _agro_context(pid, req.year)


@app.post("/api/polygon/{pid}/weather-refresh")
def refresh_weather(pid: str) -> dict:
    """Обновляет только погоду сохранённого поля, без повторной загрузки спутников."""
    from shapely.geometry import shape

    from service.weather_source import SOURCE, collect_weather, weather_records
    report = field_store.read_report(pid)
    if not report or not report.get("geometry"):
        raise HTTPException(400, "У анонимного примера нет координат для получения новой погоды")
    years = sorted(map(int, report["years"]))
    center = shape(report["geometry"]).centroid
    try:
        weather = collect_weather(center.y, center.x, range(years[0], years[-1] + 1))
    except Exception as exc:
        raise HTTPException(502, f"Не удалось обновить ERA5: {type(exc).__name__}") from exc
    report["_weather"] = weather_records(weather)
    report["weather_source"] = SOURCE
    report["weather_warnings"] = weather.attrs.get("warnings", [])
    field_store.save_report(report)
    return {"status": "ok", "days": len(weather), "warnings": report["weather_warnings"]}


@app.get("/api/polygon/{pid}/imagery")
def imagery(pid: str, year: int) -> dict:
    """Сохранённая карта пиксельных индексов Sentinel-2, если она уже собрана."""
    from service.imagery import read, ndmi_context
    manifest = read(pid, year)
    if manifest is None:
        return {"available": False, "year": year}
    return {"available": True, "manifest": manifest, "ndmi": ndmi_context(pid, year)}


@app.get("/api/polygon/{pid}/imagery/{year}/{date}/{index}.png")
def imagery_file(pid: str, year: int, date: str, index: str) -> FileResponse:
    from service.imagery import image_path, PALETTES
    if index not in PALETTES:
        raise HTTPException(404, "Неизвестный индекс")
    path = image_path(pid, year, date, index)
    if path is None:
        raise HTTPException(404, "Снимок не найден")
    return FileResponse(path, media_type="image/png", headers={"Cache-Control": "public, max-age=86400"})


@app.post("/api/polygon/{pid}/imagery")
def collect_imagery(pid: str, year: int) -> dict:
    """Собирает карту Sentinel-2 по запросу для поля с координатами."""
    global _pool
    report = field_store.read_report(pid)
    if not report or not report.get("geometry"):
        raise HTTPException(400, "У этого примера нет координат поля")
    if str(year) not in {str(y) for y in report["years"]}:
        raise HTTPException(400, "Выберите сезон из отчёта поля")
    from service.imagery import collect
    from service.imagery import read
    if (cached := read(pid, year)) is not None:
        return {"available": True, "manifest": cached}
    if _pool is None:
        _pool = ProcessPoolExecutor(max_workers=1)
    try:
        manifest = _pool.submit(collect, pid, report["geometry"], year).result(timeout=1200)
    except Exception as exc:
        raise HTTPException(502, f"Не удалось собрать карту: {type(exc).__name__}: {exc}") from exc
    return {"available": True, "manifest": manifest}


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
        bounds = parse_bbox(bbox)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    try:
        return osm_fields(bounds)
    except Exception as exc:
        raise HTTPException(502, f"Overpass API недоступен: {exc}") from exc


def _run_collect(geometry: dict, name: str, start_year: int, end_year: int, job: str | None = None) -> dict:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    from service.collect import analyze_geometry
    return analyze_geometry(geometry, name, start_year, end_year, job)


@app.get("/api/analyze/progress/{job}")
def analyze_progress(job: str) -> dict:
    """Ход сбора по заданию: сезоны и сцены по каждому источнику, этап, последние строки журнала."""
    state = read_progress(job)
    if state is None:
        raise HTTPException(404, "задание не найдено или ещё не начато")
    return state


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
        future = _pool.submit(_run_collect, req.geometry, req.name, req.start_year, req.end_year, req.job)
        result = future.result(timeout=1200)
    except Exception as exc:        # ошибки внешних API отдаём пользователю понятным текстом
        raise HTTPException(502, f"не удалось собрать данные: {type(exc).__name__}: {exc}") from exc
    field_store.save_report(result)                  # отчёт доступен по GET /api/polygon/{pid}
    entry = user_polygons.save(result, req.name)     # поле попадает в набор пользователя
    return JSONResponse(field_store.public_report(result) | {"uid": entry["uid"], "name": entry["name"]})


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
    return field_store.public_report(result)


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
        return FileResponse(WEB_INDEX, headers={"Cache-Control": "no-store"})
