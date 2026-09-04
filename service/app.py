"""Веб-сервис мониторинга вегетации: API + статический интерфейс.

Запуск: uv run uvicorn service.app:app --host 127.0.0.1 --port 8000
Интерфейс: http://127.0.0.1:8000/  (полигоны из данных кейса — списком, координат у них нет;
новая территория — рисуется на карте, данные собираются автоматически, см. service.collect).
"""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from service.data import Store

STATIC_DIR = Path(__file__).resolve().parent / "static"

app = FastAPI(title="NDVI-мониторинг полей", version="0.1")
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


@app.post("/api/analyze")
def analyze(req: AnalyzeRequest) -> JSONResponse:
    """Новая территория: сбор данных из открытых источников и анализ тем же пайплайном."""
    try:
        from service.collect import analyze_geometry
    except ImportError as exc:      # группа geo не установлена
        raise HTTPException(501, f"сбор данных недоступен: {exc}") from exc
    try:
        return JSONResponse(analyze_geometry(req.geometry, req.name, req.start_year, req.end_year))
    except Exception as exc:        # ошибки внешних API отдаём пользователю понятным текстом
        raise HTTPException(502, f"не удалось собрать данные: {exc}") from exc


app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
