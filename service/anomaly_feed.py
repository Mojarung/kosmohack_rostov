"""Поля кейса и сохранённые территории: отклонения без повторного анализа."""
import json
import math
from typing import Literal
from fastapi import APIRouter, Query
from service import polygons

EPISODE_KEYS = ("pid", "year", "start", "end", "days", "n_obs", "min_z", "severity", "cause",
                "confidence", "text", "reasons", "norm_source")


def finite(value):
    return isinstance(value, (int, float)) and math.isfinite(value)


def personal_fields():
    """Тот же набор, что в «Новой территории», включая прежний формат хранения."""
    out, seen = [], set()
    for item in polygons.list_saved():
        report = polygons.load(item["uid"])
        if report is None or report["pid"] in seen:
            continue
        seen.add(report["pid"])
        out.append({"key": f"mine:{report['pid']}", "pid": report["pid"], "uid": item["uid"],
                    "name": item["name"], "source": "mine", "years": item["years"],
                    "episodes": report.get("episodes", []),
                    "assessed": [int(y) for y, season in report.get("years", {}).items()
                                 if any(finite(p.get("value")) for p in season.get("z", []))]})
    return out


def case_fields(store):
    """Читает готовые сводки сезонов вместо пересчёта кривых каждого поля."""
    by_pid, assessed = {}, {}
    for episode in json.loads(store.episodes.to_json(orient="records")):
        by_pid.setdefault(episode["pid"], []).append(episode)
    for season in json.loads(store.seasons.to_json(orient="records")):
        if finite(season.get("mean_z")):
            assessed.setdefault(season["pid"], []).append(int(season["year"]))
    return [{"key": f"case:{pid}", "pid": pid, "uid": None, "name": pid, "source": "case",
             "years": sorted(map(int, rows["year"].unique())), "episodes": by_pid.get(pid, []),
             "assessed": assessed.get(pid, [])} for pid, rows in store.obs.groupby("pid")]


def season_field(field, year):
    """Отсутствие оценки не означает норму; ошибки данных отделены от угнетения."""
    episodes = [{k: e.get(k) for k in EPISODE_KEYS} for e in field["episodes"] if e.get("year") == year]
    episodes.sort(key=lambda e: (0 if e.get("severity") == "критическая" else 1,
                                 e["min_z"] if finite(e.get("min_z")) else 0))
    real = [e for e in episodes if e.get("cause") not in {"data_suspect", "crop_rotation"}]
    if real:
        level = "high" if any(e.get("severity") == "критическая" for e in real) else "medium"
    elif episodes:
        level = "data" if any(e.get("cause") == "data_suspect" for e in episodes) else "context"
    else:
        level = "clear" if year in field["assessed"] else "unknown"
    return {k: field[k] for k in ("key", "pid", "uid", "name", "source", "years")} | {
        "level": level, "has_season": year in field["years"], "episodes": episodes}


def build_feed(get_store, source="mine", year=None):
    fields = personal_fields() if source in {"mine", "all"} else []
    if source in {"case", "all"}:
        fields.extend(case_fields(get_store()))
    years = sorted({y for field in fields for y in field["years"]}, reverse=True)
    selected = year if year is not None else (years[0] if years else None)
    rows = [season_field(field, selected) for field in fields]
    order = {"high": 0, "medium": 1, "data": 2, "context": 3, "unknown": 4, "clear": 5}
    rows.sort(key=lambda field: (order[field["level"]], field["name"].casefold()))
    return {"years": years, "year": selected, "fields": rows}


def router(get_store):
    routes = APIRouter()

    @routes.get("/api/anomaly-fields")
    def anomaly_fields(source: Literal["mine", "case", "all"] = "mine",
                       year: int | None = Query(default=None, ge=1980, le=2200)):
        return build_feed(get_store, source, year)

    return routes
