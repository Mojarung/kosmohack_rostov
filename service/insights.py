"""Краткие выводы из наблюдений; правила детектора аномалий не меняются."""

from datetime import date, timedelta
from math import isfinite
from statistics import median


def _finite(value):
    return isinstance(value, (int, float)) and isfinite(value)


def deviation_trend(season: dict) -> dict:
    """Медиана наблюдаемого Z в двух окнах по 14 дней, минимум 2 даты в каждом.

    На одну дату берём один сенсор в приоритете S2 → Landsat → MODIS.
    Восстановления и сглаженная кривая не добавляют наблюдений.
    """
    result = {"status": "insufficient", "label": "Мало наблюдений", "available": False,
              "window_days": 14, "change_threshold": 0.5, "before_count": 0, "after_count": 0}
    points = {}
    priority = {"Sentinel-2": 0, "Landsat": 1, "MODIS": 2}
    for point in sorted(season.get("observations", []), key=lambda p: priority.get(p.get("sensor"), 3)):
        if not point.get("artifact") and _finite(point.get("harmonized")):
            points.setdefault(point["date"], point["harmonized"])
    if not points:
        return result
    end = date.fromisoformat(max(points))
    start, split = end - timedelta(days=27), end - timedelta(days=13)
    result.update(start=start.isoformat(), split=split.isoformat(), end=end.isoformat())
    norms = {p["date"]: p["value"] for p in season.get("norm_mean", [])}
    stds = {p["date"]: p["value"] for p in season.get("norm_std", [])}
    before, after = [], []
    for day, value in sorted(points.items()):
        if not start.isoformat() <= day <= end.isoformat():
            continue
        normal, std = norms.get(day), stds.get(day)
        if not _finite(normal) or not _finite(std) or std <= 0:
            continue
        (before if day < split.isoformat() else after).append((value - normal) / std)
    result.update(before_count=len(before), after_count=len(after))
    if min(len(before), len(after)) < 2:
        return result
    previous, recent = median(before), median(after)
    delta = recent - previous
    if previous >= -1 and recent >= -1:
        status, label = "within", "Без снижения к ориентиру"
    elif delta >= 0.5:
        status, label = ("recovered", "Вернулось к ориентиру") if recent >= -1 else ("easing", "Ослабевает")
    elif delta <= -0.5:
        status, label = "worsening", "Усиливается"
    else:
        status, label = "stable", "Без заметных изменений"
    return result | {"available": True, "status": status, "label": label,
                     "before_z": round(previous, 3), "after_z": round(recent, 3), "delta_z": round(delta, 3)}


def with_insights(report: dict) -> dict:
    return report | {"insights": {str(year): deviation_trend(season)
                                 for year, season in report.get("years", {}).items()}}
