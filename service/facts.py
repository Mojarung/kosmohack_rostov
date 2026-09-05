"""Факты о поле в компактном виде: общий слой для агента, MCP-инструментов и отчёта.

Здесь нет ни сети, ни языковой модели — только выжимка из уже посчитанных данных.
Всё, что отсюда возвращается, проверяемо: числа берутся из рядов NDVI, нормы, погоды и эпизодов,
ничего не додумывается. Поэтому один и тот же набор фактов можно и показать пользователю,
и отдать модели как контекст, и вставить в отчёт.
"""

from __future__ import annotations

from typing import Any

# Сколько эпизодов и наблюдений отдаём по умолчанию: контекст модели не резиновый,
# а для ответа на вопрос обычно хватает самых тяжёлых эпизодов и краёв ряда.
MAX_EPISODES = 12
MAX_POINTS = 40

SEVERITY_ORDER = {"критическая": 0, "умеренная": 1}

# Причины детектор хранит кодами; людям и модели нужны слова. Те же подписи, что в интерфейсе.
CAUSE_LABEL = {
    "weather_drought": "погодный стресс",
    "unsown_or_changed": "не засеяно или другая культура",
    "early_decline": "ранний спад",
    "weak_season": "ослабленный сезон",
    "late_start": "поздний старт",
    "crop_rotation": "севооборот, не угнетение",
    "data_suspect": "вероятная ошибка данных",
}


def _round(value: Any, digits: int = 3) -> Any:
    """Округление, устойчивое к None и нечисловым значениям."""
    try:
        return round(float(value), digits)
    except (TypeError, ValueError):
        return value


def _by_year(mapping: dict | None, year: int) -> dict | None:
    """Сезоны приходят и с целыми ключами (Store), и со строковыми (JSON из field_store)."""
    if not mapping:
        return None
    return mapping.get(year) or mapping.get(str(year))


def _thin(points: list[dict], limit: int = MAX_POINTS) -> list[dict]:
    """Прореживание ряда до limit точек с сохранением первой и последней."""
    if len(points) <= limit:
        return points
    step = len(points) / limit
    picked = [points[int(i * step)] for i in range(limit)]
    if picked[-1] is not points[-1]:
        picked[-1] = points[-1]
    return picked


def episode_facts(episode: dict) -> dict:
    """Один эпизод угнетения: период, глубина, фаза, причина и её обоснование."""
    return {
        "год": episode.get("year"),
        "период": f"{episode.get('start')} — {episode.get('end')}",
        "дней": episode.get("days"),
        "наблюдений": episode.get("n_obs"),
        "тяжесть": episode.get("severity"),
        "минимум_отклонения_сигм": _round(episode.get("min_z"), 2),
        "среднее_отклонения_сигм": _round(episode.get("mean_z"), 2),
        "худший_день": episode.get("worst_date"),
        "ndvi_в_худший_день": _round(episode.get("ndvi_at_worst")),
        "норма_в_худший_день": _round(episode.get("norm_at_worst")),
        "фаза_сезона": episode.get("phase"),
        "причина": CAUSE_LABEL.get(episode.get("cause"), episode.get("cause")),
        "код_причины": episode.get("cause"),
        "уверенность": _round(episode.get("confidence"), 2),
        "аргументы": episode.get("reasons"),
        "соседние_поля_ниже_нормы_доля": _round(episode.get("region_share_depressed"), 2),
        "отклонение_региона_сигм": _round(episode.get("region_z"), 2),
        "источник_нормы": episode.get("norm_source"),
    }


def season_facts(year: int, season: dict, weather: dict | None = None) -> dict:
    """Сводка одного сезона: сколько наблюдений, как шла кривая относительно нормы, какая погода."""
    curve = season.get("curve") or []
    norm = season.get("norm_mean") or []
    z = season.get("z") or []
    peak = max(curve, key=lambda p: p["value"]) if curve else None
    worst = min(z, key=lambda p: p["value"]) if z else None
    facts: dict[str, Any] = {
        "год": year,
        "наблюдений": len(season.get("observations") or []),
        "восстановлено_точек": len(season.get("restored") or []),
        "источник_нормы": season.get("norm_source"),
        "пик_ndvi": _round(peak["value"]) if peak else None,
        "дата_пика": peak["date"] if peak else None,
        "худшее_отклонение_сигм": _round(worst["value"], 2) if worst else None,
        "дата_худшего_отклонения": worst["date"] if worst else None,
    }
    if norm:
        peak_norm = max(norm, key=lambda p: p["value"])
        facts["пик_нормы"] = _round(peak_norm["value"])
        facts["дата_пика_нормы"] = peak_norm["date"]
    if weather and weather.get("date"):
        precip = [v for v in (weather.get("precip") or []) if isinstance(v, (int, float))]
        temp = [v for v in (weather.get("temp") or []) if isinstance(v, (int, float))]
        facts["осадки_за_сезон_мм"] = _round(sum(precip), 1) if precip else None
        facts["средняя_температура_c"] = _round(sum(temp) / len(temp), 1) if temp else None
        facts["дней_погоды"] = len(weather["date"])
    return facts


def field_facts(detail: dict, year: int | None = None) -> dict:
    """Полный набор фактов о поле: паспорт, сезоны, эпизоды и ряд выбранного года.

    year задаёт, по какому сезону отдавать подробный ряд; без него берётся последний.
    """
    years = sorted(int(y) for y in (detail.get("years") or {}))
    chosen = year if year in years else (years[-1] if years else None)
    episodes = list(detail.get("episodes") or [])
    episodes.sort(key=lambda e: (SEVERITY_ORDER.get(e.get("severity"), 9), e.get("min_z", 0)))

    facts: dict[str, Any] = {
        "поле": detail.get("name") or detail.get("pid"),
        "идентификатор": detail.get("pid"),
        "культура": detail.get("crop"),
        "происхождение": detail.get("kind"),
        "сезоны": years,
        "всего_эпизодов": len(detail.get("episodes") or []),
        "критических_эпизодов": sum(1 for e in (detail.get("episodes") or []) if e.get("severity") == "критическая"),
        "эпизоды": [episode_facts(e) for e in episodes[:MAX_EPISODES]],
        "сводка_по_сезонам": [
            season_facts(y, _by_year(detail.get("years"), y) or {}, _by_year(detail.get("weather"), y))
            for y in years
        ],
    }
    if chosen is not None:
        season = _by_year(detail.get("years"), chosen) or {}
        facts["выбранный_сезон"] = chosen
        facts["ряд_выбранного_сезона"] = {
            "восстановленная_кривая": _thin([
                {"дата": p["date"], "ndvi": _round(p["value"])} for p in (season.get("curve") or [])
            ]),
            "норма": _thin([
                {"дата": p["date"], "ndvi": _round(p["value"])} for p in (season.get("norm_mean") or [])
            ]),
            "наблюдения": _thin([
                {"дата": o["date"], "ndvi": _round(o["value"]), "спутник": o.get("sensor")}
                for o in (season.get("observations") or []) if not o.get("artifact")
            ]),
        }
    if detail.get("collected"):
        facts["собрано"] = detail["collected"]
    if detail.get("weather_source"):
        facts["источник_погоды"] = detail["weather_source"]
    return facts


def method_facts(meta: dict) -> dict:
    """Как получены числа: метрики решения и источники данных. Нужны, чтобы ответ был проверяемым."""
    task1 = meta.get("task1", {})
    task2 = meta.get("task2", {})
    return {
        "восстановление_пропусков": {
            "rmse": _round(task1.get("rmse_val"), 4),
            "rmse_без_калибровки": _round(task1.get("rmse_model_only"), 4),
            "gap_score": _round(task1.get("gap_score"), 2),
            "rmse_baseline_среднее_соседей": _round(task1.get("baseline_rmse"), 3),
            "контрольных_точек": task1.get("n_gaps"),
            "модель": task1.get("models"),
            "оговорка": "это отложенная выборка, а не приватный лидерборд",
        },
        "поиск_эпизодов": {
            "полей": task2.get("n_polygons"),
            "сезонов": task2.get("n_seasons"),
            "эпизодов": task2.get("n_episodes"),
            "доля_сезонов_с_эпизодом": _round(task2.get("seasons_with_episode"), 2),
            "по_причинам": task2.get("by_cause"),
        },
        "источники": [f"{s.get('name')}: {s.get('detail')}" for s in (meta.get("sources") or [])],
    }
