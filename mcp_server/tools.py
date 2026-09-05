"""Инструменты MCP: чистые функции над данными кейса.

Вынесены отдельно от транспорта, поэтому проверяются обычными тестами без запуска сервера.
Каждая возвращает данные, а не текст: клиент сам решает, как их показать.
Тяжёлый сбор данных по новой территории сюда намеренно не вынесен — он идёт минуты
и требует сети; для него есть маршрут POST /api/analyze в сервисе.
"""

from __future__ import annotations

from typing import Any

from service.facts import CAUSE_LABEL, field_facts, method_facts

# Сколько строк отдаём в списках по умолчанию: клиент MCP кладёт ответ в контекст модели.
DEFAULT_LIMIT = 20
MAX_LIMIT = 200


def _store():
    """Хранилище данных кейса. Импорт внутри функции: он тянет pandas и читает CSV."""
    from service.data import Store
    return Store()


def _meta() -> dict:
    from service.meta import build_meta
    store = _store()
    return build_meta(n_gaps=int(len(store.gaps)))


def list_fields(crop: str | None = None, only_critical: bool = False, limit: int = DEFAULT_LIMIT) -> dict:
    """Поля кейса со сводкой: культура, сезоны, сколько найдено периодов снижения."""
    rows = _store().polygons()
    if crop:
        needle = crop.lower()
        rows = [r for r in rows if needle in str(r.get("crop", "")).lower()]
    if only_critical:
        rows = [r for r in rows if r.get("n_critical", 0) > 0]
    rows.sort(key=lambda r: (-r.get("n_critical", 0), -r.get("n_episodes", 0)))
    limited = rows[: max(1, min(limit, MAX_LIMIT))]
    return {"всего": len(rows), "показано": len(limited), "поля": limited}


def field_summary(pid: str, year: int | None = None) -> dict:
    """Полная выжимка по одному полю: паспорт, сезоны, периоды снижения и ряд выбранного года."""
    store = _store()
    if pid not in set(store.obs["pid"]):
        raise ValueError(f"поля {pid} нет в данных кейса")
    return field_facts(store.polygon(pid), year)


def find_episodes(year: int | None = None, cause: str | None = None, severity: str | None = None,
                  limit: int = DEFAULT_LIMIT) -> dict:
    """Периоды снижения по всем полям с фильтрами по году, причине и тяжести."""
    frame = _store().episodes.copy()
    if year:
        frame = frame[frame["year"] == int(year)]
    if severity:
        frame = frame[frame["severity"] == severity]
    if cause:
        codes = [code for code, label in CAUSE_LABEL.items() if cause.lower() in label.lower()]
        frame = frame[frame["cause"].isin(codes or [cause])]
    frame = frame.sort_values("min_z")
    limited = frame.head(max(1, min(limit, MAX_LIMIT)))
    items = [
        {
            "поле": row["pid"], "год": int(row["year"]),
            "период": f"{row['start']} — {row['end']}", "дней": int(row["days"]),
            "тяжесть": row["severity"],
            "причина": CAUSE_LABEL.get(row["cause"], row["cause"]),
            "минимум_отклонения_сигм": round(float(row["min_z"]), 2),
            "объяснение": row.get("text"),
        }
        for _, row in limited.iterrows()
    ]
    return {"всего": int(len(frame)), "показано": len(items), "эпизоды": items}


def ask_about_field(pid: str, question: str, year: int | None = None) -> dict:
    """Вопрос о поле своими словами. С ключом OLLAMA_API_KEY отвечает модель, иначе — правила."""
    from service.agent import ask
    store = _store()
    if pid not in set(store.obs["pid"]):
        raise ValueError(f"поля {pid} нет в данных кейса")
    return ask(store.polygon(pid), question, meta=_meta(), year=year)


def field_report(pid: str, year: int | None = None) -> dict:
    """Готовый HTML-отчёт по полю: самодостаточный файл, печатается в PDF из браузера."""
    from service.report_html import build_report
    store = _store()
    if pid not in set(store.obs["pid"]):
        raise ValueError(f"поля {pid} нет в данных кейса")
    html = build_report(store.polygon(pid), year=year, meta=_meta())
    return {"поле": pid, "сезон": year, "размер_символов": len(html), "html": html}


def solution_metrics() -> dict:
    """Метрики решения и источники данных: чем измеряли и на чём проверяли."""
    return method_facts(_meta())


# Описания инструментов для MCP: имя, что делает, какие аргументы принимает.
TOOL_SPECS: list[dict[str, Any]] = [
    {
        "name": "list_fields",
        "description": "Список полей кейса со сводкой: культура, сезоны, число периодов снижения.",
        "handler": list_fields,
        "schema": {
            "type": "object",
            "properties": {
                "crop": {"type": "string", "description": "фильтр по культуре, часть названия"},
                "only_critical": {"type": "boolean", "description": "только поля с тяжёлыми периодами"},
                "limit": {"type": "integer", "description": f"сколько строк вернуть, по умолчанию {DEFAULT_LIMIT}"},
            },
        },
    },
    {
        "name": "field_summary",
        "description": "Выжимка по одному полю: паспорт, сезоны, периоды снижения, ряд выбранного года.",
        "handler": field_summary,
        "schema": {
            "type": "object",
            "properties": {
                "pid": {"type": "string", "description": "идентификатор поля, например AOI-0001"},
                "year": {"type": "integer", "description": "сезон для подробного ряда"},
            },
            "required": ["pid"],
        },
    },
    {
        "name": "find_episodes",
        "description": "Периоды снижения по всем полям с фильтрами по году, причине и тяжести.",
        "handler": find_episodes,
        "schema": {
            "type": "object",
            "properties": {
                "year": {"type": "integer"},
                "cause": {"type": "string", "description": "часть названия причины, например «погодный»"},
                "severity": {"type": "string", "enum": ["критическая", "умеренная"]},
                "limit": {"type": "integer"},
            },
        },
    },
    {
        "name": "ask_about_field",
        "description": "Вопрос о поле своими словами; ответ строится по проверяемым фактам этого поля.",
        "handler": ask_about_field,
        "schema": {
            "type": "object",
            "properties": {
                "pid": {"type": "string"},
                "question": {"type": "string"},
                "year": {"type": "integer"},
            },
            "required": ["pid", "question"],
        },
    },
    {
        "name": "field_report",
        "description": "HTML-отчёт по полю для владельца: графики, периоды снижения и что проверить.",
        "handler": field_report,
        "schema": {
            "type": "object",
            "properties": {"pid": {"type": "string"}, "year": {"type": "integer"}},
            "required": ["pid"],
        },
    },
    {
        "name": "solution_metrics",
        "description": "Метрики решения и источники данных: чем измеряли качество и на чём проверяли.",
        "handler": solution_metrics,
        "schema": {"type": "object", "properties": {}},
    },
]

HANDLERS = {spec["name"]: spec["handler"] for spec in TOOL_SPECS}
