"""Тесты MCP-инструментов и сборки сервера без запуска процесса stdio.

Хранилище кейса подменяется маленьким набором в памяти: тесты не читают CSV и не ходят в сеть.
Сам сервер проверяется через объект MCPServer — транспорт для этого поднимать не нужно.
"""

import asyncio
import json

import pandas as pd
import pytest

from mcp_server import server, tools


def _season(year: int) -> dict:
    """Сезон в том же виде, что отдаёт service.data.Store.polygon."""
    def points(base: float) -> list[dict]:
        return [{"date": f"{year}-06-{1 + 3 * i:02d}", "value": round(base + 0.05 * i, 3)} for i in range(8)]

    return {"norm_source": "история поля",
            "observations": [{"date": f"{year}-06-01", "value": 0.31, "sensor": "Sentinel-2", "artifact": False}],
            "restored": [], "curve": points(0.20), "norm_mean": points(0.25), "z": points(-1.5)}


POLYGONS = [
    {"pid": "AOI-0001", "crop": "озимая пшеница", "kind": "полигон train", "years": [2023, 2024],
     "n_obs": 120, "n_episodes": 2, "n_critical": 1, "has_weather": True, "n_gaps": 3},
    {"pid": "AOI-0002", "crop": "подсолнечник", "kind": "полигон test с историей", "years": [2024],
     "n_obs": 60, "n_episodes": 1, "n_critical": 0, "has_weather": False, "n_gaps": 0},
    {"pid": "AOI-0003", "crop": "яровая пшеница", "kind": "полигон test", "years": [2024],
     "n_obs": 40, "n_episodes": 0, "n_critical": 0, "has_weather": False, "n_gaps": 0},
]

EPISODES = pd.DataFrame([
    {"pid": "AOI-0001", "year": 2024, "start": "2024-06-04", "end": "2024-06-16", "days": 12,
     "severity": "критическая", "cause": "weather_drought", "min_z": -2.6, "text": "сухо и жарко"},
    {"pid": "AOI-0001", "year": 2023, "start": "2023-05-01", "end": "2023-05-20", "days": 19,
     "severity": "умеренная", "cause": "late_start", "min_z": -1.4, "text": "поздний сев"},
    {"pid": "AOI-0002", "year": 2024, "start": "2024-07-01", "end": "2024-07-14", "days": 13,
     "severity": "умеренная", "cause": "crop_rotation", "min_z": -1.8, "text": "смена культуры"},
])

DETAIL = {
    "pid": "AOI-0001", "crop": "озимая пшеница", "kind": "полигон train",
    "years": {2023: _season(2023), 2024: _season(2024)},
    "episodes": EPISODES.loc[EPISODES["pid"] == "AOI-0001"].to_dict(orient="records"),
    "weather": {2024: {"date": ["2024-06-01"], "precip": [12.0], "temp": [21.0]}},
}


class _FakeStore:
    """Хранилище кейса в памяти: те же поля и методы, которыми пользуются инструменты."""

    def __init__(self) -> None:
        self.obs = pd.DataFrame({"pid": [p["pid"] for p in POLYGONS]})
        self.episodes = EPISODES.copy()
        self.gaps = pd.DataFrame({"pid": ["AOI-0001"] * 3})

    def polygons(self) -> list[dict]:
        return [dict(row) for row in POLYGONS]          # список сортируется вызывающим кодом

    def polygon(self, pid: str) -> dict:
        return DETAIL | {"pid": pid}


@pytest.fixture
def store(monkeypatch) -> _FakeStore:
    """Инструменты работают с набором в памяти вместо CSV кейса."""
    fake = _FakeStore()
    monkeypatch.setattr(tools, "_store", lambda: fake)
    return fake


def test_list_tools_describes_every_tool():
    """Шесть инструментов с непустыми описаниями и схемами аргументов."""
    listed = server.list_tools()
    assert [t["name"] for t in listed] == ["list_fields", "field_summary", "find_episodes",
                                           "ask_about_field", "field_report", "solution_metrics"]
    assert len(listed) == 6 and set(tools.HANDLERS) == {t["name"] for t in listed}
    for tool in listed:
        assert tool["description"].strip() and tool["inputSchema"]["type"] == "object"
        assert isinstance(tool["inputSchema"].get("properties"), dict)
    schemas = {t["name"]: t["inputSchema"] for t in listed}
    assert schemas["field_summary"]["required"] == ["pid"]
    assert schemas["ask_about_field"]["required"] == ["pid", "question"]
    assert schemas["solution_metrics"]["properties"] == {}


def test_call_tool_reports_unknown_name():
    """Неизвестное имя инструмента — понятный JSON, а не исключение."""
    payload = json.loads(server.call_tool("нет_такого"))
    assert "ошибка" in payload and "нет_такого" in payload["ошибка"]


def test_call_tool_reports_wrong_arguments(store):
    """Неверные аргументы тоже возвращаются ответом, а не рвут соединение."""
    лишний = json.loads(server.call_tool("field_summary", {"поле": "AOI-0001"}))
    assert "неверные аргументы" in лишний["ошибка"]
    без_обязательного = json.loads(server.call_tool("field_summary", {}))
    assert "неверные аргументы" in без_обязательного["ошибка"]
    нечисло = json.loads(server.call_tool("list_fields", {"limit": "много"}))
    assert "ошибка" in нечисло


def test_call_tool_reports_missing_field(store):
    """Несуществующее поле объясняется словами, с именем класса ошибки для отладки."""
    payload = json.loads(server.call_tool("field_summary", {"pid": "AOI-9999"}))
    assert payload["ошибка"].startswith("ValueError") and "нет в данных кейса" in payload["ошибка"]
    with pytest.raises(ValueError, match="AOI-9999"):
        tools.field_summary("AOI-9999")


def test_list_fields_filters_by_crop_and_severity(store):
    """Фильтры по культуре и тяжести, сортировка по числу тяжёлых периодов, лимит строк."""
    полный = tools.list_fields()
    assert полный["всего"] == 3 and полный["показано"] == 3
    assert [f["pid"] for f in полный["поля"]] == ["AOI-0001", "AOI-0002", "AOI-0003"]

    пшеница = tools.list_fields(crop="ПШЕНИЦА")               # регистр не важен, ищется подстрока
    assert пшеница["всего"] == 2 and {f["pid"] for f in пшеница["поля"]} == {"AOI-0001", "AOI-0003"}

    тяжёлые = tools.list_fields(only_critical=True)
    assert тяжёлые["всего"] == 1 and тяжёлые["поля"][0]["pid"] == "AOI-0001"
    assert tools.list_fields(crop="рис")["поля"] == []

    урезанный = tools.list_fields(limit=2)
    assert урезанный["всего"] == 3 and урезанный["показано"] == 2
    assert tools.list_fields(limit=999)["показано"] == 3      # лимит ограничен сверху MAX_LIMIT


def test_find_episodes_filters_by_year_severity_and_cause(store):
    """Периоды снижения фильтруются по году, тяжести и русскому названию причины."""
    все = tools.find_episodes()
    assert все["всего"] == 3 and все["эпизоды"][0]["минимум_отклонения_сигм"] == -2.6
    assert все["эпизоды"][0]["поле"] == "AOI-0001" and все["эпизоды"][0]["год"] == 2024
    assert все["эпизоды"][0]["период"] == "2024-06-04 — 2024-06-16"
    assert все["эпизоды"][0]["объяснение"] == "сухо и жарко"

    assert tools.find_episodes(year=2023)["всего"] == 1
    assert tools.find_episodes(severity="критическая")["всего"] == 1
    assert tools.find_episodes(year=2024, severity="умеренная")["всего"] == 1

    погода = tools.find_episodes(cause="погодный")
    assert погода["всего"] == 1 and погода["эпизоды"][0]["причина"] == "погодный стресс"
    assert tools.find_episodes(cause="севооборот")["эпизоды"][0]["поле"] == "AOI-0002"
    assert tools.find_episodes(cause="нашествие марсиан")["всего"] == 0

    урезанный = tools.find_episodes(limit=1)
    assert урезанный["всего"] == 3 and урезанный["показано"] == 1


def test_field_summary_returns_facts_of_chosen_season(store):
    """Выжимка по полю — те же факты, что показывает интерфейс."""
    summary = tools.field_summary("AOI-0001", year=2023)
    assert summary["идентификатор"] == "AOI-0001" and summary["культура"] == "озимая пшеница"
    assert summary["сезоны"] == [2023, 2024] and summary["выбранный_сезон"] == 2023
    assert summary["всего_эпизодов"] == 2 and summary["критических_эпизодов"] == 1
    assert summary["эпизоды"][0]["причина"] == "погодный стресс"
    assert len(summary["ряд_выбранного_сезона"]["восстановленная_кривая"]) == 8


def test_field_report_returns_self_contained_html(store):
    """Отчёт отдаётся готовым HTML вместе с его размером."""
    payload = json.loads(server.call_tool("field_report", {"pid": "AOI-0001", "year": 2024}))
    assert payload["поле"] == "AOI-0001" and payload["сезон"] == 2024
    assert payload["размер_символов"] == len(payload["html"])
    assert payload["html"].startswith("<!doctype html>") and "<svg" in payload["html"]
    assert "https://" not in payload["html"]


def test_solution_metrics_describes_method(store):
    """Метрики решения собираются из артефактов репозитория и числа контрольных точек."""
    metrics = json.loads(server.call_tool("solution_metrics"))
    assert set(metrics) == {"восстановление_пропусков", "поиск_эпизодов", "источники"}
    assert metrics["восстановление_пропусков"]["контрольных_точек"] == 3
    assert "отложенная выборка" in metrics["восстановление_пропусков"]["оговорка"]
    assert metrics["источники"] and all(": " in s for s in metrics["источники"])


def test_ask_about_field_without_key_uses_rules(store, monkeypatch):
    """Без ключа модели инструмент отвечает по правилам и не ходит в сеть."""
    monkeypatch.delenv("OLLAMA_API_KEY", raising=False)
    payload = json.loads(server.call_tool("ask_about_field",
                                          {"pid": "AOI-0001", "question": "Почему поле просело?"}))
    assert payload["source"] == "rules" and payload["tools_used"] == []
    assert "погодный стресс" in payload["answer"]
    with pytest.raises(ValueError, match="нет в данных кейса"):
        tools.ask_about_field("AOI-9999", "Что случилось?")


def test_safe_wrapper_returns_json_and_keeps_signature():
    """Обёртка инструмента отдаёт JSON и на успехе, и на ошибке, сохраняя имя функции."""
    def ok(pid: str) -> dict:
        """Успешный инструмент."""
        return {"поле": pid}

    def bad(pid: str) -> dict:
        raise ValueError("нет данных")

    assert json.loads(server._safe(ok)("AOI-1")) == {"поле": "AOI-1"}
    assert "ValueError: нет данных" in json.loads(server._safe(bad)("AOI-1"))["ошибка"]
    assert server._safe(ok).__name__ == "ok" and server._safe(ok).__doc__ == "Успешный инструмент."


def test_build_server_registers_all_tools():
    """Сервер собирается со всеми инструментами и схемами из аннотаций, без запуска транспорта."""
    pytest.importorskip("mcp")
    built = server.build_server()
    assert built.name == "vegetation" and "Ростовской области" in built.instructions

    registered = asyncio.run(built.list_tools())
    assert [t.name for t in registered] == [spec["name"] for spec in tools.TOOL_SPECS]
    assert all(t.description for t in registered)
    schemas = {t.name: t.input_schema for t in registered}
    assert schemas["field_summary"]["required"] == ["pid"]
    assert set(schemas["find_episodes"]["properties"]) == {"year", "cause", "severity", "limit"}
    assert schemas["solution_metrics"]["properties"] == {}
