"""Тесты агента-агронома: ответы по правилам, инструменты и диалог с моделью без сети.

Модель не вызывается ни разу: ветка с языковой моделью проверяется подменой
service.agent._call_model, ветка правил — снятой переменной ANTHROPIC_API_KEY.
"""

import pytest

from service import agent
from service.agent import (
    MAX_TOOL_STEPS,
    ask,
    make_tools,
    rule_based_answer,
    year_in_question,
)
from service.facts import field_facts


def _season(year: int) -> dict:
    """Сезон в том же виде, что отдаёт service.data.Store."""
    def points(values: tuple[float, ...]) -> list[dict]:
        return [{"date": f"{year}-06-{1 + 2 * i:02d}", "value": v} for i, v in enumerate(values)]

    return {"norm_source": "история поля",
            "observations": [{"date": f"{year}-06-01", "value": 0.31, "sensor": "Sentinel-2", "artifact": False}],
            "restored": [], "curve": points((0.30, 0.60, 0.50)),
            "norm_mean": points((0.35, 0.65, 0.55)), "z": points((-0.4, -1.2, -0.6))}


def _episode(year: int, severity: str, min_z: float, cause: str) -> dict:
    return {"pid": "AOI-0001", "year": year, "start": f"{year}-06-01", "end": f"{year}-06-21",
            "days": 21, "n_obs": 5, "severity": severity, "min_z": min_z, "mean_z": min_z + 0.5,
            "worst_date": f"{year}-06-11", "ndvi_at_worst": 0.21, "norm_at_worst": 0.51,
            "phase": "налив", "cause": cause, "confidence": 0.78, "reasons": ["осадки ниже нормы"]}


def _weather(year: int, precip: float) -> dict:
    return {"date": [f"{year}-06-0{i}" for i in (1, 2, 3)], "precip": [precip, 0.0, 0.0],
            "temp": [20.0, 22.0, 24.0]}


DETAIL = {
    "pid": "AOI-0001", "name": "Поле у балки", "crop": "озимая пшеница", "kind": "полигон train",
    "years": {2023: _season(2023), 2024: _season(2024)},
    "episodes": [_episode(2024, "критическая", -2.5, "weather_drought"),
                 _episode(2023, "умеренная", -1.4, "late_start")],
    "weather": {2023: _weather(2023, 10.0), 2024: _weather(2024, 40.0)},
}
DRY_FIELD = {"pid": "AOI-0002", "years": {2024: _season(2024)}, "episodes": []}
META = {"task1": {"rmse_val": 0.0543, "gap_score": 13.7, "n_gaps": 2323},
        "task2": {"n_episodes": 318}, "sources": [{"name": "ERA5", "detail": "Open-Meteo"}]}


class _Block:
    """Блок ответа модели: текст или запрос инструмента (форма как у SDK anthropic)."""

    def __init__(self, type: str, text: str | None = None, name: str | None = None,
                 input: dict | None = None, id: str | None = None) -> None:
        self.type, self.text, self.name, self.input, self.id = type, text, name, input or {}, id


class _Response:
    """Ответ модели: причина остановки и список блоков."""

    def __init__(self, stop_reason: str, content: list[_Block]) -> None:
        self.stop_reason, self.content = stop_reason, content


@pytest.fixture
def facts() -> dict:
    return field_facts(DETAIL)


@pytest.fixture
def no_key(monkeypatch):
    """Ключа модели нет: агент обязан отвечать по правилам и никуда не ходить."""
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setattr(agent, "_call_model", lambda *a: pytest.fail("модель не должна вызываться"))


@pytest.mark.parametrize("question", [
    "Что случилось с полем?", "Как считается точность?", "Сколько выпало осадков?",
    "Расскажи про 2024 год", "Привет",
])
def test_without_api_key_answer_is_always_rule_based(no_key, question):
    """Без ключа источник ответа всегда «rules», инструменты не вызываются."""
    result = ask(DETAIL, question, meta=META)
    assert result["source"] == "rules" and result["tools_used"] == []
    assert result["answer"] and isinstance(result["answer"], str)


def test_empty_question_returns_hint(no_key):
    """Пустой вопрос — подсказка, а не пустой ответ и не обращение к модели."""
    assert ask(DETAIL, "   ")["answer"] == "Задайте вопрос о поле."
    assert ask(DETAIL, None)["source"] == "rules"


@pytest.mark.parametrize("question, expected", [
    ("Каким методом это посчитано?", "Sentinel-2"),
    ("Насколько можно доверять числам?", "Точность восстановления"),
    ("Сколько выпало осадков за сезон?", "мм осадков"),
    ("Была ли засуха?", "мм осадков"),
    ("Какой год был самый плохой?", "Самый тяжёлый период"),
    ("Почему просела зелёность?", "вероятные причины"),
    ("Что не так с полем?", "вероятные причины"),
    ("Расскажи про 2024 год", "Сезон 2024 на поле"),
    ("Здравствуйте", "Спросите про конкретный год"),
])
def test_rule_based_answer_picks_branch_by_keywords(facts, question, expected):
    """Ветка ответа выбирается ключевыми словами вопроса."""
    assert expected in rule_based_answer(facts, question)


def test_rule_based_answer_about_weather(facts):
    """Погодный ответ берёт последний сезон и отдельно называет самый сухой."""
    answer = rule_based_answer(facts, "Сколько было осадков?")
    assert "40.0 мм осадков" in answer and "22.0 °C" in answer
    assert "Самым сухим" in answer and "2023" in answer
    без_погоды = rule_based_answer(field_facts(DRY_FIELD), "Сколько было осадков?")
    assert "метеоряда нет" in без_погоды


def test_rule_based_answer_without_episodes():
    """Поле без эпизодов не выдумывает угнетение ни в «худшем», ни в «причинах»."""
    facts = field_facts(DRY_FIELD)
    assert "не найдено" in rule_based_answer(facts, "Какой самый худший период?")
    assert "не найдено" in rule_based_answer(facts, "Почему поле просело?")
    assert "Найдено периодов снижения: 0" in rule_based_answer(facts, "Привет")


def test_rule_based_answer_narrows_to_named_year(facts):
    """Названный в вопросе год сужает и эпизоды, и сезоны."""
    answer = rule_based_answer(facts, "Что было худшего в 2023 году?")
    assert "в 2023 году" in answer and "2023-06-01 — 2023-06-21" in answer
    assert "поздний старт" in answer          # причина эпизода именно 2023 года


@pytest.mark.parametrize("question, known, expected", [
    ("что было в 2019 и 2024 годах", [2023, 2024], 2024),
    ("сравни 2023 и 2024", [2023, 2024], 2023),
    ("в 1999 году", [2023, 2024], None),
    ("без года", [2023], None),
    ("в 2024", [], None),
])
def test_year_in_question_takes_only_known_season(question, known, expected):
    """Из вопроса берётся только тот год, который у поля действительно есть."""
    assert year_in_question(question, known) == expected


def test_tools_return_facts_and_report_missing_season():
    """Инструменты отдают проверяемые факты, а несуществующий сезон — понятную ошибку."""
    tools = make_tools(DETAIL, META)
    assert set(tools) == {"season", "episodes", "method"}

    missing = tools["season"]({"year": 1999})
    assert "ошибка" in missing and "1999" in missing["ошибка"] and "2023" in missing["ошибка"]
    assert "ошибка" in tools["season"]({})            # год не назван — сезона 0 тоже нет

    season = tools["season"]({"year": 2024})
    assert season["год"] == 2024 and season["осадки_за_сезон_мм"] == 40.0

    assert len(tools["episodes"]({})) == 2
    only_2023 = tools["episodes"]({"year": 2023})
    assert len(only_2023) == 1 and only_2023[0]["год"] == 2023
    assert tools["method"]({})["восстановление_пропусков"]["rmse"] == 0.0543


def test_tools_work_with_string_year_keys():
    """Поле из field_store хранит годы строками — инструмент должен его понимать."""
    detail = {"pid": "NEW:1", "years": {"2024": _season(2024)}, "weather": {"2024": _weather(2024, 5.0)}}
    assert make_tools(detail)["season"]({"year": 2024})["год"] == 2024


def test_episodes_tool_answers_when_nothing_found():
    """Пустой список эпизодов заменяется понятным ответом, а не пустотой."""
    assert make_tools(DRY_FIELD)["episodes"]({}) == {"ответ": "периодов снижения не найдено"}


def test_model_calls_tool_and_returns_its_own_answer(monkeypatch):
    """Модель просит инструмент, получает факты и отвечает текстом: источник — llm."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "ключ-для-теста")
    seen: list[dict] = []

    def fake_call(client, model, messages, tools):
        seen.append(messages[-1])
        assert model == "claude-test" and [t["name"] for t in tools] == ["season", "episodes", "method"]
        if len(seen) == 1:
            return _Response("tool_use", [_Block("tool_use", name="season", input={"year": 2024}, id="t1")])
        return _Response("end_turn", [_Block("text", text="Сезон 2024 прошёл ниже нормы.")])

    monkeypatch.setattr(agent, "_call_model", fake_call)
    result = ask(DETAIL, "Что было в 2024 году?", meta=META, model="claude-test")

    assert result == {"answer": "Сезон 2024 прошёл ниже нормы.", "source": "llm", "tools_used": ["season"]}
    assert "Факты о поле" in seen[0]["content"] and "Поле у балки" in seen[0]["content"]
    tool_result = seen[1]["content"][0]
    assert tool_result["type"] == "tool_result" and tool_result["tool_use_id"] == "t1"
    assert '"год": 2024' in tool_result["content"]          # инструменту вернулись факты сезона


def test_model_failure_falls_back_to_rules(monkeypatch):
    """Отказ модели (сеть, лимиты) не оставляет пользователя без ответа."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "ключ-для-теста")

    def boom(*args):
        raise RuntimeError("нет сети")

    monkeypatch.setattr(agent, "_call_model", boom)
    result = ask(DETAIL, "Почему просела зелёность?", meta=META)
    assert result["source"] == "rules" and result["tools_used"] == []
    assert "вероятные причины" in result["answer"]


def test_empty_model_text_falls_back_to_rules(monkeypatch):
    """Ответ модели без текста считается неудачей: отвечаем правилами."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "ключ-для-теста")
    monkeypatch.setattr(agent, "_call_model",
                        lambda *a: _Response("end_turn", [_Block("text", text="   ")]))
    result = ask(DETAIL, "Какой год самый тяжёлый?", meta=META)
    assert result["source"] == "rules" and "Самый тяжёлый период" in result["answer"]


def test_endless_tool_requests_are_stopped(monkeypatch):
    """Модель, которая только просит инструменты, останавливается на MAX_TOOL_STEPS."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "ключ-для-теста")
    monkeypatch.setattr(agent, "_call_model", lambda *a: _Response(
        "tool_use", [_Block("tool_use", name="нет_такого", input={}, id="t9")]))
    result = ask(DETAIL, "Привет", meta=META)
    assert result["source"] == "rules"
    assert result["tools_used"] == ["нет_такого"] * MAX_TOOL_STEPS
