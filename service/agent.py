"""Агент-агроном: отвечает на вопросы о поле по посчитанным фактам.

Модель не видит сырых данных и не считает NDVI — она получает выжимку из service.facts и может
дозапросить подробности через инструменты (сезон, эпизоды, метод). Всё, что она говорит,
опирается на уже проверенные числа.

Без ключа модели (см. service.llm) работает разбор по правилам: он отвечает на частые вопросы
теми же фактами, только без связного текста. Провайдер и модель настраиваются переменными
окружения NVIDIA_API_KEY и NDVI_LLM_MODEL. Установка: uv sync --group agent.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from typing import Any

from service import llm
from service.facts import _by_year, episode_facts, field_facts, method_facts, season_facts

MAX_TOOL_STEPS = 4          # больше четырёх дозапросов на один вопрос не нужно и дорого
MAX_TOKENS = 1500

SYSTEM_PROMPT = """Ты агроном-аналитик спутникового мониторинга полей. Отвечаешь на вопросы владельца поля
на русском языке.

Правила:
- Опирайся только на факты, которые тебе дали или которые вернули инструменты. Ничего не выдумывай.
- Если для ответа не хватает данных, так и скажи и объясни, каких именно.
- Пиши коротко и по делу: 2-5 предложений, конкретные числа и даты из фактов.
- Без формул и жаргона. NDVI объясняй как «зелёность поля по снимку», сигмы — как «насколько сильно
  сезон отличается от обычного для этого поля».
- Отклонение ниже нормы — это сигнал, а не диагноз: упоминай, чем его можно проверить в поле.
- Не давай обещаний про урожай и не ставь агрономических диагнозов, которых нет в фактах."""

TOOLS = [
    {
        "name": "season",
        "description": "Сводка по одному сезону поля: наблюдения, пик кривой, худшее отклонение, осадки и температура.",
        "parameters": {
            "type": "object",
            "properties": {"year": {"type": "integer", "description": "год сезона"}},
            "required": ["year"],
        },
    },
    {
        "name": "episodes",
        "description": "Периоды снижения на поле с причинами и аргументами. Без года — все, с годом — только за него.",
        "parameters": {
            "type": "object",
            "properties": {"year": {"type": "integer", "description": "год, необязательно"}},
        },
    },
    {
        "name": "method",
        "description": "Как получены числа: метрики восстановления пропусков, статистика эпизодов и источники данных.",
        "parameters": {"type": "object", "properties": {}},
    },
]


def make_tools(detail: dict, meta: dict | None = None) -> dict[str, Callable[[dict], Any]]:
    """Инструменты агента поверх одного поля. Чистые функции: ни сети, ни модели внутри нет."""

    def season(args: dict) -> Any:
        year = int(args.get("year", 0))
        payload = _by_year(detail.get("years"), year)
        if not payload:
            known = sorted(int(y) for y in (detail.get("years") or {}))
            return {"ошибка": f"сезона {year} нет; есть {known}"}
        return season_facts(year, payload, _by_year(detail.get("weather"), year))

    def episodes(args: dict) -> Any:
        items = list(detail.get("episodes") or [])
        if args.get("year"):
            items = [e for e in items if e.get("year") == int(args["year"])]
        return [episode_facts(e) for e in items] or {"ответ": "периодов снижения не найдено"}

    def method(_: dict) -> Any:
        return method_facts(meta or {})

    return {"season": season, "episodes": episodes, "method": method}


def year_in_question(question: str, known: list[int]) -> int | None:
    """Год, названный в вопросе, если такой сезон у поля есть."""
    for match in re.findall(r"\b(?:19|20)\d{2}\b", question):
        year = int(match)
        if year in known:
            return year
    return None


def _about_method() -> str:
    return ("Ряд зелёности собирается из снимков Sentinel-2, Landsat и MODIS, пропуски восстанавливает модель, "
            "а сезон сравнивается с нормой этого же поля по его прошлым годам. "
            "Точность восстановления и список источников показаны на экране «Как это работает».")


def _about_weather(field: str, seasons: list[dict]) -> str:
    with_weather = [s for s in seasons if s.get("осадки_за_сезон_мм") is not None]
    if not with_weather:
        return f"Для поля {field} метеоряда нет, поэтому про осадки и температуру сказать нечего."
    driest = min(with_weather, key=lambda s: s["осадки_за_сезон_мм"])
    last = with_weather[-1]
    answer = (f"За сезон {last['год']} на поле {field} выпало {last['осадки_за_сезон_мм']} мм осадков "
              f"при средней температуре {last.get('средняя_температура_c')} °C.")
    if driest["год"] != last["год"]:
        answer += f" Самым сухим из наблюдаемых был {driest['год']} год: {driest['осадки_за_сезон_мм']} мм."
    return answer


def _about_worst(field: str, episodes: list[dict], year: int | None) -> str:
    where = f" в {year} году" if year else ""
    if not episodes:
        return f"На поле {field}{where} устойчивых отклонений ниже нормы не найдено."
    worst = episodes[0]
    return (f"Самый тяжёлый период на поле {field}{where} — {worst['период']}, "
            f"{worst['дней']} дней, тяжесть «{worst['тяжесть']}». "
            f"В худший день {worst['худший_день']} зелёность была {worst['ndvi_в_худший_день']} "
            f"при обычной для этой даты {worst['норма_в_худший_день']}. "
            f"Вероятная причина: {worst['причина']}. "
            "Это сигнал, а не диагноз: стоит сверить с записями по полю за те даты.")


def _about_causes(field: str, episodes: list[dict], year: int | None) -> str:
    where = f" в {year} году" if year else ""
    if not episodes:
        return f"На поле {field}{where} отклонений, требующих объяснения, не найдено."
    lines = [f"{e['период']}: {e['причина']} (тяжесть «{e['тяжесть']}», уверенность {e['уверенность']})"
             for e in episodes[:3]]
    return f"Периоды снижения на поле {field}{where} и их вероятные причины:\n- " + "\n- ".join(lines)


def _about_season(field: str, season: dict, episodes: list[dict]) -> str:
    tail = (f"Периодов снижения за этот год: {len(episodes)}." if episodes
            else "Периодов снижения за этот год не найдено.")
    return (f"Сезон {season.get('год')} на поле {field}: {season.get('наблюдений')} снимков, "
            f"пик зелёности {season.get('пик_ndvi')} — {season.get('дата_пика')}, "
            f"худшее отклонение {season.get('худшее_отклонение_сигм')} сигм "
            f"({season.get('дата_худшего_отклонения')}). " + tail)


def rule_based_answer(facts: dict, question: str) -> str:
    """Ответ без языковой модели: те же факты, выбранные по ключевым словам вопроса."""
    q = question.lower()
    field = facts.get("поле")
    episodes = facts.get("эпизоды") or []
    seasons = facts.get("сводка_по_сезонам") or []
    year = year_in_question(question, facts.get("сезоны") or [])
    if year:
        episodes = [e for e in episodes if e.get("год") == year]
        seasons = [s for s in seasons if s.get("год") == year] or seasons

    if any(w in q for w in ("метод", "как счита", "точност", "rmse", "модел", "откуда данн",
                            "надёжн", "надежн", "довер", "проверя")):
        return _about_method()
    if any(w in q for w in ("погод", "осадк", "дожд", "температур", "засух", "полив", "влаг")):
        return _about_weather(field, seasons)
    if any(w in q for w in ("худш", "хуже", "плох", "тяжёл", "тяжел", "критич", "серьёзн", "серьезн")):
        return _about_worst(field, episodes, year)
    if any(w in q for w in ("почему", "причин", "что случилось", "из-за чего", "что не так", "проблем")):
        return _about_causes(field, episodes, year)
    if year and seasons:
        return _about_season(field, seasons[0], episodes)

    years = facts.get("сезоны") or []
    span = f"{years[0]}–{years[-1]}" if years else "нет данных"
    return (f"Поле {field}, культура: {facts.get('культура')}. Сезоны {span}. "
            f"Найдено периодов снижения: {facts.get('всего_эпизодов', 0)}, "
            f"из них тяжёлых: {facts.get('критических_эпизодов', 0)}. "
            "Спросите про конкретный год, про причины или про погоду — отвечу подробнее.")


def _run_tool_loop(messages: list, handlers: dict, facts: dict, question: str, model: str | None) -> dict:
    """Диалог с моделью: пока она просит инструменты — отдаём факты, иначе возвращаем ответ.

    Формат сообщений — OpenAI (service.llm сам переводит его под провайдера).
    """
    used: list[str] = []
    for _ in range(MAX_TOOL_STEPS):
        try:
            reply = llm.chat(messages, SYSTEM_PROMPT, TOOLS, model=model, max_tokens=MAX_TOKENS)
        except Exception:                       # сеть, лимиты, отказ модели — отвечаем по правилам
            return {"answer": rule_based_answer(facts, question), "source": "rules", "tools_used": used}
        if not reply.wants_tools:
            text = (reply.text or "").strip()
            if not text:            # пустой ответ модели считаем неудачей
                break
            return {"answer": text, "source": "llm", "tools_used": used}

        messages.append({
            "role": "assistant",
            "content": reply.text or None,
            "tool_calls": [{"id": c.id, "type": "function",
                            "function": {"name": c.name, "arguments": json.dumps(c.arguments, ensure_ascii=False)}}
                           for c in reply.tool_calls],
        })
        for call in reply.tool_calls:
            used.append(call.name)
            handler = handlers.get(call.name)
            payload = handler(call.arguments) if handler else {"ошибка": "неизвестный инструмент"}
            messages.append({"role": "tool", "tool_call_id": call.id,
                             "content": json.dumps(payload, ensure_ascii=False, default=str)})
    # шаги кончились, а ответа нет — не оставляем пользователя ни с чем
    return {"answer": rule_based_answer(facts, question), "source": "rules", "tools_used": used}


def ask(detail: dict, question: str, meta: dict | None = None, year: int | None = None,
        model: str | None = None) -> dict:
    """Отвечает на вопрос о поле: текст ответа, источник (llm или rules) и вызванные инструменты."""
    question = (question or "").strip()
    facts = field_facts(detail, year)
    if not question:
        return {"answer": "Задайте вопрос о поле.", "source": "rules", "tools_used": []}
    if not llm.available():
        return {"answer": rule_based_answer(facts, question), "source": "rules", "tools_used": []}

    messages: list[dict] = [{
        "role": "user",
        "content": ("Факты о поле (JSON):\n" + json.dumps(facts, ensure_ascii=False)
                    + f"\n\nВопрос владельца поля: {question}"),
    }]
    return _run_tool_loop(messages, make_tools(detail, meta), facts, question, model)
