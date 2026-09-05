"""Языковая модель для объяснений и агента: Ollama Cloud.

Ключ и модель берутся из окружения, в коде их нет:

    OLLAMA_API_KEY   — ключ Ollama Cloud (без него сервис работает по правилам);
    NDVI_LLM_MODEL   — модель, по умолчанию gemma4:31b;
    OLLAMA_BASE_URL  — адрес, по умолчанию https://ollama.com/v1.

Ollama говорит по протоколу OpenAI, поэтому берём готовый клиент `openai`.
Наружу торчит одна функция `chat`: принимает сообщения и инструменты в формате OpenAI,
возвращает разобранный ответ `Reply`. Без ключа `available()` возвращает False,
и вызывающий код работает по правилам — сервис не должен зависеть от внешней модели.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any

BASE_URL = "https://ollama.com/v1"
# Замер на бесплатных облачных моделях (короткий ответ / вызов инструмента):
#   gemma4:31b        0.8 с / 0.5 с   — выбрана
#   gpt-oss:20b       2.7 с / 1.3 с
#   nemotron-3-super  3.0 с / 2.3 с
# kimi-k3 отвечает минутами и в интерактивном сценарии не годится.
DEFAULT_MODEL = "gemma4:31b"
DEFAULT_MAX_TOKENS = 1200
DEFAULT_TEMPERATURE = 0.4       # объяснения должны быть предсказуемыми, а не творческими
TIMEOUT_S = 90.0


@dataclass(frozen=True)
class ToolCall:
    """Запрос модели на вызов инструмента."""

    id: str
    name: str
    arguments: dict[str, Any]


@dataclass(frozen=True)
class Reply:
    """Разобранный ответ модели: текст и/или запрошенные инструменты."""

    text: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    raw: Any = None

    @property
    def wants_tools(self) -> bool:
        return bool(self.tool_calls)


def model_name() -> str:
    """Имя модели из окружения или значение по умолчанию."""
    return os.environ.get("NDVI_LLM_MODEL") or DEFAULT_MODEL


def available() -> bool:
    """Есть ли ключ и установлен ли клиент."""
    if not os.environ.get("OLLAMA_API_KEY"):
        return False
    try:
        import openai  # noqa: F401
    except ImportError:
        return False
    return True


def _parse_arguments(raw: str | dict | None) -> dict:
    """Аргументы инструмента приходят строкой JSON; на битой строке отдаём пустой словарь."""
    if isinstance(raw, dict):
        return raw
    if not raw:
        return {}
    try:
        parsed = json.loads(raw)
    except (TypeError, json.JSONDecodeError):
        return {}
    return parsed if isinstance(parsed, dict) else {}


def chat(messages: list[dict], system: str, tools: list[dict] | None = None,
         model: str | None = None, max_tokens: int = DEFAULT_MAX_TOKENS,
         temperature: float = DEFAULT_TEMPERATURE) -> Reply:
    """Один запрос к модели. Сообщения и инструменты — в формате OpenAI.

    Бросает исключение при сетевой ошибке: вызывающий код сам решает, откатываться ли на правила.
    """
    if not os.environ.get("OLLAMA_API_KEY"):
        raise RuntimeError("не задан OLLAMA_API_KEY")
    from openai import OpenAI

    client = OpenAI(base_url=os.environ.get("OLLAMA_BASE_URL", BASE_URL),
                    api_key=os.environ["OLLAMA_API_KEY"], timeout=TIMEOUT_S)
    payload: dict[str, Any] = {
        "model": model or model_name(),
        "messages": [{"role": "system", "content": system}, *messages],
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    if tools:
        payload["tools"] = [{"type": "function", "function": tool} for tool in tools]
    response = client.chat.completions.create(**payload)
    message = response.choices[0].message
    calls = [ToolCall(id=call.id, name=call.function.name, arguments=_parse_arguments(call.function.arguments))
             for call in (message.tool_calls or [])]
    return Reply(text=(message.content or "").strip(), tool_calls=calls, raw=message)
