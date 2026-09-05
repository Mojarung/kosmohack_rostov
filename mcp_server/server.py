"""MCP-сервер проекта: те же данные, что в веб-интерфейсе, доступны любому MCP-клиенту.

Запуск (stdio, как ждут Claude Desktop и Claude Code):
    uv run --no-sync python -m mcp_server

Подключение в Claude Code:
    claude mcp add vegetation -- uv run --no-sync python -m mcp_server

Транспорт здесь тонкий: вся работа — в mcp_server.tools, поэтому инструменты проверяются
обычными тестами без запуска сервера. Схемы аргументов MCP SDK 2.x выводит из аннотаций типов.
"""

from __future__ import annotations

import functools
import json
import logging
from collections.abc import Callable
from typing import Any

from mcp_server.tools import HANDLERS, TOOL_SPECS

log = logging.getLogger("mcp_server")

# Описания инструментов в формате MCP: то же, что в TOOL_SPECS, только без ссылок на функции.
TOOLS = [
    {"name": spec["name"], "description": spec["description"], "inputSchema": spec["schema"]}
    for spec in TOOL_SPECS
]


def list_tools() -> list[dict[str, Any]]:
    """Список инструментов сервера."""
    return TOOLS


def call_tool(name: str, arguments: dict | None = None) -> str:
    """Вызывает инструмент и возвращает результат текстом (JSON).

    Ошибки наружу не пробрасываются: клиент получает понятное сообщение, а не разрыв соединения.
    """
    handler = HANDLERS.get(name)
    if handler is None:
        return json.dumps({"ошибка": f"неизвестный инструмент {name}"}, ensure_ascii=False)
    try:
        result = handler(**(arguments or {}))
    except TypeError as exc:
        return json.dumps({"ошибка": f"неверные аргументы: {exc}"}, ensure_ascii=False)
    except Exception as exc:
        log.exception("Инструмент %s завершился ошибкой", name)
        return json.dumps({"ошибка": f"{type(exc).__name__}: {exc}"}, ensure_ascii=False)
    return json.dumps(result, ensure_ascii=False, default=str)


def _safe(handler: Callable[..., Any]) -> Callable[..., str]:
    """Обёртка инструмента: результат всегда текст JSON, ошибка не рвёт соединение.

    functools.wraps сохраняет сигнатуру, поэтому MCP SDK выводит схему аргументов
    из аннотаций самой функции, а не из **kwargs обёртки.
    """

    @functools.wraps(handler)
    def run(*args: Any, **kwargs: Any) -> str:
        try:
            return json.dumps(handler(*args, **kwargs), ensure_ascii=False, default=str)
        except Exception as exc:
            log.exception("Инструмент %s завершился ошибкой", handler.__name__)
            return json.dumps({"ошибка": f"{type(exc).__name__}: {exc}"}, ensure_ascii=False)

    return run


def build_server():
    """Собирает MCPServer и регистрирует инструменты из TOOL_SPECS."""
    from mcp.server.mcpserver import MCPServer

    server = MCPServer(
        name="vegetation",
        title="Вегетация: мониторинг полей",
        version="1.0",
        instructions=("Данные спутникового мониторинга сельхозполей Ростовской области: ряды NDVI, "
                      "периоды снижения с причинами, погода ERA5 и метрики решения. "
                      "Числа проверяемые — они посчитаны сервисом, а не придуманы."),
    )
    for spec in TOOL_SPECS:
        server.add_tool(_safe(spec["handler"]), name=spec["name"], description=spec["description"])
    return server


def main() -> None:
    """Точка входа: uv run --no-sync python -m mcp_server."""
    logging.basicConfig(level=logging.WARNING, format="%(name)s: %(message)s")
    build_server().run(transport="stdio")
