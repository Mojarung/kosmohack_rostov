"""MCP-сервер проекта: инструменты мониторинга полей для любого MCP-клиента.

Позволяет спрашивать про поля кейса из Claude Desktop, Claude Code или другого клиента,
не открывая веб-интерфейс. Инструменты возвращают те же проверяемые факты, что и сервис.
"""

from mcp_server.server import TOOLS, call_tool, list_tools

__all__ = ["TOOLS", "call_tool", "list_tools"]
