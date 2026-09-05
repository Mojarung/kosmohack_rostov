"""Запуск MCP-сервера: uv run --no-sync --group agent python -m mcp_server."""

from mcp_server.server import main

if __name__ == "__main__":
    main()
