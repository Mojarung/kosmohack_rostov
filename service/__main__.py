"""Запуск полного веб-сервиса: uv run --locked --group service python -m service."""

import argparse
import logging
from multiprocessing import freeze_support

import uvicorn


def main() -> None:
    """Единая точка запуска для Windows и Linux."""
    parser = argparse.ArgumentParser(description="Веб-сервис мониторинга NDVI")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    uvicorn.run("service.app:app", host=args.host, port=args.port)


if __name__ == "__main__":
    freeze_support()
    main()
