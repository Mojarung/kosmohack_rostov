"""Проверка полного окружения перед приёмом запросов веб-сервисом."""

import importlib
from pathlib import Path

START_COMMAND = "uv run --locked --group service python -m service"


def plotly_bundle() -> Path:
    """JS графиков поставляется установленным пакетом Plotly и не зависит от CDN."""
    package = importlib.import_module("plotly")
    return Path(package.__file__).parent / "package_data" / "plotly.min.js"


def check_runtime() -> None:
    """Импортирует сборщик заранее, чтобы ошибка установки не возникала после клика по карте."""
    try:
        importlib.import_module("service.collect")
        if not plotly_bundle().is_file():
            raise ImportError("не найден plotly.min.js в установленном пакете plotly")
    except ImportError as exc:
        raise RuntimeError(f"Не установлены зависимости сбора данных ({exc}). Запустите: {START_COMMAND}") from exc
