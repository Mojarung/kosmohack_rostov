"""Единый стиль графиков и сохранение фигур."""

from __future__ import annotations

from pathlib import Path

import matplotlib

# Только запись файлов, без GUI-бэкенда
matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import seaborn as sns  # noqa: E402

from eda.config import FIG_DIR  # noqa: E402

DPI = 130


def setup_style() -> None:
    """Общий стиль: светлый фон, читаемые подписи, кириллица через DejaVu Sans."""
    sns.set_theme(style="whitegrid", context="notebook", font="DejaVu Sans")
    plt.rcParams.update({
        "figure.dpi": DPI,
        "savefig.dpi": DPI,
        "axes.titlesize": 12,
        "axes.titleweight": "bold",
        "axes.labelsize": 10,
        "legend.fontsize": 9,
        "figure.constrained_layout.use": True,
    })


def save_fig(fig: plt.Figure, name: str) -> Path:
    """Сохраняет фигуру в reports/eda/figures/<name>.png и закрывает её."""
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    path = FIG_DIR / f"{name}.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path
