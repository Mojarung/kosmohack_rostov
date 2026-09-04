"""Объяснение эпизода языковой моделью по структурированным фактам (Claude через официальный SDK).

Без ключа (`ANTHROPIC_API_KEY`) или при ошибке API возвращается текст по правилам из anomaly.report —
детектор и интерпретация работают и без LLM, модель лишь делает объяснение связным и добавляет рекомендации.
Установка: uv sync --group agent. Модель задаётся переменной окружения ANOMALY_LLM_MODEL (по умолчанию claude-opus-5).
"""

from __future__ import annotations

import json
import os

DEFAULT_MODEL = "claude-opus-5"
SYSTEM_PROMPT = """Ты агроном-аналитик спутникового мониторинга полей. Тебе дают структурированные факты об эпизоде
снижения NDVI на поле: даты, глубину отклонения от климатической нормы поля, фенологию сезона, погоду ERA5
(осадки и температура против нормы тех же дат), поведение соседних полей региона и предварительную причину,
поставленную правилами. Напиши объяснение для агронома на русском языке: 3–6 предложений, без формул и
жаргона, с конкретными числами из фактов. Скажи, насколько серьёзен эпизод, какова наиболее вероятная причина
и почему (опирайся только на факты), что могло бы её опровергнуть, и одну-две практические рекомендации
(что проверить в поле или в документах). Не выдумывай данных, которых нет в фактах."""


def facts_for_llm(row: dict) -> dict:
    """Компактный набор фактов эпизода для промпта (из строки episodes.csv или словаря run.py)."""
    keys = ["pid", "year", "start", "end", "days", "n_obs", "min_z", "mean_z", "critical_days", "max_deficit",
            "worst_date", "ndvi_at_worst", "norm_at_worst", "phase", "severity", "cause", "confidence",
            "norm_source", "weather_source", "region_z", "region_share_depressed", "reasons"]
    facts = {k: row.get(k) for k in keys if row.get(k) is not None}
    weather = row.get("weather")
    if isinstance(weather, str):
        try:
            weather = json.loads(weather)
        except json.JSONDecodeError:
            weather = None
    if weather:
        facts["weather"] = {k: v for k, v in weather.items() if k in ("combined", "before", "during", "source")}
    return facts


def explain_with_llm(row: dict, fallback_text: str, model: str | None = None) -> tuple[str, str]:
    """Возвращает (текст, источник): источник 'llm' при успехе, иначе 'rules' с текстом по правилам."""
    if not os.environ.get("ANTHROPIC_API_KEY"):
        return fallback_text, "rules"
    try:
        import anthropic
    except ImportError:
        return fallback_text, "rules"
    client = anthropic.Anthropic()
    prompt = ("Факты эпизода (JSON):\n" + json.dumps(facts_for_llm(row), ensure_ascii=False, indent=1)
              + "\n\nЧерновик объяснения по правилам:\n" + fallback_text)
    try:
        response = client.messages.create(
            model=model or os.environ.get("ANOMALY_LLM_MODEL", DEFAULT_MODEL),
            max_tokens=2000,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": prompt}],
        )
    except anthropic.APIError:
        return fallback_text, "rules"
    if response.stop_reason == "refusal":
        return fallback_text, "rules"
    text = "".join(block.text for block in response.content if block.type == "text").strip()
    return (text, "llm") if text else (fallback_text, "rules")
