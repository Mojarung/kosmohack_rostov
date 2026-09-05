"""Объяснение эпизода языковой моделью по структурированным фактам (Claude через официальный SDK).

Без ключа модели или при ошибке API возвращается текст по правилам из anomaly.report — детектор
и интерпретация работают и без модели, она лишь делает объяснение связным и добавляет рекомендации.
Ключ и модель настраиваются в service.llm (OLLAMA_API_KEY, NDVI_LLM_MODEL).
Установка: uv sync --group agent.
"""

from __future__ import annotations

import json

SYSTEM_PROMPT = """Ты агроном-аналитик спутникового мониторинга полей. Тебе дают структурированные факты об эпизоде
снижения NDVI на поле: даты, глубину отклонения от климатической нормы поля, фенологию сезона, погоду ERA5
(осадки и температура против нормы тех же дат), поведение соседних полей региона и предварительную причину,
поставленную правилами. Напиши объяснение для агронома на русском языке: 3–6 предложений, без формул и
жаргона, с конкретными числами из фактов. Скажи, насколько серьёзен эпизод, какова наиболее вероятная причина
и почему (опирайся только на факты), что могло бы её опровергнуть, и одну-две практические рекомендации
(что проверить в поле или в документах). Не выдумывай данных, которых нет в фактах."""


def facts_for_llm(row: dict) -> dict:
    """Компактный набор фактов эпизода для промпта (из строки episodes.csv или словаря run.py)."""
    keys = ["pid", "name", "year", "start", "end", "days", "n_obs", "min_z", "mean_z", "critical_days", "max_deficit",
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
    """Возвращает (текст, источник): источник 'llm' при успехе, иначе 'rules' с текстом по правилам.

    Провайдер и модель выбираются в service.llm по переменным окружения; здесь только промпт.
    """
    from service import llm

    if not llm.available():
        return fallback_text, "rules"
    prompt = ("Факты эпизода (JSON):" + chr(10) + json.dumps(facts_for_llm(row), ensure_ascii=False, indent=1)
              + chr(10) * 2 + "Черновик объяснения по правилам:" + chr(10) + fallback_text)
    try:
        reply = llm.chat([{"role": "user", "content": prompt}], SYSTEM_PROMPT, model=model, max_tokens=2000)
    except Exception:
        return fallback_text, "rules"
    text = (reply.text or "").strip()
    return (text, "llm") if text else (fallback_text, "rules")
