"""Объяснения периодов снижения языковой моделью — фоном, после анализа.

Детектор и правила остаются на месте: они находят период, ставят причину и пишут черновик,
который пользователь видит сразу. Модель только переписывает черновик связным текстом
по тем же фактам — числа она не меняет.

Почему фоном: модель с рассуждением отвечает десятками секунд на эпизод, а эпизодов у поля
бывает с десяток. Ждать её в момент анализа нельзя, поэтому анализ отдаёт правиловый текст
сразу, а объяснения догружаются: интерфейс запрашивает `GET /api/explanations/{pid}`
и подменяет текст, когда тот готов.

Без ключа `NVIDIA_API_KEY` ничего не запускается и статус сразу `off`.
Модель задаётся переменной NDVI_LLM_MODEL. Установка: uv sync --group agent.
"""

from __future__ import annotations

import logging
import threading
from concurrent.futures import ThreadPoolExecutor

from service import llm

log = logging.getLogger("service.explain")

# Запросы независимы и ждут сеть, поэтому потоков больше, чем ядер, но без фанатизма:
# у API есть лимиты, а эпизодов на поле обычно единицы.
MAX_WORKERS = 3
# Больше не переписываем: у поля с длинной историей эпизодов много, а пользы от текста
# по каждому мало. Остальные остаются с правиловым объяснением.
MAX_EPISODES = 8

_lock = threading.Lock()
_jobs: dict[str, dict] = {}     # pid -> {"status": ..., "items": {ключ эпизода: {"text", "source"}}}


def episode_key(episode: dict) -> str:
    """Ключ эпизода для сопоставления с интерфейсом: год и границы периода."""
    return f"{episode.get('year')}:{episode.get('start')}:{episode.get('end')}"


def llm_available() -> bool:
    """Есть ли ключ и клиент. Отдельная функция, чтобы удобно подменять в тестах."""
    return llm.available()


def _explain_one(episode: dict, model: str | None) -> tuple[str, str]:
    """Переписывает объяснение одного эпизода: (текст, источник)."""
    from anomaly.llm import explain_with_llm

    fallback = episode.get("text", "")
    try:
        return explain_with_llm(episode, fallback, model)
    except Exception:                       # сеть, лимиты, неожиданный ответ — остаёмся на правилах
        log.exception("Не удалось объяснить эпизод %s %s", episode.get("pid"), episode.get("start"))
        return fallback, "rules"


def explain_episodes(episodes: list[dict], model: str | None = None,
                     max_workers: int = MAX_WORKERS) -> list[dict]:
    """Синхронный вариант: объяснения всех эпизодов сразу. Для скриптов и тестов.

    Порядок сохраняется, входной список не изменяется — возвращается новый.
    """
    if not episodes:
        return []
    if not llm_available():
        return [e | {"text_source": "rules"} for e in episodes]

    head, tail = episodes[:MAX_EPISODES], episodes[MAX_EPISODES:]
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        results = list(pool.map(lambda e: _explain_one(e, model), head))
    rewritten = [e | {"text": text, "text_source": source} for e, (text, source) in zip(head, results)]
    written = sum(1 for e in rewritten if e["text_source"] == "llm")
    log.info("Объяснения моделью: %d из %d эпизодов", written, len(head))
    return rewritten + [e | {"text_source": "rules"} for e in tail]


def _run_job(pid: str, episodes: list[dict], model: str | None) -> None:
    """Фоновая задача: переписывает объяснения и складывает их в память процесса."""
    try:
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
            results = list(pool.map(lambda e: _explain_one(e, model), episodes))
        items = {episode_key(e): {"text": text, "source": source}
                 for e, (text, source) in zip(episodes, results)}
        status = "ready"
    except Exception:                       # задача не должна ронять сервер
        log.exception("Фоновые объяснения для %s не удались", pid)
        items, status = {}, "error"
    with _lock:
        _jobs[pid] = {"status": status, "items": items}
    log.info("Объяснения для %s: %s, готово %d", pid, status, len(items))


def request(pid: str, episodes: list[dict], model: str | None = None) -> dict:
    """Запускает фоновое объяснение, если оно ещё не запускалось. Возвращает текущее состояние."""
    if not llm_available():
        return {"status": "off", "items": {}}
    if not episodes:
        return {"status": "ready", "items": {}}
    with _lock:
        current = _jobs.get(pid)
        if current is not None:
            return current
        _jobs[pid] = {"status": "pending", "items": {}}
    threading.Thread(target=_run_job, args=(pid, episodes[:MAX_EPISODES], model),
                     name=f"explain:{pid}", daemon=True).start()
    return {"status": "pending", "items": {}}


def state(pid: str) -> dict:
    """Текущее состояние объяснений поля, без запуска задачи."""
    if not llm_available():
        return {"status": "off", "items": {}}
    with _lock:
        return _jobs.get(pid) or {"status": "idle", "items": {}}


def reset(pid: str | None = None) -> None:
    """Сбрасывает кэш объяснений: целиком или по одному полю. Нужно в тестах и при пересчёте."""
    with _lock:
        if pid is None:
            _jobs.clear()
        else:
            _jobs.pop(pid, None)
