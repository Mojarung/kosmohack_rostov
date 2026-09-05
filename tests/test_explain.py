"""Тесты объяснений языковой моделью на этапе анализа (service.explain).

Сети здесь нет: доступность модели и сам вызов подменяются. Проверяется главное —
без ключа анализ работает как раньше, с моделью текст переписывается фоном и подхватывается
интерфейсом, а любая ошибка модели не роняет анализ и оставляет правиловое объяснение.
"""

from __future__ import annotations

import time

import pytest

from service import explain


def episode(start: str = "2020-05-01", text: str = "черновик по правилам") -> dict:
    """Минимальный эпизод: для объяснений важны только текст и опознавательные поля."""
    return {"pid": "AOI-0001", "year": 2020, "start": start, "end": "2020-06-01", "text": text}


def test_без_ключа_текст_остаётся_правиловым(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OLLAMA_API_KEY", raising=False)
    result = explain.explain_episodes([episode(), episode("2020-07-01", "второй черновик")])
    assert [e["text"] for e in result] == ["черновик по правилам", "второй черновик"]
    assert {e["text_source"] for e in result} == {"rules"}


def test_пустой_список_возвращает_пустой(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OLLAMA_API_KEY", raising=False)
    assert explain.explain_episodes([]) == []


def test_входной_список_не_меняется(monkeypatch: pytest.MonkeyPatch) -> None:
    """Правило проекта: входные структуры не мутируем."""
    monkeypatch.delenv("OLLAMA_API_KEY", raising=False)
    source = [episode()]
    explain.explain_episodes(source)
    assert "text_source" not in source[0]


def test_модель_переписывает_текст(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(explain, "llm_available", lambda: True)
    monkeypatch.setattr("anomaly.llm.explain_with_llm",
                        lambda ep, fallback, model=None: (f"связный текст про {ep['start']}", "llm"))
    result = explain.explain_episodes([episode(), episode("2020-07-01")])
    assert [e["text"] for e in result] == ["связный текст про 2020-05-01", "связный текст про 2020-07-01"]
    assert {e["text_source"] for e in result} == {"llm"}


def test_порядок_эпизодов_сохраняется(monkeypatch: pytest.MonkeyPatch) -> None:
    """Запросы идут параллельно, но порядок в списке должен остаться исходным."""
    monkeypatch.setattr(explain, "llm_available", lambda: True)
    monkeypatch.setattr("anomaly.llm.explain_with_llm", lambda ep, fallback, model=None: (ep["start"], "llm"))
    starts = [f"2020-0{i}-01" for i in range(1, 8)]
    result = explain.explain_episodes([episode(s) for s in starts])
    assert [e["text"] for e in result] == starts


def test_ошибка_модели_оставляет_правиловый_текст(monkeypatch: pytest.MonkeyPatch) -> None:
    def falls(ep: dict, fallback: str, model: str | None = None):
        raise RuntimeError("сеть недоступна")

    monkeypatch.setattr(explain, "llm_available", lambda: True)
    monkeypatch.setattr("anomaly.llm.explain_with_llm", falls)
    result = explain.explain_episodes([episode()])
    assert result[0]["text"] == "черновик по правилам"
    assert result[0]["text_source"] == "rules"


def test_отказ_модели_помечается_как_правила(monkeypatch: pytest.MonkeyPatch) -> None:
    """explain_with_llm сам возвращает источник rules, когда модель не ответила."""
    monkeypatch.setattr(explain, "llm_available", lambda: True)
    monkeypatch.setattr("anomaly.llm.explain_with_llm", lambda ep, fallback, model=None: (fallback, "rules"))
    result = explain.explain_episodes([episode()])
    assert result[0]["text_source"] == "rules"


def test_переписываются_только_первые_эпизоды(monkeypatch: pytest.MonkeyPatch) -> None:
    """Длинная история не должна стоить десятков запросов: хвост остаётся на правилах."""
    monkeypatch.setattr(explain, "llm_available", lambda: True)
    monkeypatch.setattr("anomaly.llm.explain_with_llm", lambda ep, fallback, model=None: ("модель", "llm"))
    result = explain.explain_episodes([episode(f"2020-{i:02d}-01") for i in range(1, explain.MAX_EPISODES + 4)])
    assert [e["text_source"] for e in result[: explain.MAX_EPISODES]] == ["llm"] * explain.MAX_EPISODES
    assert {e["text_source"] for e in result[explain.MAX_EPISODES:]} == {"rules"}
    assert {e["text"] for e in result[explain.MAX_EPISODES:]} == {"черновик по правилам"}


def test_llm_available_без_ключа(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OLLAMA_API_KEY", raising=False)
    assert explain.llm_available() is False


def test_llm_available_с_ключом_но_без_клиента(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ключ есть, а клиент не установлен: анализ всё равно должен работать."""
    import builtins

    monkeypatch.setenv("OLLAMA_API_KEY", "тест")
    real_import = builtins.__import__

    def no_openai(name: str, *args, **kwargs):
        if name == "openai":
            raise ImportError("нет пакета")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_openai)
    assert explain.llm_available() is False


# --- ленивая (фоновая) подгрузка объяснений ---


def test_ключ_эпизода_из_года_и_границ() -> None:
    assert explain.episode_key(episode()) == "2020:2020-05-01:2020-06-01"


def test_без_ключа_фоновая_задача_не_запускается(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(explain, "llm_available", lambda: False)
    assert explain.request("AOI-0001", [episode()]) == {"status": "off", "items": {}}
    assert explain.state("AOI-0001") == {"status": "off", "items": {}}


def test_состояние_до_запуска(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(explain, "llm_available", lambda: True)
    explain.reset()
    assert explain.state("AOI-0001") == {"status": "idle", "items": {}}


def test_пустой_список_эпизодов_сразу_готов(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(explain, "llm_available", lambda: True)
    explain.reset()
    assert explain.request("AOI-0002", []) == {"status": "ready", "items": {}}


def test_фоновая_задача_доводит_объяснения(monkeypatch: pytest.MonkeyPatch) -> None:
    """Первый запрос отдаёт pending, дальше в кэше появляется текст модели."""
    monkeypatch.setattr(explain, "llm_available", lambda: True)
    monkeypatch.setattr("anomaly.llm.explain_with_llm",
                        lambda ep, fallback, model=None: (f"модель про {ep['start']}", "llm"))
    explain.reset()

    assert explain.request("AOI-0003", [episode()])["status"] == "pending"
    for _ in range(100):                      # задача идёт в отдельном потоке
        current = explain.state("AOI-0003")
        if current["status"] != "pending":
            break
        time.sleep(0.05)
    assert current["status"] == "ready"
    assert current["items"]["2020:2020-05-01:2020-06-01"] == {"text": "модель про 2020-05-01", "source": "llm"}


def test_повторный_запрос_не_плодит_задачи(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(explain, "llm_available", lambda: True)
    calls: list[str] = []

    def slow(ep: dict, fallback: str, model: str | None = None):
        calls.append(ep["start"])
        time.sleep(0.2)
        return "готово", "llm"

    monkeypatch.setattr("anomaly.llm.explain_with_llm", slow)
    explain.reset()
    for _ in range(5):
        explain.request("AOI-0004", [episode()])
    for _ in range(100):
        if explain.state("AOI-0004")["status"] != "pending":
            break
        time.sleep(0.05)
    assert calls == ["2020-05-01"]            # задача запустилась ровно один раз


def test_сброс_кэша_по_полю(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(explain, "llm_available", lambda: True)
    monkeypatch.setattr("anomaly.llm.explain_with_llm", lambda ep, fallback, model=None: ("текст", "llm"))
    explain.reset()
    explain.request("AOI-0005", [episode()])
    explain.reset("AOI-0005")
    assert explain.state("AOI-0005")["status"] == "idle"


def test_ошибка_в_фоне_не_роняет_сервис(monkeypatch: pytest.MonkeyPatch) -> None:
    """Исключение внутри задачи должно оставаться в ней и превращаться в понятный статус."""
    monkeypatch.setattr(explain, "llm_available", lambda: True)
    monkeypatch.setattr(explain, "_explain_one", lambda *a: (_ for _ in ()).throw(RuntimeError("сломалось")))
    explain.reset()
    explain.request("AOI-0006", [episode()])
    for _ in range(100):
        current = explain.state("AOI-0006")
        if current["status"] != "pending":
            break
        time.sleep(0.05)
    assert current["status"] == "error" and current["items"] == {}
