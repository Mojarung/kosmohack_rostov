"""Ход сбора данных для нового полигона: сборщик пишет JSON, сервер и интерфейс его читают.

Сбор идёт в отдельном процессе, поэтому общая память не подходит: состояние лежит в файле
artifacts/service/progress/<job>.json, куда потоки сборщика дописывают завершённые годы,
а GET /api/analyze/progress/<job> отдаёт файл как есть. Идентификатор задания задаёт клиент
до отправки POST /api/analyze, так что опрашивать прогресс можно, пока основной запрос ещё висит.
"""

from __future__ import annotations

import datetime as dt
import json
import re
import threading
import time
from pathlib import Path

from gapfill.config import ARTIFACTS_DIR
from service.field_store import _write

PROGRESS_DIR = ARTIFACTS_DIR / "service" / "progress"
JOB_RE = re.compile(r"^[A-Za-z0-9-]{8,64}$")     # только безопасные символы: идентификатор становится именем файла
KEEP_SECONDS = 24 * 3600                          # файлы завершённых заданий живут сутки
TITLES = {"S2": "Sentinel-2", "Landsat": "Landsat 8/9", "MODIS": "MODIS", "ERA5": "Погода ERA5"}


def _now() -> str:
    return dt.datetime.now(dt.UTC).isoformat(timespec="seconds")


def is_valid_job(job: str | None) -> bool:
    return bool(job) and JOB_RE.match(job) is not None


class Progress:
    """Состояние одного задания. Без идентификатора все методы — пустые операции (пакетный запуск, тесты)."""

    def __init__(self, job: str | None):
        self.path = PROGRESS_DIR / f"{job}.json" if is_valid_job(job) else None
        self.lock = threading.Lock()
        self.state = {"job": job, "started": _now(), "stage": "collect", "sources": {}, "log": []}
        if self.path is not None:
            _prune_old()
            self._write()

    def start(self, label: str, total: int) -> None:
        """Источник начал работу: известно, сколько сезонов предстоит загрузить."""
        with self.lock:
            self.state["sources"][label] = {"title": TITLES.get(label, label), "total": total, "done": 0,
                                            "scenes": 0, "status": "running", "note": ""}
            self._write()

    def year_done(self, label: str, year: int, scenes: int, seconds: float, ok: bool = True) -> None:
        """Один сезон источника загружен (или не загружен): счётчик и строка журнала."""
        with self.lock:
            src = self.state["sources"].setdefault(label, {"title": TITLES.get(label, label), "total": 0, "done": 0,
                                                          "scenes": 0, "status": "running", "note": ""})
            src["done"] += 1
            src["scenes"] += scenes
            text = f"{src['title']} {year}: {scenes} сцен за {seconds:.0f} с" if ok else f"{src['title']} {year}: не загружен"
            self._log(text)
            self._write()

    def finish_source(self, label: str, note: str = "", ok: bool = True) -> None:
        with self.lock:
            src = self.state["sources"].setdefault(label, {"title": TITLES.get(label, label), "total": 0, "done": 0,
                                                          "scenes": 0, "status": "running", "note": ""})
            src["status"] = "done" if ok else "failed"
            src["note"] = note
            if ok:
                src["done"] = max(src["done"], src["total"])    # ERA5 грузится одним блоком без счётчика по годам
            self._write()

    def stage(self, name: str, text: str = "") -> None:
        """Крупный этап: collect → analysis → done | error."""
        with self.lock:
            self.state["stage"] = name
            if text:
                self._log(text)
            self._write()

    def _log(self, text: str) -> None:
        self.state["log"] = (self.state["log"] + [text])[-12:]

    def _write(self) -> None:
        if self.path is None:
            return
        self.state["updated"] = _now()
        _write(self.path, self.state)


def read_progress(job: str) -> dict | None:
    """Текущее состояние задания или None, если сборщик ещё не создал файл."""
    if not is_valid_job(job):
        return None
    path = PROGRESS_DIR / f"{job}.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):    # файл в момент атомарной замены
        return None


def _prune_old() -> None:
    """Удаляет файлы заданий старше суток, чтобы каталог не рос бесконечно."""
    if not PROGRESS_DIR.exists():
        return
    threshold = time.time() - KEEP_SECONDS
    for path in PROGRESS_DIR.glob("*.json"):
        try:
            if path.stat().st_mtime < threshold:
                path.unlink()
        except OSError:
            pass
