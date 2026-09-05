"""Локальное сохранение отчётов и заметок: атомарные файлы с устойчивыми идентификаторами."""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
import threading
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / "artifacts" / "fields"
LOCK = threading.RLock()


def field_id(geometry: dict) -> str:
    """Имя поля не определяет его идентичность: одинаковые имена не перезаписывают разные контуры."""
    return "FIELD-" + hashlib.sha256(json.dumps(geometry, sort_keys=True).encode()).hexdigest()[:20]


def _path(pid: str, kind: str) -> Path:
    return ROOT / kind / (hashlib.sha256(pid.encode()).hexdigest() + ".json")


def _write(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f".{uuid.uuid4().hex}.tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    os.replace(temporary, path)


def read_report(pid: str) -> dict | None:
    path = _path(pid, "reports")
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def save_report(report: dict) -> None:
    """Записывает готовый отчёт отдельно от изменяемых пользователем параметров."""
    with LOCK:
        report = report | {"saved_at": dt.datetime.now(dt.UTC).isoformat(), "report_version": "farmer-1"}
        _write(_path(report["pid"], "reports"), report)


def public_report(report: dict) -> dict:
    return {key: value for key, value in report.items() if not key.startswith("_")}


def list_fields() -> list[dict]:
    out = []
    for path in (ROOT / "reports").glob("*.json"):
        report = json.loads(path.read_text(encoding="utf-8"))
        out.append({"pid": report["pid"], "name": report.get("name", report["pid"]),
                    "years": sorted(map(int, report["years"])), "saved_at": report.get("saved_at"),
                    "n_episodes": len(report.get("episodes", [])), "geometry": report.get("geometry")})
    return sorted(out, key=lambda item: item.get("saved_at") or "", reverse=True)


def read_settings(pid: str, year: int) -> dict:
    path = _path(pid, "settings")
    all_settings = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    return all_settings.get(str(year), {})


def save_settings(pid: str, year: int, settings: dict) -> None:
    with LOCK:
        path = _path(pid, "settings")
        values = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
        values[str(year)] = settings
        _write(path, values)
