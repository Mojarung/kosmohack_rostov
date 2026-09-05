"""Набор полигонов пользователя: сохранение результатов анализа новых территорий, список, открытие, удаление.

Хранилище — по одному JSON-файлу на полигон в artifacts/service/polygons/<uid>.json (не в git): геометрия,
имя и полный результат анализа, чтобы открывать сохранённое поле без повторного сбора спутниковых данных.
"""

from __future__ import annotations

import datetime as dt
import json
import re
import uuid
from pathlib import Path

from gapfill.config import ARTIFACTS_DIR
from service import field_store

POLYGONS_DIR = ARTIFACTS_DIR / "service" / "polygons"
UID_RE = re.compile(r"^[0-9a-f]{12}$")            # идентификатор — только hex, чтобы имя файла было безопасным
FIELD_RE = re.compile(r"^FIELD-[0-9a-f]{20}$")
SEVERITY_RANK = {"критическая": 2, "умеренная": 1}
STATUS_BY_RANK = {2: "критическая", 1: "умеренная", 0: "норма"}


def summary_of(result: dict) -> dict:
    """Краткая сводка для списка и карты: эпизоды за всё время и состояние последнего сезона."""
    episodes = result.get("episodes", [])
    years = sorted(int(y) for y in result.get("years", {}))
    last_year = years[-1] if years else None
    worst = max((SEVERITY_RANK.get(e.get("severity"), 0) for e in episodes if e.get("year") == last_year), default=0)
    return {"years": years, "n_episodes": len(episodes),
            "n_critical": sum(e.get("severity") == "критическая" for e in episodes),
            "last_year": last_year, "last_year_status": STATUS_BY_RANK[worst]}


def entry_of(record: dict) -> dict:
    """Запись списка: идентификатор, имя, время, геометрия и сводка (без полного результата)."""
    return {"uid": record["uid"], "name": record["name"], "created_at": record["created_at"],
            "geometry": record["result"].get("geometry"), **summary_of(record["result"])}


def save(result: dict, name: str, base: Path | None = None) -> dict:
    """Сохраняет результат анализа новой территории и возвращает запись списка."""
    base = base or POLYGONS_DIR
    base.mkdir(parents=True, exist_ok=True)
    record = {"uid": uuid.uuid4().hex[:12], "name": name or result.get("pid", "поле"),
              "created_at": dt.datetime.now().isoformat(timespec="seconds"), "result": result}
    (base / f"{record['uid']}.json").write_text(json.dumps(record, ensure_ascii=False), encoding="utf-8")
    return entry_of(record)


def list_saved(base: Path | None = None) -> list[dict]:
    """Единый список нового и прежнего интерфейса, без повторов одного отчёта."""
    records = _records(base)
    entries = [entry_of(r) for r in records]
    known = {r["result"].get("pid") for r in records}
    if base is None:
        for item in field_store.list_fields():
            if item["pid"] not in known:
                report = field_store.read_report(item["pid"])
                entries.append({"uid": item["pid"], "name": item["name"],
                                "created_at": item.get("saved_at") or "", "geometry": item["geometry"],
                                **summary_of(report)})
    return sorted(entries, key=lambda e: e["created_at"], reverse=True)


def _records(base: Path | None = None) -> list[dict]:
    return [json.loads(p.read_text(encoding="utf-8")) for p in (base or POLYGONS_DIR).glob("*.json")]


def find_report(pid: str) -> dict | None:
    """Совместимость с полями, собранными до появления общего хранилища отчётов."""
    return next((r["result"] for r in _records() if r["result"].get("pid") == pid), None)


def load(uid: str, base: Path | None = None) -> dict | None:
    """Полный результат анализа сохранённого полигона (None, если нет)."""
    if base is None and FIELD_RE.fullmatch(uid):
        report = field_store.read_report(uid)
        return report | {"uid": uid} if report else None
    path = (base or POLYGONS_DIR) / f"{uid}.json"
    if not UID_RE.match(uid) or not path.exists():
        return None
    record = json.loads(path.read_text(encoding="utf-8"))
    report = field_store.read_report(record["result"]["pid"]) if base is None else None
    return (report or record["result"]) | {"uid": record["uid"], "name": record["name"], "created_at": record["created_at"]}


def delete(uid: str, base: Path | None = None) -> bool:
    """Удаляет полигон из набора; False, если такого нет."""
    if base is None and FIELD_RE.fullmatch(uid):
        return field_store.delete_report(uid)
    path = (base or POLYGONS_DIR) / f"{uid}.json"
    if not UID_RE.match(uid) or not path.exists():
        return False
    record = json.loads(path.read_text(encoding="utf-8"))
    path.unlink()
    pid = record["result"].get("pid")
    if base is None and pid and not any(r["result"].get("pid") == pid for r in _records()):
        field_store.delete_report(pid)
    return True
