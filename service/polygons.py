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

POLYGONS_DIR = ARTIFACTS_DIR / "service" / "polygons"
UID_RE = re.compile(r"^[0-9a-f]{12}$")            # идентификатор — только hex, чтобы имя файла было безопасным
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
    """Все сохранённые полигоны, новые сверху."""
    base = base or POLYGONS_DIR
    if not base.exists():
        return []
    records = [json.loads(p.read_text(encoding="utf-8")) for p in base.glob("*.json")]
    return sorted((entry_of(r) for r in records), key=lambda e: e["created_at"], reverse=True)


def load(uid: str, base: Path | None = None) -> dict | None:
    """Полный результат анализа сохранённого полигона (None, если нет)."""
    path = (base or POLYGONS_DIR) / f"{uid}.json"
    if not UID_RE.match(uid) or not path.exists():
        return None
    record = json.loads(path.read_text(encoding="utf-8"))
    return record["result"] | {"uid": record["uid"], "name": record["name"], "created_at": record["created_at"]}


def delete(uid: str, base: Path | None = None) -> bool:
    """Удаляет полигон из набора; False, если такого нет."""
    path = (base or POLYGONS_DIR) / f"{uid}.json"
    if not UID_RE.match(uid) or not path.exists():
        return False
    path.unlink()
    return True
