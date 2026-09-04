"""Хранилище пользовательских полигонов — SQLite, чтобы не тянуть внешнюю БД."""

from __future__ import annotations

import json
import sqlite3
import time
from contextlib import contextmanager

from ndvi.paths import ROOT

DB_PATH = ROOT / "artifacts" / "polygons.sqlite"

SCHEMA = """
CREATE TABLE IF NOT EXISTS polygons (
    id          TEXT PRIMARY KEY,
    name        TEXT NOT NULL,
    crop_type   TEXT,
    source      TEXT,           -- osm | draw | demo
    area_ha     REAL,
    geometry    TEXT NOT NULL,  -- GeoJSON
    created_at  REAL
);
"""


@contextmanager
def connect():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        conn.executescript(SCHEMA)
        yield conn
        conn.commit()
    finally:
        conn.close()


def list_polygons() -> list[dict]:
    with connect() as c:
        rows = c.execute("SELECT * FROM polygons ORDER BY created_at DESC").fetchall()
    return [_row_to_dict(r) for r in rows]


def get_polygon(pid: str) -> dict | None:
    with connect() as c:
        row = c.execute("SELECT * FROM polygons WHERE id = ?", (pid,)).fetchone()
    return _row_to_dict(row) if row else None


def save_polygon(pid: str, name: str, geometry: dict, crop_type: str = "не указана",
                 source: str = "draw", area_ha: float = 0.0) -> dict:
    with connect() as c:
        c.execute(
            "INSERT OR REPLACE INTO polygons (id, name, crop_type, source, area_ha, geometry, created_at)"
            " VALUES (?,?,?,?,?,?,?)",
            (pid, name, crop_type, source, area_ha, json.dumps(geometry), time.time()))
    return get_polygon(pid)


def delete_polygon(pid: str) -> bool:
    with connect() as c:
        cur = c.execute("DELETE FROM polygons WHERE id = ?", (pid,))
    return cur.rowcount > 0


def _row_to_dict(row: sqlite3.Row) -> dict:
    d = dict(row)
    d["geometry"] = json.loads(d["geometry"])
    return d
