"""Контракт объединённого списка: источники, сезоны, дубли и отсутствие оценки."""
from types import SimpleNamespace

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from service import anomaly_feed, field_store, polygons
from service.app import app
from tests.e2e.anomalies_fixtures import seed_saved


@pytest.fixture
def saved(tmp_path, monkeypatch):
    monkeypatch.setattr(polygons, "POLYGONS_DIR", tmp_path / "polygons")
    monkeypatch.setattr(field_store, "ROOT", tmp_path / "fields")
    return seed_saved()


def test_mine_uses_saved_reports_without_case_loading(saved):
    def forbidden():
        pytest.fail("Мои поля не должны загружать датасет кейса")
    result = anomaly_feed.build_feed(forbidden)
    assert result["year"] == 2025 and result["years"] == [2025, 2024]
    assert len(result["fields"]) == 3
    levels = {f["name"]: f["level"] for f in result["fields"]}
    assert levels == {"Южное": "clear", "У реки": "data", "Северное": "unknown"}
    assert next(f for f in result["fields"] if f["name"] == "Северное")["has_season"] is False
    old = anomaly_feed.build_feed(forbidden, year=2024)
    assert old["fields"][0]["name"] == "Южное"
    assert old["fields"][0]["level"] == "high" and len(old["fields"][0]["episodes"]) == 2
    assert next(f for f in old["fields"] if f["name"] == "Северное")["level"] == "unknown"


def test_sources_and_missing_year(saved):
    store = SimpleNamespace(obs=pd.DataFrame({"pid": ["AOI-0001"], "year": [2023]}),
                            episodes=pd.DataFrame(), seasons=pd.DataFrame())
    case = anomaly_feed.build_feed(lambda: store, "case")
    assert case["year"] == 2023 and len(case["fields"]) == 1
    combined = anomaly_feed.build_feed(lambda: store, "all", 2024)
    assert len(combined["fields"]) == 4 and combined["years"] == [2025, 2024, 2023]
    assert len({f["key"] for f in combined["fields"]}) == 4
    assert all(f["level"] == "unknown" for f in anomaly_feed.build_feed(lambda: store, "all", 2022)["fields"])


def test_api_and_deletion(saved):
    client = TestClient(app)
    response = client.get("/api/anomaly-fields")
    assert response.status_code == 200 and len(response.json()["fields"]) == 3
    for entry in saved:
        polygons.delete(entry["uid"])
    assert client.get("/api/anomaly-fields").json() == {"years": [], "year": None, "fields": []}


@pytest.mark.parametrize("query", ["source=bad", "year=abc", "year=1500"])
def test_invalid_query(query):
    assert TestClient(app).get(f"/api/anomaly-fields?{query}").status_code == 422
