"""Контракт поиска, геометрия, общий кэш, лимит и отказ внешнего источника."""
import io
import json
import urllib.error
from concurrent.futures import ThreadPoolExecutor

import pytest
from fastapi.testclient import TestClient

from service import geocoding
from service.app import app


@pytest.fixture
def client(monkeypatch, tmp_path):
    monkeypatch.setattr(geocoding, "CACHE_PATH", tmp_path / "geocoding.sqlite")
    monkeypatch.setenv("NOMINATIM_URL", "https://geocoder.example/search")
    return TestClient(app)


def place():
    return {"osm_type": "way", "osm_id": 123, "display_name": "Большая Садовая, Ростов-на-Дону",
            "lat": "47.22", "lon": "39.72", "boundingbox": ["47.21", "47.23", "39.70", "39.74"],
            "address": {"road": "Большая Садовая", "state": "Ростовская область"}, "type": "residential"}


def test_search_contract_and_cache(client, monkeypatch):
    calls = []
    def upstream(req, **kwargs):
        calls.append(req)
        return io.BytesIO(json.dumps([place()]).encode())
    monkeypatch.setattr(geocoding.urllib.request, "urlopen", upstream)
    response = client.get("/api/places", params={"q": "Большая Садовая, Ростов-на-Дону"})
    assert response.status_code == 200
    found = response.json()[0]
    assert found["bbox"] == [39.70, 47.21, 39.74, 47.23]
    assert found["center"] == [39.72, 47.22]
    assert found["address"]["state"] == "Ростовская область"
    assert client.get("/api/places", params={"q": "большая  садовая, Ростов-на-Дону"}).json() == response.json()
    assert len(calls) == 1 and calls[0].get_header("User-agent")
    assert client.get("/api/places", params={"q": "Москва"}).status_code == 429


@pytest.mark.parametrize("query,status", [("", 422), ("а", 422), ("  ", 400), ("а" * 201, 422),
                                          ("91, 30", 400), ("45, 181", 400)])
def test_invalid_query(client, query, status):
    assert client.get("/api/places", params={"q": query}).status_code == status


def test_coordinates_skip_network(client, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("Координаты не должны обращаться к сети")
    monkeypatch.setattr(geocoding.urllib.request, "urlopen", forbidden)
    for query in ("47.22, 39.72", "47.22 39.72", "47.22;39.72"):
        response = client.get("/api/places", params={"q": query})
        assert response.json()[0]["center"] == [39.72, 47.22]


def test_failure_then_retry(client, monkeypatch):
    now = [100.0]
    monkeypatch.setattr(geocoding.time, "time", lambda: now[0])
    def offline(*args, **kwargs):
        raise urllib.error.URLError("offline")
    monkeypatch.setattr(geocoding.urllib.request, "urlopen", offline)
    assert client.get("/api/places?q=Ростов").status_code == 502
    now[0] += 2
    monkeypatch.setattr(geocoding.urllib.request, "urlopen", lambda *a, **k: io.BytesIO(b"[]"))
    assert client.get("/api/places?q=Ростов").json() == []
    assert client.get("/api/places?q=Ростов").status_code == 200


def test_bad_geometry():
    assert geocoding.normalize_place({**place(), "lat": "nan"}) is None
    assert geocoding.normalize_place({**place(), "boundingbox": [0, 1, 2, 3]}) is None


def test_concurrent_requests_share_limit(client):
    def reserve(index):
        try:
            geocoding.cached_or_reserve("https://geocoder.example", str(index))
            return 200
        except geocoding.HTTPException as exc:
            return exc.status_code
    with ThreadPoolExecutor(max_workers=4) as pool:
        assert sorted(pool.map(reserve, range(4))) == [200, 429, 429, 429]
