"""Неизвестные API и файлы не подменяются HTML для клиентской маршрутизации."""
import pytest
from fastapi.testclient import TestClient

from service.app import app


@pytest.mark.parametrize("path", ["/api", "/api/missing-route", "/api/places-typo", "/missing.js", "/fonts/missing.woff2"])
def test_missing_route_returns_json_404(path):
    response = TestClient(app).get(path)
    assert response.status_code == 404
    assert response.headers["content-type"].startswith("application/json")
    assert "detail" in response.json()


def test_explore_still_serves_spa():
    response = TestClient(app).get("/explore")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/html")


def test_places_returns_data_not_spa():
    response = TestClient(app).get("/api/places", params={"q": "47.22, 39.72"})
    assert response.status_code == 200
    assert response.json()[0]["center"] == [39.72, 47.22]
