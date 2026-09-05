"""Регрессии запуска и поиска полей: зависимости сборщика и ошибки рамки карты."""

import io
import json
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from service.osm import osm_fields
from service.runtime import check_runtime


@pytest.fixture(autouse=True)
def clear_osm_cache():
    """Сетевые заглушки не должны использовать ответы предыдущего теста."""
    from service.osm import _cached_fields
    _cached_fields.cache_clear()
    yield
    _cached_fields.cache_clear()


def test_osm_falls_back_and_caches_success(monkeypatch):
    """Отказ одного Overpass не прерывает поиск; повторный клик не делает новый сетевой запрос."""
    import urllib.error
    calls = []
    def response(req, **kwargs):
        calls.append(req.full_url)
        if len(calls) == 1:
            raise urllib.error.HTTPError(req.full_url, 504, "Gateway Timeout", {}, None)
        return io.BytesIO(b'{"elements": []}')
    monkeypatch.setattr("service.osm.urllib.request.urlopen", response)
    monkeypatch.setattr("service.osm.time.monotonic", lambda: 1000)
    assert osm_fields((47.3, 39.8, 47.4, 39.9)) == []
    assert osm_fields((47.3, 39.8, 47.4, 39.9)) == []
    assert len(calls) == 2 and calls[0] != calls[1]


def test_osm_import_without_satellite_packages():
    """OSM импортируется даже при недоступных библиотеках спутникового сборщика."""
    code = textwrap.dedent("""
        import sys
        class BlockSatelliteImports:
            def find_spec(self, fullname, path=None, target=None):
                if fullname.split('.')[0] in {'planetary_computer', 'pystac_client', 'odc', 'rasterio', 'shapely'}:
                    raise ModuleNotFoundError(fullname)
        sys.meta_path.insert(0, BlockSatelliteImports())
        from service.osm import osm_fields
        assert callable(osm_fields)
    """)
    result = subprocess.run([sys.executable, "-c", code], cwd=Path(__file__).resolve().parents[1],
                            capture_output=True, text=True, timeout=20, check=False)
    assert result.returncode == 0, result.stderr


def test_osm_returns_closed_geojson_contours(monkeypatch):
    """В карту попадают замкнутые контуры с правильным порядком долгота/широта."""
    ring = [{"lon": 39.8, "lat": 47.3}, {"lon": 39.9, "lat": 47.3},
            {"lon": 39.9, "lat": 47.4}, {"lon": 39.8, "lat": 47.3}]
    payload = {"elements": [{"id": 1, "geometry": ring}, {"id": 2, "geometry": ring[:-1]}]}
    monkeypatch.setattr("service.osm.urllib.request.urlopen", lambda *a, **k: io.BytesIO(json.dumps(payload).encode()))
    fields = osm_fields((47.3, 39.8, 47.4, 39.9))
    assert len(fields) == 1
    assert fields[0]["geometry"] == {"type": "Polygon", "coordinates": [[[39.8, 47.3], [39.9, 47.3],
                                                                        [39.9, 47.4], [39.8, 47.3]]]}


def test_runtime_explains_missing_collection_dependency(monkeypatch):
    """Ошибка неполной установки содержит воспроизводимую команду запуска."""
    def missing(name):
        raise ModuleNotFoundError("No module named 'planetary_computer'")
    monkeypatch.setattr("service.runtime.importlib.import_module", missing)
    with pytest.raises(RuntimeError, match="--group service python -m service"):
        check_runtime()


@pytest.fixture
def service_client(monkeypatch):
    """API без сетевых запросов и дорогого импорта геостека; проверка импорта тестируется отдельно."""
    pytest.importorskip("fastapi")
    pytest.importorskip("httpx")
    from fastapi.testclient import TestClient

    import service.app as module
    monkeypatch.setattr(module, "check_runtime", lambda: None)
    with TestClient(module.app) as client:
        yield client


@pytest.mark.parametrize("bbox", ["wrong", "1,2,3", "nan,39,48,40", "48,39,47,40", "47,39,49,41"])
def test_invalid_bbox_is_not_reported_as_overpass_failure(service_client, bbox, monkeypatch):
    """Неверные координаты дают 400 до обращения к внешнему API."""
    def unexpected_request(bounds):
        pytest.fail("внешний API не должен вызываться для неверной рамки")
    monkeypatch.setattr("service.app.osm_fields", unexpected_request)
    response = service_client.get("/api/fields", params={"bbox": bbox})
    assert response.status_code == 400
    assert "Overpass" not in response.json()["detail"]


def test_health_and_fields_api(service_client, monkeypatch):
    """Готовность и поиск контуров доступны после старта приложения."""
    monkeypatch.setattr("service.app.osm_fields", lambda bounds: [{"id": 7}])
    assert service_client.get("/api/health").json() == {"status": "ok", "collection": True}
    response = service_client.get("/api/fields", params={"bbox": "47.3,39.8,47.4,39.9"})
    assert response.status_code == 200 and response.json() == [{"id": 7}]


def test_incomplete_environment_stops_startup(monkeypatch):
    """Приложение не объявляет готовность с отсутствующим сборщиком."""
    pytest.importorskip("fastapi")
    pytest.importorskip("httpx")
    from fastapi.testclient import TestClient

    import service.app as module
    def missing():
        raise RuntimeError("не установлены зависимости сборщика")
    monkeypatch.setattr(module, "check_runtime", missing)
    with pytest.raises(RuntimeError, match="зависимости сборщика"), TestClient(module.app):
        pytest.fail("запуск с неполным окружением должен завершиться ошибкой")


def test_plotly_is_served_locally(service_client):
    """Графики получают JS с сервера, даже когда внешний CDN недоступен."""
    pytest.importorskip("plotly")
    html = service_client.get("/").text
    assert 'src="/vendor/plotly.min.js"' in html
    response = service_client.get("/vendor/plotly.min.js")
    assert response.status_code == 200 and len(response.content) > 1_000_000
    assert "javascript" in response.headers["content-type"]


def test_s2_uses_public_cogs_without_mutating_catalog_item():
    """Снятые с общего псевдонима JP2 не должны перехватывать каналы публичного COG."""
    pytest.importorskip("planetary_computer")
    import datetime as dt

    import pystac
    from odc.stac import parse_items

    from service.collect import public_s2_items

    item = pystac.Item("scene", geometry=None, bbox=None, datetime=dt.datetime(2020, 6, 1, tzinfo=dt.UTC), properties={})
    item.collection_id = "sentinel-2-l2a"
    for key, href, media in [("red", "https://example.com/B04.tif", pystac.MediaType.COG),
                              ("red-jp2", "s3://requester-pays/B04.jp2", pystac.MediaType.JPEG2000)]:
        item.add_asset(key, pystac.Asset(href, media_type=media, roles=["data"],
                                         extra_fields={"eo:bands": [{"name": "red", "common_name": "red"}]}))
    for key in ("nir", "scl"):
        item.add_asset(key, pystac.Asset(f"https://example.com/{key}.tif", media_type=pystac.MediaType.COG, roles=["data"]))
    private = item.clone()
    private.assets["red"].href = "s3://requester-pays/B04.jp2"
    public = public_s2_items([item, private])
    assert len(public) == 1
    assert "red-jp2" in item.assets and "red-jp2" not in public[0].assets
    parsed = next(iter(parse_items(public)))
    assert parsed.bands[("red", 1)].uri == "https://example.com/B04.tif"


def test_saved_field_and_season_settings_survive_reopening(service_client, monkeypatch, tmp_path):
    """Поля с одинаковыми названиями и заметки разных сезонов не перезаписывают друг друга."""
    from service import field_store
    monkeypatch.setattr(field_store, "ROOT", tmp_path)
    for pid in ("FIELD-A", "FIELD-B"):
        field_store.save_report({"pid": pid, "name": "Одинаковое имя", "years": {"2024": {}, "2025": {}},
                                 "episodes": [], "_weather": []})
    assert len(service_client.get("/api/saved-fields").json()) == 2
    assert "_weather" not in service_client.get("/api/polygon/FIELD-A").json()
    settings = {"year": 2025, "profile": "custom", "base": 10, "sowing_date": "2025-04-01",
                "review_status": "На проверке", "comment": "Проверить полив"}
    assert service_client.post("/api/polygon/FIELD-A/agro", json=settings).status_code == 200
    saved = service_client.get("/api/polygon/FIELD-A/agro?year=2025").json()
    assert saved["settings"]["comment"] == "Проверить полив"
    assert not saved["heat"]["available"]
    assert service_client.get("/api/polygon/FIELD-A/agro?year=2024").json()["settings"] == {}
    assert service_client.get("/api/polygon/FIELD-B/agro?year=2025").json()["settings"] == {}
    assert service_client.post("/api/polygon/FIELD-A/agro", json=settings | {"sowing_date": "2024-04-01"}).status_code == 422


def test_imagery_cache_and_year_validation(service_client, monkeypatch, tmp_path):
    """Повторный сбор использует кэш; отсутствующий сезон не запускает сетевую работу."""
    from service import field_store, imagery
    monkeypatch.setattr(field_store, "ROOT", tmp_path / "fields")
    monkeypatch.setattr(imagery, "ROOT", tmp_path / "imagery")
    field_store.save_report({"pid": "FIELD-A", "years": {"2025": {}}, "episodes": [],
                             "geometry": {"type": "Polygon", "coordinates": [[[39, 47], [39.01, 47], [39.01, 47.01], [39, 47]]]}})
    assert not service_client.get("/api/polygon/FIELD-A/imagery?year=2025").json()["available"]
    assert service_client.post("/api/polygon/FIELD-A/imagery?year=1900").status_code == 400
    assert service_client.post("/api/polygon/AOI-0001/imagery?year=2025").status_code == 400
    manifest = {"pid": "FIELD-A", "year": 2025, "generation": "test", "scenes": []}
    field_store._write(imagery.directory("FIELD-A", 2025) / "manifest.json", manifest)
    assert service_client.post("/api/polygon/FIELD-A/imagery?year=2025").json()["manifest"] == manifest
    response = service_client.get("/api/polygon/FIELD-A")
    assert response.json()["insights"]["2025"]["status"] == "insufficient"
    assert service_client.get("/api/polygon/FIELD-A/imagery/2025/2025-10-01/invalid.png").status_code == 404


@pytest.mark.parametrize("payload", [
    {"geometry": {"type": "Point", "coordinates": [39, 47]}},
    {"geometry": {"type": "Polygon", "coordinates": [[[39, 47], [40, 47], [40, 48], [39, 47]]]},
     "start_year": 2025, "end_year": 2019},
])
def test_invalid_collection_inputs_fail_before_collection(service_client, payload):
    """Неверный объект/интервал не попадает в процесс сборщика."""
    assert service_client.post("/api/analyze", json=payload).status_code == 422
