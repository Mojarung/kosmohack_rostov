"""Тесты слоя сервиса без сети: набор полигонов пользователя."""

from service.polygons import delete, list_saved, load, save, summary_of

RESULT = {"pid": "NEW:поле", "geometry": {"type": "Polygon", "coordinates": [[[39.8, 47.3], [39.9, 47.3], [39.9, 47.4], [39.8, 47.3]]]},
          "years": {"2023": {}, "2024": {}},
          "episodes": [{"year": 2023, "severity": "умеренная"}, {"year": 2024, "severity": "критическая"}]}


def test_summary_reports_last_season_status():
    s = summary_of(RESULT)
    assert s["years"] == [2023, 2024] and s["n_episodes"] == 2 and s["n_critical"] == 1
    assert s["last_year"] == 2024 and s["last_year_status"] == "критическая"
    assert summary_of({"years": {}, "episodes": []})["last_year_status"] == "норма"


def test_save_list_load_delete_roundtrip(tmp_path):
    entry = save(RESULT, "моё поле", base=tmp_path)
    assert entry["name"] == "моё поле" and entry["geometry"] == RESULT["geometry"]
    listed = list_saved(base=tmp_path)
    assert [e["uid"] for e in listed] == [entry["uid"]]
    full = load(entry["uid"], base=tmp_path)
    assert full["pid"] == "NEW:поле" and full["name"] == "моё поле" and len(full["episodes"]) == 2
    assert load("../etc/passwd", base=tmp_path) is None and not delete("zzz", base=tmp_path)
    assert delete(entry["uid"], base=tmp_path) and list_saved(base=tmp_path) == []


def test_user_polygon_endpoints(tmp_path, monkeypatch):
    """REST-обёртка набора полигонов: список, открытие, удаление, 404 для чужого идентификатора."""
    from fastapi.testclient import TestClient

    import service.polygons as store
    from service import field_store
    from service.app import app

    monkeypatch.setattr(store, "POLYGONS_DIR", tmp_path)
    monkeypatch.setattr(field_store, "ROOT", tmp_path / "fields")
    client = TestClient(app)
    assert client.get("/api/user-polygons").json() == []
    entry = store.save(RESULT, "поле у дороги")
    listed = client.get("/api/user-polygons").json()
    assert len(listed) == 1 and listed[0]["name"] == "поле у дороги" and listed[0]["last_year_status"] == "критическая"
    full = client.get(f"/api/user-polygons/{entry['uid']}").json()
    assert full["pid"] == "NEW:поле" and full["geometry"] == RESULT["geometry"]
    assert client.get("/api/user-polygons/000000000000").status_code == 404
    assert client.delete(f"/api/user-polygons/{entry['uid']}").json() == {"deleted": entry["uid"]}
    assert client.delete(f"/api/user-polygons/{entry['uid']}").status_code == 404


def test_legacy_fields_visible_with_weather_and_no_duplicates(tmp_path, monkeypatch):
    """Старое поле открывается в React со снимками по прежнему pid, без повторного сбора."""
    from fastapi.testclient import TestClient

    from service import field_store, polygons
    from service.app import app
    monkeypatch.setattr(polygons, "POLYGONS_DIR", tmp_path / "polygons")
    monkeypatch.setattr(field_store, "ROOT", tmp_path / "fields")
    pid = field_store.field_id(RESULT["geometry"])
    report = RESULT | {"pid": pid, "name": "Поле из прежнего интерфейса", "_weather": []}
    field_store.save_report(report)
    client = TestClient(app)
    listed = client.get("/api/user-polygons").json()
    assert len(listed) == 1 and listed[0]["uid"] == pid
    opened = client.get(f"/api/user-polygons/{pid}").json()
    assert opened["pid"] == pid and "insights" in opened and "_weather" not in opened
    assert client.get(f"/api/polygon/{pid}/agro?year=2024").status_code == 200
    entry = polygons.save(report, "То же поле")
    assert len(client.get("/api/user-polygons").json()) == 1
    assert client.delete(f"/api/user-polygons/{entry['uid']}").status_code == 200
    assert client.get("/api/user-polygons").json() == []
    field_store.save_report(report)
    assert client.delete(f"/api/user-polygons/{pid}").status_code == 200
    assert client.get("/api/user-polygons").json() == []
