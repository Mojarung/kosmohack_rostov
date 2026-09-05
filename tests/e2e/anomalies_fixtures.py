"""Небольшие отчёты для изолированного прогона UI; не записываются в данные пользователя."""
from copy import deepcopy

from service import field_store, polygons


def season(year, assessed=True):
    points = [{"date": f"{year}-{month:02}-15", "value": value}
              for month, value in [(4, .2), (5, .4), (6, .55), (7, .3), (8, .4), (9, .3), (10, .2)]]
    return {"norm_source": "история поля", "curve": points, "norm_mean": points,
            "norm_std": [p | {"value": .1} for p in points],
            "z": [p | {"value": 0 if assessed else None} for p in points], "restored": [],
            "observations": [p | {"harmonized": p["value"], "sensor": "Sentinel-2", "artifact": False} for p in points]}


def episode(year=2024, cause="weather_drought", start="06-01", end="06-30", severity="критическая"):
    return {"pid": "NEW:south", "year": year, "start": f"{year}-{start}", "end": f"{year}-{end}",
            "days": 30, "min_z": -2.1, "mean_z": -1.5, "critical_days": 5, "n_obs": 5,
            "cause": cause, "severity": severity, "confidence": .7, "norm_source": "история поля",
            "text": "Тестовый эпизод снижения зелёности", "reasons": "мало осадков"}


def seed_saved():
    report = {"pid": "NEW:south", "name": "Южное", "crop": "пшеница", "kind": "новая территория",
              "years": {"2024": season(2024), "2025": season(2025)}, "geometry": None,
              "weather": {}, "_weather": [], "shape": [],
              "episodes": [episode(), episode(cause="early_decline", start="08-01", end="08-30", severity="умеренная")]}
    entries = [polygons.save(report, "Южное")]
    # Отчёт есть в обоих хранилищах, но в общем списке должен появиться один раз.
    field_store.save_report(report)
    river = deepcopy(report)
    river.update(pid="NEW:river", name="У реки", episodes=[episode(2025, "data_suspect")])
    entries.append(polygons.save(river, "У реки"))
    north = deepcopy(report)
    north.update(pid="NEW:north", name="Северное", years={"2024": season(2024, False)}, episodes=[])
    entries.append(polygons.save(north, "Северное"))
    return entries
