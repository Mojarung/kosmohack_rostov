"""Преобразование исходной погоды в данные дополнительных графиков."""


def legacy_weather_records(report: dict) -> list[dict]:
    """Существующие AOI дают только температуру и осадки, без выдуманной ET₀/Tmin/Tmax."""
    records = []
    for values in report.get("weather", {}).values():
        records.extend({"date": date, "era5_temp_c": temp, "era5_precip_mm": precip}
                       for date, temp, precip in zip(values["date"], values["temp"], values["precip"]))
    return records
