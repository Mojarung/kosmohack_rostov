"""Тесты HTML-отчёта по полю: самодостаточность файла, экранирование и светофор сезона.

Отчёт должен открываться без интернета, поэтому в разметке не должно быть ни одной внешней ссылки.
"""

import pytest

from service.facts import field_facts
from service.report_html import (
    _episode_block,
    _seasons_table,
    build_report,
    season_chart,
    season_status,
)


def _season(year: int) -> dict:
    """Сезон с кривой и нормой из восьми точек: этого хватает на график."""
    def points(base: float) -> list[dict]:
        return [{"date": f"{year}-06-{1 + 3 * i:02d}", "value": round(base + 0.05 * i, 3)} for i in range(8)]

    return {"norm_source": "история поля",
            "observations": [{"date": f"{year}-06-01", "value": 0.31, "sensor": "Sentinel-2", "artifact": False}],
            "restored": [], "curve": points(0.20), "norm_mean": points(0.25),
            "z": points(-1.5)}


def _episode(year: int, severity: str, cause: str) -> dict:
    return {"pid": "AOI-0001", "year": year, "start": f"{year}-06-04", "end": f"{year}-06-16",
            "days": 12, "n_obs": 4, "severity": severity, "min_z": -2.5, "mean_z": -2.0,
            "worst_date": f"{year}-06-10", "ndvi_at_worst": 0.21, "norm_at_worst": 0.51,
            "phase": "налив", "cause": cause, "confidence": 0.78, "reasons": ["осадки ниже нормы"]}


DETAIL = {
    "pid": "AOI-0001", "name": "Поле у балки", "crop": "озимая пшеница",
    "years": {2023: _season(2023), 2024: _season(2024)},
    "episodes": [_episode(2024, "критическая", "weather_drought"),
                 _episode(2023, "умеренная", "late_start")],
    "weather": {2024: {"date": ["2024-06-01", "2024-06-02"], "precip": [12.0, 3.0], "temp": [20.0, 24.0]}},
    "collected": "S2 120 сцен, Landsat 40 сцен",
}
META = {"task1": {"rmse_val": 0.054, "baseline_rmse": 0.093}}


@pytest.fixture
def report() -> str:
    return build_report(DETAIL, year=2024, meta=META, generated_on="2026-09-05")


def test_report_is_valid_self_contained_html(report):
    """Готовый файл — цельный HTML без единой внешней ссылки, со встроенным графиком."""
    assert report.startswith("<!doctype html>") and report.rstrip().endswith("</html>")
    assert "http://" not in report and "https://" not in report
    assert "<svg" in report and "</svg>" in report
    assert report.count("<style>") == 1 and "<script" not in report.lower()


def test_report_shows_field_and_season(report):
    """В отчёте есть имя поля, культура, дата формирования и разделы по сезону."""
    assert "Поле у балки" in report and "озимая пшеница" in report
    assert "отчёт от 2026-09-05" in report and "сезоны 2023–2024" in report
    assert "Сезон 2024: критическое" in report
    assert "Как шёл сезон 2024" in report and "Все сезоны" in report
    assert "S2 120 сцен" in report                     # пометка о сборе данных
    assert "0.054" in report and "0.093" in report      # метрики из meta


def test_report_lists_episodes_of_chosen_season(report):
    """Показываются периоды выбранного сезона, а не всей истории поля."""
    assert report.count('class="episode"') == 1
    assert "2024-06-04 — 2024-06-16" in report and "погодный стресс" in report
    assert "2023-06-04" not in report
    спокойный = build_report(DETAIL, year=2023, meta=META, generated_on="2026-09-05")
    assert "Сезон 2023: требует внимания" in спокойный and "поздний старт" in спокойный


@pytest.mark.parametrize("name, forbidden, expected", [
    ("<script>alert(1)</script>", "<script", "&lt;script&gt;alert(1)&lt;/script&gt;"),
    ('Поле "Южное" & Ко', '"Южное"', "&quot;Южное&quot; &amp; Ко"),
    ("<b onclick='x'>поле</b>", "<b ", "&lt;b onclick=&#x27;x&#x27;&gt;"),
])
def test_report_escapes_field_name(name, forbidden, expected):
    """Имя поля вводит пользователь: разметка из него в отчёт попадать не должна."""
    report = build_report(DETAIL | {"name": name}, year=2024, generated_on="2026-09-05")
    assert forbidden not in report and expected in report


def test_report_of_field_without_seasons():
    """Поле без сезонов даёт корректный отчёт с объяснением, а не ошибку."""
    report = build_report({"pid": "AOI-9999"}, generated_on="2026-09-05")
    assert report.startswith("<!doctype html>") and "AOI-9999" in report
    assert "Данных за сезон недостаточно." in report
    assert "недостаточно наблюдений, чтобы построить график" in report


@pytest.mark.parametrize("year, status", [(2024, "критическое"), (2023, "требует внимания"), (2022, "в норме")])
def test_season_status_traffic_light(year, status):
    """Светофор сезона: тяжёлый эпизод — критическое, любой другой — внимание, иначе норма."""
    facts = field_facts(DETAIL)
    assert season_status(facts, year) == status


def test_season_status_without_episodes():
    """Пустая выжимка не роняет светофор."""
    assert season_status({}, 2024) == "в норме"
    assert season_status({"эпизоды": []}, 2024) == "в норме"


def test_season_chart_without_points_returns_text():
    """Сезон без наблюдений даёт понятный текст, а не пустой или сломанный SVG."""
    for season in ({}, {"curve": []}, {"curve": [{"date": "2024-06-01", "value": 0.4}]}):
        chart = season_chart(season, [])
        assert "<svg" not in chart and "недостаточно наблюдений" in chart


def test_season_chart_draws_curve_norm_and_bands():
    """На графике есть кривая, пунктир нормы и полоса периода снижения нужного цвета."""
    facts = field_facts(DETAIL, 2024)
    chart = season_chart(DETAIL["years"][2024], facts["эпизоды"][:1])
    assert chart.startswith("<svg") and chart.endswith("</svg>")
    assert "stroke-dasharray" in chart and chart.count("<path") == 2
    assert '<rect' in chart and "#c8423f" in chart      # цвет критической тяжести
    assert "aria-label" in chart and "06-01" in chart   # подписи оси


def test_season_chart_survives_missing_norm():
    """Без нормы график всё равно строится — только без пунктира."""
    season = {"curve": DETAIL["years"][2024]["curve"]}
    chart = season_chart(season, [])
    assert chart.startswith("<svg") and "stroke-dasharray" not in chart


def test_episode_block_explains_what_to_check():
    """К каждой причине даётся своя подсказка, к незнакомой — общая."""
    facts = field_facts(DETAIL, 2024)
    block = _episode_block(facts["эпизоды"][0])
    assert "сверьте с записями по осадкам и поливу" in block
    assert "погодный стресс" in block and "критическая" in block and "#c8423f" in block
    assert "2024-06-10" in block and "0.21" in block
    чужая = _episode_block({"причина": "нашествие марсиан", "тяжесть": "умеренная"})
    assert "сверьте период с записями по полю" in чужая and "#e8a33d" in чужая


def test_episode_block_escapes_values():
    """Значения эпизода тоже экранируются: они приходят из данных, а не из кода."""
    block = _episode_block({"период": "<b>2024</b>", "причина": "погодный стресс"})
    assert "<b>2024</b>" not in block and "&lt;b&gt;2024&lt;/b&gt;" in block


def test_seasons_table_has_row_per_season():
    """Таблица сезонов: строка на год, статус и осадки, прочерк вместо пустой погоды."""
    table = _seasons_table(field_facts(DETAIL))
    assert table.count("<tr>") == 3                    # заголовок и два сезона
    assert ">2023<" in table and ">2024<" in table
    assert "критическое" in table and "требует внимания" in table
    assert ">15.0<" in table and ">—<" in table         # осадки 2024 и прочерк 2023
    assert _seasons_table({}).count("<tr>") == 1        # без сезонов остаётся только заголовок
