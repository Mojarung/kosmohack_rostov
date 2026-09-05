"""Тесты выжимки фактов о поле: ключи сезонов, прореживание рядов, подписи причин.

Данные кейса с диска здесь не читаются: все входы собираются вручную в том же виде,
в каком их отдают service.data.Store (целые ключи годов) и service.field_store (строковые).
"""

from service.facts import (
    CAUSE_LABEL,
    MAX_EPISODES,
    MAX_POINTS,
    _by_year,
    _thin,
    episode_facts,
    field_facts,
    method_facts,
    season_facts,
)


def _points(year: int, values: tuple[float, ...]) -> list[dict]:
    """Ряд «дата — значение» с шагом в двое суток: тот же вид, что в service.data."""
    return [{"date": f"{year}-06-{1 + 2 * i:02d}", "value": v} for i, v in enumerate(values)]


def _season(year: int) -> dict:
    """Один сезон: два наблюдения (одно бракованное), восстановленная точка, кривая, норма и Z."""
    return {
        "norm_source": "история поля",
        "observations": [
            {"date": f"{year}-06-01", "value": 0.31, "sensor": "Sentinel-2", "artifact": False},
            {"date": f"{year}-06-05", "value": 0.99, "sensor": "MODIS", "artifact": True},
        ],
        "restored": [{"date": f"{year}-07-01", "value": 0.52}],
        "curve": _points(year, (0.30, 0.60, 0.50)),
        "norm_mean": _points(year, (0.35, 0.65, 0.55)),
        "z": _points(year, (-0.40, -1.20, -0.60)),
    }


EPISODE = {
    "pid": "AOI-0001", "year": 2024, "start": "2024-06-01", "end": "2024-06-21", "days": 21,
    "n_obs": 5, "severity": "критическая", "min_z": -2.512345, "mean_z": -1.987654,
    "worst_date": "2024-06-11", "ndvi_at_worst": 0.211111, "norm_at_worst": 0.512345,
    "phase": "налив", "cause": "weather_drought", "confidence": 0.777777,
    "reasons": ["осадки ниже нормы"], "region_share_depressed": 0.123456, "region_z": -0.567891,
    "norm_source": "история поля",
}

WEATHER = {"date": ["2024-06-01", "2024-06-02", "2024-06-03"],
           "precip": [1.5, None, 2.5], "temp": [20.0, 22.0, 24.0]}


def test_empty_field_has_no_seasons_and_no_series():
    """Поле без сезонов не роняет выжимку и не выдумывает выбранный год."""
    facts = field_facts({"pid": "AOI-0000", "crop": "пшеница"})
    assert facts["поле"] == "AOI-0000" and facts["идентификатор"] == "AOI-0000"
    assert facts["сезоны"] == [] and facts["эпизоды"] == [] and facts["сводка_по_сезонам"] == []
    assert facts["всего_эпизодов"] == 0 and facts["критических_эпизодов"] == 0
    assert "выбранный_сезон" not in facts and "ряд_выбранного_сезона" not in facts


def test_by_year_understands_int_and_str_keys():
    """Store отдаёт годы числами, field_store — строками из JSON; читаться должны оба."""
    assert _by_year({2024: {"a": 1}}, 2024) == {"a": 1}
    assert _by_year({"2024": {"a": 1}}, 2024) == {"a": 1}
    assert _by_year({2024: {"a": 1}}, 2023) is None
    assert _by_year(None, 2024) is None and _by_year({}, 2024) is None


def test_field_facts_same_for_int_and_str_year_keys():
    """Выжимка по полю не зависит от того, каким пришёл ключ года."""
    seasons = {2023: _season(2023), 2024: _season(2024)}
    from_store = field_facts({"pid": "AOI-0001", "years": seasons, "episodes": [EPISODE]})
    from_json = field_facts({"pid": "AOI-0001", "episodes": [EPISODE],
                             "years": {str(y): s for y, s in seasons.items()}})
    assert from_store == from_json
    assert from_store["сезоны"] == [2023, 2024] and from_store["выбранный_сезон"] == 2024


def test_field_facts_picks_requested_or_last_season():
    """Запрошенный сезон берётся как есть, несуществующий — заменяется последним."""
    detail = {"pid": "AOI-0001", "years": {2023: _season(2023), 2024: _season(2024)}}
    assert field_facts(detail, 2023)["выбранный_сезон"] == 2023
    assert field_facts(detail, 1999)["выбранный_сезон"] == 2024
    assert field_facts(detail)["выбранный_сезон"] == 2024


def test_thin_keeps_first_and_last_point_within_limit():
    """Прореживание не длиннее лимита и сохраняет края ряда: сезон не обрезается посередине."""
    points = [{"i": i} for i in range(137)]
    thinned = _thin(points)
    assert len(thinned) == MAX_POINTS
    assert thinned[0] is points[0] and thinned[-1] is points[-1]

    short = points[:MAX_POINTS]
    assert _thin(short) is short                       # короткий ряд возвращается без копирования

    tiny = _thin(points, limit=3)
    assert len(tiny) == 3 and tiny[0] is points[0] and tiny[-1] is points[-1]


def test_episode_facts_translates_cause_and_keeps_code():
    """Причина показывается словами, а исходный код остаётся в поле «код_причины»."""
    facts = episode_facts(EPISODE)
    assert facts["причина"] == CAUSE_LABEL["weather_drought"] == "погодный стресс"
    assert facts["код_причины"] == "weather_drought"
    assert facts["период"] == "2024-06-01 — 2024-06-21" and facts["дней"] == 21
    assert facts["минимум_отклонения_сигм"] == -2.51 and facts["среднее_отклонения_сигм"] == -1.99
    assert facts["ndvi_в_худший_день"] == 0.211 and facts["норма_в_худший_день"] == 0.512
    assert facts["уверенность"] == 0.78 and facts["соседние_поля_ниже_нормы_доля"] == 0.12
    assert facts["аргументы"] == ["осадки ниже нормы"]


def test_episode_facts_survives_unknown_cause_and_gaps():
    """Незнакомый код причины и пустой эпизод не роняют выжимку."""
    unknown = episode_facts(EPISODE | {"cause": "нет_такого_кода"})
    assert unknown["причина"] == unknown["код_причины"] == "нет_такого_кода"
    empty = episode_facts({})
    assert empty["период"] == "None — None" and empty["причина"] is None
    assert empty["минимум_отклонения_сигм"] is None


def test_season_facts_without_weather():
    """Без метеоряда сезон описывается только снимками, погодных ключей в ответе нет."""
    facts = season_facts(2024, _season(2024))
    assert facts["год"] == 2024 and facts["наблюдений"] == 2 and facts["восстановлено_точек"] == 1
    assert facts["источник_нормы"] == "история поля"
    assert facts["пик_ndvi"] == 0.6 and facts["дата_пика"] == "2024-06-03"
    assert facts["худшее_отклонение_сигм"] == -1.2 and facts["дата_худшего_отклонения"] == "2024-06-03"
    assert facts["пик_нормы"] == 0.65 and facts["дата_пика_нормы"] == "2024-06-03"
    assert "осадки_за_сезон_мм" not in facts and "дней_погоды" not in facts


def test_season_facts_with_weather():
    """С метеорядом добавляются сумма осадков, средняя температура и число дней; пропуски отбрасываются."""
    facts = season_facts(2024, _season(2024), WEATHER)
    assert facts["осадки_за_сезон_мм"] == 4.0 and facts["средняя_температура_c"] == 22.0
    assert facts["дней_погоды"] == 3
    без_чисел = season_facts(2024, _season(2024), {"date": ["2024-06-01"], "precip": [None], "temp": []})
    assert без_чисел["осадки_за_сезон_мм"] is None and без_чисел["средняя_температура_c"] is None
    assert "осадки_за_сезон_мм" not in season_facts(2024, _season(2024), {"date": []})


def test_season_facts_of_empty_season():
    """Сезон без кривой и наблюдений даёт нули и None, а не исключение."""
    facts = season_facts(2020, {})
    assert facts["наблюдений"] == 0 and facts["восстановлено_точек"] == 0
    assert facts["пик_ndvi"] is None and facts["худшее_отклонение_сигм"] is None


def test_field_facts_sorts_episodes_and_hides_artifacts():
    """Тяжёлые эпизоды идут первыми, бракованные наблюдения в ряд не попадают."""
    episodes = [
        EPISODE | {"severity": "умеренная", "min_z": -3.0, "start": "2023-05-01", "year": 2023},
        EPISODE | {"severity": "критическая", "min_z": -1.0, "start": "2024-07-01"},
        EPISODE | {"severity": "критическая", "min_z": -2.5},
    ]
    facts = field_facts({"pid": "AOI-0001", "years": {2024: _season(2024)}, "episodes": episodes}, 2024)
    assert facts["всего_эпизодов"] == 3 and facts["критических_эпизодов"] == 2
    assert [e["минимум_отклонения_сигм"] for e in facts["эпизоды"]] == [-2.5, -1.0, -3.0]

    series = facts["ряд_выбранного_сезона"]
    assert [o["дата"] for o in series["наблюдения"]] == ["2024-06-01"]      # artifact=True отброшен
    assert series["наблюдения"][0]["спутник"] == "Sentinel-2"
    assert len(series["восстановленная_кривая"]) == 3 and len(series["норма"]) == 3


def test_field_facts_limits_episode_list():
    """В контекст модели уходит не больше MAX_EPISODES эпизодов, счётчик остаётся полным."""
    facts = field_facts({"pid": "AOI-0001", "episodes": [EPISODE] * (MAX_EPISODES + 5)})
    assert facts["всего_эпизодов"] == MAX_EPISODES + 5 and len(facts["эпизоды"]) == MAX_EPISODES


def test_field_facts_keeps_name_and_collection_notes():
    """Имя поля пользователя важнее идентификатора; пометки сбора попадают в выжимку."""
    facts = field_facts({"pid": "NEW:1", "name": "Поле у балки", "crop": "подсолнечник",
                         "kind": "новая территория", "collected": "S2 120 сцен",
                         "weather_source": "ERA5 (Open-Meteo)"})
    assert facts["поле"] == "Поле у балки" and facts["идентификатор"] == "NEW:1"
    assert facts["культура"] == "подсолнечник" and facts["происхождение"] == "новая территория"
    assert facts["собрано"] == "S2 120 сцен" and facts["источник_погоды"] == "ERA5 (Open-Meteo)"


def test_season_summary_uses_weather_by_year():
    """Погода подтягивается к своему сезону по тем же правилам ключей, что и сезоны."""
    detail = {"pid": "AOI-0001", "years": {"2024": _season(2024), "2023": _season(2023)},
              "weather": {"2024": WEATHER}}
    summary = {s["год"]: s for s in field_facts(detail)["сводка_по_сезонам"]}
    assert summary[2024]["осадки_за_сезон_мм"] == 4.0
    assert "осадки_за_сезон_мм" not in summary[2023]


def test_method_facts_reads_metrics_and_sources():
    """Метод описывается числами из meta; без meta блок остаётся пустым, но валидным."""
    meta = {
        "task1": {"rmse_val": 0.054321, "rmse_model_only": 0.056789, "gap_score": 13.71,
                  "baseline_rmse": 0.09341, "n_gaps": 2323, "models": "LightGBM + SeasonNet"},
        "task2": {"n_polygons": 52, "n_seasons": 610, "n_episodes": 318,
                  "seasons_with_episode": 0.4123, "by_cause": {"weather_drought": 90}},
        "sources": [{"name": "ERA5", "detail": "Open-Meteo, суточные ряды"}],
    }
    facts = method_facts(meta)
    task1 = facts["восстановление_пропусков"]
    assert task1["rmse"] == 0.0543 and task1["rmse_без_калибровки"] == 0.0568
    assert task1["gap_score"] == 13.71 and task1["rmse_baseline_среднее_соседей"] == 0.093
    assert task1["контрольных_точек"] == 2323 and task1["модель"] == "LightGBM + SeasonNet"
    assert "отложенная выборка" in task1["оговорка"]
    assert facts["поиск_эпизодов"]["доля_сезонов_с_эпизодом"] == 0.41
    assert facts["источники"] == ["ERA5: Open-Meteo, суточные ряды"]

    пустой = method_facts({})
    assert пустой["источники"] == [] and пустой["восстановление_пропусков"]["rmse"] is None


def test_cause_dictionaries_cover_the_same_codes():
    """Классы причин описаны в пяти местах (детектор, факты, три файла интерфейса).

    Общего словаря у Python и TypeScript быть не может, поэтому расхождение ловится здесь:
    новый класс причины, забытый в интерфейсе, показывался бы пользователю как сырой код.
    """
    from pathlib import Path

    from anomaly.report import CAUSE_TEXT

    assert set(CAUSE_LABEL) == set(CAUSE_TEXT)
    web = Path(__file__).resolve().parents[1] / "web" / "src" / "lib"
    for name in ("plain.ts", "format.ts"):
        text = (web / name).read_text(encoding="utf-8")
        missing = [code for code in CAUSE_TEXT if code not in text]
        assert not missing, f"в {name} нет причин: {missing}"
