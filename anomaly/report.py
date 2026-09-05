"""Интерпретация эпизода: причина по правилам и текст на русском из структурированных фактов.

Классы причин (по ТЗ нужно отделять погодный стресс от облаков, смены фазы и ошибок данных):
  weather_drought   — дефицит осадков и/или жара до и во время эпизода;
  unsown_or_changed — с начала сезона кривая плоская, пик много ниже нормы: поле не засеяно / пар / другая культура;
  early_decline     — сезон стартовал нормально, спад начался заметно раньше нормы (ранняя уборка, полегание, болезнь);
  weak_season       — сезон в целом слабее нормы без явного погодного сигнала (агротехника, почва, вредители);
  late_start        — всходы и рост запаздывают относительно нормы;
  crop_rotation     — сезон начался поздно, но пик достигнут позже и не ниже нормы: на поле яровая культура вместо
                      привычной озимой (смена фазы, а не угнетение);
  data_suspect      — эпизод опирается на мало наблюдений или окружён артефактами; вероятна ошибка данных.
Региональный контекст: если другие поля региона в те же дни тоже ниже нормы — явление региональное (погода),
если соседи в норме — причина локальная (агротехника, поле).
"""

from __future__ import annotations

from anomaly.config import EARLY_DECLINE_DAYS, FLAT_PEAK_NDVI, LOW_PEAK_RATIO, PRECIP_DEFICIT_PCT, TEMP_ANOMALY_C


def plural(n: int, one: str, few: str, many: str) -> str:
    """Русское склонение существительного при числе: 1 день, 2 дня, 15 дней, 62 дня."""
    n = abs(int(n))
    if n % 10 == 1 and n % 100 != 11:
        return one
    if 2 <= n % 10 <= 4 and not 12 <= n % 100 <= 14:
        return few
    return many


def days(n: int) -> str:
    """«N дней» с правильным склонением."""
    return f"{int(n)} {plural(n, 'день', 'дня', 'дней')}"


SEVERITY = {"критическая": -2.0, "умеренная": -1.0}
REGION_LOW_Z = -0.7          # медианный Z соседей ниже — явление региональное
REGION_NORMAL_Z = -0.3       # выше — соседи в норме, причина локальная
ROTATION_PEAK_SHIFT = 25     # пик позже нормы на столько дней при нормальной высоте — другая (яровая) культура


def severity_label(ep: dict) -> str:
    """Тяжесть эпизода по минимальному Z и доле критических дней."""
    if ep["min_z"] < -2 and ep["critical_days"] >= 5:
        return "критическая"
    return "умеренная"


MIN_WEATHER_DAYS = 20   # решение о погодном стрессе принимается по окну не короче этого
LONG_DRY_SPELL_DAYS = 20  # сухая серия (< 1 мм/день) такой длины считается погодным стрессом сама по себе


def _weather_signal(weather: dict) -> tuple[bool, list[str]]:
    """Есть ли погодный стресс: решение по объединённому окну (30 дней до + эпизод), детали — по частям."""
    notes, stress = [], False
    comb = weather.get("combined") or {}
    if "precip_deficit_pct" in comb and comb.get("n_days", 0) >= MIN_WEATHER_DAYS:
        dry = comb["precip_deficit_pct"] >= PRECIP_DEFICIT_PCT
        hot = comb["temp_anomaly_c"] >= TEMP_ANOMALY_C and comb["hot_days"] >= 5
        long_dry_spell = comb.get("max_dry_spell", 0) >= LONG_DRY_SPELL_DAYS   # затяжная сухая серия — тоже стресс
        stress = dry or hot or long_dry_spell
        if dry:
            notes.append(f"за 30 дней до и во время эпизода осадков {comb['precip_mm']:.0f} мм при норме "
                         f"{comb['precip_norm_mm']:.0f} мм (дефицит {comb['precip_deficit_pct']:.0f} %)")
        if hot:
            notes.append(f"температура выше нормы на {comb['temp_anomaly_c']:.1f} °C ({comb['temp_c']:.1f} против "
                         f"{comb['temp_norm_c']:.1f} °C), {days(comb['hot_days'])} с жарой")
        if comb.get("max_dry_spell", 0) >= 15:
            notes.append(f"сухая серия {days(comb['max_dry_spell'])} подряд")
    for name, label in (("before", "за 30 дней до эпизода"), ("during", "во время эпизода")):
        b = weather.get(name)
        if stress and b and "precip_deficit_pct" in b and abs(b["precip_deficit_pct"]) >= PRECIP_DEFICIT_PCT:
            notes.append(f"{label}: осадков {b['precip_mm']:.0f} мм при норме {b['precip_norm_mm']:.0f} мм")
    return stress, notes


def _region_note(region: dict) -> tuple[str, list[str]]:
    """Региональный контекст: 'regional', 'local' или 'unknown' и формулировка."""
    if not region.get("available"):
        return "unknown", []
    z, share, n = region["region_z"], region["share_polygons_depressed"], region["n_polygons"]
    if z <= REGION_LOW_Z or share >= 0.5:
        return "regional", [f"соседние поля региона в эти дни тоже ниже нормы (медианный Z {z:.1f}, "
                            f"ниже нормы {share:.0%} из {n} полей) — явление региональное"]
    if z >= REGION_NORMAL_Z:
        return "local", [f"другие поля региона в эти дни в норме (медианный Z {z:.1f}, ниже нормы {share:.0%} из {n}) — "
                         "причина локальная, на самом поле"]
    return "mixed", [f"соседние поля региона в эти дни чуть ниже нормы (медианный Z {z:.1f})"]


def classify(ep: dict, pheno_dev: dict, pheno: dict, weather: dict, artifacts_near: int,
             region: dict | None = None) -> tuple[str, float, list[str]]:
    """Причина эпизода, уверенность 0–1 и список аргументов."""
    stress, notes = _weather_signal(weather)
    scope, rnotes = _region_note(region or {})
    context = rnotes + notes            # региональный контекст, затем погода — после главного аргумента причины
    peak_ratio = pheno_dev.get("peak_ratio")
    if ep["n_obs"] <= 2 and artifacts_near >= 2:
        obs_txt = f"{ep['n_obs']} {plural(ep['n_obs'], 'наблюдение', 'наблюдения', 'наблюдений')}"
        art_txt = f"{artifacts_near} {plural(artifacts_near, 'артефакт', 'артефакта', 'артефактов')}"
        return "data_suspect", 0.4, [f"внутри эпизода всего {obs_txt}, рядом {art_txt}"] + context
    if (peak_ratio is not None and peak_ratio >= 0.85 and pheno_dev.get("peak_shift_days", 0) >= ROTATION_PEAK_SHIFT
            and ep["end_doy"] <= 200):
        return "crop_rotation", 0.7, [f"пик достигнут ({peak_ratio:.0%} нормы), но на {days(pheno_dev['peak_shift_days'])} "
                                      "позже обычного: на поле яровая культура вместо привычной озимой, весеннее "
                                      "отставание — смена фазы, а не угнетение"] + context
    flat = pheno.get("valid") and (pheno["peak"] < FLAT_PEAK_NDVI or (peak_ratio is not None and peak_ratio < 0.6))
    if flat and ep["start_doy"] <= 165:
        norm_peak = pheno["peak"] / max(peak_ratio, 1e-6) if peak_ratio else float("nan")
        return "unsown_or_changed", 0.8, [f"пик главного сезона всего {pheno['peak']:.2f} при норме {norm_peak:.2f}: "
                                          "кривая плоская с весны — поле не засеяно, под паром или занято другой культурой"] + context
    if peak_ratio is not None and peak_ratio >= 0.85 and pheno_dev.get("decline_shift_days", 0) <= -EARLY_DECLINE_DAYS:
        cause = "weather_drought" if stress else "early_decline"
        main = [f"пик на уровне нормы ({peak_ratio:.0%}), но спад начался на {days(-pheno_dev['decline_shift_days'])} раньше обычного"]
        return cause, 0.75 if stress else 0.6, (notes + main + rnotes) if stress else (main + context)
    if pheno_dev.get("sos_shift_days", 0) >= 15 and ep["start_doy"] <= 150 and (peak_ratio or 0) >= LOW_PEAK_RATIO:
        return "late_start", 0.6, [f"рост начался на {days(pheno_dev['sos_shift_days'])} позже нормы, "
                                   f"но пик достигнут ({peak_ratio:.0%} нормы)"] + context
    if stress:
        return "weather_drought", 0.8 if scope == "regional" else 0.65, notes + rnotes
    if scope == "regional":
        return "weather_drought", 0.5, ["явного дефицита осадков в ERA5 нет, но угнетены все поля региона — "
                                        "вероятен региональный погодный фактор (заморозки, суховей, град)"] + context
    if peak_ratio is not None and peak_ratio < LOW_PEAK_RATIO:
        return "weak_season", 0.6 if scope == "local" else 0.55, [f"пик сезона {peak_ratio:.0%} от нормы без явного погодного сигнала"] + context
    return "weak_season", 0.45 if scope == "local" else 0.4, ["погодного сигнала нет; вероятны агротехнические причины"] + context


CAUSE_TEXT = {
    "weather_drought": "погодный стресс: дефицит осадков и/или жара",
    "unsown_or_changed": "поле не засеяно, под паром или сменилась культура",
    "early_decline": "ранний спад вегетации (уборка, полегание, болезнь)",
    "weak_season": "ослабленный сезон без явной погодной причины",
    "late_start": "запаздывание всходов и роста",
    "crop_rotation": "смена культуры: яровая вместо озимой, весеннее отставание — не угнетение",
    "data_suspect": "вероятная ошибка данных (мало наблюдений, артефакты рядом)",
}


def describe(pid: str, year: int, ep: dict, cause: str, confidence: float, reasons: list[str], norm_source: str,
             display_name: str | None = None) -> str:
    """Текст объяснения эпизода на русском из фактов (без LLM).

    `display_name` — имя поля для заголовка (у новых полей пользователя вместо технического FIELD-…);
    по умолчанию используется идентификатор `pid`, как для полигонов кейса AOI-xxxx.
    """
    sev = severity_label(ep)
    obs_txt = f"{ep['n_obs']} {plural(ep['n_obs'], 'наблюдение', 'наблюдения', 'наблюдений')}"
    head = (f"{display_name or pid}, сезон {year}: {sev} аномалия с {ep['start']} по {ep['end']} ({days(ep['days'])}, {obs_txt}), "
            f"фаза — {ep['phase']}. NDVI в худший день {ep['worst_date']}: "
            f"{ep['ndvi_at_worst']:.2f} при норме {ep['norm_at_worst']:.2f} (Z = {ep['min_z']:.1f}).")
    body = f" Вероятная причина — {CAUSE_TEXT[cause]} (уверенность {confidence:.0%})."
    tail = (" Аргументы: " + "; ".join(reasons) + ".") if reasons else ""
    return head + body + tail + f" Норма: {norm_source}."
