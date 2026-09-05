"""Отчёт по полю одним HTML-файлом: для владельца поля и агронома, а не для дата-сайентиста.

Файл самодостаточный: графики нарисованы прямо в SVG, внешних скриптов и картинок нет,
поэтому его можно отправить почтой, открыть без интернета и распечатать в PDF
(в браузере «Печать» → «Сохранить как PDF»; вёрстка под печать задана в @page и @media print).

Числа берутся из service.facts, то есть ровно те же, что показывает интерфейс: отчёт ничего
не пересчитывает и ничего не додумывает.
"""

from __future__ import annotations

import html
from datetime import date
from typing import Any

from service.facts import _by_year, field_facts

# Размеры холста графика в единицах SVG; на печати он масштабируется по ширине страницы.
CHART_W, CHART_H = 900, 260
PAD_L, PAD_R, PAD_T, PAD_B = 46, 14, 14, 26

SEVERITY_COLOR = {"критическая": "#c8423f", "умеренная": "#e8a33d"}
STATUS_COLOR = {"критическое": "#c8423f", "требует внимания": "#e8a33d", "в норме": "#3f7d45"}


def _esc(value: Any) -> str:
    """Экранирование текста: в отчёт попадают названия полей, введённые пользователем."""
    return html.escape("" if value is None else str(value))


def season_status(facts: dict, year: int) -> str:
    """Светофор сезона: критическое, требует внимания или в норме."""
    episodes = [e for e in (facts.get("эпизоды") or []) if e.get("год") == year]
    if any(e.get("тяжесть") == "критическая" for e in episodes):
        return "критическое"
    return "требует внимания" if episodes else "в норме"


def _scale(points: list[dict], key: str, lo: float, hi: float, x0: float, x1: float) -> list[tuple[float, float]]:
    """Перевод точек ряда в координаты SVG. Ось X — порядковый номер дня, ось Y — значение."""
    if not points:
        return []
    n = max(len(points) - 1, 1)
    span = max(hi - lo, 1e-6)
    return [
        (x0 + (x1 - x0) * i / n, PAD_T + (CHART_H - PAD_T - PAD_B) * (1 - (p[key] - lo) / span))
        for i, p in enumerate(points)
    ]


def _path(coords: list[tuple[float, float]]) -> str:
    return " ".join(("M" if i == 0 else "L") + f"{x:.1f},{y:.1f}" for i, (x, y) in enumerate(coords))


def season_chart(season: dict, episodes: list[dict]) -> str:
    """График сезона в SVG: норма полосой, восстановленная кривая линией, периоды снижения заливкой."""
    curve = [{"date": p["date"], "v": p["value"]} for p in (season.get("curve") or []) if p.get("value") is not None]
    norm = [{"date": p["date"], "v": p["value"]} for p in (season.get("norm_mean") or []) if p.get("value") is not None]
    if len(curve) < 2:
        return '<p class="muted">Для этого сезона недостаточно наблюдений, чтобы построить график.</p>'

    x0, x1 = PAD_L, CHART_W - PAD_R
    lo, hi = 0.0, 1.0
    dates = [p["date"] for p in curve]
    curve_xy = _scale(curve, "v", lo, hi, x0, x1)
    norm_xy = _scale(norm, "v", lo, hi, x0, x1) if norm else []

    def x_for(day: str) -> float:
        """Позиция даты на оси: ближайший день ряда."""
        nearest = min(range(len(dates)), key=lambda i: abs(_ordinal(dates[i]) - _ordinal(day)))
        return x0 + (x1 - x0) * nearest / max(len(dates) - 1, 1)

    bands = "".join(
        f'<rect x="{min(x_for(e["период"].split(" — ")[0]), x_for(e["период"].split(" — ")[1])):.1f}" '
        f'y="{PAD_T}" width="{abs(x_for(e["период"].split(" — ")[1]) - x_for(e["период"].split(" — ")[0])):.1f}" '
        f'height="{CHART_H - PAD_T - PAD_B}" fill="{SEVERITY_COLOR.get(e.get("тяжесть"), "#999")}" opacity="0.13"/>'
        for e in episodes
    )
    grid = "".join(
        f'<line x1="{x0}" x2="{x1}" y1="{PAD_T + (CHART_H - PAD_T - PAD_B) * (1 - v):.1f}" '
        f'y2="{PAD_T + (CHART_H - PAD_T - PAD_B) * (1 - v):.1f}" stroke="#e2ded4"/>'
        f'<text x="{x0 - 8}" y="{PAD_T + (CHART_H - PAD_T - PAD_B) * (1 - v) + 4:.1f}" '
        f'text-anchor="end" class="ax">{v:.1f}</text>'
        for v in (0.0, 0.25, 0.5, 0.75, 1.0)
    )
    ticks = "".join(
        f'<text x="{x0 + (x1 - x0) * i / max(len(dates) - 1, 1):.1f}" y="{CHART_H - 6}" '
        f'text-anchor="middle" class="ax">{_esc(dates[i][5:])}</text>'
        for i in range(0, len(dates), max(len(dates) // 6, 1))
    )
    norm_line = f'<path d="{_path(norm_xy)}" fill="none" stroke="#9aa091" stroke-width="2" stroke-dasharray="5 4"/>' if norm_xy else ""
    return (f'<svg viewBox="0 0 {CHART_W} {CHART_H}" class="chart" role="img" '
            f'aria-label="Зелёность поля за сезон">{grid}{bands}{norm_line}'
            f'<path d="{_path(curve_xy)}" fill="none" stroke="#1a1d16" stroke-width="2.4"/>{ticks}</svg>')


def _ordinal(day: str) -> int:
    """Дата вида 2024-06-10 в число дней — для позиционирования по оси."""
    try:
        y, m, d = (int(part) for part in day.split("-"))
        return date(y, m, d).toordinal()
    except (ValueError, AttributeError):
        return 0


def _episode_block(episode: dict) -> str:
    """Один период снижения: что и когда, насколько глубоко, вероятная причина и что проверить."""
    color = SEVERITY_COLOR.get(episode.get("тяжесть"), "#999")
    checks = {
        "погодный стресс": "сверьте с записями по осадкам и поливу за эти даты",
        "не засеяно или другая культура": "проверьте по севообороту, что и когда сеяли на этом поле",
        "севооборот, не угнетение": "проверьте по севообороту: смена культуры сдвигает сроки, урожаю это не мешает",
        "ранний спад": "уточните дату уборки: ранняя уборка выглядит на снимке как спад",
        "поздний старт": "уточните дату сева и всходов",
        "ослабленный сезон": "осмотрите поле и сверьте с историей внесения удобрений",
        "вероятная ошибка данных": "снимков за период мало, значение может быть искажено облаком",
    }
    hint = checks.get(episode.get("причина"), "сверьте период с записями по полю")
    return f"""<article class="episode" style="border-left-color:{color}">
  <div class="episode-head">
    <strong>{_esc(episode.get('период'))}</strong>
    <span class="pill" style="background:{color}1f;color:{color}">{_esc(episode.get('тяжесть'))}</span>
    <span class="muted">{_esc(episode.get('дней'))} дней, снимков {_esc(episode.get('наблюдений'))}</span>
  </div>
  <p><b>Вероятная причина:</b> {_esc(episode.get('причина'))}
     (уверенность {_esc(episode.get('уверенность'))}).</p>
  <p>В худший день {_esc(episode.get('худший_день'))} зелёность была
     <b>{_esc(episode.get('ndvi_в_худший_день'))}</b> при обычной для этой даты
     {_esc(episode.get('норма_в_худший_день'))}.</p>
  <p class="muted">Что проверить: {hint}.</p>
</article>"""


def _seasons_table(facts: dict) -> str:
    """Все сезоны одной таблицей: где было спокойно, а где стоит посмотреть внимательнее."""
    rows = []
    for season in facts.get("сводка_по_сезонам") or []:
        year = season.get("год")
        status = season_status(facts, year)
        rows.append(
            f'<tr><td class="num">{_esc(year)}</td><td class="num">{_esc(season.get("наблюдений"))}</td>'
            f'<td class="num">{_esc(season.get("пик_ndvi"))}</td>'
            f'<td class="num">{_esc(season.get("осадки_за_сезон_мм") or "—")}</td>'
            f'<td><span class="pill" style="background:{STATUS_COLOR[status]}1f;'
            f'color:{STATUS_COLOR[status]}">{status}</span></td></tr>'
        )
    return ("<table><thead><tr><th>Сезон</th><th>Снимков</th><th>Пик зелёности</th>"
            "<th>Осадки, мм</th><th>Состояние</th></tr></thead><tbody>"
            + "".join(rows) + "</tbody></table>")


def build_report(detail: dict, year: int | None = None, meta: dict | None = None,
                 generated_on: str | None = None) -> str:
    """Собирает готовый HTML-отчёт по полю."""
    facts = field_facts(detail, year)
    chosen = facts.get("выбранный_сезон")
    season = _by_year(detail.get("years"), chosen) or {}
    episodes_year = [e for e in (facts.get("эпизоды") or []) if e.get("год") == chosen]
    status = season_status(facts, chosen) if chosen else "нет данных"
    weather = next((s for s in (facts.get("сводка_по_сезонам") or []) if s.get("год") == chosen), {})
    stamp = generated_on or date.today().isoformat()

    summary = (f"За сезон {chosen} найдено периодов снижения: {len(episodes_year)}."
               if chosen else "Данных за сезон недостаточно.")
    if not episodes_year and chosen:
        summary = (f"За сезон {chosen} поле держалось в пределах обычного для себя коридора: "
                   "устойчивых периодов снижения не найдено.")

    method = ""
    if meta:
        task1 = meta.get("task1", {})
        method = (f"<p>Пропуски в ряду восстанавливает модель: ошибка на отложенной выборке "
                  f"{_esc(task1.get('rmse_val'))} по шкале зелёности "
                  f"(у простого способа «среднее двух соседних дат» — {_esc(task1.get('baseline_rmse'))}). "
                  "Это проверка на скрытых точках, а не результат независимого конкурса.</p>")

    return f"""<!doctype html>
<html lang="ru"><head><meta charset="utf-8">
<title>Отчёт по полю {_esc(facts.get('поле'))}</title>
<style>{_STYLE}</style></head>
<body>
<header>
  <div>
    <div class="eyebrow">Спутниковый мониторинг поля</div>
    <h1>{_esc(facts.get('поле'))}</h1>
    <p class="muted">{_esc(facts.get('культура'))} · сезоны {_esc((facts.get('сезоны') or ['—'])[0])}–{_esc((facts.get('сезоны') or ['—'])[-1])} · отчёт от {_esc(stamp)}</p>
  </div>
  <div class="status" style="background:{STATUS_COLOR.get(status, '#999')}1f;color:{STATUS_COLOR.get(status, '#999')}">
    Сезон {_esc(chosen)}: {status}
  </div>
</header>

<section>
  <h2>Коротко</h2>
  <p>{summary}</p>
  <ul>
    <li>Всего за историю наблюдений найдено периодов снижения: <b>{_esc(facts.get('всего_эпизодов'))}</b>,
        из них тяжёлых: <b>{_esc(facts.get('критических_эпизодов'))}</b>.</li>
    <li>Снимков за сезон {_esc(chosen)}: <b>{_esc(weather.get('наблюдений'))}</b>,
        пик зелёности <b>{_esc(weather.get('пик_ndvi'))}</b> {_esc(weather.get('дата_пика') or '')}.</li>
    {f"<li>Осадков за сезон: <b>{_esc(weather.get('осадки_за_сезон_мм'))} мм</b>, средняя температура {_esc(weather.get('средняя_температура_c'))} °C.</li>" if weather.get('осадки_за_сезон_мм') is not None else ""}
  </ul>
</section>

<section>
  <h2>Как шёл сезон {_esc(chosen)}</h2>
  {season_chart(season, episodes_year)}
  <p class="legend"><span class="key line"></span> зелёность поля по снимкам
     <span class="key dash"></span> обычный для этого поля ход сезона
     <span class="key band"></span> период снижения</p>
  <p class="muted">Зелёность — это NDVI, показатель по спутниковому снимку: чем выше, тем больше живой
     зелёной массы на поле. Обычный ход сезона считается по прошлым годам этого же поля, а не по среднему
     по региону, поэтому сравнение честное для конкретного участка.</p>
</section>

<section>
  <h2>Периоды снижения</h2>
  {"".join(_episode_block(e) for e in episodes_year) if episodes_year
    else '<p class="muted">За выбранный сезон устойчивых периодов снижения не найдено.</p>'}
</section>

<section class="break">
  <h2>Все сезоны</h2>
  {_seasons_table(facts)}
</section>

<section>
  <h2>Откуда числа</h2>
  <p>Ряд собран из снимков Sentinel-2, Landsat 8/9 и MODIS: облака и тени убираются масками,
     значения разных спутников приводятся к одной шкале. Погода — ERA5 по центру поля.</p>
  {method}
  <p class="muted">Найденный период снижения — это сигнал посмотреть, а не диагноз. Причина ставится
     по правилам: погода тех же дат против нормы, фаза сезона и поведение соседних полей. Проверять
     её нужно записями по полю.</p>
  <p class="muted">{_esc(facts.get('собрано') or '')}</p>
</section>

<footer>Космохакатон · Ростовская область · отчёт сформирован сервисом «Вегетация»</footer>
</body></html>"""


_STYLE = """
:root { --ink:#1a1d16; --muted:#7a7c71; --line:#e4e1d8; --canvas:#f7f6f3; }
* { box-sizing:border-box; }
body { margin:0; padding:28px clamp(16px,4vw,44px); background:var(--canvas); color:var(--ink);
  font:14px/1.55 "Segoe UI", system-ui, sans-serif; max-width:1000px; }
h1 { font-size:30px; margin:2px 0 4px; letter-spacing:-.02em; }
h2 { font-size:18px; margin:0 0 10px; letter-spacing:-.01em; }
p { margin:0 0 8px; }
ul { margin:6px 0 0; padding-left:18px; }
li { margin-bottom:4px; }
.eyebrow { font-size:10.5px; letter-spacing:.14em; text-transform:uppercase; color:var(--muted); }
.muted { color:var(--muted); font-size:12.5px; }
header { display:flex; justify-content:space-between; align-items:flex-start; gap:16px;
  padding-bottom:16px; border-bottom:1px solid var(--line); margin-bottom:18px; }
.status { padding:8px 14px; border-radius:999px; font-size:13px; white-space:nowrap; }
section { background:#fff; border-radius:16px; padding:18px 20px; margin-bottom:14px; }
.chart { width:100%; height:auto; }
.ax { font-size:10px; fill:#7a7c71; }
.legend { font-size:12px; color:var(--muted); display:flex; gap:14px; flex-wrap:wrap; align-items:center; }
.key { display:inline-block; width:22px; height:0; margin-right:5px; vertical-align:middle; }
.key.line { border-top:2.4px solid #1a1d16; }
.key.dash { border-top:2px dashed #9aa091; }
.key.band { height:10px; background:#c8423f26; border-radius:2px; }
.episode { border-left:3px solid #999; padding:10px 0 10px 14px; margin-bottom:12px; }
.episode-head { display:flex; gap:10px; align-items:center; flex-wrap:wrap; margin-bottom:6px; }
.pill { padding:2px 10px; border-radius:999px; font-size:11px; }
table { width:100%; border-collapse:collapse; font-size:13px; }
th { text-align:left; font-weight:400; font-size:10px; letter-spacing:.12em; text-transform:uppercase;
  color:var(--muted); padding:6px 8px; border-bottom:1px solid var(--line); }
td { padding:6px 8px; border-bottom:1px solid #f0eee8; }
td.num { font-variant-numeric:tabular-nums; }
footer { color:var(--muted); font-size:11.5px; padding-top:10px; }
@page { size:A4; margin:14mm; }
@media print {
  body { background:#fff; padding:0; max-width:none; font-size:11pt; }
  section { border:1px solid var(--line); border-radius:8px; page-break-inside:avoid; }
  .break { page-break-before:always; }
}
"""
