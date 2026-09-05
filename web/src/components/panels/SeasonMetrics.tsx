/** Краткие показатели сезона простыми словами; число остаётся мелким шрифтом.
 *  Дата наведения общая с графиками: при движении курсора фразы пересчитываются на выбранный день. */
import type { AgroContext, WeatherMetric } from "../../api/analytics";
import type { SeasonYear } from "../../api/types";
import { shortDate } from "../../lib/format";
import { comparison, dayAt, finite, metricPoint, number } from "../../lib/metrics";
import { dryPlain, heatPlain, ndviPlain, rainPlain, type Tone } from "../../lib/plain";

function Metric({ label, phrase, tone, value, note, id, units }: {
  label: string; phrase: string; tone: Tone; value: string; note: string; id: string; units?: string;
}) {
  return <div className="season-metric" data-testid={`metric-${id}`} data-tone={tone}>
    <span className="meta">{label}</span>
    <strong className="season-metric-phrase">{phrase}</strong>
    <span className="meta season-metric-figure"><span className="num">{value}{units && <small> {units}</small>}</span> · {note}</span>
  </div>;
}

function WeatherCard({ metric, day, id, label, units, plain }: {
  metric: WeatherMetric; day: string | null; id: string; label: string; units: string;
  plain: (value: number | null | undefined, mean: number | null | undefined) => { text: string; tone: Tone };
}) {
  if (!metric.available) return null;
  const point = metricPoint(metric, day);
  const words = plain(point.value, point.mean);
  return <Metric id={id} label={`${label} · ${point.date ? shortDate(point.date) : "—"}`} phrase={words.text} tone={words.tone}
    value={number(point.value)} units={units} note={comparison(point.value, point.mean, units).replace("к среднему", "к норме")} />;
}

export function SeasonMetrics({ season, weather, hover }: {
  season: SeasonYear; weather?: AgroContext; hover: number | null;
}) {
  const day = dayAt(hover);
  const latest = season.observations.filter(p => !p.artifact && finite(p.harmonized))
    .sort((a, b) => a.date.localeCompare(b.date)).at(-1);
  const target = day ?? latest?.date;
  const value = day ? season.curve.find(p => p.date === day)?.value : latest?.harmonized;
  const normal = season.norm_mean.find(p => p.date === target)?.value;
  const greens = ndviPlain(value, normal);
  const dry = weather?.dry_spell;
  const dryWords = dry?.available ? dryPlain(dry.days) : null;
  return <div className="season-metrics" aria-label="Показатели поля">
    <Metric id="ndvi" label={`${day ? "Зелень поля" : "Последний снимок"} · ${target ? shortDate(target) : "—"}`}
      phrase={greens.text} tone={greens.tone}
      value={`NDVI ${number(value, 2)}`} note={comparison(value, normal, "", 2).replace("к среднему", "к обычному")} />
    {weather && <>
      <WeatherCard metric={weather.rain} day={day} id="rain" label="Дожди за 30 дней" units="мм" plain={rainPlain} />
      <WeatherCard metric={weather.thermal} day={day} id="heat" label="Тепло с 1 апреля" units="°C·дни" plain={heatPlain} />
    </>}
    {dry?.available && dryWords && <Metric id="dry" label="Самый долгий сухой период" phrase={dryWords.text} tone={dryWords.tone}
      value={`${dry.days} дн.`}
      note={dry.start && dry.end ? `${shortDate(dry.start)} — ${shortDate(dry.end)}${dry.complete ? "" : " · неполные данные"}` : "сухих дней не было"} />}
  </div>;
}
