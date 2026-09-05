/** Краткие показатели сезона простыми словами; число остаётся мелким шрифтом.
 *  Дата наведения общая с графиками: при движении курсора фразы пересчитываются на выбранный день. */
import type { AgroContext, WeatherMetric } from "../../api/analytics";
import type { SeasonYear } from "../../api/types";
import { ms, shortDate } from "../../lib/format";
import { comparison, dayAt, finite, historySpan, metricPoint, number } from "../../lib/metrics";
import { dryPlain, heatPlain, ndviPlain, rainPlain, valueNear, type Tone } from "../../lib/plain";

export type MetricZoom = { chart: "ndvi" | "rain" | "thermal"; from: number; to: number };
const DAY = 86400000;

function Metric({ label, phrase, tone, value, note, id, units, onZoom }: {
  label: string; phrase: string; tone: Tone; value: string; note: string; id: string; units?: string;
  onZoom?: () => void;
}) {
  const Tag = onZoom ? "button" : "div";
  return <Tag className="season-metric" data-testid={`metric-${id}`} data-tone={tone}
    {...(onZoom ? { type: "button" as const, onClick: onZoom } : {})}>
    <span className="meta">{label}</span>
    <strong className="season-metric-phrase">{phrase}</strong>
    <span className="meta season-metric-figure"><span className="num">{value}{units && <small> {units}</small>}</span> · {note}</span>
    {onZoom && <span className="season-metric-action">Показать на графике ↗</span>}
  </Tag>;
}

function WeatherCard({ metric, day, id, label, units, plain, onZoom }: {
  metric: WeatherMetric; day: string | null; id: string; label: string; units: string;
  plain: (value: number | null | undefined, mean: number | null | undefined) => { text: string; tone: Tone };
  onZoom: (selection: MetricZoom) => void;
}) {
  if (!metric.available) return null;
  const point = metricPoint(metric, day);
  const words = plain(point.value, point.mean);
  return <Metric id={id} label={`${label} · ${point.date ? shortDate(point.date) : "—"}`} phrase={words.text} tone={words.tone}
    value={number(point.value)} units={units}
    note={comparison(point.value, point.mean, units).replace("к среднему", `к норме${historySpan(metric)}`)}
    onZoom={point.date && finite(point.value) ? () => onZoom({ chart: id === "heat" ? "thermal" : "rain",
      from: id === "heat" ? ms(`${point.date.slice(0, 4)}-04-01`) : ms(point.date) - 29 * DAY,
      to: ms(point.date) }) : undefined} />;
}

export function SeasonMetrics({ season, weather, hover, onZoom }: {
  season: SeasonYear; weather?: AgroContext; hover: number | null;
  onZoom: (selection: MetricZoom) => void;
}) {
  const day = dayAt(hover);
  const latest = season.observations.filter(p => !p.artifact && finite(p.harmonized))
    .sort((a, b) => a.date.localeCompare(b.date)).at(-1);
  const target = day ?? latest?.date;
  const value = day ? season.curve.find(p => p.date === day)?.value : latest?.harmonized;
  const normal = target ? valueNear(season.norm_mean, target) : undefined;
  const spread = target ? valueNear(season.norm_std, target) : undefined;
  const greens = ndviPlain(value, normal, spread);
  const dry = weather?.dry_spell;
  const dryWords = dry?.available ? dryPlain(dry.days) : null;
  // Сезон длится до конца октября, поэтому в текущем году серия «не завершена» по определению.
  // Вместо пугающего «неполные данные» говорим, до какого дня есть погода.
  const weatherUntil = weather?.rain.available ? metricPoint(weather.rain, null).date : undefined;
  const dryNote = dry?.complete ? "" : weatherUntil ? ` · погода до ${shortDate(weatherUntil)}` : " · сезон ещё идёт";
  return <div className="season-metrics" aria-label="Показатели поля">
    <Metric id="ndvi" label={`${day ? "Зелень поля" : "Последний снимок"} · ${target ? shortDate(target) : "—"}`}
      phrase={greens.text} tone={greens.tone}
      value={`NDVI ${number(value, 2)}`} note={comparison(value, normal, "", 2).replace("к среднему", "к обычному")}
      onZoom={target && finite(value) ? () => onZoom({ chart: "ndvi", from: ms(target) - 14 * DAY, to: ms(target) + 14 * DAY }) : undefined} />
    {weather && <>
      <WeatherCard metric={weather.rain} day={day} id="rain" label="Дожди за 30 дней" units="мм" plain={rainPlain} onZoom={onZoom} />
      <WeatherCard metric={weather.thermal} day={day} id="heat" label="Тепло с 1 апреля" units="°C·дни" plain={heatPlain} onZoom={onZoom} />
    </>}
    {dry?.available && dryWords && <Metric id="dry" label="Самый долгий сухой период" phrase={dryWords.text} tone={dryWords.tone}
      value={`${dry.days} дн.`}
      note={dry.start && dry.end ? `${shortDate(dry.start)} — ${shortDate(dry.end)}${dryNote}` : "сухих дней не было"}
      onZoom={dry.start && dry.end ? () => onZoom({ chart: "rain", from: ms(dry.start!) - 3 * DAY, to: ms(dry.end!) + 3 * DAY }) : undefined} />}
  </div>;
}
