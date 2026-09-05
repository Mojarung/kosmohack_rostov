/** Краткие показатели сезона; дата наведения общая с графиками. */
import type { AgroContext, WeatherMetric } from "../../api/analytics";
import type { SeasonYear } from "../../api/types";
import { shortDate } from "../../lib/format";
import { comparison, dayAt, finite, metricPoint, number } from "../../lib/metrics";

function Metric({ label, value, note, id, units }: { label: string; value: string; note: string; id: string; units?: string }) {
  return <div className="season-metric" data-testid={`metric-${id}`}>
    <span className="meta">{label}</span><strong className="num">{value}{units && <small> {units}</small>}</strong><span className="meta">{note}</span>
  </div>;
}

function WeatherCard({ metric, day, id, label, units }: {
  metric: WeatherMetric; day: string | null; id: string; label: string; units: string;
}) {
  if (!metric.available) return null;
  const point = metricPoint(metric, day);
  return <Metric id={id} label={`${label} · ${point.date ? shortDate(point.date) : "—"}`}
    value={number(point.value)} units={units} note={comparison(point.value, point.mean, units)} />;
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
  const dry = weather?.dry_spell;
  return <div className="season-metrics" aria-label="Показатели поля">
    <Metric id="ndvi" label={`${day ? "Кривая NDVI" : "Последний снимок"} · ${target ? shortDate(target) : "—"}`}
      value={number(value, 3)} note={comparison(value, normal, "", 3).replace("к среднему", "к ориентиру")} />
    {weather && <>
      <WeatherCard metric={weather.rain} day={day} id="rain" label="Осадки за 30 дней" units="мм" />
      <WeatherCard metric={weather.thermal} day={day} id="heat" label="Накопленное тепло" units="°C·дни" />
    </>}
    {dry?.available && <Metric id="dry" label="Сухой период · максимум за сезон" value={`${dry.days} дн.`}
      note={dry.start && dry.end ? `${shortDate(dry.start)} — ${shortDate(dry.end)}${dry.complete ? "" : " · неполные данные"}` : "Сухих дней не было"} />}
  </div>;
}
