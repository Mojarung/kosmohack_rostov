/** Единая панель сезона: оформление main, погодные показатели и карты из field-insights. */
import { useMemo, useState, type ReactNode } from "react";
import { useQuery } from "@tanstack/react-query";
import { api } from "../../api/client";
import type { PolygonDetail } from "../../api/types";
import { SENSOR_COLOR, ms, plural } from "../../lib/format";
import { SeasonChart } from "../charts/SeasonChart";
import { WeatherChart } from "../charts/WeatherChart";
import { ZChart } from "../charts/ZChart";
import { useDateRange } from "../charts/range";
import { EpisodeCard } from "./EpisodeCard";
import { FieldVerdict } from "./FieldVerdict";
import { SeasonMetrics } from "./SeasonMetrics";
import { AgroPanel } from "./AgroPanel";
import { FieldInsights } from "./FieldInsights";

function Legend() {
  return <div className="metric-legend">
    <span style={{ color: "var(--ink)" }}>━ Зелень поля в этом году</span><span>━ Как обычно (коридор нормы)</span>
    {Object.entries(SENSOR_COLOR).map(([label, color]) => <span key={label} style={{ color }}>● {label}</span>)}
  </div>;
}

function SeasonContent({ detail, year, onYear, aside }: { detail: PolygonDetail; year: number; onYear: (year: number) => void; aside?: (year: number) => ReactNode }) {
  const season = detail.years[String(year)], weather = detail.weather?.[String(year)];
  const episodes = detail.episodes.filter(e => e.year === year);
  const shape = (detail.shape ?? []).filter(s => s.year === year);
  const bounds = useMemo(() => ({ from: ms(`${year}-04-01`), to: ms(`${year}-10-30`) }), [year]);
  const control = useDateRange(bounds);
  const [hover, setHover] = useState<number | null>(null), [showDetails, setShowDetails] = useState(false);
  const agro = useQuery({ queryKey: ["agro", detail.pid, year, detail.saved_at],
    queryFn: ({ signal }) => api.agro(detail.pid, year, signal), retry: false });
  if (!season) return <p className="meta">Нет наблюдений за этот сезон.</p>;
  return <>
    <section className="pane season-main">
    <SeasonHeader detail={detail} year={year} onYear={onYear} />
    <div className="pane-body season-content">
      <FieldVerdict season={season} episodes={episodes} trend={detail.insights?.[year]} weather={agro.data} />
      <SeasonMetrics season={season} weather={agro.data} hover={hover} />
      <div className="season-toolbar">
        <h3>Как росло поле по снимкам</h3>
        <button type="button" className="btn btn--sm btn--ghost" onClick={control.reset} disabled={!control.zoomed}>Весь сезон</button>
      </div>
      <Legend />
      <div data-testid="ndvi-chart" data-from={control.view.from} data-to={control.view.to}>
        <SeasonChart season={season} episodes={episodes} control={control} hover={hover} onHover={setHover} />
      </div>
      <p className="meta chart-instruction">Линия — зелень поля, серая полоса — как бывает обычно. Выделите период мышью, чтобы приблизить; двойной клик — весь сезон.</p>
      <FieldInsights detail={detail} year={year} />
      <AgroPanel context={agro.data} loading={agro.isPending} error={agro.isError} retry={() => void agro.refetch()}
        year={year} control={control} hover={hover} onHover={setHover} />
      <details className="season-calculation" onToggle={e => setShowDetails(e.currentTarget.open)}>
        <summary>Подробности для агронома: данные и расчёт</summary>
        {showDetails && <div className="stack" style={{ gap: 12 }}>
          <p className="meta">Z = (NDVI − среднее истории) / разброс. Жёлтый порог −1, красный −2; детектор также учитывает длительность и реальные наблюдения.</p>
          <div data-testid="z-chart"><ZChart season={season} episodes={episodes} control={control} hover={hover} onHover={setHover} /></div>
          <h3>Исходные наблюдения и восстановленные пропуски</h3>
          <p className="meta">Точки — исходные значения спутников, ромбы — контрольные восстановления модели, крестики — исключённые артефакты.</p>
          <div data-testid="raw-ndvi-chart"><SeasonChart raw season={season} episodes={episodes} control={control} hover={hover} onHover={setHover} /></div>
          {weather && <><h3>Температура и осадки по дням</h3><div data-testid="raw-weather-chart">
            <WeatherChart weather={weather} control={control} hover={hover} onHover={setHover} /></div></>}
          <p className="meta">{detail.collected || "Исходные наблюдения и контрольные восстановления из данных кейса."}</p>
        </div>}
      </details>
    </div>
    </section>
    <section className="pane season-episodes">
      <div className="pane-head" style={{ gap: 8, flexWrap: "wrap" }}>
        <h2 className="pane-title">Когда поле отставало</h2><span className="meta">{episodes.length
          ? `${episodes.length} ${plural(episodes.length, "период", "периода", "периодов")} в ${year} году` : `в ${year} году не найдено`}</span>
      </div>
      <div className="pane-body pane-body--pad stack" style={{ gap: 10 }}>
      {!episodes.length && <p className="meta">{season.z.some(p => Number.isFinite(p.value))
        ? "Поле весь сезон держалось в пределах обычного." : "Недостаточно снимков, чтобы судить о поле."}</p>}
      {[...episodes].sort((a, b) => a.min_z - b.min_z).map(e =>
        <EpisodeCard compact key={`${e.start}-${e.end}`} episode={e} onZoom={() => {
          control.select(ms(e.start) - 30 * 86400000, ms(e.end) + 10 * 86400000);
          document.querySelector('[data-testid="ndvi-chart"]')?.scrollIntoView({ behavior: "smooth", block: "center" });
        }} />)}
      {shape.map(item => <div key={`${item.year}-${item.shape_direction}`} className="card card--sunk">
        <div className="eyebrow">Сезон прошёл необычно · {item.shape_direction}</div>
        <p className="meta">{item.shape_reasons}</p>
      </div>)}
      </div>
    </section>
    {aside?.(year)}
  </>;
}

function SeasonHeader({ detail, year, onYear }: { detail: PolygonDetail; year: number; onYear: (year: number) => void }) {
  const years = Object.keys(detail.years).map(Number).sort((a, b) => a - b);
  const season = detail.years[String(year)];
  return <div className="pane-head season-heading">
      <div className="row" style={{ gap: 8, flexWrap: "wrap" }}>
        <span className="pane-title">Сезон {year}</span>
        <span className="meta">{season ? `${season.observations.length} наблюдений · ${season.norm_source}` : "Нет данных"}</span>
      </div>
      <div className="season-years" role="group" aria-label="Сезон">
        {years.map(y => {
          const episodes = detail.episodes.filter(e => e.year === y);
          const critical = episodes.some(e => e.severity === "критическая");
          return <button key={y} type="button" className="chip chip--year" aria-label={String(y)} data-year={y} aria-pressed={y === year}
            data-level={critical ? "critical" : episodes.length ? "moderate" : "none"} onClick={() => onYear(y)}>{String(y).slice(2)}</button>;
        })}
      </div>
  </div>;
}

export function SeasonPanel({ detail, aside }: { detail: PolygonDetail; aside?: (year: number) => ReactNode }) {
  const years = Object.keys(detail.years).map(Number).sort((a, b) => a - b);
  const [year, setYear] = useState(() => years.at(-1) ?? 0);
  return <div className="season-layout season-panel" data-testid="season-panel" data-pid={detail.pid} data-year={year}>
    <SeasonContent key={`${detail.pid}:${year}`} detail={detail} year={year} onYear={setYear} aside={aside} />
  </div>;
}
