/** Единая панель сезона: оформление main, погодные показатели и карты из field-insights. */
import { useMemo, useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { api } from "../../api/client";
import type { PolygonDetail } from "../../api/types";
import { SENSOR_COLOR, ms, plural } from "../../lib/format";
import { SeasonChart } from "../charts/SeasonChart";
import { WeatherChart } from "../charts/WeatherChart";
import { ZChart } from "../charts/ZChart";
import { useDateRange } from "../charts/range";
import { EpisodeCard } from "./EpisodeCard";
import { SeasonMetrics } from "./SeasonMetrics";
import { AgroPanel } from "./AgroPanel";
import { FieldInsights } from "./FieldInsights";

function Legend() {
  return <div className="metric-legend">
    <span style={{ color: "var(--ink)" }}>━ Кривая NDVI</span><span>━ Ориентир ±1σ</span>
    {Object.entries(SENSOR_COLOR).map(([label, color]) => <span key={label} style={{ color }}>● {label}</span>)}
  </div>;
}

function SeasonContent({ detail, year }: { detail: PolygonDetail; year: number }) {
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
    <div className="season-content">
      <SeasonMetrics season={season} weather={agro.data} hover={hover} />
      <div className="season-toolbar">
        <h3>Развитие растительности</h3>
        <button type="button" className="btn btn--sm btn--ghost" onClick={control.reset} disabled={!control.zoomed}>Весь сезон</button>
      </div>
      <Legend />
      <div data-testid="ndvi-chart" data-from={control.view.from} data-to={control.view.to}>
        <SeasonChart season={season} episodes={episodes} control={control} hover={hover} onHover={setHover} />
      </div>
      <p className="meta chart-instruction">Точки в шкале Sentinel-2. Выделите период мышью для приближения; двойной клик — весь сезон.</p>
      <FieldInsights detail={detail} year={year} />
      <AgroPanel context={agro.data} loading={agro.isPending} error={agro.isError} retry={() => void agro.refetch()}
        year={year} control={control} hover={hover} onHover={setHover} />
      <details className="season-calculation" onToggle={e => setShowDetails(e.currentTarget.open)}>
        <summary>Данные и расчёт</summary>
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
    <div className="season-episodes stack">
      <div className="spread" style={{ gap: 8, flexWrap: "wrap" }}>
        <h2>Эпизоды угнетения</h2><span className="meta">{episodes.length
          ? `${episodes.length} ${plural(episodes.length, "эпизод", "эпизода", "эпизодов")} в ${year} году` : `в ${year} году не найдено`}</span>
      </div>
      {!episodes.length && <p className="meta">{season.z.some(p => Number.isFinite(p.value))
        ? "Устойчивых или сильных отклонений по правилам детектора не обнаружено." : "Недостаточно данных для оценки отклонений."}</p>}
      {[...episodes].sort((a, b) => a.min_z - b.min_z).map(e =>
        <EpisodeCard key={`${e.start}-${e.end}`} episode={e} onZoom={() => {
          control.select(ms(e.start) - 30 * 86400000, ms(e.end) + 10 * 86400000);
        }} />)}
      {shape.map(item => <div key={`${item.year}-${item.shape_direction}`} className="card card--sunk">
        <div className="eyebrow">Нетипичная форма сезона · {item.shape_direction}</div>
        <p className="meta">{item.shape_reasons}</p>
      </div>)}
    </div>
  </>;
}

export function SeasonPanel({ detail }: { detail: PolygonDetail }) {
  const years = Object.keys(detail.years).map(Number).sort((a, b) => a - b);
  const [year, setYear] = useState(() => years.at(-1) ?? 0);
  const season = detail.years[String(year)];
  return <div className="card card--flush season-panel" data-testid="season-panel" data-pid={detail.pid} data-year={year}>
    <div className="season-heading spread">
      <div><div className="eyebrow">Сезон</div><div className="row" style={{ gap: 8 }}>
        <span className="season-year">{year}</span>
        <span className="meta">{season ? `${season.observations.length} наблюдений · ${season.norm_source}` : "Нет данных"}</span>
      </div></div>
      <div className="season-years" role="group" aria-label="Сезон">
        {years.map(y => {
          const episodes = detail.episodes.filter(e => e.year === y);
          const critical = episodes.some(e => e.severity === "критическая");
          return <button key={y} type="button" className="mono" data-year={y} aria-pressed={y === year}
            data-level={critical ? "critical" : episodes.length ? "moderate" : "none"} onClick={() => setYear(y)}>{y}</button>;
        })}
      </div>
    </div>
    <SeasonContent key={`${detail.pid}:${year}`} detail={detail} year={year} />
  </div>;
}
