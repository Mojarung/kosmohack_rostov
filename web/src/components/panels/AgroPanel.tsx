/** Один погодный график, три показателя; недоступные показатели не предлагаются. */
import { useState } from "react";
import type { AgroContext } from "../../api/analytics";
import { shortDate } from "../../lib/format";
import { comparison, dayAt, finite, historySpan, metricPoint, number } from "../../lib/metrics";
import { AgroChart } from "../charts/AgroChart";
import { RangeToolbar } from "../charts/RangeToolbar";
import type { RangeControl } from "../charts/range";

const MODES = {
  rain: { label: "Осадки", color: "#3a6ea5", units: "мм", description: "Сумма осадков за 30 полных дней" },
  thermal: { label: "Тепло", color: "#c2703a", units: "°C·дни", description: "Накопленное тепло выше +10 °C от 1 апреля" },
  water: { label: "Баланс влаги", color: "#427b78", units: "мм", description: "Осадки минус испарение ET₀ за 30 дней" },
};
export type WeatherMode = keyof typeof MODES;

export function AgroPanel({ context, loading, error, retry, year, control, hover, onHover, preferred, onMode }: {
  context?: AgroContext; loading: boolean; error: boolean; retry: () => void; year: number;
  control: RangeControl; hover: number | null; onHover: (time: number | null) => void;
  preferred: WeatherMode; onMode: (mode: WeatherMode) => void;
}) {
  const [kolobokEnabled, setKolobokEnabled] = useState(false);
  if (loading || error) return <div className="season-weather" role="status">
    <span className="meta">{loading ? "Загружаем погоду…" : "Погода не загрузилась."}</span>
    {error && <button className="btn btn--sm btn--ghost" onClick={retry}>Повторить загрузку погоды</button>}
  </div>;
  if (!context) return null;
  const available = (Object.keys(MODES) as WeatherMode[]).filter(key => context[key].available);
  if (!available.length) return null;
  const mode = available.includes(preferred) ? preferred : available[0];
  const spec = MODES[mode], metric = context[mode], point = metricPoint(metric, dayAt(hover));
  const history = metric.mean.some(finite);
  return <section className="season-weather" aria-label="Погодный контекст">
    <div className="spread" style={{ gap: 10, flexWrap: "wrap" }}>
      <h3>Погодный контекст</h3>
      <div className="metric-tabs" role="group" aria-label="Погодный показатель">
        {available.map(key => <button key={key} type="button" aria-pressed={key === mode}
          data-weather={key} onClick={() => onMode(key)}>{MODES[key].label}</button>)}
      </div>
    </div>
    <p className="meta weather-description">{spec.description} · {spec.units}</p>
    <p className="weather-readout num" data-testid="weather-readout" data-date={point.date}>
      {point.date ? shortDate(point.date) : "—"} · {number(point.value)} {spec.units}
      {finite(point.mean) ? ` · ${comparison(point.value, point.mean, spec.units)}${historySpan(metric)}` : " · истории для сравнения мало"}
    </p>
    <div className="metric-legend"><span style={{ color: spec.color }}>● {year}</span>
      {history && <span>━ Среднее{historySpan(metric) || " прошлых лет"} · полоса 10–90%</span>}
      {mode === "rain" && <button type="button" className="kolobok-toggle"
        aria-pressed={kolobokEnabled} title={kolobokEnabled ? "Спрятать колобка" : "Покатить колобка"}
        onClick={() => setKolobokEnabled(enabled => !enabled)}>
        <svg width="10" height="10" viewBox="0 0 10 10" aria-hidden="true">
          <circle cx="5" cy="5" r="3.5" stroke="currentColor" fill={kolobokEnabled ? "currentColor" : "none"} />
        </svg>
        Колобок
      </button>}
    </div>
    <AgroChart metric={metric} color={spec.color} units={spec.units} year={year}
      control={control} hover={hover} onHover={onHover} showKolobok={mode === "rain" && kolobokEnabled} />
    <RangeToolbar control={control} />
  </section>;
}
