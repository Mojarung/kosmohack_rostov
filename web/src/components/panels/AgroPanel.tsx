/** Один погодный график, три показателя; недоступные показатели не предлагаются. */
import { useState } from "react";
import type { AgroContext } from "../../api/analytics";
import { shortDate } from "../../lib/format";
import { comparison, dayAt, finite, metricPoint, number } from "../../lib/metrics";
import { AgroChart } from "../charts/AgroChart";
import type { RangeControl } from "../charts/range";

const MODES = {
  rain: { label: "Осадки", color: "#3a6ea5", units: "мм", description: "Сумма осадков за 30 полных дней" },
  thermal: { label: "Тепло", color: "#c2703a", units: "°C·дни", description: "Накопленное тепло выше +10 °C от 1 апреля" },
  water: { label: "Баланс влаги", color: "#427b78", units: "мм", description: "Осадки минус испарение ET₀ за 30 дней" },
};
type Mode = keyof typeof MODES;

export function AgroPanel({ context, loading, error, retry, year, control, hover, onHover }: {
  context?: AgroContext; loading: boolean; error: boolean; retry: () => void; year: number;
  control: RangeControl; hover: number | null; onHover: (time: number | null) => void;
}) {
  const [preferred, setPreferred] = useState<Mode>("rain");
  if (loading || error) return <div className="season-weather" role="status">
    <span className="meta">{loading ? "Загружаем погоду…" : "Погода не загрузилась."}</span>
    {error && <button className="btn btn--sm btn--ghost" onClick={retry}>Повторить загрузку погоды</button>}
  </div>;
  if (!context) return null;
  const available = (Object.keys(MODES) as Mode[]).filter(key => context[key].available);
  if (!available.length) return null;
  const mode = available.includes(preferred) ? preferred : available[0];
  const spec = MODES[mode], metric = context[mode], point = metricPoint(metric, dayAt(hover));
  const history = metric.mean.some(finite);
  return <section className="season-weather" aria-label="Погодный контекст">
    <div className="spread" style={{ gap: 10, flexWrap: "wrap" }}>
      <h3>Погодный контекст</h3>
      <div className="metric-tabs" role="group" aria-label="Погодный показатель">
        {available.map(key => <button key={key} type="button" aria-pressed={key === mode}
          data-weather={key} onClick={() => setPreferred(key)}>{MODES[key].label}</button>)}
      </div>
    </div>
    <p className="meta weather-description">{spec.description} · {spec.units}</p>
    <p className="weather-readout num" data-testid="weather-readout" data-date={point.date}>
      {point.date ? shortDate(point.date) : "—"} · {number(point.value)} {spec.units}
      {finite(point.mean) ? ` · ${comparison(point.value, point.mean, spec.units)}` : " · истории для сравнения мало"}
    </p>
    <div className="metric-legend"><span style={{ color: spec.color }}>● {year}</span>
      {history && <span>━ Среднее прошлых лет · полоса 10–90%</span>}</div>
    <AgroChart metric={metric} color={spec.color} units={spec.units} year={year}
      control={control} hover={hover} onHover={onHover} />
    <details className="metric-help"><summary>Как читать погодный график</summary><div className="stack">
      <p>Сравниваем одинаковые календарные даты прошлых лет, без текущего года. Для ориентира нужны минимум 3 года. Пропуски остаются разрывами.</p>
      <p>Тепло: сумма max(Tср − 10 °C, 0) от 1 апреля. Это общий показатель сезона, без привязки к фазам конкретной культуры.</p>
      <p>Сухой период: максимум дней подряд с осадками менее 1 мм за сезон. Неизвестный день прерывает серию.</p>
      {context.water.available && <p>Баланс влаги: осадки − ET₀. ET₀ учитывает температуру, влажность воздуха, ветер и солнечную радиацию. Ниже среднего — погода суше обычного. Это не влажность почвы и не норма полива.</p>}
      <p>{context.source || "ERA5 из данных кейса"}. Погода помогает объяснить NDVI и не меняет пороги аномалий.</p>
    </div></details>
  </section>;
}
