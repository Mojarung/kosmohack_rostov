/** Погодный показатель и диапазон 10–90% прошлых лет на общей шкале NDVI. */
import { useId } from "react";
import { ChartsContainer } from "@mui/x-charts/ChartsContainer";
import { ChartsXAxis } from "@mui/x-charts/ChartsXAxis";
import { ChartsYAxis } from "@mui/x-charts/ChartsYAxis";
import { ChartsGrid } from "@mui/x-charts/ChartsGrid";
import { ChartsTooltip } from "@mui/x-charts/ChartsTooltip";
import { LinePlot } from "@mui/x-charts/LineChart";
import { useDrawingArea, useXScale, useYScale } from "@mui/x-charts/hooks";
import type { WeatherMetric } from "../../api/analytics";
import { ms } from "../../lib/format";
import { finite, number } from "../../lib/metrics";
import { BrushLayer } from "./BrushLayer";
import { PlotClip } from "./PlotClip";
import { RollingKolobok } from "./RollingKolobok";
import type { RangeControl } from "./range";

function HistoryBand({ metric }: { metric: WeatherMetric }) {
  const x = useXScale("agro-x"), y = useYScale("agro-y");
  const area = useDrawingArea(), clip = useId();
  // Пропуск разрывает полосу: нельзя дорисовывать неизвестную историю.
  const segments: number[][] = [[]];
  metric.date.forEach((_, i) => {
    if (finite(metric.low[i]) && finite(metric.high[i])) segments.at(-1)!.push(i);
    else if (segments.at(-1)!.length) segments.push([]);
  });
  return <g>
    <defs><clipPath id={clip}><rect x={area.left} y={area.top} width={area.width} height={area.height} /></clipPath></defs>
    <g clipPath={`url(#${clip})`}>{segments.filter(s => s.length > 1).map((indices, i) =>
      <polygon key={i} fill="#9aa091" opacity={0.16} points={[
        ...indices.map(j => `${x(ms(metric.date[j]))},${y(metric.high[j]!)}`),
        ...[...indices].reverse().map(j => `${x(ms(metric.date[j]))},${y(metric.low[j]!)}`),
      ].join(" ")} />)}</g>
  </g>;
}

export function AgroChart({ metric, color, units, year, control, hover, onHover, showKolobok = false }: {
  metric: WeatherMetric; color: string; units: string; year: number;
  showKolobok?: boolean;
  control: RangeControl; hover: number | null; onHover: (time: number | null) => void;
}) {
  const history = metric.mean.some(finite);
  return <div data-testid="weather-chart" data-from={control.view.from} data-to={control.view.to}>
    <ChartsContainer<"line"> height={230} margin={{ left: 8, right: 12, top: 12, bottom: 4 }}
      series={[
        ...(history ? [{ type: "line" as const, id: "history", label: "Среднее прошлых лет", data: metric.mean,
          color: "#9aa091", showMark: false, curve: "linear" as const,
          valueFormatter: (v: number | null) => `${number(v)} ${units}` }] : []),
        { type: "line", id: "weather", label: String(year), data: metric.value, color, showMark: false,
          curve: "linear", valueFormatter: v => `${number(v)} ${units}` },
      ]}
      xAxis={[{ id: "agro-x", data: metric.date.map(d => new Date(ms(d))), scaleType: "utc",
        min: new Date(control.view.from), max: new Date(control.view.to), tickNumber: 6,
        valueFormatter: (d: Date, ctx) => d.toLocaleDateString("ru-RU", {
          month: "short", ...(ctx.location === "tick" ? {} : { day: "numeric" }),
        }),
      }]}
      yAxis={[{ id: "agro-y", width: 54, tickNumber: 5, valueFormatter: (v: number) => number(v, 0) }]}>
      <ChartsGrid horizontal /><PlotClip><HistoryBand metric={metric} /><LinePlot /></PlotClip>
      <ChartsXAxis axisId="agro-x" /><ChartsYAxis axisId="agro-y" /><ChartsTooltip />
      <BrushLayer axisId="agro-x" control={control} hover={hover} onHover={onHover} />
      {showKolobok && <RollingKolobok metric={metric} />}
    </ChartsContainer>
  </div>;
}
