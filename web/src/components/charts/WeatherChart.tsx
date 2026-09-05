/** Погода сезона по ERA5: суточные осадки столбиками и температура линией на второй оси.
 *
 *  Шкала времени та же, что у графиков NDVI и отклонения, поэтому окно просмотра и курсор
 *  общие для всех трёх: выделили период на любом графике — приблизились все. Столбики осадков
 *  рисуются своим слоем, потому что столбчатый график MUI требует категориальной шкалы. */

import { useMemo } from "react";
import { ChartsContainer } from "@mui/x-charts/ChartsContainer";
import { ChartsXAxis } from "@mui/x-charts/ChartsXAxis";
import { ChartsYAxis } from "@mui/x-charts/ChartsYAxis";
import { ChartsTooltip } from "@mui/x-charts/ChartsTooltip";
import { LinePlot } from "@mui/x-charts/LineChart";
import { useDrawingArea, useXScale, useYScale } from "@mui/x-charts/hooks";

import type { WeatherYear } from "../../api/types";
import { BrushLayer } from "./BrushLayer";
import type { RangeControl } from "./range";

const X_AXIS = "wx-x";
const RAIN = "rain";
const TEMP = "temp";
const DAY_MS = 24 * 3600 * 1000;

interface WeatherChartProps {
  weather: WeatherYear;
  height?: number;
  control: RangeControl;
  hover: number | null;
  onHover: (value: number | null) => void;
}

/** Столбики осадков: ширина равна суткам в текущем масштабе, минимум один пиксель. */
function RainBars({ dates, precip }: { dates: Date[]; precip: number[] }) {
  const xScale = useXScale(X_AXIS);
  const yScale = useYScale(RAIN);
  const { top, height } = useDrawingArea();
  const zero = yScale(0) ?? top + height;
  const step = Math.abs((xScale(new Date(DAY_MS * 2)) ?? 0) - (xScale(new Date(DAY_MS)) ?? 0));
  const barWidth = Math.max(1, Math.min(14, step * 0.72));

  return (
    <g>
      {dates.map((date, index) => {
        const value = precip[index];
        if (!value) return null;
        const x = xScale(date);
        const y = yScale(value);
        if (x === undefined || y === undefined) return null;
        return (
          <rect
            key={date.getTime()}
            x={x - barWidth / 2}
            y={Math.min(y, zero)}
            width={barWidth}
            height={Math.max(1, Math.abs(zero - y))}
            fill="#3a6ea5"
            opacity={0.85}
            rx={barWidth > 3 ? 1.5 : 0}
          >
            <title>{`${date.toISOString().slice(0, 10)} · осадки ${value.toFixed(1)} мм`}</title>
          </rect>
        );
      })}
    </g>
  );
}

export function WeatherChart({ weather, height = 160, control, hover, onHover }: WeatherChartProps) {
  const data = useMemo(
    () => ({
      dates: weather.date.map((d) => new Date(`${d}T00:00:00Z`)),
      precip: weather.precip,
      temp: weather.temp,
      maxRain: Math.max(1, ...weather.precip.filter((v) => Number.isFinite(v))),
    }),
    [weather],
  );
  if (!data.dates.length) return null;

  return (
    <ChartsContainer<"line">
      height={height}
      margin={{ left: 8, right: 8, top: 10, bottom: 4 }}
      series={[
        {
          type: "line",
          id: "temp",
          label: "температура, °C",
          data: data.temp,
          color: "#c2703a",
          yAxisId: TEMP,
          showMark: false,
          curve: "monotoneX",
          valueFormatter: (v) => (v == null ? "" : `${v.toFixed(1)} °C`),
        },
      ]}
      xAxis={[
        {
          id: X_AXIS,
          data: data.dates,
          scaleType: "time",
          min: new Date(control.view.from),
          max: new Date(control.view.to),
          tickNumber: 6,
          valueFormatter: (date: Date, context) =>
            context.location === "tick"
              ? date.toLocaleDateString("ru-RU", { month: "short", timeZone: "UTC" })
              : date.toLocaleDateString("ru-RU", { day: "numeric", month: "long", timeZone: "UTC" }),
        },
      ]}
      yAxis={[
        { id: RAIN, position: "left", width: 42, label: "мм", min: 0, max: Math.ceil(data.maxRain * 1.1) },
        { id: TEMP, position: "right", width: 42, label: "°C" },
      ]}
    >
      <RainBars dates={data.dates} precip={data.precip} />
      <LinePlot />
      <ChartsXAxis axisId={X_AXIS} />
      <ChartsYAxis axisId={RAIN} />
      <ChartsYAxis axisId={TEMP} />
      <ChartsTooltip />
      <BrushLayer axisId={X_AXIS} control={control} hover={hover} onHover={onHover} />
    </ChartsContainer>
  );
}
