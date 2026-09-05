/** Погода сезона по ERA5: суточные осадки столбиками и температура линией на второй оси. */

import { useMemo } from "react";
import { ChartsContainer } from "@mui/x-charts/ChartsContainer";
import { ChartsXAxis } from "@mui/x-charts/ChartsXAxis";
import { ChartsYAxis } from "@mui/x-charts/ChartsYAxis";
import { ChartsTooltip } from "@mui/x-charts/ChartsTooltip";
import { BarPlot } from "@mui/x-charts/BarChart";
import { LinePlot } from "@mui/x-charts/LineChart";

import type { WeatherYear } from "../../api/types";

const X_AXIS = "wx-x";
const RAIN = "rain";
const TEMP = "temp";

export function WeatherChart({ weather, height = 150 }: { weather: WeatherYear; height?: number }) {
  const data = useMemo(
    () => ({
      labels: weather.date.map((d) => d.slice(5)),
      precip: weather.precip,
      temp: weather.temp,
    }),
    [weather],
  );
  if (!data.labels.length) return null;

  return (
    <ChartsContainer<"bar" | "line">
      height={height}
      margin={{ left: 8, right: 8, top: 8, bottom: 4 }}
      series={[
        {
          type: "bar",
          id: "precip",
          label: "осадки, мм",
          data: data.precip,
          color: "#3a6ea5",
          yAxisId: RAIN,
          valueFormatter: (v) => (v == null ? "" : `${v.toFixed(1)} мм`),
        },
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
      xAxis={[{ id: X_AXIS, data: data.labels, scaleType: "band", tickNumber: 6 }]}
      yAxis={[
        { id: RAIN, position: "left", width: 42, label: "мм" },
        { id: TEMP, position: "right", width: 42, label: "°C" },
      ]}
    >
      <BarPlot />
      <LinePlot />
      <ChartsXAxis axisId={X_AXIS} />
      <ChartsYAxis axisId={RAIN} />
      <ChartsYAxis axisId={TEMP} />
      <ChartsTooltip />
    </ChartsContainer>
  );
}
