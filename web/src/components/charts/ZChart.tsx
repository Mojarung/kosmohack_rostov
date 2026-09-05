/** Отклонение от нормы в единицах σ: пороги −1 и −2 из ТЗ, полосы эпизодов те же, что на графике сезона. */

import { useMemo } from "react";
import { ChartsContainer } from "@mui/x-charts/ChartsContainer";
import { ChartsXAxis } from "@mui/x-charts/ChartsXAxis";
import { ChartsYAxis } from "@mui/x-charts/ChartsYAxis";
import { ChartsReferenceLine } from "@mui/x-charts/ChartsReferenceLine";
import { ChartsTooltip } from "@mui/x-charts/ChartsTooltip";
import { LinePlot } from "@mui/x-charts/LineChart";
import { useDrawingArea, useXScale } from "@mui/x-charts/hooks";

import type { Episode, SeasonYear } from "../../api/types";
import { axisDate, localDate, ms } from "../../lib/format";
import { BrushLayer } from "./BrushLayer";
import { PlotClip } from "./PlotClip";
import type { RangeControl } from "./range";

const X_AXIS = "z-x";
const Y_AXIS = "z-y";

function Bands({ episodes }: { episodes: Episode[] }) {
  const xScale = useXScale(X_AXIS);
  const { top, height } = useDrawingArea();
  return (
    <g>
      {episodes.map((episode) => {
        const x1 = xScale(ms(episode.start));
        const x2 = xScale(ms(episode.end));
        if (x1 === undefined || x2 === undefined) return null;
        return (
          <rect
            key={episode.start}
            x={Math.min(x1, x2)}
            y={top}
            width={Math.max(2, Math.abs(x2 - x1))}
            height={height}
            fill={episode.severity === "критическая" ? "#c8423f" : "#e8a33d"}
            opacity={0.13}
          />
        );
      })}
    </g>
  );
}

interface ZChartProps {
  season: SeasonYear;
  episodes: Episode[];
  height?: number;
  control: RangeControl;
  hover: number | null;
  onHover: (value: number | null) => void;
}

export function ZChart({ season, episodes, height = 210, control, hover, onHover }: ZChartProps) {
  const data = useMemo(
    () => {
      const values = season.z.map(p => Number.isFinite(p.value) ? p.value : null);
      let low = 0, high = 0;
      for (const value of values) {
        if (value === null) continue;
        low = Math.min(low, value);
        high = Math.max(high, value);
      }
      // Одна шкала на сезон: при приближении линия не меняет высоту, пики не обрезаются.
      const padding = Math.max(0.3, (high - low) * 0.08);
      return {
        dates: season.z.map(p => localDate(p.date)), values,
        min: Math.min(-4, Math.floor(low - padding)),
        max: Math.max(3, Math.ceil(high + padding)),
      };
    },
    [season],
  );
  if (data.dates.length < 2) return null;

  return (
    <ChartsContainer<"line">
      height={height}
      margin={{ left: 8, right: 12, top: 8, bottom: 4 }}
      series={[
        {
          type: "line",
          id: "z",
          label: "отклонение от нормы, σ",
          data: data.values,
          color: "#1a1d16",
          showMark: false,
          curve: "monotoneX",
          valueFormatter: (v) => (v == null ? "нет" : `${v.toFixed(2)} σ`),
        },
      ]}
      xAxis={[
        {
          id: X_AXIS,
          data: data.dates,
          scaleType: "utc",
          min: new Date(control.view.from),
          max: new Date(control.view.to),
          tickNumber: 6,
          valueFormatter: (date: Date, context) => axisDate(date, context.location),
        },
      ]}
      yAxis={[{ id: Y_AXIS, min: data.min, max: data.max, width: 54, tickNumber: 5 }]}
    >
      <PlotClip><Bands episodes={episodes} />
      <ChartsReferenceLine y={-1} lineStyle={{ stroke: "#e8a33d", strokeDasharray: "4 4" }} />
      <ChartsReferenceLine y={-2} lineStyle={{ stroke: "#c8423f", strokeDasharray: "4 4" }} />
      <LinePlot /></PlotClip>
      <ChartsXAxis axisId={X_AXIS} />
      <ChartsYAxis axisId={Y_AXIS} label="σ" />
      <ChartsTooltip />
      <BrushLayer axisId={X_AXIS} control={control} hover={hover} onHover={onHover} />
    </ChartsContainer>
  );
}
