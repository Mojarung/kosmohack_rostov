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
import { ms } from "../../lib/format";

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

export function ZChart({ season, episodes, height = 150 }: { season: SeasonYear; episodes: Episode[]; height?: number }) {
  const data = useMemo(
    () => ({
      dates: season.z.map((p) => new Date(`${p.date}T00:00:00Z`)),
      values: season.z.map((p) => p.value),
    }),
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
          scaleType: "time",
          tickNumber: 6,
          valueFormatter: (date: Date, context) =>
            context.location === "tick"
              ? date.toLocaleDateString("ru-RU", { month: "short", timeZone: "UTC" })
              : date.toLocaleDateString("ru-RU", { day: "numeric", month: "long", timeZone: "UTC" }),
        },
      ]}
      yAxis={[{ id: Y_AXIS, min: -4, max: 3, width: 42, tickNumber: 4 }]}
    >
      <Bands episodes={episodes} />
      <ChartsReferenceLine y={-1} lineStyle={{ stroke: "#e8a33d", strokeDasharray: "4 4" }} />
      <ChartsReferenceLine y={-2} lineStyle={{ stroke: "#c8423f", strokeDasharray: "4 4" }} />
      <LinePlot />
      <ChartsXAxis axisId={X_AXIS} />
      <ChartsYAxis axisId={Y_AXIS} label="σ" />
      <ChartsTooltip />
    </ChartsContainer>
  );
}
