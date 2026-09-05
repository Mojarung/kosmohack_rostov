/** Главный график сезона: норма ±1σ, восстановленная кривая, исходные наблюдения по сенсорам,
 *  восстановленные контрольные точки и полосы эпизодов угнетения.
 *
 *  Собран композицией MUI X Charts: линии рисует библиотека, а слои нормы, наблюдений и эпизодов —
 *  собственные SVG-оверлеи поверх тех же шкал (useXScale/useYScale), чтобы каждый сенсор имел свой
 *  цвет и форму, а артефакты были видны отдельно. */

import { useMemo } from "react";
import { ChartsContainer } from "@mui/x-charts/ChartsContainer";
import { ChartsXAxis } from "@mui/x-charts/ChartsXAxis";
import { ChartsYAxis } from "@mui/x-charts/ChartsYAxis";
import { ChartsGrid } from "@mui/x-charts/ChartsGrid";
import { ChartsTooltip } from "@mui/x-charts/ChartsTooltip";
import { LinePlot } from "@mui/x-charts/LineChart";
import { useDrawingArea, useXScale, useYScale } from "@mui/x-charts/hooks";

import type { Episode, SeasonYear } from "../../api/types";
import { SENSOR_COLOR, axisDate, localDate, ms } from "../../lib/format";
import { BrushLayer } from "./BrushLayer";
import { PlotClip } from "./PlotClip";
import type { RangeControl } from "./range";

const X_AXIS = "season-x";
const Y_AXIS = "ndvi-y";

interface SeasonChartProps {
  season: SeasonYear;
  episodes: Episode[];
  height?: number;
  /** Общее окно просмотра трёх графиков сезона. */
  control: RangeControl;
  hover: number | null;
  onHover: (value: number | null) => void;
  raw?: boolean;
}

/** Допустимый суточный скачок нормы: среднее нескольких гладких кривых так резко не меняется. */
const NORM_MAX_DAILY_STEP = 0.03;
/** Сколько дней у каждого края сезона проверяем на скачки. */
const NORM_EDGE_DAYS = 7;

/** Обрезает края нормы, где она собрана не по всем годам и потому скачет день ото дня.
 *
 *  В первые и последние дни сезона часть кривых прошлых лет ещё ненадёжна и в среднее
 *  не входит, поэтому норма у краёв «ступенчатая» (0,24 → 0,37 за сутки). Отрезаем край
 *  до последнего скачка в пределах недели. Детектор это не трогает: Z уже посчитан
 *  на сервере, здесь только не рисуем ступеньки на графике. */
function trimNormEdges(points: { date: string; value: number }[]): { date: string; value: number }[] {
  const sorted = [...points].sort((a, b) => a.date.localeCompare(b.date));
  const jumpy = (i: number) => Math.abs(sorted[i].value - sorted[i - 1].value) > NORM_MAX_DAILY_STEP;
  let start = 0;
  for (let i = 1; i < Math.min(sorted.length, NORM_EDGE_DAYS); i += 1) if (jumpy(i)) start = i;
  let end = sorted.length;
  for (let i = Math.max(start + 1, sorted.length - NORM_EDGE_DAYS); i < sorted.length; i += 1) if (jumpy(i)) { end = i - 1; break; }
  return sorted.slice(start, end);
}

/** Ежедневная сетка дат сезона и значения кривой/нормы, выровненные по ней. */
function useSeasonGrid(season: SeasonYear) {
  return useMemo(() => {
    const keys = new Set<string>();
    season.curve.forEach((p) => keys.add(p.date));
    season.norm_mean.forEach((p) => keys.add(p.date));
    const dates = [...keys].sort();
    const curveMap = new Map(season.curve.map((p) => [p.date, p.value]));
    const normMap = new Map(trimNormEdges(season.norm_mean).map((p) => [p.date, p.value]));
    const stdMap = new Map(season.norm_std.map((p) => [p.date, p.value]));
    return {
      dates: dates.map(localDate),
      iso: dates,
      curve: dates.map((d) => curveMap.get(d) ?? null),
      norm: dates.map((d) => normMap.get(d) ?? null),
      band: dates
        .map((d) => {
          const mean = normMap.get(d);
          const std = stdMap.get(d);
          return mean === undefined || std === undefined ? null : { date: d, low: mean - std, high: mean + std };
        })
        .filter((v): v is { date: string; low: number; high: number } => v !== null),
    };
  }, [season]);
}

/** Полупрозрачная лента нормы ±1σ. */
function NormBand({ band }: { band: { date: string; low: number; high: number }[] }) {
  const xScale = useXScale(X_AXIS);
  const yScale = useYScale(Y_AXIS);
  if (band.length < 2) return null;
  const top = band.map((p) => `${xScale(ms(p.date))},${yScale(p.high)}`);
  const bottom = [...band].reverse().map((p) => `${xScale(ms(p.date))},${yScale(p.low)}`);
  return <polygon points={[...top, ...bottom].join(" ")} fill="#9aa091" opacity={0.16} />;
}

/** Вертикальные полосы найденных эпизодов: цвет — тяжесть. */
function EpisodeBands({ episodes }: { episodes: Episode[] }) {
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
            key={`${episode.start}-${episode.end}`}
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

/** Исходные наблюдения: круг по сенсору, крест — отбракованный артефакт. */
function ObservationDots({ season, raw = false }: { season: SeasonYear; raw?: boolean }) {
  const xScale = useXScale(X_AXIS);
  const yScale = useYScale(Y_AXIS);
  return (
    <g>
      {season.observations.filter(obs => raw || !obs.artifact).map((obs) => {
        const x = xScale(ms(obs.date));
        const y = yScale(raw ? obs.value : obs.harmonized);
        if (x === undefined || y === undefined) return null;
        const color = SENSOR_COLOR[obs.sensor] ?? "#666";
        const label = `${obs.date} · ${obs.sensor} · ${obs.value.toFixed(3)}${
          obs.artifact ? " · отбраковано как артефакт" : ` · в шкале S2 ${obs.harmonized.toFixed(3)}`
        }`;
        if (obs.artifact) {
          return (
            <g key={`a${obs.date}-${obs.sensor}`} stroke="#c8423f" strokeWidth={1.5} opacity={0.85}>
              <title>{label}</title>
              <line x1={x - 4} y1={y - 4} x2={x + 4} y2={y + 4} />
              <line x1={x - 4} y1={y + 4} x2={x + 4} y2={y - 4} />
            </g>
          );
        }
        return (
          <circle key={`o${obs.date}-${obs.sensor}`} cx={x} cy={y} r={3.4} fill={color} stroke="#fff" strokeWidth={1}>
            <title>{label}</title>
          </circle>
        );
      })}
    </g>
  );
}

/** Восстановленные контрольные точки (то, что уходит в submission). */
function RestoredDots({ season }: { season: SeasonYear }) {
  const xScale = useXScale(X_AXIS);
  const yScale = useYScale(Y_AXIS);
  return (
    <g>
      {season.restored.map((point) => {
        const x = xScale(ms(point.date));
        const y = yScale(point.value);
        if (x === undefined || y === undefined) return null;
        return (
          <g key={`r${point.date}`} transform={`translate(${x} ${y}) rotate(45)`}>
            <title>{`${point.date} · восстановлено моделью · ${point.value.toFixed(3)}`}</title>
            <rect x={-4} y={-4} width={8} height={8} fill="#b03a76" stroke="#fff" strokeWidth={1.2} />
          </g>
        );
      })}
    </g>
  );
}

export function SeasonChart({ season, episodes, height = 300, control, hover, onHover, raw = false }: SeasonChartProps) {
  const grid = useSeasonGrid(season);
  if (grid.dates.length === 0) {
    return <div className="meta">В этом сезоне нет наблюдений.</div>;
  }

  return (
    <ChartsContainer<"line">
      height={height}
      margin={{ left: 8, right: 12, top: 12, bottom: 4 }}
      dataset={undefined}
      series={[
        {
          type: "line",
          id: "norm",
          label: `норма (${season.norm_source})`,
          data: grid.norm,
          color: "#9aa091",
          showMark: false,
          curve: "monotoneX",
          valueFormatter: (v) => (v == null ? "нет" : v.toFixed(3)),
        },
        {
          type: "line",
          id: "curve",
          label: "Кривая NDVI",
          data: grid.curve,
          color: "#1a1d16",
          showMark: false,
          curve: "monotoneX",
          valueFormatter: (v) => (v == null ? "нет" : v.toFixed(3)),
        },
      ]}
      xAxis={[
        {
          id: X_AXIS,
          data: grid.dates,
          scaleType: "utc",
          min: new Date(control.view.from),
          max: new Date(control.view.to),
          tickNumber: 6,
          valueFormatter: (date: Date, context) => axisDate(date, context.location),
        },
      ]}
      yAxis={[{ id: Y_AXIS, min: 0, max: 1, width: 46, tickNumber: 5, valueFormatter: (v: number) => v.toFixed(1) }]}
    >
      <ChartsGrid horizontal />
      <PlotClip><EpisodeBands episodes={episodes} />
      <NormBand band={grid.band} />
      <LinePlot />
      <ObservationDots season={season} raw={raw} />
      {/* Восстановленные моделью точки видны и на главном графике: по ТЗ эксперт должен видеть
          исходный и восстановленный ряд рядом, а не открывать для этого «подробности». */}
      <RestoredDots season={season} /></PlotClip>
      <ChartsXAxis axisId={X_AXIS} />
      <ChartsYAxis axisId={Y_AXIS} label="NDVI" />
      <ChartsTooltip />
      <BrushLayer axisId={X_AXIS} control={control} hover={hover} onHover={onHover} />
    </ChartsContainer>
  );
}
