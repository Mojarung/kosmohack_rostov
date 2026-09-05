/** Экран сезона: слева графики, справа список эпизодов. Обе колонки во всю высоту рабочей области,
 *  прокручивается только содержимое. Используется и для полигонов кейса, и для собранных территорий. */

import { useEffect, useMemo, useRef, useState } from "react";

import type { PolygonDetail, SeasonYear, WeatherYear } from "../../api/types";
import { SENSOR_COLOR, ms, plural } from "../../lib/format";
import { SeasonChart } from "../charts/SeasonChart";
import { WeatherChart } from "../charts/WeatherChart";
import { ZChart } from "../charts/ZChart";
import { useDateRange, type DateRange } from "../charts/range";
import { EpisodeCard } from "./EpisodeCard";

/** Высоты вспомогательных графиков; остальное место отдаётся главному. */
const Z_HEIGHT = 104;
const WEATHER_HEIGHT = 116;
const NDVI_MIN = 180;

/** Границы сезона по всем рядам: кривая, наблюдения и погода на одной шкале. */
function seasonBounds(season: SeasonYear | undefined, weather: WeatherYear | undefined): DateRange {
  const times: number[] = [];
  season?.curve.forEach((p) => times.push(ms(p.date)));
  season?.observations.forEach((p) => times.push(ms(p.date)));
  weather?.date.forEach((d) => times.push(ms(d)));
  if (!times.length) return { from: 0, to: 1 };
  return { from: Math.min(...times), to: Math.max(...times) };
}

/** Значение ряда в ближайшей к курсору дате. */
function nearest(points: { date: string; value: number }[] | undefined, at: number | null) {
  if (!points?.length || at === null) return null;
  let best = points[0];
  let bestGap = Math.abs(ms(best.date) - at);
  for (const point of points) {
    const gap = Math.abs(ms(point.date) - at);
    if (gap < bestGap) {
      best = point;
      bestGap = gap;
    }
  }
  return bestGap > 5 * 24 * 3600 * 1000 ? null : best;
}

/** Высота блока: главный график занимает всё, что осталось от вспомогательных. */
function useAvailableHeight(): [React.RefObject<HTMLDivElement | null>, number] {
  const ref = useRef<HTMLDivElement>(null);
  const [height, setHeight] = useState(340);
  useEffect(() => {
    const node = ref.current;
    if (!node) return;
    const observer = new ResizeObserver(([entry]) => setHeight(entry.contentRect.height));
    observer.observe(node);
    return () => observer.disconnect();
  }, []);
  return [ref, height];
}

function Legend() {
  const items = [
    ...Object.entries(SENSOR_COLOR).map(([name, color]) => ({ name, color, shape: "dot" as const })),
    { name: "кривая", color: "#1a1d16", shape: "line" as const },
    { name: "восстановлено", color: "#b03a76", shape: "diamond" as const },
    { name: "норма ±1σ", color: "#9aa091", shape: "band" as const },
    { name: "артефакт", color: "#c8423f", shape: "cross" as const },
  ];
  return (
    <div className="season-legend">
      {items.map((item) => (
        <span key={item.name} className="season-legend-item">
          <svg width="12" height="12" viewBox="0 0 14 14" aria-hidden>
            {item.shape === "dot" && <circle cx="7" cy="7" r="4" fill={item.color} />}
            {item.shape === "line" && <path d="M1 9c3-6 9 2 12-4" stroke={item.color} strokeWidth="1.8" fill="none" />}
            {item.shape === "diamond" && (
              <rect x="3" y="3" width="8" height="8" transform="rotate(45 7 7)" fill={item.color} />
            )}
            {item.shape === "band" && <rect x="1" y="4" width="12" height="6" fill={item.color} opacity="0.35" />}
            {item.shape === "cross" && (
              <g stroke={item.color} strokeWidth="1.8">
                <path d="M3 3l8 8M11 3l-8 8" />
              </g>
            )}
          </svg>
          {item.name}
        </span>
      ))}
    </div>
  );
}

/** Строка показателей под курсором: то же, что на графиках, но числами. */
function Readout({ season, weather, at }: { season: SeasonYear; weather?: WeatherYear; at: number | null }) {
  const curve = nearest(season.curve, at);
  const norm = nearest(season.norm_mean, at);
  const z = nearest(season.z, at);
  const rain = weather ? nearest(weather.date.map((d, i) => ({ date: d, value: weather.precip[i] })), at) : null;
  const temp = weather ? nearest(weather.date.map((d, i) => ({ date: d, value: weather.temp[i] })), at) : null;

  const cells = [
    { label: "NDVI", value: curve ? curve.value.toFixed(3) : "—" },
    { label: "норма", value: norm ? norm.value.toFixed(3) : "—" },
    { label: "отклонение", value: z ? `${z.value.toFixed(1)} σ` : "—" },
    { label: "осадки", value: rain ? `${rain.value.toFixed(1)} мм` : "—" },
    { label: "температура", value: temp ? `${temp.value.toFixed(1)} °C` : "—" },
  ];

  return (
    <div className="season-readout">
      <span className="season-readout-date mono">
        {at === null
          ? "наведите на график"
          : new Date(at).toLocaleDateString("ru-RU", { day: "numeric", month: "long", timeZone: "UTC" })}
      </span>
      {cells.map((cell) => (
        <span key={cell.label} className="season-readout-cell">
          <span className="eyebrow">{cell.label}</span>
          <span className="num">{cell.value}</span>
        </span>
      ))}
    </div>
  );
}

export function SeasonPanel({ detail }: { detail: PolygonDetail }) {
  const years = useMemo(() => Object.keys(detail.years).map(Number).sort((a, b) => a - b), [detail.years]);
  const withEpisodes = useMemo(() => new Set(detail.episodes.map((e) => e.year)), [detail.episodes]);
  const critical = useMemo(
    () => new Set(detail.episodes.filter((e) => e.severity === "критическая").map((e) => e.year)),
    [detail.episodes],
  );
  const [year, setYear] = useState<number>(() => {
    const criticalYears = years.filter((y) => critical.has(y));
    return criticalYears.at(-1) ?? years.at(-1) ?? 0;
  });

  const season = detail.years[String(year)];
  const episodes = detail.episodes.filter((e) => e.year === year);
  const weather = detail.weather?.[String(year)];
  const shape = (detail.shape ?? []).filter((item) => item.year === year);

  const bounds = useMemo(() => seasonBounds(season, weather), [season, weather]);
  const control = useDateRange(bounds);
  const [hover, setHover] = useState<number | null>(null);
  const [chartsRef, chartsHeight] = useAvailableHeight();

  const ndviHeight = Math.max(
    NDVI_MIN,
    chartsHeight - Z_HEIGHT - (weather ? WEATHER_HEIGHT : 34) - 26,
  );

  return (
    <div className="season-layout">
      <section className="pane">
        <div className="pane-head">
          <span className="row" style={{ gap: 10 }}>
            <span className="pane-title">Сезон {year}</span>
            <span className="meta">
              {season ? `${season.observations.length} наблюдений · норма: ${season.norm_source}` : "нет данных"}
            </span>
          </span>
          <div className="chips season-years">
            {years.map((y) => (
              <button
                key={y}
                type="button"
                className={"chip chip--year" + (y === year ? " is-active" : "")}
                onClick={() => setYear(y)}
                title={
                  critical.has(y) ? "есть критический эпизод" : withEpisodes.has(y) ? "есть эпизод" : "без эпизодов"
                }
              >
                {String(y).slice(2)}
                {(withEpisodes.has(y) || critical.has(y)) && (
                  <span
                    aria-hidden
                    className="chip-mark"
                    style={{ background: critical.has(y) ? "#c8423f" : "#e8a33d" }}
                  />
                )}
              </button>
            ))}
          </div>
        </div>

        {season ? (
          <>
            <Readout season={season} weather={weather} at={hover} />
            <div className="season-toolbar">
              <Legend />
              <button type="button" className="btn btn--sm btn--ghost" onClick={control.reset} disabled={!control.zoomed}>
                Весь сезон
              </button>
            </div>

            <div ref={chartsRef} className="season-charts">
              <SeasonChart
                season={season}
                episodes={episodes}
                height={ndviHeight}
                control={control}
                hover={hover}
                onHover={setHover}
              />
              <ZChart
                season={season}
                episodes={episodes}
                height={Z_HEIGHT}
                control={control}
                hover={hover}
                onHover={setHover}
              />
              {weather ? (
                <WeatherChart weather={weather} height={WEATHER_HEIGHT} control={control} hover={hover} onHover={setHover} />
              ) : (
                <p className="meta" style={{ padding: "6px 16px" }}>
                  Метеоряда для этого поля нет: погода в объяснениях берётся по соседним полям региона.
                </p>
              )}
            </div>
          </>
        ) : (
          <p className="meta" style={{ padding: 16 }}>
            Нет наблюдений за этот сезон.
          </p>
        )}
      </section>

      <section className="pane">
        <div className="pane-head">
          <span className="pane-title">Эпизоды угнетения</span>
          <span className="meta">
            {episodes.length
              ? `${episodes.length} ${plural(episodes.length, "эпизод", "эпизода", "эпизодов")}`
              : "не найдено"}
          </span>
        </div>
        <div className="pane-body pane-body--pad stack" style={{ gap: 10 }}>
          {episodes.length === 0 && (
            <p className="meta">
              Устойчивых отклонений ниже нормы нет: кривая сезона держится в пределах ±1σ от нормы этого поля.
            </p>
          )}
          {episodes
            .slice()
            .sort((a, b) => a.min_z - b.min_z)
            .map((episode) => (
              <EpisodeCard
                key={`${episode.start}-${episode.end}`}
                episode={episode}
                compact
                onZoom={() => control.select(ms(episode.start), ms(episode.end))}
              />
            ))}
          {shape.map((item) => (
            <div key={`${item.year}-${item.shape_direction}`} className="card card--sunk">
              <div className="eyebrow">Нетипичная форма сезона · {item.shape_direction}</div>
              <p style={{ marginTop: 6, color: "var(--ink-soft)", fontSize: 13 }}>{item.shape_reasons}</p>
            </div>
          ))}
        </div>
      </section>
    </div>
  );
}
