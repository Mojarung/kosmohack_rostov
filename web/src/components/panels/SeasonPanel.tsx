/** Экран сезона: переключатель годов, три графика и список эпизодов с объяснениями.
 *  Используется и для полигонов кейса, и для территорий, собранных сервисом. */

import { useMemo, useState } from "react";

import type { PolygonDetail, SeasonYear, WeatherYear } from "../../api/types";
import { SENSOR_COLOR, ms, plural } from "../../lib/format";
import { SeasonChart } from "../charts/SeasonChart";
import { WeatherChart } from "../charts/WeatherChart";
import { ZChart } from "../charts/ZChart";
import { useDateRange, type DateRange } from "../charts/range";
import { EpisodeCard } from "./EpisodeCard";

/** Границы сезона по всем рядам: кривая, наблюдения и погода на одной шкале. */
function seasonBounds(season: SeasonYear | undefined, weather: WeatherYear | undefined): DateRange {
  const times: number[] = [];
  season?.curve.forEach((p) => times.push(ms(p.date)));
  season?.observations.forEach((p) => times.push(ms(p.date)));
  weather?.date.forEach((d) => times.push(ms(d)));
  if (!times.length) {
    const now = Date.now();
    return { from: now, to: now + 1 };
  }
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

/** Строка показателей под курсором: то же, что видно на графиках, но числами. */
function Readout({
  season,
  weather,
  at,
}: {
  season: SeasonYear;
  weather: WeatherYear | undefined;
  at: number | null;
}) {
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
          : new Date(at).toLocaleDateString("ru-RU", { day: "numeric", month: "long", year: "numeric", timeZone: "UTC" })}
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

function Legend() {
  const items = [
    ...Object.entries(SENSOR_COLOR).map(([name, color]) => ({ name, color, shape: "dot" as const })),
    { name: "восстановленная кривая", color: "#1a1d16", shape: "line" as const },
    { name: "восстановленные пропуски", color: "#b03a76", shape: "diamond" as const },
    { name: "норма ±1σ", color: "#9aa091", shape: "band" as const },
    { name: "артефакт, исключён", color: "#c8423f", shape: "cross" as const },
  ];
  return (
    <div className="row" style={{ gap: 14, flexWrap: "wrap" }}>
      {items.map((item) => (
        <span key={item.name} className="row meta" style={{ gap: 6 }}>
          <svg width="14" height="14" viewBox="0 0 14 14" aria-hidden>
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

export function SeasonPanel({ detail }: { detail: PolygonDetail }) {
  const years = useMemo(
    () => Object.keys(detail.years).map(Number).sort((a, b) => a - b),
    [detail.years],
  );
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
  const shape = (detail.shape ?? []).filter((s) => s.year === year);

  // общее окно просмотра и общий курсор трёх графиков
  const bounds = useMemo(() => seasonBounds(season, weather), [season, weather]);
  const control = useDateRange(bounds);
  const [hover, setHover] = useState<number | null>(null);

  return (
    <div className="stack" style={{ gap: 16 }}>
      <div className="card card--flush">
        <div
          className="spread"
          style={{ padding: "14px var(--pad)", borderBottom: "1px solid var(--line)", flexWrap: "wrap", gap: 10 }}
        >
          <div>
            <div className="eyebrow">Сезон</div>
            <div className="row" style={{ gap: 8, marginTop: 2 }}>
              <span style={{ fontFamily: "var(--font-serif)", fontSize: 22 }}>{year}</span>
              <span className="meta">
                {season ? `${season.observations.length} наблюдений · норма: ${season.norm_source}` : "нет данных"}
              </span>
            </div>
          </div>
          <div className="row" style={{ gap: 4, flexWrap: "wrap" }}>
            {years.map((y) => (
              <button
                key={y}
                type="button"
                onClick={() => setYear(y)}
                className="mono"
                style={{
                  cursor: "pointer",
                  border: `1px solid ${y === year ? "var(--ink)" : "var(--line)"}`,
                  background: y === year ? "var(--ink)" : "var(--surface)",
                  color: y === year ? "#fff" : "var(--ink-soft)",
                  borderRadius: 6,
                  padding: "3px 8px",
                  fontSize: 12,
                  position: "relative",
                  transition: "background .16s var(--ease), color .16s var(--ease)",
                }}
                title={
                  critical.has(y)
                    ? "есть критический эпизод"
                    : withEpisodes.has(y)
                      ? "есть эпизод угнетения"
                      : "без эпизодов"
                }
              >
                {y}
                {(withEpisodes.has(y) || critical.has(y)) && (
                  <span
                    aria-hidden
                    style={{
                      position: "absolute",
                      left: 6,
                      right: 6,
                      bottom: 2,
                      height: 2,
                      borderRadius: 2,
                      background: critical.has(y) ? "#c8423f" : "#e8a33d",
                    }}
                  />
                )}
              </button>
            ))}
          </div>
        </div>

        <div style={{ padding: "12px var(--pad) 4px" }}>
          <Legend />
        </div>

        {season ? (
          <>
            <Readout season={season} weather={weather} at={hover} />

            <div className="season-toolbar">
              <span className="meta">
                Выделите период мышью, чтобы приблизить. Колесо — масштаб, двойной клик — весь сезон.
              </span>
              <button
                type="button"
                className="btn btn--sm btn--ghost"
                onClick={control.reset}
                disabled={!control.zoomed}
              >
                Весь сезон
              </button>
            </div>

            <div style={{ padding: "0 8px" }}>
              <SeasonChart season={season} episodes={episodes} control={control} hover={hover} onHover={setHover} />
            </div>
            <div style={{ padding: "0 8px 8px" }}>
              <ZChart season={season} episodes={episodes} control={control} hover={hover} onHover={setHover} />
            </div>
            <div style={{ padding: "8px var(--pad) 14px" }}>
              <div className="eyebrow">Погода ERA5</div>
              {weather ? (
                <WeatherChart weather={weather} control={control} hover={hover} onHover={setHover} />
              ) : (
                <p className="meta" style={{ marginTop: 6 }}>
                  Для этого поля метеоряда нет: погода в объяснениях берётся по соседним полям региона.
                </p>
              )}
            </div>
          </>
        ) : (
          <p className="meta" style={{ padding: "var(--pad)" }}>
            Нет наблюдений за этот сезон.
          </p>
        )}
      </div>

      <div className="stack" style={{ gap: 10 }}>
        <div className="spread">
          <h2>Эпизоды угнетения</h2>
          <span className="meta">
            {episodes.length
              ? `${episodes.length} ${plural(episodes.length, "эпизод", "эпизода", "эпизодов")} в ${year} году`
              : `в ${year} году не найдено`}
          </span>
        </div>
        {episodes.length === 0 && (
          <div className="card card--sunk">
            <p className="meta">
              Устойчивых или сильных отклонений ниже нормы не обнаружено: кривая сезона держится в пределах ±1σ
              от нормы этого поля.
            </p>
          </div>
        )}
        {episodes
          .slice()
          .sort((a, b) => a.min_z - b.min_z)
          .map((episode) => (
            <EpisodeCard
              key={`${episode.start}-${episode.end}`}
              episode={episode}
              onZoom={() => control.select(ms(episode.start), ms(episode.end))}
            />
          ))}
        {shape.map((item) => (
          <div key={`${item.year}-${item.shape_direction}`} className="card card--sunk">
            <div className="eyebrow">Нетипичная форма сезона · {item.shape_direction}</div>
            <p style={{ marginTop: 6, color: "var(--ink-soft)" }}>{item.shape_reasons}</p>
          </div>
        ))}
      </div>
    </div>
  );
}
