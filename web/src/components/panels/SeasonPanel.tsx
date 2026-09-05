/** Экран сезона: переключатель годов, три графика и список эпизодов с объяснениями.
 *  Используется и для полигонов кейса, и для территорий, собранных сервисом. */

import { useMemo, useState } from "react";

import type { PolygonDetail } from "../../api/types";
import { SENSOR_COLOR, plural } from "../../lib/format";
import { SeasonChart } from "../charts/SeasonChart";
import { WeatherChart } from "../charts/WeatherChart";
import { ZChart } from "../charts/ZChart";
import { EpisodeCard } from "./EpisodeCard";

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
            <div style={{ padding: "0 8px" }}>
              <SeasonChart season={season} episodes={episodes} />
            </div>
            <div style={{ padding: "0 8px 8px" }}>
              <ZChart season={season} episodes={episodes} />
            </div>
            <div style={{ padding: "8px var(--pad)", borderTop: "1px solid var(--line)" }}>
              <div className="eyebrow">Погода ERA5</div>
              {weather ? (
                <WeatherChart weather={weather} />
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
            <EpisodeCard key={`${episode.start}-${episode.end}`} episode={episode} />
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
