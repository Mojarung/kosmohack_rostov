/** Карточка эпизода угнетения: период, тяжесть, причина и разобранные аргументы. */

import type { Episode } from "../../api/types";
import { CAUSE_LABEL, dateRange, plural, severityTone } from "../../lib/format";
import { CurveArt, DroughtArt, FieldArt, GapArt, SproutArt } from "../art/Art";

const CAUSE_ART: Record<string, typeof DroughtArt> = {
  weather_drought: DroughtArt,
  unsown_or_changed: FieldArt,
  crop_rotation: SproutArt,
  data_suspect: GapArt,
};

interface EpisodeCardProps {
  episode: Episode;
  compact?: boolean;
  /** Приблизить период эпизода на графиках сезона. */
  onZoom?: () => void;
}

export function EpisodeCard({ episode, compact = false, onZoom }: EpisodeCardProps) {
  const tone = severityTone(episode.severity);
  const Art = CAUSE_ART[episode.cause] ?? CurveArt;
  const reasons = episode.reasons ? episode.reasons.split(" | ").filter(Boolean) : [];

  return (
    <article
      className="card"
      style={{
        padding: compact ? 16 : 20,
        borderLeft: `3px solid ${tone === "crit" ? "#c8423f" : tone === "warn" ? "#e8a33d" : "#3f7d45"}`,
        display: "grid",
        gridTemplateColumns: compact ? "1fr" : "40px 1fr",
        gap: compact ? 10 : 16,
        alignItems: "start",
      }}
    >
      {!compact && <Art size={40} />}
      <div className="stack" style={{ gap: 8, minWidth: 0 }}>
        <div className="spread" style={{ flexWrap: "wrap", gap: 8 }}>
          <div className="row" style={{ gap: 8, flexWrap: "wrap" }}>
            {onZoom ? (
              <button type="button" className="episode-zoom mono" onClick={onZoom} title="приблизить период на графиках">
                {dateRange(episode.start, episode.end)}
              </button>
            ) : (
              <strong className="mono" style={{ fontSize: 13 }}>
                {dateRange(episode.start, episode.end)}
              </strong>
            )}
            <span className="meta">
              {episode.days} {plural(episode.days, "день", "дня", "дней")} · {episode.n_obs}{" "}
              {plural(episode.n_obs, "наблюдение", "наблюдения", "наблюдений")}
            </span>
          </div>
          <div className="row" style={{ gap: 6 }}>
            <span className={`tag tag--${tone}`}>{episode.severity}</span>
            <span className="tag">
              Z<sub>min</sub> {episode.min_z.toFixed(1)}
            </span>
          </div>
        </div>

        <div className="row" style={{ gap: 8, flexWrap: "wrap" }}>
          <span className="tag tag--info">{CAUSE_LABEL[episode.cause] ?? episode.cause}</span>
          <span className="meta">уверенность {Math.round(episode.confidence * 100)} %</span>
        </div>

        <p style={{ color: "var(--ink-soft)" }}>{episode.text}</p>

        {reasons.length > 0 && (
          <details>
            <summary className="meta" style={{ cursor: "pointer" }}>
              разобрать по аргументам ({reasons.length})
            </summary>
            <ul className="stack" style={{ gap: 4, margin: "6px 0 0", paddingLeft: 16, color: "var(--muted)", fontSize: 12.5 }}>
              {reasons.map((reason, index) => (
                <li key={index}>{reason}</li>
              ))}
            </ul>
          </details>
        )}

        <div className="row" style={{ gap: 12, flexWrap: "wrap" }}>
          <span className="meta">норма: {episode.norm_source}</span>
          {episode.weather_source && <span className="meta">погода: {episode.weather_source}</span>}
          {episode.region_share_depressed != null && (
            <span className="meta">
              соседних полей ниже нормы: {Math.round(episode.region_share_depressed * 100)} %
            </span>
          )}
        </div>
      </div>
    </article>
  );
}
