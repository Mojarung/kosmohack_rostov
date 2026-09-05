/** Карточка эпизода угнетения: сначала фраза для фермера и совет, технический разбор под спойлером. */

import type { Episode } from "../../api/types";
import { CAUSE_LABEL, dateRange, plural, severityTone } from "../../lib/format";
import { CAUSE_ADVICE, confidenceWord, episodeSentence } from "../../lib/plain";
import { CurveArt, DroughtArt, FieldArt, GapArt, SproutArt } from "../art/Art";

const CAUSE_ART: Record<string, typeof DroughtArt> = {
  weather_drought: DroughtArt,
  unsown_or_changed: FieldArt,
  crop_rotation: SproutArt,
  data_suspect: GapArt,
};

const SEVERITY_PLAIN: Record<string, string> = { критическая: "сильное отставание", умеренная: "заметное отставание" };

interface EpisodeCardProps {
  episode: Episode;
  compact?: boolean;
  /** Приблизить период эпизода на графиках сезона. */
  onZoom?: () => void;
}

/** Доля соседей ниже нормы словами: «соседи в порядке, просело только это поле» или «просели все поля вокруг».
 *  Для смены культуры и ошибок данных сравнение с соседями не показываем: оно только путает. */
function neighboursPlain(episode: Episode): string | null {
  const share = episode.region_share_depressed;
  if (share == null || !Number.isFinite(share) || episode.cause === "crop_rotation" || episode.cause === "data_suspect") return null;
  if (share >= 0.6) return "Просели почти все поля вокруг: причина общая, скорее всего погода.";
  if (share >= 0.35) return "Часть соседних полей в это время тоже просела.";
  return "Соседние поля в это время были в порядке: просело только это.";
}

export function EpisodeCard({ episode, compact = false, onZoom }: EpisodeCardProps) {
  const tone = severityTone(episode.severity);
  const Art = CAUSE_ART[episode.cause] ?? CurveArt;
  const reasons = episode.reasons ? episode.reasons.split(" | ").filter(Boolean) : [];
  const neighbours = neighboursPlain(episode);

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
              <button type="button" className="episode-zoom mono" onClick={onZoom} title="показать этот период на графике">
                {dateRange(episode.start, episode.end)}
              </button>
            ) : (
              <strong className="mono" style={{ fontSize: 13 }}>
                {dateRange(episode.start, episode.end)}
              </strong>
            )}
            <span className="meta">
              {episode.days} {plural(episode.days, "день", "дня", "дней")}
            </span>
          </div>
          <span className={`tag tag--${tone}`}>{SEVERITY_PLAIN[episode.severity] ?? episode.severity}</span>
        </div>

        <p className="episode-plain">{episodeSentence(episode)}</p>
        {neighbours && <p className="meta">{neighbours}</p>}
        <p className="episode-advice">
          <span className="eyebrow">Что делать</span> {CAUSE_ADVICE[episode.cause] ?? "Осмотрите поле лично."}
        </p>

        <details>
          <summary className="meta" style={{ cursor: "pointer" }}>
            подробности для агронома{reasons.length ? ` (${reasons.length} ${plural(reasons.length, "аргумент", "аргумента", "аргументов")})` : ""}
          </summary>
          <div className="stack" style={{ gap: 6, marginTop: 6 }}>
            <div className="row" style={{ gap: 6, flexWrap: "wrap" }}>
              <span className="tag tag--info">{CAUSE_LABEL[episode.cause] ?? episode.cause}</span>
              <span className="tag">Z<sub>min</sub> {episode.min_z.toFixed(1)}</span>
              <span className="meta">
                уверенность {Math.round(episode.confidence * 100)} % ({confidenceWord(episode.confidence)}) ·{" "}
                {episode.n_obs} {plural(episode.n_obs, "наблюдение", "наблюдения", "наблюдений")}
              </span>
            </div>
            <p style={{ color: "var(--ink-soft)", fontSize: 12.5, margin: 0 }}>{episode.text}</p>
            {reasons.length > 0 && (
              <ul className="stack" style={{ gap: 4, margin: 0, paddingLeft: 16, color: "var(--muted)", fontSize: 12.5 }}>
                {reasons.map((reason, index) => (
                  <li key={index}>{reason}</li>
                ))}
              </ul>
            )}
            <div className="row" style={{ gap: 12, flexWrap: "wrap" }}>
              <span className="meta">норма: {episode.norm_source}</span>
              {episode.weather_source && <span className="meta">погода: {episode.weather_source}</span>}
              {episode.region_share_depressed != null && (
                <span className="meta">соседних полей ниже нормы: {Math.round(episode.region_share_depressed * 100)} %</span>
              )}
            </div>
          </div>
        </details>
      </div>
    </article>
  );
}
