/** Все найденные эпизоды по всем полям: фильтры по году, причине и тяжести. */

import { useEffect, useMemo, useRef, useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { Link } from "react-router-dom";

import { api } from "../api/client";
import { CAUSE_LABEL, dateRange, plural, severityTone } from "../lib/format";
import { Reveal } from "../components/motion/Reveal";
import { ErrorNote, Loader } from "../components/ui/Loader";

/** Сколько строк таблицы показывать сразу и на сколько прибавлять при прокрутке.
 *  Разом отрисованные шестьсот с лишним эпизодов задерживали переключение вкладки почти на полсекунды. */
const PAGE = 40;

export default function AnomaliesPage() {
  const [year, setYear] = useState<number | undefined>();
  const [cause, setCause] = useState<string | undefined>();
  const [severity, setSeverity] = useState<string | undefined>();

  const summary = useQuery({ queryKey: ["summary"], queryFn: api.summary });
  const episodes = useQuery({
    queryKey: ["episodes", year, cause, severity],
    queryFn: () => api.episodes({ year, cause, severity }),
  });

  const years = useMemo(
    () => Object.keys(summary.data?.by_year ?? {}).map(Number).sort((a, b) => a - b),
    [summary.data],
  );
  const causes = useMemo(() => Object.keys(summary.data?.by_cause ?? {}), [summary.data]);

  // видимая часть списка; при смене фильтров начинаем сначала
  const [visible, setVisible] = useState(PAGE);
  const sentinelRef = useRef<HTMLDivElement>(null);
  useEffect(() => setVisible(PAGE), [year, cause, severity]);
  useEffect(() => {
    const node = sentinelRef.current;
    if (!node) return;
    const observer = new IntersectionObserver(
      ([entry]) => { if (entry.isIntersecting) setVisible((n) => n + PAGE); },
      { rootMargin: "300px" },
    );
    observer.observe(node);
    return () => observer.disconnect();
  }, [episodes.data]);
  const maxByYear = Math.max(1, ...Object.values(summary.data?.by_year ?? { a: 1 }));

  return (
    <div className="stack" style={{ gap: 18 }}>
      <div className="stack" style={{ gap: 6 }}>
        <span className="eyebrow">Задача 2</span>
        <h1 style={{ fontSize: 34 }}>Периоды угнетения</h1>
        <p className="meta" style={{ maxWidth: "72ch" }}>
          Эпизод — это участок сезона, где кривая поля устойчиво или сильно ниже собственной нормы: не менее
          14 дней со средним отклонением ниже −1.2σ либо минимум ниже −1.5σ. Причина подбирается по правилам:
          погода ERA5 против нормы тех же дат, фенология сезона, поведение соседних полей.
        </p>
      </div>

      {summary.data && years.length > 0 && (
        <Reveal>
          <section className="card">
            <div className="eyebrow" style={{ marginBottom: 10 }}>
              Эпизодов по годам
            </div>
            <div className="row" style={{ gap: 6, alignItems: "flex-end", height: 96 }}>
              {years.map((y) => {
                const count = summary.data.by_year[String(y)] ?? 0;
                const active = year === y;
                return (
                  <button
                    key={y}
                    type="button"
                    onClick={() => setYear(active ? undefined : y)}
                    title={`${y}: ${count} ${plural(count, "эпизод", "эпизода", "эпизодов")}`}
                    style={{
                      flex: 1,
                      background: "transparent",
                      border: 0,
                      cursor: "pointer",
                      display: "flex",
                      flexDirection: "column",
                      alignItems: "center",
                      gap: 4,
                      height: "100%",
                      justifyContent: "flex-end",
                    }}
                  >
                    <span
                      style={{
                        width: "100%",
                        height: `${(count / maxByYear) * 74}px`,
                        background: active ? "var(--ink)" : "var(--line-strong)",
                        borderRadius: 3,
                        transition: "background .18s var(--ease), transform .18s var(--ease)",
                        transform: active ? "scaleY(1.02)" : undefined,
                        transformOrigin: "bottom",
                      }}
                    />
                    <span className="mono" style={{ fontSize: 10, color: active ? "var(--ink)" : "var(--muted)" }}>
                      {String(y).slice(2)}
                    </span>
                  </button>
                );
              })}
            </div>
          </section>
        </Reveal>
      )}

      <div className="row" style={{ gap: 8, flexWrap: "wrap" }}>
        <button
          type="button"
          className={`btn btn--sm ${severity ? "btn--ghost" : ""}`}
          onClick={() => setSeverity(undefined)}
        >
          все
        </button>
        <button
          type="button"
          className={`btn btn--sm ${severity === "критическая" ? "" : "btn--ghost"}`}
          onClick={() => setSeverity("критическая")}
        >
          критические
        </button>
        <button
          type="button"
          className={`btn btn--sm ${severity === "умеренная" ? "" : "btn--ghost"}`}
          onClick={() => setSeverity("умеренная")}
        >
          умеренные
        </button>
        <span style={{ width: 12 }} />
        <button type="button" className={`btn btn--sm ${cause ? "btn--ghost" : ""}`} onClick={() => setCause(undefined)}>
          любая причина
        </button>
        {causes.map((item) => (
          <button
            key={item}
            type="button"
            className={`btn btn--sm ${cause === item ? "" : "btn--ghost"}`}
            onClick={() => setCause(cause === item ? undefined : item)}
          >
            {CAUSE_LABEL[item] ?? item}
          </button>
        ))}
        {year && (
          <button type="button" className="btn btn--sm btn--ghost" onClick={() => setYear(undefined)}>
            сбросить {year}
          </button>
        )}
      </div>

      {episodes.isLoading && <Loader label="Загружаем эпизоды" />}
      {episodes.isError && <ErrorNote error={episodes.error} />}

      {episodes.data && (
        <>
          <p className="meta">
            Найдено {episodes.data.length} {plural(episodes.data.length, "эпизод", "эпизода", "эпизодов")}
          </p>
          <div className="card card--flush scroll" style={{ maxHeight: "62vh" }}>
            <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 13 }}>
              <thead style={{ position: "sticky", top: 0, background: "var(--surface)", zIndex: 1 }}>
                <tr style={{ textAlign: "left" }}>
                  {["Поле", "Год", "Период", "Дней", "Zmin", "Тяжесть", "Причина", "Объяснение"].map((title) => (
                    <th
                      key={title}
                      className="eyebrow"
                      style={{ padding: "10px 12px", borderBottom: "1px solid var(--line)", fontWeight: 400 }}
                    >
                      {title}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {episodes.data.slice(0, visible).map((episode, index) => (
                  <tr key={`${episode.pid}-${episode.start}-${index}`} style={{ borderBottom: "1px solid var(--line)" }}>
                    <td style={{ padding: "9px 12px" }}>
                      <Link to={`/field/${episode.pid}`} className="mono" style={{ textDecoration: "none" }}>
                        {episode.pid}
                      </Link>
                    </td>
                    <td className="mono" style={{ padding: "9px 12px" }}>
                      {episode.year}
                    </td>
                    <td style={{ padding: "9px 12px", whiteSpace: "nowrap" }}>
                      {dateRange(episode.start, episode.end)}
                    </td>
                    <td className="num" style={{ padding: "9px 12px" }}>
                      {episode.days}
                    </td>
                    <td className="num" style={{ padding: "9px 12px" }}>
                      {episode.min_z.toFixed(1)}
                    </td>
                    <td style={{ padding: "9px 12px" }}>
                      <span className={`tag tag--${severityTone(episode.severity)}`}>{episode.severity}</span>
                    </td>
                    <td style={{ padding: "9px 12px" }}>{CAUSE_LABEL[episode.cause] ?? episode.cause}</td>
                    <td style={{ padding: "9px 12px", color: "var(--muted)", maxWidth: 460 }}>
                      <span
                        title="нажмите, чтобы развернуть"
                        onClick={(event) => {
                          const node = event.currentTarget;
                          node.style.webkitLineClamp = node.style.webkitLineClamp === "unset" ? "3" : "unset";
                        }}
                        style={{
                          display: "-webkit-box",
                          WebkitLineClamp: 3,
                          WebkitBoxOrient: "vertical",
                          overflow: "hidden",
                          cursor: "pointer",
                        }}
                      >
                        {episode.text}
                      </span>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
            {visible < episodes.data.length && (
              <div ref={sentinelRef} className="meta" style={{ padding: "14px 12px" }}>
                Показано {visible} из {episodes.data.length}; прокрутите ниже, чтобы догрузить.
              </div>
            )}
          </div>
        </>
      )}
    </div>
  );
}
