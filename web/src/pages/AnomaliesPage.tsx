/** Все найденные эпизоды: годы столбиками сверху, фильтры и таблица во всю оставшуюся высоту. */

import { useEffect, useMemo, useRef, useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { useNavigate } from "react-router-dom";

import { api } from "../api/client";
import { CAUSE_LABEL, dateRange, plural, severityTone } from "../lib/format";
import { ErrorNote } from "../components/ui/Loader";
import { PagePending } from "../components/ui/PagePending";

/** Порция строк таблицы: разом отрисованные шестьсот эпизодов задерживали переход на вкладку. */
const PAGE = 40;

export default function AnomaliesPage() {
  const navigate = useNavigate();
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
  const maxByYear = Math.max(1, ...Object.values(summary.data?.by_year ?? { a: 1 }));

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
  }, [episodes.data, visible]);

  if (summary.isLoading) return <PagePending label="Загружаем эпизоды" />;
  if (summary.isError) return <ErrorNote error={summary.error} />;

  return (
    <div className="workspace workspace--anomalies">
      <section className="pane">
        <div className="pane-head">
          <span className="row" style={{ gap: 10 }}>
            <span className="pane-title">Периоды угнетения</span>
            <span className="meta">
              не менее 14 дней ниже −1.2σ либо минимум ниже −1.5σ от нормы своего поля
            </span>
          </span>
          <span className="meta">
            {episodes.data?.length ?? 0} {plural(episodes.data?.length ?? 0, "эпизод", "эпизода", "эпизодов")}
          </span>
        </div>

        <div className="year-bars">
          {years.map((y) => {
            const value = summary.data?.by_year?.[String(y)] ?? 0;
            const active = year === y;
            return (
              <button
                key={y}
                type="button"
                className={"year-bar" + (active ? " is-active" : "")}
                onClick={() => setYear(active ? undefined : y)}
                title={`${y}: ${value} ${plural(value, "эпизод", "эпизода", "эпизодов")}`}
              >
                <span className="year-bar-track">
                  <span className="year-bar-fill" style={{ height: `${(value / maxByYear) * 100}%` }} />
                </span>
                <span className="year-bar-label mono">{String(y).slice(2)}</span>
              </button>
            );
          })}
        </div>

        <div className="pane-tools">
          <div className="chips">
            <button type="button" className={"chip" + (!severity ? " is-active" : "")} onClick={() => setSeverity(undefined)}>
              любая тяжесть
            </button>
            {["критическая", "умеренная"].map((item) => (
              <button
                key={item}
                type="button"
                className={"chip" + (severity === item ? " is-active" : "")}
                onClick={() => setSeverity(severity === item ? undefined : item)}
              >
                {item}
              </button>
            ))}
          </div>
          <span className="tools-divider" aria-hidden />
          <div className="chips">
            <button type="button" className={"chip" + (!cause ? " is-active" : "")} onClick={() => setCause(undefined)}>
              любая причина
            </button>
            {causes.map((item) => (
              <button
                key={item}
                type="button"
                className={"chip" + (cause === item ? " is-active" : "")}
                onClick={() => setCause(cause === item ? undefined : item)}
              >
                {CAUSE_LABEL[item] ?? item}
              </button>
            ))}
          </div>
        </div>

        <div className="pane-body">
          {episodes.isError && <ErrorNote error={episodes.error} />}
          <table className="rows">
            <thead>
              <tr>
                <th>Поле</th>
                <th>Год</th>
                <th>Период</th>
                <th>Дней</th>
                <th>Zmin</th>
                <th>Тяжесть</th>
                <th>Причина</th>
                <th>Объяснение</th>
              </tr>
            </thead>
            <tbody>
              {(episodes.data ?? []).slice(0, visible).map((episode, index) => (
                <tr
                  key={`${episode.pid}-${episode.start}-${index}`}
                  onClick={() => navigate(`/field/${episode.pid}`)}
                >
                  <td className="mono">{episode.pid}</td>
                  <td className="mono">{episode.year}</td>
                  <td style={{ whiteSpace: "nowrap" }}>{dateRange(episode.start, episode.end)}</td>
                  <td className="num">{episode.days}</td>
                  <td className="num">{episode.min_z.toFixed(1)}</td>
                  <td>
                    <span className={`tag tag--${severityTone(episode.severity)}`}>{episode.severity}</span>
                  </td>
                  <td>{CAUSE_LABEL[episode.cause] ?? episode.cause}</td>
                  <td className="cell-text"><span className="cell-clamp">{episode.text}</span></td>
                </tr>
              ))}
            </tbody>
          </table>
          {episodes.data && visible < episodes.data.length && (
            <div ref={sentinelRef} className="meta" style={{ padding: "12px 14px" }}>
              Показано {visible} из {episodes.data.length}; прокрутите ниже, чтобы догрузить.
            </div>
          )}
        </div>
      </section>
    </div>
  );
}
