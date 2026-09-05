/** Поля кейса: список полей слева, метрики решения и причины угнетения справа.
 *  Экран умещается целиком, прокручивается только список полей. */

import { useMemo, useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { useNavigate } from "react-router-dom";

import { api } from "../api/client";
import { CAUSE_LABEL, plural } from "../lib/format";
import { ErrorNote } from "../components/ui/Loader";
import { PagePending } from "../components/ui/PagePending";

type Filter = "all" | "critical" | "quiet";

const FILTERS: { key: Filter; label: string }[] = [
  { key: "all", label: "все" },
  { key: "critical", label: "с критическими" },
  { key: "quiet", label: "без эпизодов" },
];

/** Горизонтальная полоса доли: причина угнетения и сколько эпизодов на неё приходится. */
function CauseBar({ label, value, total }: { label: string; value: number; total: number }) {
  const share = total ? value / total : 0;
  return (
    <div className="cause-row">
      <span className="cause-label">{label}</span>
      <span className="cause-track">
        <span className="cause-fill" style={{ width: `${Math.max(2, share * 100)}%` }} />
      </span>
      <span className="cause-value num">{value}</span>
    </div>
  );
}

export default function OverviewPage() {
  const navigate = useNavigate();
  const [query, setQuery] = useState("");
  const [filter, setFilter] = useState<Filter>("all");

  const polygons = useQuery({ queryKey: ["polygons"], queryFn: api.polygons });
  const meta = useQuery({ queryKey: ["meta"], queryFn: api.meta });
  const summary = useQuery({ queryKey: ["summary"], queryFn: api.summary });

  const rows = useMemo(() => {
    const list = polygons.data ?? [];
    const needle = query.trim().toLowerCase();
    return list
      .filter((item) => (needle ? item.pid.toLowerCase().includes(needle) || item.crop.toLowerCase().includes(needle) : true))
      .filter((item) =>
        filter === "critical" ? item.n_critical > 0 : filter === "quiet" ? item.n_episodes === 0 : true,
      )
      .sort((a, b) => b.n_critical - a.n_critical || b.n_episodes - a.n_episodes);
  }, [polygons.data, query, filter]);

  const causes = useMemo(() => {
    const entries = Object.entries(summary.data?.by_cause ?? {});
    const total = entries.reduce((sum, [, value]) => sum + value, 0);
    return { entries: entries.sort((a, b) => b[1] - a[1]), total };
  }, [summary.data]);

  if (polygons.isLoading) return <PagePending label="Загружаем поля кейса" />;
  if (polygons.isError) return <ErrorNote error={polygons.error} />;

  const task1 = meta.data?.task1;
  const task2 = meta.data?.task2;

  return (
    <div className="workspace workspace--overview">
      <section className="pane">
        <div className="pane-head">
          <span className="pane-title">Поля кейса</span>
          <span className="meta">
            {rows.length} из {polygons.data?.length ?? 0}
          </span>
        </div>

        <div className="pane-tools">
          <input
            className="input"
            placeholder="поиск по номеру поля или культуре"
            value={query}
            onChange={(event) => setQuery(event.target.value)}
            style={{ maxWidth: 300 }}
          />
          <div className="chips">
            {FILTERS.map((item) => (
              <button
                key={item.key}
                type="button"
                className={"chip" + (filter === item.key ? " is-active" : "")}
                onClick={() => setFilter(item.key)}
              >
                {item.label}
              </button>
            ))}
          </div>
        </div>

        <div className="pane-body">
          <table className="rows">
            <thead>
              <tr>
                <th>Поле</th>
                <th>Культура</th>
                <th>Сезоны</th>
                <th>Наблюдений</th>
                <th>Эпизодов</th>
                <th>Критических</th>
                <th>Погода</th>
              </tr>
            </thead>
            <tbody>
              {rows.map((item) => (
                <tr key={item.pid} onClick={() => navigate(`/field/${item.pid}`)}>
                  <td className="mono">{item.pid}</td>
                  <td>{item.crop}</td>
                  <td className="num">
                    {item.years[0]}–{item.years.at(-1)}
                  </td>
                  <td className="num">{item.n_obs}</td>
                  <td className="num">{item.n_episodes}</td>
                  <td>
                    {item.n_critical > 0 ? (
                      <span className="tag tag--crit">{item.n_critical}</span>
                    ) : (
                      <span className="meta">—</span>
                    )}
                  </td>
                  <td className="meta">{item.has_weather ? "ERA5" : "нет"}</td>
                </tr>
              ))}
              {rows.length === 0 && (
                <tr>
                  <td colSpan={7} className="meta" style={{ padding: 18 }}>
                    Ничего не нашлось. Измените запрос или снимите фильтр.
                  </td>
                </tr>
              )}
            </tbody>
          </table>
        </div>
      </section>

      <div className="workspace-side">
        <section className="pane">
          <div className="pane-head">
            <span className="pane-title">Результат</span>
            <span className="meta">на данных кейса</span>
          </div>
          <div className="kpi-grid">
            <div className="kpi">
              <span className="eyebrow">RMSE восстановления</span>
              <span className="kpi-value">{task1 ? task1.rmse_val.toFixed(3) : "—"}</span>
              <span className="kpi-note">
                у baseline «среднее соседей» {task1 ? task1.baseline_rmse.toFixed(3) : "—"}
              </span>
            </div>
            <div className="kpi">
              <span className="eyebrow">GapScore</span>
              <span className="kpi-value">{task1 ? task1.gap_score.toFixed(1) : "—"}</span>
              <span className="kpi-note">из 30 возможных</span>
            </div>
            <div className="kpi">
              <span className="eyebrow">Контрольных точек</span>
              <span className="kpi-value">{task1 ? task1.n_gaps.toLocaleString("ru-RU") : "—"}</span>
              <span className="kpi-note">скрытых значений в тесте</span>
            </div>
            <div className="kpi">
              <span className="eyebrow">Эпизодов найдено</span>
              <span className="kpi-value">{task2 ? task2.n_episodes.toLocaleString("ru-RU") : "—"}</span>
              <span className="kpi-note">
                в {task2?.n_seasons ?? 0} {plural(task2?.n_seasons ?? 0, "сезоне", "сезонах", "сезонах")}
              </span>
            </div>
          </div>
        </section>

        <section className="pane">
          <div className="pane-head">
            <span className="pane-title">Причины угнетения</span>
            <span className="meta">{causes.total} эпизодов</span>
          </div>
          <div className="pane-body pane-body--pad">
            {causes.entries.map(([cause, value]) => (
              <CauseBar key={cause} label={CAUSE_LABEL[cause] ?? cause} value={value} total={causes.total} />
            ))}
          </div>
        </section>
      </div>
    </div>
  );
}
