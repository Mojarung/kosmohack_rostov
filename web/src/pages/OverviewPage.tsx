/** Обзор: что делает сервис, метрики решения и список полей кейса с быстрым поиском. */

import { useMemo, useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { Link } from "react-router-dom";

import { api } from "../api/client";
import { CAUSE_LABEL, plural } from "../lib/format";
import { CurveArt, GapArt, LayersArt, SatelliteArt } from "../components/art/Art";
import { HeroArt } from "../components/art/HeroArt";
import { Reveal } from "../components/motion/Reveal";
import { ErrorNote, Loader } from "../components/ui/Loader";

function Hero() {
  return (
    <section
      className="card"
      style={{
        display: "grid",
        gridTemplateColumns: "minmax(0, 1.25fr) minmax(260px, 0.75fr)",
        gap: "clamp(20px, 3vw, 48px)",
        alignItems: "center",
        padding: "clamp(26px, 3.4vw, 52px)",
        position: "relative",
        overflow: "hidden",
      }}
    >
      <div className="stack" style={{ gap: 18, minWidth: 0 }}>
        <span className="eyebrow">Мониторинг вегетационной динамики</span>
        <h1>
          Спутниковый ряд поля без пропусков и с объяснением, что пошло не так
        </h1>
        <p style={{ color: "var(--ink-soft)", maxWidth: "62ch", fontSize: 15 }}>
          Сервис собирает снимки Sentinel-2, Landsat и MODIS по контуру участка, приводит их к одной шкале,
          восстанавливает пропущенные даты и сравнивает сезон с нормой этого же поля. Периоды угнетения
          показываются на графике и объясняются погодой, фенологией и состоянием соседних полей.
        </p>
        <div className="row" style={{ gap: 10, flexWrap: "wrap" }}>
          <Link to="/explore" className="btn" style={{ textDecoration: "none" }}>
            Выбрать поле на карте
          </Link>
          <Link to="/anomalies" className="btn btn--ghost" style={{ textDecoration: "none" }}>
            Смотреть найденные аномалии
          </Link>
        </div>
      </div>
      <div className="row" style={{ justifyContent: "center" }}>
        <HeroArt />
      </div>
    </section>
  );
}

function MetricTile({
  art,
  title,
  value,
  note,
  index,
}: {
  art: React.ReactNode;
  title: string;
  value: string;
  note: string;
  index: number;
}) {
  return (
    <Reveal index={index}>
      <div className="card stack" style={{ gap: 10, height: "100%" }}>
        {art}
        <div className="eyebrow">{title}</div>
        <div className="num" style={{ fontSize: 30, lineHeight: 1.1 }}>
          {value}
        </div>
        <p className="meta">{note}</p>
      </div>
    </Reveal>
  );
}

const PAGE = 24;   // сколько карточек полей показывать сразу

export default function OverviewPage() {
  const meta = useQuery({ queryKey: ["meta"], queryFn: api.meta });
  const polygons = useQuery({ queryKey: ["polygons"], queryFn: api.polygons });
  const summary = useQuery({ queryKey: ["summary"], queryFn: api.summary });
  const [search, setSearch] = useState("");
  const [onlyCritical, setOnlyCritical] = useState(false);
  const [limit, setLimit] = useState(PAGE);

  const list = useMemo(() => {
    const items = polygons.data ?? [];
    const query = search.trim().toLowerCase();
    return items
      .filter((item) => (query ? `${item.pid} ${item.crop}`.toLowerCase().includes(query) : true))
      .filter((item) => (onlyCritical ? item.n_critical > 0 : true))
      .sort((a, b) => b.n_critical - a.n_critical || b.n_episodes - a.n_episodes || a.pid.localeCompare(b.pid));
  }, [polygons.data, search, onlyCritical]);

  const topCauses = useMemo(() => {
    const by = summary.data?.by_cause ?? {};
    return Object.entries(by)
      .sort((a, b) => b[1] - a[1])
      .slice(0, 4);
  }, [summary.data]);

  return (
    <div className="stack" style={{ gap: 22 }}>
      <Hero />

      {meta.isError && <ErrorNote error={meta.error} />}
      {meta.data && (
        <div
          style={{
            display: "grid",
            gap: 14,
            gridTemplateColumns: "repeat(auto-fit, minmax(230px, 1fr))",
          }}
        >
          <MetricTile
            index={0}
            art={<GapArt size={36} />}
            title="Восстановление пропусков"
            value={`RMSE ${meta.data.task1.rmse_val.toFixed(3)}`}
            note={`GapScore ${meta.data.task1.gap_score.toFixed(1)} из 30 против ${meta.data.task1.baseline_gap_score.toFixed(1)} у baseline «среднее соседей» (RMSE ${meta.data.task1.baseline_rmse.toFixed(3)}).`}
          />
          <MetricTile
            index={1}
            art={<CurveArt size={36} />}
            title="Контрольные точки"
            value={meta.data.task1.n_gaps.toLocaleString("ru-RU")}
            note={`Скрытые значения, которые модель восстанавливает в файле ${meta.data.data.test_file}.`}
          />
          <MetricTile
            index={2}
            art={<LayersArt size={36} />}
            title="Обучающих наблюдений"
            value={meta.data.task1.train_points.toLocaleString("ru-RU")}
            note={meta.data.task1.models}
          />
          <MetricTile
            index={3}
            art={<SatelliteArt size={36} />}
            title="Найдено эпизодов"
            value={String(meta.data.task2.n_episodes)}
            note={`В ${meta.data.task2.n_seasons} сезонах ${meta.data.task2.n_polygons} полей; эпизод хотя бы раз встречается в ${Math.round(meta.data.task2.seasons_with_episode * 100)} % сезонов.`}
          />
        </div>
      )}

      {topCauses.length > 0 && (
        <Reveal>
          <section className="card">
            <div className="spread" style={{ marginBottom: 12 }}>
              <h2>Причины угнетения по всем полям</h2>
              <Link to="/anomalies" className="meta" style={{ textDecoration: "none" }}>
                перейти к списку →
              </Link>
            </div>
            <div style={{ display: "grid", gap: 10, gridTemplateColumns: "repeat(auto-fit, minmax(190px, 1fr))" }}>
              {topCauses.map(([cause, count]) => {
                const total = Object.values(summary.data?.by_cause ?? {}).reduce((a, b) => a + b, 0) || 1;
                return (
                  <div key={cause} className="stack" style={{ gap: 6 }}>
                    <div className="spread">
                      <span style={{ fontSize: 13 }}>{CAUSE_LABEL[cause] ?? cause}</span>
                      <span className="num meta">{count}</span>
                    </div>
                    <div style={{ height: 4, background: "var(--surface-sunk)", borderRadius: 4 }}>
                      <div
                        style={{
                          height: "100%",
                          width: `${(count / total) * 100}%`,
                          background: "var(--ink)",
                          opacity: 0.65,
                          borderRadius: 4,
                        }}
                      />
                    </div>
                  </div>
                );
              })}
            </div>
          </section>
        </Reveal>
      )}

      <section className="stack" style={{ gap: 12 }}>
        <div className="spread" style={{ flexWrap: "wrap", gap: 10 }}>
          <div>
            <h2>Поля кейса</h2>
            <p className="meta">
              Анонимные участки из данных организаторов: координат у них нет, поэтому они показаны списком.
            </p>
          </div>
          <div className="row" style={{ gap: 8 }}>
            <input
              className="input"
              style={{ width: 220 }}
              type="search"
              placeholder="поиск по номеру или культуре"
              value={search}
              onChange={(event) => setSearch(event.target.value)}
            />
            <button
              type="button"
              className={`btn btn--sm ${onlyCritical ? "" : "btn--ghost"}`}
              onClick={() => setOnlyCritical((value) => !value)}
            >
              только критические
            </button>
          </div>
        </div>

        {polygons.isLoading && <Loader label="Загружаем поля" />}
        {polygons.isError && <ErrorNote error={polygons.error} />}

        <div style={{ display: "grid", gap: 12, gridTemplateColumns: "repeat(auto-fill, minmax(268px, 1fr))" }}>
          {list.slice(0, limit).map((item) => (
            <Link
              key={item.pid}
              to={`/field/${item.pid}`}
              className="card stack poly-card"
              style={{ gap: 10, textDecoration: "none", height: "100%" }}
            >
              <div className="spread">
                <span className="mono" style={{ fontSize: 14 }}>
                  {item.pid}
                </span>
                {item.n_critical > 0 ? (
                  <span className="tag tag--crit">{item.n_critical} критических</span>
                ) : item.n_episodes > 0 ? (
                  <span className="tag tag--warn">{item.n_episodes} эпизодов</span>
                ) : (
                  <span className="tag tag--ok">без эпизодов</span>
                )}
              </div>
              <div style={{ fontSize: 13, color: "var(--ink-soft)" }}>{item.crop}</div>
              <div className="meta">
                {item.years[0]}–{item.years.at(-1)} · {item.n_obs}{" "}
                {plural(item.n_obs, "наблюдение", "наблюдения", "наблюдений")}
                {item.n_gaps > 0 && ` · ${item.n_gaps} восстановлено`}
              </div>
              <div className="meta" style={{ fontSize: 11.5 }}>
                {item.kind}
              </div>
            </Link>
          ))}
        </div>
        {!polygons.isLoading && list.length === 0 && <p className="meta">Ничего не нашлось.</p>}
        {list.length > limit && (
          <div className="row" style={{ justifyContent: "center" }}>
            <button type="button" className="btn btn--ghost" onClick={() => setLimit((value) => value + PAGE)}>
              Показать ещё {Math.min(PAGE, list.length - limit)} из {list.length - limit}
            </button>
          </div>
        )}
      </section>
    </div>
  );
}
