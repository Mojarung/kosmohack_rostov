/** Новая территория: выбор контура на карте (готовый из OpenStreetMap или нарисованный),
 *  автоматический сбор спутниковых и метеоданных, анализ и сохранение поля в набор пользователя. */

import { Suspense, lazy, useEffect, useMemo, useRef, useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useNavigate, useParams } from "react-router-dom";

import { api } from "../api/client";
import type { CollectSource, OsmField, PolygonDetail } from "../api/types";
import { plural, severityTone } from "../lib/format";
import { FieldArt, SearchFieldsArt } from "../components/art/Art";
import { SeasonPanel } from "../components/panels/SeasonPanel";
import { ErrorNote } from "../components/ui/Loader";
import { PagePending } from "../components/ui/PagePending";

const FieldMap = lazy(() => import("../components/map/FieldMap"));

const YEARS = { start: 2019, end: new Date().getFullYear() };

/** Примерная площадь полигона в гектарах (сферическая аппроксимация, достаточно для подсказки). */
function areaHa(polygon: GeoJSON.Polygon): number {
  const ring = polygon.coordinates[0] ?? [];
  if (ring.length < 4) return 0;
  const lat0 = (ring.reduce((sum, p) => sum + p[1], 0) / ring.length) * (Math.PI / 180);
  const mx = 111320 * Math.cos(lat0);
  const my = 110540;
  let sum = 0;
  for (let i = 0; i < ring.length - 1; i += 1) {
    const [x1, y1] = ring[i];
    const [x2, y2] = ring[i + 1];
    sum += x1 * mx * (y2 * my) - x2 * mx * (y1 * my);
  }
  return Math.abs(sum / 2) / 10_000;
}

/** Идентификатор задания сбора: клиент задаёт его заранее и опрашивает прогресс, пока POST ещё выполняется. */
function newJobId(): string {
  const raw = typeof crypto.randomUUID === "function" ? crypto.randomUUID() : `${Date.now()}-${Math.random()}`;
  return raw.replace(/[^A-Za-z0-9-]/g, "").slice(0, 36).padEnd(8, "0");
}

const SOURCE_ORDER = ["S2", "Landsat", "MODIS", "ERA5"];

/** Реальный ход сбора с сервера: сезоны и сцены по каждому источнику, затем разбор сезонов. */
function CollectProgress({ startedAt, job }: { startedAt: number; job: string }) {
  const [elapsed, setElapsed] = useState(0);
  useEffect(() => {
    const timer = window.setInterval(() => setElapsed(Math.round((Date.now() - startedAt) / 1000)), 1000);
    return () => window.clearInterval(timer);
  }, [startedAt]);
  const progress = useQuery({
    queryKey: ["collect-progress", job],
    queryFn: () => api.analyzeProgress(job),
    refetchInterval: 2000,
    retry: false,
  });
  const state = progress.data;
  const sources = SOURCE_ORDER.map((key) => state?.sources[key]).filter((s): s is CollectSource => Boolean(s));
  const lastLine = state?.log.at(-1);

  return (
    <div className="card card--sunk stack" style={{ gap: 8 }} data-testid="collect-progress">
      <div className="spread">
        <span className="eyebrow">Собираем данные</span>
        <span className="num meta">{Math.floor(elapsed / 60)} мин {elapsed % 60} с</span>
      </div>
      <span className="meta">{YEARS.start}–{YEARS.end} · сезонов: {YEARS.end - YEARS.start + 1}</span>
      {sources.length === 0 && <span className="meta">Запускаем сборщик, запрашиваем каталоги снимков…</span>}
      {sources.map((source) => (
        <div key={source.title} className="stack" style={{ gap: 3 }}>
          <div className="spread" style={{ fontSize: 12 }}>
            <span>{source.title}</span>
            <span className="meta num">
              {source.status === "failed"
                ? source.note
                : `${source.done} / ${source.total}${source.scenes ? ` · ${source.scenes} сцен` : ""}`}
            </span>
          </div>
          <div style={{ height: 3, borderRadius: 2, background: "var(--line)", overflow: "hidden" }}>
            <div
              style={{
                height: "100%",
                width: `${source.total ? Math.round((100 * source.done) / source.total) : 0}%`,
                background: source.status === "failed" ? "var(--crit-ink)" : "var(--ok-ink)",
                transition: "width 0.4s",
              }}
            />
          </div>
        </div>
      ))}
      {state?.stage === "analysis" && <span className="meta">Спутники и погода собраны, разбираем сезоны…</span>}
      {lastLine && state?.stage === "collect" && <span className="meta">{lastLine}</span>}
      <p className="meta">
        Источники грузятся одновременно; время зависит от скорости хранилищ снимков.
        Дождитесь завершения на этой странице: отчёт откроется автоматически.
      </p>
    </div>
  );
}

export default function ExplorePage() {
  const { uid } = useParams();
  const navigate = useNavigate();
  const queryClient = useQueryClient();

  const [selection, setSelection] = useState<GeoJSON.Polygon | null>(null);
  const [drawing, setDrawing] = useState(false);
  const [name, setName] = useState("новое поле");
  const [osmFields, setOsmFields] = useState<OsmField[]>([]);
  const [osmError, setOsmError] = useState<string | null>(null);
  const [osmLoading, setOsmLoading] = useState(false);
  const bboxRef = useRef<[number, number, number, number] | null>(null);
  const [result, setResult] = useState<PolygonDetail | null>(null);
  const [startedAt, setStartedAt] = useState(0);
  const [job, setJob] = useState("");

  const saved = useQuery({ queryKey: ["user-polygons"], queryFn: api.userPolygons });
  const opened = useQuery({
    queryKey: ["user-polygon", uid],
    queryFn: () => api.userPolygon(uid!),
    enabled: Boolean(uid),
  });

  const analyze = useMutation({
    mutationFn: (jobId: string) =>
      api.analyze({
        geometry: selection!,
        name: name.trim() || "новое поле",
        start_year: YEARS.start,
        end_year: YEARS.end,
        job: jobId,
      }),
    onMutate: (jobId) => {
      setStartedAt(Date.now());
      setJob(jobId);
    },
    onSuccess: async (data) => {
      setResult(data);
      await queryClient.invalidateQueries({ queryKey: ["user-polygons"] });
      if (data.uid) navigate(`/explore/${data.uid}`);
    },
  });

  const remove = useMutation({
    mutationFn: (target: string) => api.deleteUserPolygon(target),
    onSuccess: async (_data, target) => {
      await queryClient.invalidateQueries({ queryKey: ["user-polygons"] });
      if (uid === target) {
        setResult(null);
        navigate("/explore");
      }
    },
  });

  async function findFields() {
    const bbox = bboxRef.current;
    if (!bbox) return;
    setOsmLoading(true);
    setOsmError(null);
    try {
      const fields = await api.fields(bbox);
      setOsmFields(fields);
      if (fields.length === 0) setOsmError("В этой области контуров OpenStreetMap нет — нарисуйте поле вручную.");
    } catch (error) {
      setOsmError(error instanceof Error ? error.message : String(error));
    } finally {
      setOsmLoading(false);
    }
  }

  const detail = uid ? opened.data : result;
  const area = selection ? areaHa(selection) : 0;
  const savedList = useMemo(() => saved.data ?? [], [saved.data]);

  // открытое поле показываем вместо карты: экран остаётся один, без прокрутки
  if (detail) {
    return (
      <div className="workspace workspace--field">
        <div className="screen-head">
          <button type="button" className="back-link" onClick={() => { setResult(null); navigate("/explore"); }}>
            ← к карте
          </button>
          <span className="screen-title">{detail.name ?? detail.pid}</span>
          <span className="meta">{detail.collected ?? detail.kind}</span>
          <span className="head-facts">
            <span className="head-fact">
              <span className="num">{Object.keys(detail.years).length}</span>
              <span className="eyebrow">{plural(Object.keys(detail.years).length, "сезон собран", "сезона собрано", "сезонов собрано")}</span>
            </span>
            <span className="head-fact">
              <span className="num">{detail.episodes.length}</span>
              <span className="eyebrow">{plural(detail.episodes.length, "эпизод", "эпизода", "эпизодов")}</span>
            </span>
          </span>
        </div>
        <SeasonPanel key={detail.pid} detail={detail} />
      </div>
    );
  }

  return (
    <div className="workspace workspace--explore">
      <section className="pane pane--plain">
        <Suspense fallback={<PagePending label="Инициализируем карту" />}>
          <FieldMap
            saved={savedList}
            osmFields={osmFields}
            selection={selection}
            drawing={drawing}
            onDrawn={(geometry) => {
              setSelection(geometry);
              setDrawing(false);
            }}
            onPickOsm={(field) => {
              setSelection(field.geometry);
              setName(field.name);
              setDrawing(false);
            }}
            onPickSaved={(target) => navigate(`/explore/${target}`)}
            onViewportChange={(bbox) => {
              bboxRef.current = bbox;
            }}
            height="100%"
          />
        </Suspense>
      </section>

      <div className="workspace-side">
        <section className="pane">
          <div className="step">
            <div className="step-head">
              <SearchFieldsArt size={28} />
              <div>
                <div className="eyebrow">Шаг 1</div>
                <div style={{ fontSize: 13.5 }}>Выберите контур</div>
              </div>
            </div>
            <button type="button" className="btn btn--ghost btn--sm" onClick={findFields} disabled={osmLoading}>
              {osmLoading ? "Ищем контуры…" : "Найти поля OSM в видимой области"}
            </button>
            <button
              type="button"
              className={drawing ? "btn btn--sm" : "btn btn--ghost btn--sm"}
              onClick={() => setDrawing((value) => !value)}
            >
              {drawing ? "Рисование включено — кликайте по карте" : "Нарисовать контур вручную"}
            </button>
            {osmFields.length > 0 && <p className="meta">Найдено контуров: {osmFields.length}. Кликните по любому.</p>}
            {osmError && <p className="meta">{osmError}</p>}
          </div>

          <div className="step">
            <div className="step-head">
              <FieldArt size={28} />
              <div>
                <div className="eyebrow">Шаг 2</div>
                <div style={{ fontSize: 13.5 }}>Соберите данные</div>
              </div>
            </div>
            <input
              className="input"
              value={name}
              onChange={(event) => setName(event.target.value)}
              placeholder="название поля"
            />
            <p className="meta">{selection ? `Контур задан, около ${area.toFixed(1)} га.` : "Контур ещё не выбран."}</p>
            <button
              type="button"
              className="btn btn--sm"
              disabled={!selection || analyze.isPending}
              onClick={() => analyze.mutate(newJobId())}
            >
              {analyze.isPending ? "Собираем…" : `Собрать за ${YEARS.start}–${YEARS.end} и разобрать`}
            </button>
            {analyze.isPending && <CollectProgress startedAt={startedAt} job={job} />}
            {analyze.isError && <ErrorNote error={analyze.error} />}
          </div>
        </section>

        <section className="pane">
          <div className="pane-head">
            <span className="pane-title">Мои поля</span>
            <span className="meta">{savedList.length}</span>
          </div>
          <div className="pane-body pane-body--pad stack" style={{ gap: 5 }}>
            {savedList.length === 0 && (
              <p className="meta">
                Пока пусто. Проанализированные территории сохраняются здесь и подсвечиваются на карте цветом
                состояния последнего сезона.
              </p>
            )}
            {savedList.map((item) => (
              <div key={item.uid} className="saved-row" onClick={() => navigate(`/explore/${item.uid}`)}>
                <div className="stack" style={{ gap: 1, minWidth: 0 }}>
                  <span className="saved-name">{item.name}</span>
                  <span className="meta" style={{ fontSize: 11 }}>
                    {item.years.length ? `${item.years[0]}–${item.years.at(-1)}` : "нет сезонов"} · {item.n_episodes}{" "}
                    {plural(item.n_episodes, "эпизод", "эпизода", "эпизодов")}
                  </span>
                </div>
                <div className="row" style={{ gap: 6 }}>
                  <span className={`tag tag--${severityTone(item.last_year_status)}`}>{item.last_year_status}</span>
                  <button
                    type="button"
                    aria-label={`удалить ${item.name}`}
                    className="saved-remove"
                    onClick={(event) => {
                      event.stopPropagation();
                      remove.mutate(item.uid);
                    }}
                  >
                    ✕
                  </button>
                </div>
              </div>
            ))}
          </div>
        </section>
      </div>

      {opened.isError && <ErrorNote error={opened.error} />}
    </div>
  );
}
