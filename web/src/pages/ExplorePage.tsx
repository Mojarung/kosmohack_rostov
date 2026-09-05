/** Новая территория: выбор контура на карте (готовый из OpenStreetMap или нарисованный),
 *  автоматический сбор спутниковых и метеоданных, анализ и сохранение поля в набор пользователя. */

import { Suspense, lazy, useEffect, useMemo, useRef, useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useNavigate, useParams } from "react-router-dom";

import { api } from "../api/client";
import type { OsmField, PolygonDetail } from "../api/types";
import { severityTone } from "../lib/format";
import { FieldArt, SearchFieldsArt } from "../components/art/Art";
import { SeasonPanel } from "../components/panels/SeasonPanel";
import { ErrorNote, Loader } from "../components/ui/Loader";

const FieldMap = lazy(() => import("../components/map/FieldMap"));

const YEARS = { start: 2019, end: new Date().getUTCFullYear() };

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

/** Прогресс сбора: реального стрима нет, поэтому показываем стадии и прошедшее время. */
function CollectProgress({ startedAt }: { startedAt: number }) {
  const [elapsed, setElapsed] = useState(0);
  useEffect(() => {
    const timer = window.setInterval(() => setElapsed(Math.round((Date.now() - startedAt) / 1000)), 1000);
    return () => window.clearInterval(timer);
  }, [startedAt]);

  const stages = [
    { label: "Поиск сцен Sentinel-2", at: 0 },
    { label: "Landsat 8/9 и маска облаков", at: 35 },
    { label: "Композиты MODIS", at: 80 },
    { label: "Погода ERA5 и разбор сезонов", at: 115 },
  ];
  const active = stages.filter((stage) => elapsed >= stage.at).length - 1;

  return (
    <div className="card card--sunk stack" style={{ gap: 10 }}>
      <div className="spread">
        <span className="eyebrow">Собираем данные</span>
        <span className="num meta">{elapsed} с</span>
      </div>
      <div className="stack" style={{ gap: 6 }}>
        {stages.map((stage, index) => (
          <div key={stage.label} className="row" style={{ gap: 8 }}>
            <span
              aria-hidden
              style={{
                width: 7,
                height: 7,
                borderRadius: "50%",
                background: index <= active ? "var(--ink)" : "var(--line-strong)",
                transition: "background .3s var(--ease)",
              }}
            />
            <span style={{ fontSize: 13, color: index <= active ? "var(--ink)" : "var(--muted)" }}>{stage.label}</span>
          </div>
        ))}
      </div>
      <p className="meta">
        Обычно занимает 1–3 минуты: для каждого сезона запрашиваются снимки трёх спутников и обрезаются по контуру
        поля. Страницу можно не перезагружать.
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

  const saved = useQuery({ queryKey: ["user-polygons"], queryFn: api.userPolygons });
  const opened = useQuery({
    queryKey: ["user-polygon", uid],
    queryFn: () => api.userPolygon(uid!),
    enabled: Boolean(uid),
  });

  const analyze = useMutation({
    mutationFn: () =>
      api.analyze({
        geometry: selection!,
        name: name.trim() || "новое поле",
        start_year: YEARS.start,
        end_year: YEARS.end,
      }),
    onMutate: () => setStartedAt(Date.now()),
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

  const detail = opened.data ?? result;
  const area = selection ? areaHa(selection) : 0;
  const savedList = useMemo(() => saved.data ?? [], [saved.data]);

  return (
    <div className="stack" style={{ gap: 18 }}>
      <div className="stack" style={{ gap: 6 }}>
        <span className="eyebrow">Сценарий без подготовки данных</span>
        <h1 style={{ fontSize: 34 }}>Новая территория</h1>
        <p className="meta" style={{ maxWidth: "70ch" }}>
          Найдите готовые сельхозконтуры OpenStreetMap в видимой области карты или обведите участок сами.
          Сервис сам соберёт снимки и погоду за {YEARS.start}–{YEARS.end} годы, построит ряд NDVI, восстановит
          пропуски и найдёт периоды угнетения.
        </p>
      </div>

      <div
        style={{
          display: "grid",
          gap: 16,
          gridTemplateColumns: "minmax(0, 1fr) minmax(280px, 340px)",
          alignItems: "start",
        }}
      >
        <div className="card card--flush" style={{ height: 536 }}>
          <Suspense fallback={<Loader label="Инициализируем карту" />}>
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
              height={520}
            />
          </Suspense>
        </div>

        <div className="stack" style={{ gap: 14 }}>
          <div className="card stack" style={{ gap: 12 }}>
            <div className="row" style={{ gap: 10 }}>
              <SearchFieldsArt size={34} />
              <div>
                <div className="eyebrow">Шаг 1</div>
                <div style={{ fontSize: 14 }}>Выберите контур</div>
              </div>
            </div>
            <button type="button" className="btn btn--ghost" onClick={findFields} disabled={osmLoading}>
              {osmLoading ? "Ищем контуры…" : "Найти поля OSM в видимой области"}
            </button>
            <button
              type="button"
              className={drawing ? "btn" : "btn btn--ghost"}
              onClick={() => setDrawing((value) => !value)}
            >
              {drawing ? "Рисование включено — кликайте по карте" : "Нарисовать контур вручную"}
            </button>
            {osmFields.length > 0 && (
              <p className="meta">Найдено контуров: {osmFields.length}. Кликните по любому на карте.</p>
            )}
            {osmError && <p className="meta">{osmError}</p>}
          </div>

          <div className="card stack" style={{ gap: 12 }}>
            <div className="row" style={{ gap: 10 }}>
              <FieldArt size={34} />
              <div>
                <div className="eyebrow">Шаг 2</div>
                <div style={{ fontSize: 14 }}>Соберите данные</div>
              </div>
            </div>
            <input
              className="input"
              value={name}
              onChange={(event) => setName(event.target.value)}
              placeholder="название поля"
            />
            <p className="meta">
              {selection ? `Контур задан, около ${area.toFixed(1)} га.` : "Контур ещё не выбран."}
            </p>
            <button
              type="button"
              className="btn"
              disabled={!selection || analyze.isPending}
              onClick={() => analyze.mutate()}
            >
              {analyze.isPending ? "Собираем…" : "Собрать данные и проанализировать"}
            </button>
            {analyze.isPending && <CollectProgress startedAt={startedAt} />}
            {analyze.isError && <ErrorNote error={analyze.error} />}
          </div>

          <div className="card stack" style={{ gap: 10 }}>
            <div className="spread">
              <div className="eyebrow">Мои поля</div>
              <span className="meta">{savedList.length}</span>
            </div>
            {savedList.length === 0 && (
              <p className="meta">
                Пока пусто. Проанализированные территории сохраняются здесь и подсвечиваются на карте цветом
                состояния последнего сезона.
              </p>
            )}
            <div className="stack scroll" style={{ gap: 6, maxHeight: 260 }}>
              {savedList.map((item) => (
                <div
                  key={item.uid}
                  className="spread"
                  style={{
                    gap: 8,
                    padding: "8px 10px",
                    borderRadius: 8,
                    border: `1px solid ${uid === item.uid ? "var(--line-strong)" : "var(--line)"}`,
                    background: uid === item.uid ? "var(--surface-sunk)" : "transparent",
                    cursor: "pointer",
                  }}
                  onClick={() => navigate(`/explore/${item.uid}`)}
                >
                  <div className="stack" style={{ gap: 2, minWidth: 0 }}>
                    <span style={{ fontSize: 13, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
                      {item.name}
                    </span>
                    <span className="meta" style={{ fontSize: 11.5 }}>
                      {item.years.length ? `${item.years[0]}–${item.years.at(-1)}` : "нет сезонов"} · эпизодов{" "}
                      {item.n_episodes}
                    </span>
                  </div>
                  <div className="row" style={{ gap: 6 }}>
                    <span className={`tag tag--${severityTone(item.last_year_status)}`}>{item.last_year_status}</span>
                    <button
                      type="button"
                      aria-label={`удалить ${item.name}`}
                      className="btn btn--ghost btn--sm"
                      style={{ padding: "2px 7px", borderColor: "transparent" }}
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
          </div>
        </div>
      </div>

      {opened.isLoading && <Loader label="Открываем сохранённое поле" />}
      {opened.isError && <ErrorNote error={opened.error} />}

      {detail && (
        <section className="stack" style={{ gap: 12 }}>
          <div className="spread" style={{ flexWrap: "wrap", gap: 10 }}>
            <div>
              <h2>{detail.name ?? detail.pid}</h2>
              <p className="meta">{detail.collected ?? detail.kind}</p>
            </div>
            <span className="meta">{Object.keys(detail.years).length} сезонов собрано</span>
          </div>
          <SeasonPanel detail={detail} />
        </section>
      )}
    </div>
  );
}
