/** Динамика реальных наблюдений и площадь снижения на паре чистых снимков. */
import { lazy, Suspense, useRef, useState } from "react";
import { useIsMutating, useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { api } from "../../api/client";
import type { ImageIndex, ImageManifest, ImageScene } from "../../api/analytics";
import type { PolygonDetail } from "../../api/types";
import { shortDate } from "../../lib/format";
import { number } from "../../lib/metrics";
import { InfoPopover } from "../ui/InfoPopover";
import { ImageDatePicker } from "./ImageDatePicker";
import { ImageryProgress } from "./ImageryProgress";

const ImageryMap = lazy(() => import("../map/ImageryMap"));
const FieldOutlinePreview = lazy(() => import("../map/FieldOutlinePreview"));
const LAYERS = {
  ndvi: { label: "NDVI", colors: "#9b8872,#c2ad85 50%,#e7d596 60%,#91b967 75%,#337750 90%,#174d38", ticks: ["−1", "0", "+1"] },
  ndmi: { label: "NDMI", colors: "#a98261,#d8b88e 30%,#e6e6c9 50%,#8ac6b4 65%,#398c98 80%,#21536f", ticks: ["−1", "0", "+1"] },
  change: { label: "Изменение", colors: "#ac524b,#d99171 37.5%,#f5f2e5 50%,#85b795 62.5%,#28765b", ticks: ["−0,4", "0", "+0,4"] },
};

function AreaEvidence({ scene }: { scene?: ImageScene }) {
  if (!scene?.change) return <p>Нужны два снимка с общими чистыми пикселями не менее чем на 60% контура.</p>;
  const change = scene.change;
  return <>
    <p>NDVI снизился минимум на {number(Math.abs(change.threshold), 2)} на {number(change.drop_area_ha)} га.
      Сравниваем {number(change.area_ha)} га, видимых без облаков в обе даты ({number(change.clear_share * 100)}% контура).</p>
    <p>Доля рассчитана от сравниваемой площади. Спад может быть связан с созреванием, уборкой или стрессом.</p>
  </>;
}

function ImageView({ manifest, scene, index, onDate, onIndex }: {
  manifest: ImageManifest; scene: ImageScene; index: ImageIndex;
  onDate: (date: string) => void; onIndex: (index: ImageIndex) => void;
}) {
  const spec = LAYERS[index], stats = index === "change" ? scene.change! : scene[index];
  const url = `/api/polygon/${encodeURIComponent(manifest.pid)}/imagery/${manifest.year}/${scene.date}/${index}.png?v=${manifest.generation}`;
  return <>
    <div className="imagery-controls">
      <ImageDatePicker dates={manifest.scenes.map(s => s.date)} value={scene.date} onChange={onDate} />
      <div className="metric-tabs" role="group" aria-label="Слой карты">
        {(Object.keys(LAYERS) as ImageIndex[]).map(key => <button key={key} type="button" data-image={key}
          aria-pressed={key === index} disabled={key === "change" && !scene.change} onClick={() => onIndex(key)}>{LAYERS[key].label}</button>)}
      </div>
    </div>
    <p className="meta" data-testid="imagery-note">
      {index === "change" ? `${shortDate(scene.change!.previous_date)} → ${shortDate(scene.date)}. Красный — NDVI снизился, зелёный — вырос.`
        : index === "ndvi" ? "Зелёный — выше NDVI, больше зелёной растительности."
          : "Сине-зелёный — выше NDMI, сигнал содержания воды в растительности. Это не влажность почвы."}
      {` Чистое покрытие: ${number(stats.clear_share * 100)}% · среднее ${spec.label === "Изменение" ? "ΔNDVI" : spec.label}: ${number(stats.mean, 3)}.`}
    </p>
    <Suspense fallback={<p className="meta">Открываем карту…</p>}><ImageryMap manifest={manifest} url={url} /></Suspense>
    <div className="imagery-legend"><div style={{ background: `linear-gradient(to right,${spec.colors})` }} />
      <div className="spread meta">{spec.ticks.map(t => <span key={t}>{t}</span>)}</div></div>
  </>;
}

export function FieldInsights({ detail, year }: { detail: PolygonDetail; year: number }) {
  const client = useQueryClient(), mapRef = useRef<HTMLElement>(null);
  const [opened, setOpened] = useState(false), [date, setDate] = useState("");
  const key = ["imagery", detail.pid, year];
  const imagery = useQuery({ queryKey: key, queryFn: ({ signal }) => api.imagery(detail.pid, year, signal),
    enabled: Boolean(detail.geometry), retry: false,
    refetchInterval: query => opened && !query.state.data?.available ? 2000 : false });
  const collection = useMutation({ mutationKey: key, mutationFn: () => api.collectImagery(detail.pid, year),
    onSuccess: result => client.setQueryData(key, result) });
  const pending = useIsMutating({ mutationKey: key }) > 0;
  const progress = imagery.data && !imagery.data.available ? imagery.data.progress : undefined;
  const collecting = Boolean(progress && ["queued", "catalog", "download", "render"].includes(progress.stage))
    || pending;
  const [preferredIndex, setIndex] = useState<ImageIndex>("ndvi");
  const manifest = imagery.data?.available ? imagery.data.manifest : undefined;
  const scene = manifest?.scenes.find(s => s.date === date) ?? manifest?.scenes.at(-1);
  const index = preferredIndex === "change" && !scene?.change ? "ndvi" : preferredIndex;
  const trend = detail.insights?.[year];
  const change = scene?.change;
  const period = change ? `${shortDate(change.previous_date)} → ${shortDate(scene!.date)}`
    : scene ? "По двум снимкам Sentinel-2" : "Считается по двум снимкам Sentinel-2: загрузите их в карте поля";
  function openMap() {
    setIndex(change ? "change" : "ndvi"); setOpened(true);
    mapRef.current?.scrollIntoView({ behavior: "smooth", block: "center" });
  }
  return <div className="field-insights">
    <div className="insight-grid">
      <article className="field-insight" data-testid="trend-card" data-status={trend?.status}>
        <div className="insight-head"><span className="meta">Куда идёт поле последние две недели</span>
          <InfoPopover title="Динамика поля">
          <p>{trend?.available ? `Медиана Z: ${number(trend.before_z, 2)} → ${number(trend.after_z, 2)}.` : "Недостаточно наблюдений с историческим ориентиром."}
            {` Пригодных дат: ${trend?.before_count || 0} и ${trend?.after_count || 0}.`}</p>
          <p>Два соседних периода по 14 дней, минимум 2 даты в каждом. Изменение от 0,5 Z считаем заметным. Восстановленные точки не участвуют.</p>
          </InfoPopover></div>
        <strong>{trend?.label || "Мало наблюдений"}</strong>
        <span className="meta">{trend?.start && trend.end ? `${shortDate(trend.start)} — ${shortDate(trend.end)}` : "Нужны наблюдения и история"}</span>
      </article>
      {detail.geometry && <article className="field-insight" data-testid="area-card">
        <div className="insight-head"><span className="meta">Сколько поля потеряло зелень</span>
          <InfoPopover title="Площадь снижения зелени"><AreaEvidence scene={scene} /></InfoPopover></div>
          <strong>{change ? `${number(change.drop_area_ha)} га · ${number((change.drop_share ?? 0) * 100)}%` : scene ? "Нужны два снимка" : "Снимки ещё не загружены"}</strong>
          <span className="meta">{period}</span>
          <button type="button" className="insight-map-link" onClick={openMap}>Открыть карту поля ↗</button>
      </article>}
    </div>
    {detail.geometry && <section className="season-imagery" ref={mapRef} aria-label="Карта поля">
      <button className="imagery-heading" type="button" aria-expanded={opened} onClick={() => setOpened(v => !v)}>
        <span>Карта поля <span className="meta">· Sentinel-2, 20 м</span></span><span aria-hidden>{opened ? "−" : "+"}</span>
      </button>
      {opened && <div className="imagery-body">
        {imagery.isPending && <p className="meta" role="status">Проверяем снимки…</p>}
        {imagery.isError && !imagery.data && <div role="status"><p className="meta">Снимки не загрузились.</p>
          <button className="btn btn--sm btn--ghost" onClick={() => void imagery.refetch()}>Повторить загрузку снимков</button></div>}
        {!imagery.isPending && (!imagery.isError || imagery.data) && !manifest && <>
          {(collecting || progress) ? <ImageryProgress progress={progress} year={year} requestedAt={collection.submittedAt}
            supported={Boolean(imagery.data && "progress" in imagery.data)} connectionError={imagery.isError} />
            : <p className="meta">Снимки этого сезона ещё не собраны.</p>}
          {!collecting && <button className="btn btn--sm" onClick={() => collection.mutate()}>
            {progress?.stage === "error" ? "Повторить сбор снимков" : "Загрузить снимки сезона"}</button>}
          {collection.isError && !collecting && progress?.stage !== "error" && <p className="meta" role="alert">{collection.error.message}</p>}
          {collecting && detail.geometry && <Suspense fallback={<p className="meta">Открываем контур поля…</p>}>
            <FieldOutlinePreview geometry={detail.geometry} /></Suspense>}
        </>}
        {manifest && scene && <ImageView manifest={manifest} scene={scene} index={index} onDate={setDate} onIndex={setIndex} />}
      </div>}
    </section>}
  </div>;
}
