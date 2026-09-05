/** Динамика реальных наблюдений и площадь снижения на паре чистых снимков. */
import { lazy, Suspense, useRef, useState } from "react";
import { useIsMutating, useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { api } from "../../api/client";
import type { ImageIndex, ImageManifest, ImageScene } from "../../api/analytics";
import type { PolygonDetail } from "../../api/types";
import { shortDate } from "../../lib/format";
import { number } from "../../lib/metrics";

const ImageryMap = lazy(() => import("../map/ImageryMap"));
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
      <label className="row meta">Снимок <select className="input" aria-label="Дата снимка" value={scene.date} onChange={e => onDate(e.target.value)}>
        {[...manifest.scenes].sort((a, b) => a.date.localeCompare(b.date)).map(s => <option key={s.date} value={s.date}>{shortDate(s.date)} {manifest.year}</option>)}
      </select></label>
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
  const key = ["imagery", detail.pid, year];
  const imagery = useQuery({ queryKey: key, queryFn: ({ signal }) => api.imagery(detail.pid, year, signal),
    enabled: Boolean(detail.geometry), retry: false });
  const collection = useMutation({ mutationKey: key, mutationFn: () => api.collectImagery(detail.pid, year),
    onSuccess: result => client.setQueryData(key, result) });
  const collecting = useIsMutating({ mutationKey: key }) > 0;
  const [opened, setOpened] = useState(false), [date, setDate] = useState("");
  const [preferredIndex, setIndex] = useState<ImageIndex>("ndvi");
  const manifest = imagery.data?.available ? imagery.data.manifest : undefined;
  const scene = manifest?.scenes.find(s => s.date === date) ?? manifest?.scenes.at(-1);
  const index = preferredIndex === "change" && !scene?.change ? "ndvi" : preferredIndex;
  const trend = detail.insights?.[year];
  const change = scene?.change;
  const period = change ? `${shortDate(change.previous_date)} → ${shortDate(scene!.date)}` : "По двум снимкам Sentinel-2";
  function openMap() {
    setIndex(change ? "change" : "ndvi"); setOpened(true);
    mapRef.current?.scrollIntoView({ behavior: "smooth", block: "center" });
  }
  return <div className="field-insights">
    <div className="insight-grid">
      <details className="field-insight" data-testid="trend-card" data-status={trend?.status}>
        <summary><span className="meta">Куда идёт поле последние две недели</span><strong>{trend?.label || "Мало наблюдений"}</strong>
          <span className="meta">{trend?.start && trend.end ? `${shortDate(trend.start)} — ${shortDate(trend.end)}` : "Нужны наблюдения и история"}</span></summary>
        <div className="insight-evidence">
          <p>{trend?.available ? `Медиана Z: ${number(trend.before_z, 2)} → ${number(trend.after_z, 2)}.` : "Недостаточно наблюдений с историческим ориентиром."}
            {` Пригодных дат: ${trend?.before_count || 0} и ${trend?.after_count || 0}.`}</p>
          <p>Два соседних периода по 14 дней, минимум 2 даты в каждом. Изменение от 0,5 Z считаем заметным. Восстановленные точки не участвуют.</p>
        </div>
      </details>
      {detail.geometry && <details className="field-insight" data-testid="area-card">
        <summary><span className="meta">Сколько поля потеряло зелень</span>
          <strong>{change ? `${number(change.drop_area_ha)} га · ${number((change.drop_share ?? 0) * 100)}%` : scene ? "Нужны два снимка" : "Откройте карту поля"}</strong>
          <span className="meta">{period}</span></summary>
        <div className="insight-evidence"><AreaEvidence scene={scene} />
          <button className="btn btn--sm btn--ghost" onClick={openMap}>Открыть карту поля</button></div>
      </details>}
    </div>
    {detail.geometry && <section className="season-imagery" ref={mapRef} aria-label="Карта поля">
      <button className="imagery-heading" type="button" aria-expanded={opened} onClick={() => setOpened(v => !v)}>
        <span>Карта поля <span className="meta">· Sentinel-2, 20 м</span></span><span aria-hidden>{opened ? "−" : "+"}</span>
      </button>
      {opened && <div className="imagery-body">
        {imagery.isPending && <p className="meta" role="status">Проверяем снимки…</p>}
        {imagery.isError && <div role="status"><p className="meta">Снимки не загрузились.</p>
          <button className="btn btn--sm btn--ghost" onClick={() => void imagery.refetch()}>Повторить загрузку снимков</button></div>}
        {!imagery.isPending && !imagery.isError && !manifest && <>
          <p className="meta" role="status">{collecting ? "Собираем чистые снимки Sentinel-2. Это может занять несколько минут." : "Снимки этого сезона ещё не собраны."}</p>
          <button className="btn btn--sm" disabled={collecting} onClick={() => collection.mutate()}>
            {collecting ? "Собираем снимки…" : "Загрузить снимки сезона"}</button>
          {collection.isError && <p className="meta" role="alert">{collection.error.message}</p>}
        </>}
        {manifest && scene && <ImageView manifest={manifest} scene={scene} index={index} onDate={setDate} onIndex={setIndex} />}
      </div>}
    </section>}
  </div>;
}
