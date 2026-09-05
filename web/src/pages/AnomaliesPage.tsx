/** Разбор отклонений: источник → сезон → поле → эпизод. */
import { useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { Link, useSearchParams } from "react-router-dom";
import { api } from "../api/client";
import type { FieldSource } from "../api/anomalies";
import { AnomalyFieldRow, LEVEL_LABELS } from "../components/panels/AnomalyFieldRow";
import { ErrorNote } from "../components/ui/Loader";
import "../styles/anomalies.css";

export default function AnomaliesPage() {
  const [params, setParams] = useSearchParams();
  const rawSource = params.get("source");
  const source: FieldSource = rawSource === "case" || rawSource === "all" ? rawSource : "mine";
  const requestedYear = Number(params.get("year"));
  const year = Number.isInteger(requestedYear) && requestedYear >= 1980 && requestedYear <= 2200 ? requestedYear : undefined;
  const search = params.get("q") ?? "", level = params.get("level") ?? "all";
  const [open, setOpen] = useState<string | null>(null);
  const [visible, setVisible] = useState(40);
  const feed = useQuery({ queryKey: ["anomaly-fields", source, year],
    queryFn: ({ signal }) => api.anomalyFields(source, year, signal), staleTime: 0, retry: false });
  const fields = (feed.data?.fields ?? []).filter(field =>
    `${field.name} ${field.pid}`.toLocaleLowerCase().includes(search.trim().toLocaleLowerCase())
    && (level === "all" || level === field.level));
  function update(key: string, value: string) {
    const next = new URLSearchParams(params);
    if (value) next.set(key, value); else next.delete(key);
    if (key === "source") next.delete("year");
    setParams(next, { replace: key === "q" }); setOpen(null); setVisible(40);
  }
  const years = [...new Set([...(feed.data?.years ?? []), ...(year ? [year] : [])])].sort((a,b) => b-a);
  return <div className="workspace workspace--anomalies" data-testid="anomalies-page">
    <section className="pane">
      <div className="anomaly-heading"><h1>Отклонения за сезон</h1>
        <p className="meta">История развития полей · не текущие предупреждения</p></div>
      <div className="anomaly-sources" role="group" aria-label="Источник полей">
        {([['mine','Мои поля'],['case','Поля кейса'],['all','Все']] as const).map(([value,label]) =>
          <button type="button" className="chip" key={value} aria-pressed={source===value}
            onClick={() => update("source",value)}>{label}</button>)}
      </div>
      <div className="anomaly-tools">
        <label>Сезон <select aria-label="Сезон отклонений" value={feed.data?.year ?? year ?? ""}
          disabled={!years.length || feed.isPending} onChange={e => update("year",e.target.value)}>
          {!years.length && <option value="">Нет сезонов</option>}
          {years.map(y=><option key={y} value={y}>{y}</option>)}
        </select></label>
        <input aria-label="Название или номер поля" placeholder="Название или номер поля"
          value={search} onChange={e=>update("q",e.target.value)} />
        <details className="anomaly-filters"><summary>Фильтры{level!=="all"?" · 1":""}</summary>
          <label>Показывать <select aria-label="Тип отклонения" value={level} onChange={e=>update("level",e.target.value)}>
            <option value="all">Все поля</option>
            {Object.entries(LEVEL_LABELS).map(([value,label])=><option key={value} value={value}>{label}</option>)}
          </select></label>
        </details>
      </div>
      <div className="anomaly-count meta" role="status">{feed.isPending ? "Загружаем поля…" : feed.isError ? "Не удалось загрузить поля" :
        `Полей: ${fields.length} · эпизодов: ${fields.reduce((n,f)=>n+f.episodes.length,0)}`}
        <span>Сначала сильные отклонения</span></div>
      <div className="pane-body anomaly-list" aria-busy={feed.isFetching}>
        {feed.isError && <div className="anomaly-empty"><ErrorNote error={feed.error} />
          <button className="btn btn--sm" onClick={()=>void feed.refetch()}>Повторить загрузку</button></div>}
        {!feed.isPending && !feed.isError && !fields.length && <div className="anomaly-empty">
          {source==="mine" && !feed.data?.fields.length ? <><p>Сохранённых полей пока нет.</p>
            <Link className="btn btn--sm" to="/explore">Добавить поле в новой территории</Link></> :
            <><p>Нет совпадений. Измените сезон, название или фильтр.</p>
            <button className="btn btn--ghost btn--sm" onClick={()=>{setParams({source});setVisible(40);}}>Сбросить фильтры</button></>}
        </div>}
        {!feed.isError && fields.slice(0,visible).map(field=><AnomalyFieldRow key={`${field.key}:${feed.data?.year}`}
          field={field} year={feed.data?.year ?? 0} expanded={open===field.key}
          onToggle={()=>setOpen(open===field.key?null:field.key)} />)}
        {!feed.isError && fields.length>visible && <button className="btn btn--ghost btn--sm anomaly-more"
          onClick={()=>setVisible(n=>n+40)}>Показать ещё · осталось {fields.length-visible}</button>}
      </div>
      <div className="anomaly-footer meta"><span>Причины — предположения по данным наблюдений.</span>
        <details><summary>Как определяем отклонения</summary><p>Сравниваем поле с историей за тот же период.
          Отклонение не доказывает потерю урожая. Отсутствие данных не считается нормой.</p></details></div>
    </section>
  </div>;
}
