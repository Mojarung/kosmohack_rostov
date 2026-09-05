/** Строка поля; подробности только у выбранного эпизода. */
import { useId, useState } from "react";
import { Link } from "react-router-dom";
import type { AnomalyField, AnomalyLevel } from "../../api/anomalies";
import { CAUSE_LABEL, dateRange } from "../../lib/format";

export const LEVEL_LABELS: Record<AnomalyLevel,string> = {
  high:"Сильное отставание", medium:"Заметное отставание", data:"Данные под вопросом",
  context:"Возможная смена культуры", unknown:"Недостаточно данных", clear:"Отклонений не найдено",
};
const ADVICE: Record<string,string> = {
  weather_drought:"Сопоставьте этот период с осадками и записями осмотров поля.",
  early_decline:"Сверьте даты уборки и осмотров: раннее снижение может быть связано с полевыми работами.",
  late_start:"Проверьте сроки сева и погодные условия перед появлением всходов.",
  crop_rotation:"Уточните культуру этого сезона: сравнение с другой культурой может объяснять отклонение.",
  data_suspect:"Сначала проверьте качество снимков; не делайте вывод о состоянии растений по одному сигналу.",
  unsown_or_changed:"Сверьте границы поля и сведения о посеве за этот сезон.",
  weak_season:"Сопоставьте график с журналом работ и итогами сезона.",
};
export function AnomalyFieldRow({field,year,expanded,onToggle}:{field:AnomalyField;year:number;expanded:boolean;onToggle:()=>void}) {
  const [index,setIndex]=useState(0);
  const id=useId(), episode=field.episodes[index], first=field.episodes[0];
  const path=field.source==="mine"?`/explore/${encodeURIComponent(field.uid!)}`:`/field/${encodeURIComponent(field.pid)}`;
  const query=new URLSearchParams({year:String(year)});
  if(episode){query.set("from",episode.start);query.set("to",episode.end);}
  return <article className="anomaly-field" data-testid="anomaly-field" data-level={field.level} data-source={field.source}>
    <button type="button" className="anomaly-row" aria-expanded={expanded} aria-controls={id} onClick={onToggle}>
      <span><strong>{field.name}</strong><small>{field.source==="mine"?"Моё поле":"Поле кейса"} · эпизодов: {field.episodes.length}</small></span>
      <span><span className={`anomaly-level anomaly-level--${field.level}`}>{LEVEL_LABELS[field.level]}</span>
        <small>{first?`Возможная причина: ${CAUSE_LABEL[first.cause]??"пока неясна"}`:
          field.has_season?"По имеющимся наблюдениям":"Нет отчёта за выбранный сезон"}</small></span>
      <span className="meta">{first?dateRange(first.start,first.end):field.has_season?`Сезон ${year}`:"—"}</span>
      <span className="anomaly-chevron" aria-hidden="true">{expanded?"⌄":"›"}</span>
    </button>
    <div id={id} hidden={!expanded} className="anomaly-detail">
      {expanded && <>
        {field.episodes.length>1 && <div className="anomaly-periods" role="group" aria-label="Период отклонения">
          {field.episodes.map((e,i)=><button type="button" className="chip" key={`${e.start}:${i}`}
            aria-pressed={index===i} onClick={()=>setIndex(i)}>{dateRange(e.start,e.end)}</button>)}
        </div>}
        {episode?<><div className="anomaly-explanation"><div><h3>Что произошло</h3>
          <p>{dateRange(episode.start,episode.end)}: отклонение от исторического уровня · {episode.days} дн.</p>
          <p className="meta">Возможная причина: {CAUSE_LABEL[episode.cause]??"неясна"}.</p></div>
          <div><h3>Что проверить</h3><p>{ADVICE[episode.cause]??"Сопоставьте снимки с журналом полевых работ."}</p></div></div>
          <details className="anomaly-tech"><summary>Числа и методика</summary>
            <p>Zmin: {episode.min_z?.toFixed(1)??"нет данных"} · уверенность: {episode.confidence==null?"не оценена":`${Math.round(episode.confidence*100)} %`}</p>
            <p>{episode.text}</p>{episode.reasons&&<p>{episode.reasons}</p>}
          </details></>:<p>{field.level==="clear"?"В отчёте за этот сезон отклонений не найдено.":
            field.has_season?"Наблюдений или исторической нормы недостаточно для оценки.":"Для этого поля нет отчёта за выбранный сезон."}</p>}
        {field.has_season && <Link className="btn btn--sm btn--ghost" to={`${path}?${query}`}>
          {episode?"Показать период на графике":"Открыть отчёт за сезон"}</Link>}
      </>}
    </div>
  </article>;
}
