/* Основной NDVI и подробности; расчёты модели и эпизодов приходят из прежнего API. */
window.FieldCharts = (() => {
  const {$,finite,format,date,esc,options}=ChartUI;
  let report,year,season,weather=null,version=0;
  const hypotheses={weather_drought:"Возможен погодный стресс. Сопоставьте осадки и полив.",
    crop_rotation:"Возможна смена культуры. Уточните культуру и сроки полевых работ.",
    late_start:"Возможен поздний старт. Уточните дату посева и сравните накопленное тепло.",
    early_decline:"Спад начался раньше обычного. Уточните дату уборки и проверьте снимки.",
    unsown_or_changed:"Возможно, изменились культура или использование поля. Уточните историю работ.",
    data_suspect:"Возможна ошибка наблюдений. Проверьте ближайшие пригодные снимки.",
    weak_season:"Развитие слабее исторического ориентира. Проверьте погоду и состояние поля."};
  function episodes() { return report.episodes.filter(e=>e.year===year).sort((a,b)=>a.start.localeCompare(b.start)); }
  function shapes() { return episodes().map(e=>({type:"rect",xref:"x",yref:"paper",x0:e.start,x1:e.end,y0:0,y1:1,
    fillcolor:e.severity==="критическая"?"#ce5350":"#e5ab48",opacity:.12,line:{width:0},layer:"below"})); }
  async function drawNDVI(token) {
    const norm=ChartUI.daily(season.norm_mean),std=new Map(season.norm_std.map(p=>[p.date,p.value]));
    const traces=[];
    if(norm.y.some(finite))traces.push(
      {x:norm.x,y:norm.y.map((v,i)=>finite(v)?v+(std.get(norm.x[i])||0):null),mode:"lines",line:{width:0},hoverinfo:"skip"},
      {x:norm.x,y:norm.y.map((v,i)=>finite(v)?v-(std.get(norm.x[i])||0):null),mode:"lines",line:{width:0},fill:"tonexty",fillcolor:"rgba(137,148,139,.14)",hoverinfo:"skip"},
      {...norm,mode:"lines",name:"Исторический ориентир",line:{color:"#9aA49c",width:1.5},hovertemplate:"Ориентир: %{y:.3f}<extra></extra>"});
    traces.push({...ChartUI.daily(season.curve),mode:"lines",name:"Кривая NDVI",line:{color:"#25362b",width:2.5},connectgaps:false,hovertemplate:"Кривая NDVI: %{y:.3f}<extra></extra>"});
    const points=season.observations.filter(p=>!p.artifact&&finite(p.harmonized));
    traces.push({x:points.map(p=>p.date),y:points.map(p=>p.harmonized),customdata:points.map(p=>[p.sensor,p.value]),mode:"markers",name:"Наблюдения",
      marker:{color:"#417ca8",size:5,opacity:.75},hovertemplate:"%{customdata[0]}: %{y:.3f} в шкале S2<br>Исходный NDVI: %{customdata[1]:.3f}<extra></extra>"});
    await Plotly.react("ndvi",traces,ChartUI.layout("NDVI",{yaxis:{title:{text:"NDVI"},range:[0,1],gridcolor:"#edf1eb"},shapes:shapes()}),options);
    if(token!==version)return;
    ChartUI.bind("ndvi");
    const latest=points.at(-1);
    $("ndvi-note").textContent=`${points.length} пригодных наблюдений${latest?` · последнее ${date(latest.date)}`:""} · ${season.norm_source}`;
  }
  function updateCards(day=null) {
    const points=season.observations.filter(p=>!p.artifact&&finite(p.harmonized)),last=points.at(-1);
    const target=day||last?.date, value=day?season.curve.find(p=>p.date===day)?.value:last?.harmonized;
    const normal=season.norm_mean.find(p=>p.date===target)?.value;
    $("metric-ndvi-label").textContent=`${day?"Кривая NDVI":"Последний снимок"} · ${date(target)}`;
    $("metric-ndvi-value").textContent=format(value,3);
    $("metric-ndvi-note").textContent=finite(value)&&finite(normal)?`${value-normal>0?"+":""}${format(value-normal,3)} к ориентиру`:"Нет оценки для сравнения";
    if(!weather)return;
    metricCard("rain",weather.rain,"мм",day);metricCard("heat",weather.thermal,"°C·дни",day);
  }
  function metricCard(id,metric,units,day) {
    $("metric-"+id).hidden=!metric.available;
    if(!metric.available)return;
    const i=day?metric.date.indexOf(day):metric.value.findLastIndex(finite),value=metric.value[i],normal=metric.mean[i];
    $("metric-"+id+"-value").textContent=`${format(value)} ${units}`;
    $("metric-"+id+"-note").textContent=finite(value)&&finite(normal)?`${value-normal>0?"+":""}${format(value-normal)} ${units} к среднему`:"Нет оценки для сравнения";
    $("metric-"+id+"-date").textContent=date(day||metric.date[i]);
  }
  function setWeather(data) {
    weather=data;
    const dry=data.dry_spell;
    $("metric-dry").hidden=!dry.available;
    $("metric-dry-value").textContent=`${dry.days} дн.`;
    $("metric-dry-note").textContent=dry.days?`${date(dry.start)} — ${date(dry.end)}${dry.complete?"":" · неполные данные"}`:"В доступных данных сухих дней нет";
    updateCards();
  }
  function renderEpisodes() {
    const items=episodes();
    $("episode-count").textContent=items.length?String(items.length):"";
    $("episode-list").innerHTML=items.map((e,i)=>`<details class="episode ${e.severity==="критическая"?"strong":""}"><summary><span>${date(e.start)} — ${date(e.end)}</span><span class="episode-level">${e.days} дн. · ${e.severity==="критическая"?"сильное":"умеренное"} отклонение</span></summary><div class="episode-body"><p>${esc(hypotheses[e.cause]||"Причина требует проверки на поле.")}</p><p class="muted">${e.n_obs} наблюдений в периоде · минимальный Z: ${format(e.min_z)}. Цвет показывает силу отклонения, а не доказанную причину.</p><button class="quiet" data-episode="${i}">Показать период на графиках</button></div></details>`).join("")||
      `<p class="muted">${season.z.some(p=>finite(p.value))?"В этом сезоне выделенных эпизодов нет.":"Недостаточно данных для оценки отклонений."}</p>`;
    for(const button of document.querySelectorAll("[data-episode]"))button.onclick=()=>{
      const e=items[Number(button.dataset.episode)],shift=(d,n)=>new Date(Date.parse(d)+n*86400000).toISOString().slice(0,10);
      ChartUI.setRange([shift(e.start,-30),shift(e.end,10)]);$("ndvi").scrollIntoView({behavior:"smooth",block:"center"});
    };
    $("shape-notes").textContent=(report.shape||[]).filter(s=>s.year===year).map(s=>s.shape_reasons).join(" ");
  }
  function details() {
    if(!season||!$("calculation").open)return;
    const thresholds=[-1,-2].map((n,i)=>({type:"line",xref:"paper",x0:0,x1:1,y0:n,y1:n,line:{color:i?"#ce5350":"#e5ab48",width:1,dash:"dash"}}));
    Plotly.react("zplot",[{...ChartUI.daily(season.z),mode:"lines",name:"Z",line:{color:"#25362b",width:1.5}}],ChartUI.layout("Z",{shapes:[...shapes(),...thresholds]}),options);
    const colors={"Sentinel-2":"#417ca8","Landsat":"#8872ba","MODIS":"#c78845"},traces=[];
    for(const [sensor,color] of Object.entries(colors)) {
      const points=season.observations.filter(p=>p.sensor===sensor);
      traces.push({x:points.map(p=>p.date),y:points.map(p=>p.value),mode:"markers",name:sensor,
        marker:{color,size:6,symbol:points.map(p=>p.artifact?"x":"circle")}});
    }
    traces.push({x:season.restored.map(p=>p.date),y:season.restored.map(p=>p.value),mode:"markers",name:"Восстановленные пропуски",marker:{symbol:"diamond",color:"#c44787",size:7}});
    Plotly.react("raw-ndvi",traces,ChartUI.layout("NDVI",{showlegend:true,legend:{orientation:"h",y:-.2},margin:{l:54,r:18,t:8,b:65}}),options);
    const w=report.weather?.[year];$("raw-weather").hidden=!w;
    if(w)Plotly.react("wx",[{x:w.date,y:w.precip,type:"bar",name:"Осадки, мм",marker:{color:"#6f9fc2"}},
      {x:w.date,y:w.temp,mode:"lines",name:"Температура, °C",line:{color:"#c78845",width:1.5},yaxis:"y2"}],
      ChartUI.layout("мм",{yaxis2:{title:{text:"°C"},overlaying:"y",side:"right",showgrid:false},margin:{l:54,r:54,t:8,b:30}}),options);
    $("source-note").textContent=report.collected||"Исходные наблюдения и контрольные восстановления из данных кейса.";
  }
  function render(data,selectedYear) {
    const token=++version;report=data;year=selectedYear;season=data.years[year];weather=null;
    ChartUI.start(year);$("calculation").open=false;
    for(const id of ["rain","heat","dry"])$("metric-"+id).hidden=true;
    for(const id of ["zplot","raw-ndvi","wx"])Plotly.purge(id);
    updateCards();renderEpisodes();drawNDVI(token);
  }
  document.addEventListener("chart-date",event=>{if(season)updateCards(event.detail);});
  document.addEventListener("DOMContentLoaded",()=>{$("calculation").ontoggle=details;});
  return {render,setWeather};
})();
