/* Один погодный график с переключателем; значения берутся из существующего API. */
window.WeatherCharts = (() => {
  const {$,finite,format,date,options}=ChartUI;
  const modes={rain:{title:"Осадки",units:"мм",color:"#417ca8",description:"Сумма осадков за последние 30 дней"},
    thermal:{title:"Тепло",units:"°C·дни",color:"#bd8850",description:"С 1 апреля · порог +10 °C · по средней температуре"},
    water:{title:"Баланс влаги",units:"мм",color:"#428f89",description:"Осадки минус испарение ET₀ за последние 30 дней"}};
  const cache=new Map();
  let context=null,year=null,mode="rain",preferred="rain",sequence=0,selectedReport=null;
  function readout(day=null) {
    if(!context)return;
    const metric=context[mode],i=day?metric.date.indexOf(day):metric.value.findLastIndex(finite);
    const value=metric.value[i],normal=metric.mean[i],units=modes[mode].units;
    $("weather-readout").textContent=`${date(day||metric.date[i])} · ${format(value)} ${units}`+
      (finite(normal)?` · среднее ${format(normal)} ${units}`:finite(value)?" · истории для сравнения мало":" · данных за этот день нет");
  }
  async function draw() {
    if(!context)return;
    const metric=context[mode],spec=modes[mode],token=sequence;
    for(const button of document.querySelectorAll("[data-weather]"))button.setAttribute("aria-pressed",String(button.dataset.weather===mode));
    $("weather-description").textContent=spec.description;
    $("weather-year-key").textContent=String(year);$("weather-year-key").style.setProperty("--weather-color",spec.color);
    $("weather-norm-key").hidden=!metric.mean.some(finite);
    readout();
    const traces=[];
    if(metric.mean.some(finite))traces.push(
      {x:metric.date,y:metric.high,mode:"lines",line:{width:0},hoverinfo:"skip",connectgaps:false},
      {x:metric.date,y:metric.low,mode:"lines",line:{width:0},fill:"tonexty",fillcolor:"rgba(137,148,139,.14)",hoverinfo:"skip",connectgaps:false},
      {x:metric.date,y:metric.mean,mode:"lines",name:"Среднее прошлых лет",line:{color:"#9aa49c",width:1.5},connectgaps:false,hovertemplate:`Среднее: %{y:.1f} ${spec.units}<extra></extra>`});
    traces.push({x:metric.date,y:metric.value,mode:"lines",name:String(year),line:{color:spec.color,width:2.2},connectgaps:false,
      hovertemplate:`${spec.title}: %{y:.1f} ${spec.units}<extra></extra>`});
    await Plotly.react("weather-plot",traces,ChartUI.layout(spec.units),options);
    if(token===sequence)ChartUI.bind("weather-plot");
  }
  function display() {
    const available=Object.keys(modes).filter(key=>context[key].available);
    FieldCharts.setWeather(context);
    $("agro-extra").hidden=!available.length;
    $("weather-content").hidden=!available.length;
    $("weather-status").textContent="";$("weather-retry").hidden=true;
    for(const button of document.querySelectorAll("[data-weather]"))button.hidden=!available.includes(button.dataset.weather);
    const years=[...new Set([...context.rain.history_years,...context.thermal.history_years])].sort();
    $("agro-source").textContent=(context.source||"ERA5 из данных кейса")+(years.length?` · история ${years[0]}–${years.at(-1)}`:"");
    $("agro-balance-note").hidden=!context.water.available;
    if(!available.length)return;
    mode=available.includes(preferred)?preferred:available[0];draw();
  }
  async function render(report,selectedYear) {
    const token=++sequence;selectedReport=report;year=selectedYear;context=null;
    $("agro-extra").hidden=true;$("weather-content").hidden=true;$("weather-status").textContent="";
    Plotly.purge("weather-plot");
    const key=`${report.pid}:${year}:${report.saved_at||"dataset"}`;
    if(!cache.has(key))cache.set(key,fetch(`/api/polygon/${encodeURIComponent(report.pid)}/agro?year=${year}`).then(response=>{
      if(!response.ok)throw new Error("weather");return response.json();
    }).catch(error=>{cache.delete(key);throw error;}));
    try {
      const result=await cache.get(key);if(token!==sequence)return;
      context=result;display();
      if(cache.size>24)cache.delete(cache.keys().next().value);
    } catch {
      if(token!==sequence)return;
      $("agro-extra").hidden=false;$("weather-retry").hidden=false;
      $("weather-status").textContent="Погода не загрузилась.";
    }
  }
  document.addEventListener("chart-date",event=>readout(event.detail));
  document.addEventListener("DOMContentLoaded",()=>{
    for(const button of document.querySelectorAll("[data-weather]"))button.onclick=()=>{mode=button.dataset.weather;preferred=mode;draw();};
    $("weather-retry").onclick=()=>render(selectedReport,year);
  });
  return {render};
})();
