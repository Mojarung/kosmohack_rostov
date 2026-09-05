/* Погодные графики сразу по имеющимся данным; NDVI и детектор не меняются. */
window.WeatherCharts = (() => {
  const $ = id => document.getElementById(id);
  const finite = value => typeof value === "number" && Number.isFinite(value);
  const format = value => value.toLocaleString("ru-RU", {maximumFractionDigits:1});
  const dateLabel = value => new Date(value+"T12:00:00").toLocaleDateString("ru-RU",{day:"numeric",month:"long"});
  let context = null, currentYear = null, sequence = 0;

  function plot(id, metric, units, color) {
    const target = $(id);
    Plotly.purge(target); target.replaceChildren();
    const traces = [];
    if (metric.mean.some(finite)) {
      traces.push({x:metric.date,y:metric.high,mode:"lines",line:{width:0},showlegend:false,hoverinfo:"skip",connectgaps:false});
      traces.push({x:metric.date,y:metric.low,mode:"lines",line:{width:0},fill:"tonexty",fillcolor:"rgba(150,150,150,.2)",showlegend:false,connectgaps:false,hoverinfo:"skip"});
      traces.push({x:metric.date,y:metric.mean,mode:"lines",name:"Среднее прошлых лет",line:{color:"#888",width:1.5},connectgaps:false});
    }
    traces.push({x:metric.date,y:metric.value,mode:"lines",name:String(currentYear),line:{color,width:2},connectgaps:false});
    Plotly.newPlot(target,traces,{
      margin:{l:60,r:15,t:5,b:50},yaxis:{title:{text:units}},
      xaxis:{range:[`${currentYear}-04-01`,`${currentYear}-10-30`]},
      legend:{orientation:"h",y:-.22},hovermode:"x unified"
    },{displayModeBar:false,responsive:true});
  }

  function latest(metric, units) {
    const i = metric.value.findLastIndex(finite);
    if (i < 0) return "";
    return `На ${dateLabel(metric.date[i])}: ${format(metric.value[i])} ${units}` +
      (finite(metric.mean[i]) ? ` · среднее прошлых лет: ${format(metric.mean[i])} ${units}` : " · истории для сравнения мало");
  }

  function renderRain() {
    const balance = $("agro-water-mode").value === "water" && context.water.available;
    const metric = balance ? context.water : context.rain;
    $("agro-rain-section").hidden = !metric.available;
    if (!metric.available) return;
    $("agro-water-title").textContent = balance ? "Осадки − испарение за 30 дней, мм" : "Осадки за 30 дней, мм";
    $("agro-water-value").textContent = latest(metric,"мм");
    plot("agro-water",metric,"мм","#2b6cb0");
  }

  function display() {
    const {rain,thermal,water,dry_spell:dry} = context;
    $("agro-extra").hidden = !rain.available && !thermal.available && !water.available;
    $("agro-water-mode").hidden = !water.available || !rain.available;
    $("agro-water-mode").value = rain.available ? "rain" : "water";
    $("agro-balance-note").hidden = !water.available;
    $("agro-status").textContent = "";
    renderRain();
    $("agro-dry-spell").hidden = !dry?.available;
    $("agro-dry-spell").textContent = dry?.available ? `Самый длинный сухой период: ${dry.days} дн.` +
      (dry.days ? ` · ${dateLabel(dry.start)} — ${dateLabel(dry.end)}` : "") + (dry.complete ? "" : " · по доступным дням") : "";
    $("agro-heat-section").hidden = !thermal.available;
    if (thermal.available) {
      $("agro-heat-value").textContent = latest(thermal,"°C·дни");
      plot("agro-heat",thermal,"°C·дни","#dd6b20");
    }
    const years = [...new Set([...rain.history_years,...thermal.history_years])].sort();
    $("agro-source").textContent = (context.source || "ERA5 из данных кейса") +
      (years.length ? ` · история ${years[0]}–${years.at(-1)}` : "");
  }

  async function render(report, year) {
    const token = ++sequence;
    currentYear = year; context = null;
    $("agro-extra").hidden = true;
    $("agro-rain-section").hidden = true; $("agro-heat-section").hidden = true;
    $("agro-extra").querySelector("details").open = false;
    for (const id of ["agro-water","agro-heat"]) { Plotly.purge(id); $(id).replaceChildren(); }
    try {
      const response = await fetch(`/api/polygon/${encodeURIComponent(report.pid)}/agro?year=${year}`);
      if (!response.ok) throw new Error("weather");
      const result = await response.json();
      if (token !== sequence) return;
      context = result; display();
    } catch {
      if (token !== sequence) return;
      $("agro-extra").hidden = false;
      $("agro-status").textContent = "Не удалось загрузить погодные графики. Обновите страницу.";
    }
  }

  document.addEventListener("DOMContentLoaded",() => {$("agro-water-mode").onchange = renderRain;});
  return {render};
})();
