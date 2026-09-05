/* Общая шкала времени, наведение и оформление графиков. */
window.ChartUI = (() => {
  const $ = id => document.getElementById(id);
  const finite = value => typeof value === "number" && Number.isFinite(value);
  const format = (value, digits=1) => finite(value) ? value.toLocaleString("ru-RU",{maximumFractionDigits:digits}) : "—";
  const date = value => value ? new Date(value+"T12:00:00Z").toLocaleDateString("ru-RU",{day:"numeric",month:"short",timeZone:"UTC"}).replace(" г.","") : "—";
  const esc = value => { const el=document.createElement("span"); el.textContent=String(value??""); return el.innerHTML; };
  let year, range, syncing=false, hovering=false;
  const options={displayModeBar:false,responsive:true,scrollZoom:false};
  function start(value) { year=value; range=[`${year}-04-01`,`${year}-10-30`]; $("reset-period").disabled=true; }
  function layout(units, extra={}) {
    return {margin:{l:54,r:18,t:12,b:30},paper_bgcolor:"#fff",plot_bgcolor:"#fff",
      font:{family:"Segoe UI, sans-serif",size:12,color:"#647069"},hovermode:"x unified",showlegend:false,
      hoverlabel:{bgcolor:"#fff",bordercolor:"#dce4dd",font:{size:12}},
      xaxis:{range:[...range],gridcolor:"#f0f3ef",zeroline:false,tickvals:[4,5,6,7,8,9,10].map(m=>`${year}-${String(m).padStart(2,"0")}-01`),
        ticktext:["апр","май","июн","июл","авг","сен","окт"],showspikes:true,spikecolor:"#75827a",spikethickness:1,spikedash:"dot"},
      yaxis:{title:{text:units,standoff:8},gridcolor:"#edf1eb",zerolinecolor:"#d5dcd5"},...extra};
  }
  async function setRange(next) {
    if(syncing)return; syncing=true; range=next.map(String);
    $("reset-period").disabled=range[0]===`${year}-04-01`&&range[1]===`${year}-10-30`;
    try { await Promise.all(["ndvi","weather-plot","zplot","raw-ndvi","wx"].filter(id=>$(id)?.data)
      .map(id=>Plotly.relayout(id,{"xaxis.range":[...range],"xaxis.autorange":false}))); }
    finally { syncing=false; }
  }
  function bind(id) {
    const target=$(id), other=()=>$(id==="ndvi"?"weather-plot":"ndvi");
    for(const [event,fn] of Object.entries(target._fieldEvents||{}))target.removeListener(event,fn);
    target._fieldEvents={
      plotly_hover:event=>{
        if(hovering)return;
        const day=String(event.points[0].x).slice(0,10); hovering=true;
        try { document.dispatchEvent(new CustomEvent("chart-date",{detail:day}));
          if(other()?.data)Plotly.Fx.hover(other(),{xval:Date.parse(day)},["xy"]); }
        finally { hovering=false; }
      },
      plotly_unhover:()=>{
        if(hovering)return; hovering=true;
        try { if(other()?.data)Plotly.Fx.unhover(other()); document.dispatchEvent(new CustomEvent("chart-date",{detail:null})); }
        finally { hovering=false; }
      },
      plotly_relayout:event=>{
        if(syncing)return;
        if(event["xaxis.autorange"])setRange([`${year}-04-01`,`${year}-10-30`]);
        else if(event["xaxis.range"])setRange(event["xaxis.range"]);
        else if(event["xaxis.range[0]"])setRange([event["xaxis.range[0]"],event["xaxis.range[1]"]]);
      }
    };
    for(const [event,fn] of Object.entries(target._fieldEvents))target.on(event,fn);
  }
  function daily(points) {
    const values=new Map(points.map(p=>[p.date,p.value])),x=[],y=[];
    for(let day=Date.parse(`${year}-04-01`);day<=Date.parse(`${year}-10-30`);day+=86400000) {
      const date=new Date(day).toISOString().slice(0,10);x.push(date);y.push(values.get(date)??null);
    }
    return {x,y};
  }
  document.addEventListener("DOMContentLoaded",()=>{$("reset-period").onclick=()=>setRange([`${year}-04-01`,`${year}-10-30`]);});
  return {$,finite,format,date,esc,options,start,layout,setRange,bind,daily};
})();
