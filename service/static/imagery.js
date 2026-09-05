/* Площадь считается по общей чистой маске двух снимков, а не по среднему NDVI. */
(() => {
  const {$,format,date}=ChartUI;
  const palettes={ndvi:{colors:"#9b8872,#c2ad85 50%,#e7d596 60%,#91b967 75%,#337750 90%,#174d38",labels:["−1","NDVI","1"]},
    ndmi:{colors:"#a98261,#d8b88e 30%,#e6e6c9 50%,#8ac6b4 65%,#398c98 80%,#21536f",labels:["−1 · суше","NDMI","1 · влажнее"]},
    change:{colors:"#ac524b,#d99171 37.5%,#f5f2e5 50%,#85b795 62.5%,#28765b",labels:["−0,4 · снижение","0","+0,4 · рост"]}};
  let report,year,manifest=null,index="ndvi",sequence=0,map,overlay,contour;
  const pending=new Map();
  const endpoint=()=>`/api/polygon/${encodeURIComponent(report.pid)}/imagery?year=${year}`;
  async function request(url,opts) {
    const response=await fetch(url,opts),data=await response.json();
    if(!response.ok)throw new Error(data.detail||"Не удалось загрузить снимки");
    return data;
  }
  function area(scene) {
    const change=scene?.change;
    $("area-value").textContent=change?`${format(change.drop_area_ha,1)} га · ${format(change.drop_share*100,1)}%`:"Нужны два снимка";
    $("area-note").textContent=change?`${date(change.previous_date)} → ${date(scene.date)} · от сравнимой площади`:"С общим чистым покрытием от 60%";
    $("area-evidence").textContent=change?
      `NDVI снизился минимум на ${format(Math.abs(change.threshold),2)}. Сравниваем ${format(change.area_ha,1)} га (${format(change.clear_share*100,1)}% поля) без облаков на обе даты. Это изменение растительности; уборка и созревание тоже могут давать снижение.`:
      "Площадь можно посчитать только по пикселям двух пригодных снимков. Пропуски и облака не считаются снижением.";
  }
  function ensureMap() {
    if(!map) {
      map=L.map("imagery-map",{scrollWheelZoom:false});
      L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png",{maxZoom:19,attribution:"© OpenStreetMap"}).addTo(map);
    }
    return map;
  }
  function draw() {
    if(!manifest)return;
    const scene=manifest.scenes.find(s=>s.date===$("imagery-date").value);
    if(!scene)return;
    const changeButton=document.querySelector('[data-image="change"]');
    changeButton.disabled=!scene.change;
    changeButton.title=scene.change?"":"Нет предыдущего снимка с достаточной общей чистой площадью";
    if(index==="change"&&!scene.change)index="ndvi";
    document.querySelectorAll("[data-image]").forEach(b=>b.setAttribute("aria-pressed",String(b.dataset.image===index)));
    area(scene);
    $("imagery-note").textContent=index==="change"?
      `${date(scene.change.previous_date)} → ${date(scene.date)} · общая чистая площадь ${format(scene.change.area_ha,1)} га (${format(scene.change.clear_share*100,1)}%) · снижение не означает доказанное повреждение.`:
      `${date(scene.date)} · видно ${format(scene[index].clear_share*100,1)}% поля · среднее ${index.toUpperCase()} ${format(scene[index].mean,3)}. Прозрачные участки — нет пригодных данных.`;
    $("imagery-scale").style.background=`linear-gradient(to right,${palettes[index].colors})`;
    $("imagery-labels").replaceChildren(...palettes[index].labels.map(t=>{const span=document.createElement("span");span.textContent=t;return span;}));
    if(!$("imagery-card").open)return;
    ensureMap();
    if(overlay)map.removeLayer(overlay);
    const path=`/api/polygon/${encodeURIComponent(report.pid)}/imagery/${year}/${scene.date}/${index}.png?v=${manifest.generation}`;
    const token=sequence;
    const layer=L.imageOverlay(path,manifest.bounds,{opacity:.9}).addTo(map);overlay=layer;
    layer.on("error",()=>{if(token===sequence&&overlay===layer)$("imagery-status").textContent="Изображение не загрузилось. Выберите дату ещё раз.";});
    layer.on("load",()=>{if(token===sequence&&overlay===layer)$("imagery-status").textContent=`${manifest.scenes.length} пригодных снимков · ${manifest.source}`;});
    if(contour)map.removeLayer(contour);
    contour=L.geoJSON(report.geometry,{style:{color:"#243e30",weight:2,fill:false}}).addTo(map);
    map.invalidateSize();map.fitBounds(contour.getBounds(),{padding:[18,18],maxZoom:17});
  }
  function display(data) {
    manifest=data.manifest;
    $("imagery-content").hidden=!data.available;
    $("imagery-load").hidden=!!data.available;
    $("imagery-load").disabled=false;
    if(!manifest) {
      $("imagery-status").textContent="Загрузите снимки сезона, чтобы увидеть изменения внутри поля. Первый сбор займёт несколько минут.";
      return;
    }
    const scenes=[...manifest.scenes].sort((a,b)=>b.date.localeCompare(a.date));
    $("imagery-date").replaceChildren(...scenes.map(s=>new Option(date(s.date),s.date)));
    $("imagery-status").textContent=`${scenes.length} пригодных снимков · ${manifest.source}`;
    draw();
  }
  async function render(data,selectedYear) {
    const token=++sequence;report=data;year=selectedYear;manifest=null;index="ndvi";
    $("imagery-card").open=false;$("area-card").open=false;
    $("imagery-card").hidden=$("area-card").hidden=!data.geometry;
    $("imagery-content").hidden=true;$("imagery-load").hidden=true;
    if(overlay&&map){map.removeLayer(overlay);overlay=null;}
    if(!data.geometry)return;
    $("area-value").textContent="Нужны снимки";
    $("area-note").textContent="Загрузить в карте поля";
    $("area-evidence").textContent="Площадь считается по пикселям Sentinel-2. Среднего NDVI по полю недостаточно.";
    $("imagery-status").textContent="Проверяем сохранённые снимки…";
    const url=endpoint();
    try {const result=await (pending.get(url)||request(url));if(token===sequence)display(result);}
    catch {if(token===sequence){$("imagery-status").textContent="Снимки не загрузились. Можно повторить.";$("imagery-load").hidden=false;$("imagery-load").disabled=false;}}
  }
  $("imagery-load").onclick=async()=>{
    const token=sequence,url=endpoint();
    $("imagery-load").disabled=true;
    $("imagery-status").textContent="Собираем Sentinel-2 за сезон… Это может занять несколько минут.";
    if(!pending.has(url))pending.set(url,request(url,{method:"POST"}));
    try {const result=await pending.get(url);if(token===sequence)display(result);}
    catch(error){if(token===sequence){$("imagery-status").textContent=error.message;$("imagery-load").disabled=false;$("imagery-load").textContent="Повторить загрузку";}}
    finally {pending.delete(url);}
  };
  $("area-open").onclick=()=>{$("imagery-card").open=true;if(manifest&&manifest.scenes.find(s=>s.date===$("imagery-date").value)?.change)index="change";draw();$("imagery-card").scrollIntoView({behavior:"smooth",block:"center"});};
  $("imagery-card").ontoggle=()=>{if($("imagery-card").open)draw();};
  $("imagery-date").onchange=draw;
  document.querySelectorAll("[data-image]").forEach(b=>b.onclick=()=>{index=b.dataset.image;draw();});
  window.renderImagery=render;
})();
