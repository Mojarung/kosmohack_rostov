// MCP Playwright browser_run_code_unsafe({filename: ...}); реальный сервис и данные кейса.
async (page, base="http://127.0.0.1:8000") => {
  let checks=0; const errors=[];
  const check=(ok,message)=>{checks++;if(!ok)throw new Error(message);};
  const close=(a,b)=>typeof a==="number"&&Math.abs(a-b)<.002;
  const api=async path=>{const r=await page.request.get(base+path);check(r.ok(),path);return r.json();};
  page.on("pageerror",e=>errors.push(e.message));
  await api("/api/health");
  await page.setViewportSize({width:1440,height:1050});
  await page.goto(base+"/legacy");
  await page.waitForFunction(()=>document.getElementById("weather-plot").data?.length>0);
  async function field(pid,withWeather=true) {
    await page.locator("#search").fill(pid);
    await page.locator(`.poly[data-pid="${pid}"]`).click();
    await page.waitForFunction(pid=>S.data?.pid===pid&&document.getElementById("ndvi").data?.length>0,pid);
    if(withWeather)await ready();
  }
  async function ready() { await page.waitForFunction(()=>document.getElementById("weather-plot").data?.at(-1).name===String(S.year)); }
  async function mode(key) {
    await page.locator(`[data-weather="${key}"]`).click();
    await page.waitForFunction(key=>document.querySelector(`[data-weather="${key}"]`).getAttribute("aria-pressed")==="true",key);
  }
  async function values() { return page.locator("#weather-plot").evaluate(el=>({value:el.data.at(-1).y.at(-1),norm:el.data.find(t=>t.name==="Среднее прошлых лет")?.y.at(-1),range:el.layout.xaxis.range,traces:el.data.length})); }

  await field("AOI-0001");await mode("rain");
  check(await page.locator("#map").isVisible(),"Исходная карта пропала");
  check(await page.locator("#calculation").evaluate(el=>!el.open),"Подробности не свёрнуты");
  check(await page.locator(".js-plotly-plot:visible").count()===2,"По умолчанию должно быть два графика");
  check(await page.locator("section form").count()===0,"Осталась обязательная форма");
  check(!await page.locator('[data-weather="water"]').isVisible(),"Предлагается ET0 без данных");
  const report=await api("/api/polygon/AOI-0001"),w=report.weather[2025],end=w.date.indexOf("2025-10-30");
  const expectedRain=w.precip.slice(end-29,end+1).reduce((a,b)=>a+b,0);
  const expectedHeat=w.temp.slice(0,end+1).reduce((a,b)=>a+Math.max(b-10,0),0);
  const historical=Object.entries(report.weather).filter(([y])=>+y<2025&&+y>=1995).map(([,v])=>{
    const i=v.date.findIndex(d=>d.endsWith("-10-30")),days=v.precip.slice(i-29,i+1);
    return i>=29&&days.length===30&&days.every(v=>v!==null)?days.reduce((a,b)=>a+b,0):null;
  }).filter(v=>v!==null);
  let run=0,dry=0;for(const v of w.precip.slice(0,end+1)){run=v!==null&&v<1?run+1:0;dry=Math.max(dry,run);}
  const rain=await values();
  check(close(rain.value,expectedRain),"Осадки не совпадают с исходными днями");
  check(close(rain.norm,historical.reduce((a,b)=>a+b,0)/historical.length),"Неверная историческая норма");
  check((await page.locator("#metric-dry-value").innerText()).includes(`${dry} дн.`),"Неверный сухой период");
  await mode("thermal");check(close((await values()).value,expectedHeat),"Тепло не совпадает с исходными температурами");
  check(await page.locator(".js-plotly-plot:visible").count()===2,"Переключатель создаёт лишние графики");

  await page.locator("#calculation>summary").click();
  await page.waitForFunction(()=>document.getElementById("raw-ndvi").data?.length>0);
  check(await page.locator("#zplot").isVisible(),"Z недоступен в подробностях");
  check(await page.locator("#wx").isVisible(),"Исходная погода недоступна");
  check(await page.locator("#raw-ndvi").evaluate(el=>el.data.some(t=>t.name==="Восстановленные пропуски"&&t.y.length>0)),"Контрольные восстановления модели потерялись");
  await page.locator("#calculation>summary").click();
  await page.locator("#ndvi").scrollIntoViewIfNeeded();
  const box=await page.locator("#ndvi").boundingBox();
  await page.mouse.move(box.x+box.width*.55,box.y+box.height*.5);
  await page.waitForFunction(()=>document.getElementById("metric-ndvi-label").textContent.includes("Кривая"));
  check((await page.locator("#weather-plot .hoverlayer").textContent()).includes("Тепло"),"Наведение не синхронизируется");
  const pickedDate=(await page.locator("#metric-ndvi-label").innerText()).split(" · ")[1];
  check((await page.locator("#weather-readout").innerText()).startsWith(pickedDate),"На графиках разные даты наведения");
  await page.mouse.move(0,0);
  await page.locator(".episode>summary").first().click();
  await page.locator("[data-episode]").first().click();
  await page.waitForFunction(()=>document.getElementById("ndvi").layout.xaxis.range[0]!=="2025-04-01");
  const episodeRange=await page.locator("#ndvi").evaluate(el=>el.layout.xaxis.range);
  check(JSON.stringify((await values()).range)===JSON.stringify(episodeRange),"Период эпизода не синхронизируется");
  await mode("rain");check(JSON.stringify((await values()).range)===JSON.stringify(episodeRange),"Переключатель теряет выбранный период");
  await page.locator("#reset-period").click();
  await page.waitForFunction(()=>document.getElementById("weather-plot").layout.xaxis.range[0]==="2025-04-01");
  check((await values()).range[1]==="2025-10-30","Сброс не возвращает весь сезон");
  await page.locator('#years button[data-y="2024"]').click();await ready();
  check(await page.locator("#weather-plot").evaluate(el=>el.data.at(-1).name)==="2024","Сезон не сменился");
  await page.locator('#years button[data-y="2025"]').click();await ready();
  check(close((await values()).value,expectedRain),"Кэш вернул другой сезон");

  await field("AOI-0005");
  await page.locator('#years button[data-y="2010"]').click();await ready();
  check((await values()).traces===1,"Выдумана историческая норма для одного сезона");
  check((await page.locator("#weather-readout").innerText()).includes("истории для сравнения мало"),"Нет пояснения к отсутствию нормы");
  const noWeatherResponse=page.waitForResponse(r=>r.url().includes("/api/polygon/AOI-0006/agro?"));
  await field("AOI-0006",false);
  await noWeatherResponse;
  await page.waitForFunction(()=>document.getElementById("metric-dry").hidden);
  check(!await page.locator("#agro-extra").isVisible(),"Показана пустая погодная панель");
  check(await page.locator("#ndvi").isVisible(),"Отсутствие погоды скрыло NDVI");

  const saved=(await api("/api/saved-fields")).find(p=>p.geometry);let balanceChecked=false;
  if(saved){
    await field(saved.pid);
    if(await page.locator('[data-weather="water"]').isVisible()){
      await mode("water");const year=await page.evaluate(()=>S.year),agro=await api(`/api/polygon/${saved.pid}/agro?year=${year}`);
      check(close((await values()).value,agro.water.value.at(-1)),"Переключатель баланса показывает осадки");balanceChecked=true;
    }
  }
  const failPattern="**/api/polygon/AOI-0037/agro?*";
  await page.route(failPattern,r=>r.fulfill({status:503,body:'{"detail":"test outage"}',contentType:"application/json"}));
  await field("AOI-0037",false);
  await page.locator("#weather-retry").waitFor({state:"visible"});
  check(await page.locator("#ndvi").isVisible(),"Ошибка погоды сломала основной график");
  await page.unroute(failPattern);await page.locator("#weather-retry").click();await ready();
  check(!await page.locator("#weather-retry").isVisible(),"Повторная загрузка не восстановила график");

  await field("AOI-0001");await mode("rain");
  await page.setViewportSize({width:390,height:844});
  await page.locator("#ndvi").scrollIntoViewIfNeeded();
  await page.waitForFunction(()=>document.documentElement.scrollWidth<=innerWidth&&document.getElementById("ndvi").clientWidth<=390);
  check(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth),"Горизонтальная прокрутка на телефоне");
  check(await page.locator("#ndvi").evaluate(el=>el.clientWidth<=390),"График выходит за экран");
  await page.setViewportSize({width:1440,height:1050});
  await page.locator("#ndvi").scrollIntoViewIfNeeded();
  check(errors.length===0,"Ошибки JavaScript: "+errors.join("; "));
  return {status:"passed",checks,rain:expectedRain,heat:expectedHeat,dryDays:dry,balanceChecked};
}
