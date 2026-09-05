// Run with MCP Playwright browser_run_code_unsafe({filename: absolutePath}).
async (page) => {
  await page.unrouteAll({behavior:"ignoreErrors"});
  let checks=0;const errors=[],base="http://127.0.0.1:8000";
  const check=(ok,message)=>{checks++;if(!ok)throw new Error(message);};
  const onError=e=>errors.push(e.message);page.on("pageerror",onError);
  const api=async path=>{const r=await page.request.get(base+path);check(r.ok(),path);return r.json();};
  const fmt=n=>n.toLocaleString("ru-RU",{minimumFractionDigits:1,maximumFractionDigits:1});
  const field=async pid=>{
    await page.locator("#search").fill(pid);await page.locator(`.poly[data-pid="${pid}"]`).click();
    await page.waitForFunction(pid=>S.data?.pid===pid&&document.getElementById("trend-note").textContent.length>0,pid);
    await page.waitForFunction(()=>document.getElementById("weather-plot").data?.at(-1)?.name===String(S.year));
  };
  const loaded=async()=>page.waitForFunction(()=>{const img=document.querySelector("#imagery-map .leaflet-image-layer");return img?.complete&&img.naturalWidth>0;});
  await page.setViewportSize({width:1440,height:1050});await page.goto(base);
  await page.waitForFunction(()=>document.getElementById("trend-note").textContent.length>0);
  await field("AOI-0030");
  const data=await api("/api/polygon/AOI-0030");
  check(await page.locator("#trend-value").innerText()===data.insights[2025].label,"Неверный статус динамики");
  check(!await page.locator("#area-card").isVisible(),"Площадь выдумана без координат");
  await page.locator("#trend-card>summary").click();
  check((await page.locator("#trend-evidence").innerText()).includes("Медиана Z"),"Нет объяснения статуса");
  await page.locator('#years [data-y="2024"]').click();
  await page.waitForFunction(()=>S.year===2024);
  check(await page.locator("#trend-value").innerText()===data.insights[2024].label,"Статус не меняется с сезоном");
  const saved=(await api("/api/saved-fields")).find(f=>f.geometry);
  check(!!saved,"Нужно сохранённое поле с реальными картами");
  await field(saved.pid);
  await page.locator('#years [data-y="2025"]').click();
  const url=`/api/polygon/${saved.pid}/imagery?year=2025`,real=await api(url);
  check(real.available&&real.manifest.scenes.length>1,"Сначала соберите реальные снимки 2025");
  await page.waitForFunction(()=>document.getElementById("area-value").textContent.includes("га"));
  const latest=real.manifest.scenes.at(-1);
  check((await page.locator("#area-value").innerText()).includes(fmt(latest.change.drop_area_ha)),"Неверная площадь снижения");
  check((await page.locator("#area-value").innerText()).includes(fmt(latest.change.drop_share*100)),"Неверная доля общей площади");
  await page.locator('#years [data-y="2024"]').click();
  await page.waitForFunction(()=>S.year===2024&&document.getElementById("area-value").textContent==="Нужны снимки");
  check(await page.locator("#imagery-content").evaluate(el=>el.hidden),"Карта прошлого сезона осталась после смены года");
  await page.locator('#years [data-y="2025"]').click();
  await page.waitForFunction(()=>document.getElementById("area-value").textContent.includes("га"));
  await page.locator("#area-card>summary").click();await page.locator("#area-open").click();await loaded();
  check(await page.locator('[data-image="change"]').getAttribute("aria-pressed")==="true","Кнопка не открывает карту изменения");
  check((await page.locator("#area-evidence").innerText()).includes("обе даты"),"Не объяснены облака и знаменатель");
  await page.locator('[data-image="ndmi"]').click();await loaded();
  check((await page.locator("#imagery-note").innerText()).includes("NDMI"),"NDMI недоступен");
  const first=real.manifest.scenes[0];await page.locator("#imagery-date").selectOption(first.date);await loaded();
  check(await page.locator('[data-image="change"]').isDisabled(),"Разница разрешена без предыдущей сцены");
  check(await page.locator("#area-value").innerText()==="Нужны два снимка","Отсутствие сравнения заменено нулём");
  await page.locator("#imagery-date").selectOption(latest.date);await page.locator('[data-image="change"]').click();await loaded();
  await page.setViewportSize({width:390,height:844});await page.locator("#area-card").scrollIntoViewIfNeeded();
  await page.waitForFunction(()=>document.documentElement.scrollWidth<=innerWidth);
  check(await page.locator("#imagery-map").evaluate(el=>el.clientWidth<=390),"Карта выходит за экран");
  await page.screenshot({path:"artifacts/e2e/insights-mobile.png",fullPage:true});
  await page.setViewportSize({width:1440,height:1050});

  // A late response for a previously selected field must not restore its area card.
  const pattern=`**/api/polygon/${saved.pid}/imagery?year=2025`;
  await field("AOI-0030");
  let delayed;const arrived=new Promise(resolve=>delayed=resolve);
  await page.route(pattern,async route=>{delayed();await page.waitForTimeout(800);await route.fulfill({json:real});});
  await field(saved.pid);await arrived;await field("AOI-0030");
  await page.waitForTimeout(1000);
  check(!await page.locator("#area-card").isVisible(),"Запоздалый ответ вернул прежнее поле");
  await page.unroute(pattern);

  // Cold cache and a failed collection request: the actual button must POST and retry.
  let posts=0;
  await page.route(pattern,route=>{
    if(route.request().method()==="GET")return route.fulfill({json:{available:false}});
    posts++;return posts===1?route.fulfill({status:503,json:{detail:"Источник временно недоступен"}}):route.fulfill({json:real});
  });
  await field(saved.pid);await page.locator("#imagery-card>summary").click();await page.locator("#imagery-load").click();
  await page.waitForFunction(()=>document.getElementById("imagery-status").textContent.includes("временно"));
  check(await page.locator("#ndvi").isVisible(),"Ошибка снимков скрыла основной график");
  await page.locator("#imagery-load").click();await loaded();
  check(posts===2,"Повторная загрузка не вызвала API");await page.unroute(pattern);
  await page.reload();await page.waitForFunction(()=>document.getElementById("trend-note").textContent.length>0);
  await field(saved.pid);await page.waitForFunction(()=>document.getElementById("area-value").textContent.includes("га"));
  await page.locator("#area-card>summary").click();await page.locator("#area-open").click();await loaded();
  check(await page.locator("#metric-rain").isVisible(),"Погодные метрики скрылись после загрузки карты");
  await page.locator("#area-card").scrollIntoViewIfNeeded();
  await page.screenshot({path:"artifacts/e2e/insights-desktop.png",fullPage:true});
  check(errors.length===0,"Ошибки JavaScript: "+errors.join("; "));page.off("pageerror",onError);
  return {status:"passed",checks,realScenes:real.manifest.scenes.length,latestDate:latest.date,area:latest.change};
}
