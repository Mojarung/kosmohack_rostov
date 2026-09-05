// Выполняется Playwright browser_run_code_unsafe с filename; нужны работающий сервис и реальные данные кейса.
async (page) => {
  const failures = [];
  let checks = 0;
  const check = (value, message) => { checks++; if (!value) throw new Error(message); };
  const close = (a, b) => typeof a === "number" && Math.abs(a - b) < .002;
  page.on("pageerror", error => failures.push(error.message));
  let available = false;
  for (let attempt=0; attempt<40 && !available; attempt++) {
    try { available = (await page.request.get("http://127.0.0.1:8000/api/health",{timeout:1000})).ok(); } catch {}
    if (!available) await new Promise(resolve=>setTimeout(resolve,500));
  }
  check(available,"Сервис не запустился за 20 секунд");
  await page.goto("http://127.0.0.1:8000/");
  await page.waitForFunction(() => document.querySelectorAll("#list .poly").length > 0);

  async function openField(pid) {
    await page.locator("#search").fill(pid);
    await page.locator(`.poly[data-pid="${pid}"]`).click();
    await page.waitForFunction(pid => S.data?.pid === pid && document.getElementById("ndvi").data?.length > 0, pid);
  }
  async function ready(year) {
    await page.waitForFunction(year => document.getElementById("agro-heat").data?.at(-1).name === String(year), year);
  }

  await openField("AOI-0001");
  await ready(2025);
  check(await page.locator("#map").isVisible(), "Исходная карта пропала");
  check(await page.locator("#agro-heat-section").isVisible(), "Тепло не появилось автоматически");
  check(await page.locator("#agro-rain-section").isVisible(), "Осадки не появились автоматически");
  check(await page.locator("#agro-extra form").count() === 0, "Осталась обязательная форма");
  check(!await page.locator("#agro-water-mode").isVisible(), "Предлагается ET0 при отсутствии данных");
  check(!await page.locator("#agro-extra details").evaluate(el => el.open), "Пояснения не свёрнуты");

  const response = await page.request.get("http://127.0.0.1:8000/api/polygon/AOI-0001");
  const report = await response.json(), weather = report.weather[2025];
  const end = weather.date.indexOf("2025-10-30");
  const expectedRain = weather.precip.slice(end-29,end+1).reduce((sum,v) => sum+v,0);
  const expectedHeat = weather.temp.slice(0,end+1).reduce((sum,v) => sum+Math.max(v-10,0),0);
  const historicRain = Object.entries(report.weather).filter(([year])=>Number(year)<2025 && Number(year)>=1995).map(([,w])=>{
    const end = w.date.findIndex(d=>d.endsWith("-10-30"));
    const values = w.precip.slice(end-29,end+1);
    return end>=29 && values.length===30 && values.every(v=>v!==null) ? values.reduce((a,b)=>a+b,0) : null;
  }).filter(v=>v!==null);
  let run = 0, expectedDry = 0;
  for (const value of weather.precip.slice(0,end+1)) { run = value !== null && value < 1 ? run+1 : 0; expectedDry = Math.max(expectedDry,run); }
  const drawn = await page.evaluate(() => ({
    rain:document.getElementById("agro-water").data.at(-1).y.at(-1),
    heat:document.getElementById("agro-heat").data.at(-1).y.at(-1),
    norm:document.getElementById("agro-water").data.find(t=>t.name==="Среднее прошлых лет").y.at(-1),
    dry:document.getElementById("agro-dry-spell").textContent,
    old:["ndvi","zplot","wx"].map(id => document.getElementById(id).data?.length)
  }));
  check(close(drawn.rain,expectedRain), "Осадки на графике не совпадают с суммой исходных дней");
  check(close(drawn.heat,expectedHeat), "Тепло на графике не совпадает с исходными температурами");
  check(close(drawn.norm,historicRain.reduce((a,b)=>a+b,0)/historicRain.length), "Историческая норма включает неверные годы");
  check(drawn.dry.includes(`${expectedDry} дн.`), "Сухой период не совпадает с исходными осадками");
  check(drawn.old.every(n=>n>0), "Один из исходных графиков сломан");
  await page.locator('#years button[data-y="2024"]').click();
  await ready(2024);
  await page.locator('#years button[data-y="2025"]').click();
  await ready(2025);

  await openField("AOI-0005");
  await ready(2025);
  check(await page.locator("#agro-water").evaluate(el=>el.data.length) === 1, "Нарисована выдуманная историческая норма");
  check((await page.locator("#agro-water-value").innerText()).includes("истории для сравнения мало"), "Не указано отсутствие истории");

  const listResponse = await page.request.get("http://127.0.0.1:8000/api/polygons");
  const noWeather = (await listResponse.json()).find(p=>!p.has_weather);
  check(noWeather, "Нет реального примера без погоды для проверки");
  const noWeatherRequest = page.waitForResponse(r=>r.url().includes(`/api/polygon/${noWeather.pid}/agro?`));
  await openField(noWeather.pid);
  await noWeatherRequest;
  await page.waitForFunction(() => !document.getElementById("agro-water").data);
  check(!await page.locator("#agro-extra").isVisible(), "Для поля без погоды показана пустая панель");

  const savedResponse = await page.request.get("http://127.0.0.1:8000/api/saved-fields");
  const saved = (await savedResponse.json()).find(p=>p.geometry);
  let savedFieldChecked = false;
  if (saved) {
    await openField(saved.pid);
    const savedYear = await page.evaluate(()=>S.year);
    await ready(savedYear);
    if (await page.locator("#agro-water-mode").isVisible()) {
      await page.locator("#agro-water-mode").selectOption("water");
      check((await page.locator("#agro-water-title").innerText()).includes("испарение"), "Не переключается водный баланс");
      const agro = await (await page.request.get(`http://127.0.0.1:8000/api/polygon/${saved.pid}/agro?year=${savedYear}`)).json();
      check(close(await page.locator("#agro-water").evaluate(el=>el.data.at(-1).y.at(-1)),agro.water.value.at(-1)), "На графике баланса остались осадки");
      savedFieldChecked = true;
    }
  }

  await openField("AOI-0001");
  await ready(2025);
  await page.locator("#agro-extra summary").click();
  check(await page.locator("#agro-source").isVisible(), "Не открылись сведения о расчёте");
  await page.locator("#agro-extra summary").click();
  await page.locator("#agro-extra").scrollIntoViewIfNeeded();
  check(failures.length === 0, "Ошибки браузера: " + failures.join("; "));
  return {status:"passed",checks,rain:drawn.rain,heat:drawn.heat,dryDays:expectedDry,noWeather:noWeather.pid,savedFieldChecked};
}
