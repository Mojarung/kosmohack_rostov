// React-сборка и настоящий API. Подменяются только явно указанные ошибки и задержки.
async (page, base = "http://127.0.0.1:8000") => {
  let checks = 0;
  const errors = [];
  const check = (value, message) => { checks++; if (!value) throw new Error(message); };
  const onError = e => errors.push(e.message);
  page.on("pageerror", onError);
  const api = async path => { const r = await page.request.get(base + path); check(r.ok(), path); return r.json(); };
  const fmt = (v, digits = 1) => v.toLocaleString("ru-RU", { maximumFractionDigits: digits });
  const loaded = async () => page.locator('[data-testid="imagery-map"][data-state="ready"]').waitFor({ timeout: 45000 });
  const panel = page.getByTestId("season-panel");
  async function open(pid, weather = true) {
    await page.goto(`${base}/field/${pid}`);
    await panel.waitFor();
    if (weather) await page.getByTestId("weather-chart").waitFor();
  }
  async function mode(key) { await page.locator(`[data-weather="${key}"]`).click(); }
  async function range(id) { return page.getByTestId(id).evaluate(el => [el.dataset.from, el.dataset.to]); }
  await page.unrouteAll({ behavior: "ignoreErrors" });
  await page.setViewportSize({ width: 1440, height: 1050 });
  await page.goto(base);
  await page.locator(".hero").waitFor();
  check(await page.locator("video").count() > 0, "Главная main потеряла видео");
  await page.getByRole("link", { name: "Поля кейса", exact: true }).click();
  await page.waitForURL("**/fields");
  await page.locator('a[href="/field/AOI-0001"]').first().waitFor({ timeout: 60000 });
  check(await page.locator(".shell-header").isVisible(), "Потеряна навигация main");
  await open("AOI-0001");
  check(await panel.getAttribute("data-year") === "2025", "Не открыт последний сезон");
  check(await page.getByTestId("z-chart").count() === 0, "Подробности не свёрнуты");
  check(await panel.locator("svg.MuiChartsSvgLayer-root").count() === 2, "Должны быть два основных графика");
  check(await page.locator('[data-weather="water"]').count() === 0, "Баланс без ET0");
  check(await page.getByTestId("area-card").count() === 0, "Площадь без координат");
  const report = await api("/api/polygon/AOI-0001"), daily = report.weather[2025];
  const end = daily.date.indexOf("2025-10-30");
  const rain = daily.precip.slice(end - 29, end + 1).reduce((a, b) => a + b, 0);
  const heat = daily.temp.slice(0, end + 1).reduce((a, b) => a + Math.max(b - 10, 0), 0);
  let run = 0, dry = 0;
  for (const p of daily.precip.slice(0, end + 1)) { run = p !== null && p < 1 ? run + 1 : 0; dry = Math.max(dry, run); }
  check((await page.getByTestId("metric-rain").innerText()).includes(`${fmt(rain)} мм`), "Осадки не совпадают с суточными данными");
  check((await page.getByTestId("metric-heat").innerText()).includes(`${fmt(heat)} °C·дни`), "Тепло не совпадает с температурами");
  check((await page.getByTestId("metric-dry").innerText()).includes(`${dry} дн.`), "Сухой период рассчитан иначе");
  check((await page.getByTestId("trend-card").innerText()).includes(report.insights[2025].label), "Статус динамики отличается от API");
  const rainPath = await page.getByTestId("weather-chart").locator(".MuiLineChart-line").last().getAttribute("d");
  await mode("thermal");
  check((await page.getByTestId("weather-readout").innerText()).includes(`${fmt(heat)} °C·дни`), "Переключение тепла");
  check(await page.getByTestId("weather-chart").locator(".MuiLineChart-line").last().getAttribute("d") !== rainPath, "Кривая погоды не переключилась");
  await page.getByTestId("ndvi-chart").scrollIntoViewIfNeeded();
  const brush = page.getByTestId("ndvi-chart").locator('rect[style*="crosshair"]'), box = await brush.boundingBox();
  await page.mouse.move(box.x + box.width * .5, box.y + box.height * .5);
  await page.waitForFunction(() => document.querySelector('[data-testid="metric-ndvi"]').textContent.includes("Кривая"));
  const day = await page.getByTestId("weather-readout").getAttribute("data-date");
  check(day.startsWith("2025-07"), "Наведение попало не в середину сезона");
  const value = report.years[2025].curve.find(p => p.date === day)?.value;
  check((await page.getByTestId("metric-ndvi").innerText()).includes(fmt(value, 3)), "NDVI и погода показывают разные даты");
  await page.mouse.move(box.x + box.width * .25, box.y + box.height * .5);
  await page.mouse.down(); await page.mouse.move(box.x + box.width * .6, box.y + box.height * .5, { steps: 8 }); await page.mouse.up();
  check(JSON.stringify(await range("ndvi-chart")) === JSON.stringify(await range("weather-chart")), "Приближение не синхронизируется");
  check((await range("ndvi-chart"))[0] !== String(Date.parse("2025-04-01")), "Период не приблизился");
  await page.mouse.move(0, 0);
  await mode("rain");
  check(JSON.stringify(await range("ndvi-chart")) === JSON.stringify(await range("weather-chart")), "Переключение сбросило масштаб");
  await page.getByRole("button", { name: "Весь сезон", exact: true }).click();
  check((await range("weather-chart"))[0] === String(Date.parse("2025-04-01")), "Сброс масштаба");
  await page.locator('.season-years [data-year="2024"]').click();
  await page.getByTestId("weather-chart").waitFor();
  check((await range("ndvi-chart"))[0] === String(Date.parse("2024-04-01")), "Сезон оставил старую шкалу");
  await page.locator('.season-years [data-year="2025"]').click();
  await page.getByTestId("weather-chart").waitFor();
  await page.locator(".season-calculation > summary").click();
  await page.getByTestId("z-chart").waitFor();
  check(await page.getByTestId("raw-weather-chart").isVisible(), "Нет исходной погоды");
  check(await page.getByTestId("raw-ndvi-chart").locator('g[transform*="rotate(45)"]').count() > 0, "Пропали восстановления модели");
  await page.locator(".season-calculation > summary").click();
  await open("AOI-0005");
  await page.locator('.season-years [data-year="2010"]').click();
  await page.getByTestId("weather-chart").waitFor();
  check((await page.getByTestId("weather-readout").innerText()).includes("истории для сравнения мало"), "Для одного года выдумана норма");
  await open("AOI-0006", false);
  await page.waitForFunction(() => !document.querySelector('[role="status"]'));
  check(await page.getByTestId("weather-chart").count() === 0, "Пустой погодный график");
  check(await page.getByTestId("ndvi-chart").isVisible(), "Нет погоды — пропал NDVI");

  const saved = (await api("/api/user-polygons")).find(p => p.uid === "FIELD-073acd129c201f502d38");
  check(!!saved, "Поле прежнего интерфейса потерялось в новом списке");
  await page.goto(`${base}/explore/${saved.uid}`);
  await page.getByTestId("area-card").waitFor();
  await page.getByTestId("weather-chart").waitFor();
  check(await page.getByRole("heading", { name: saved.name, exact: true }).isVisible(), "Поле не открылось по имени");
  const imagery = await api(`/api/polygon/${saved.uid}/imagery?year=2025`);
  check(imagery.available && imagery.manifest.scenes.length > 1, "Нужны реальные снимки");
  const latest = imagery.manifest.scenes.at(-1), first = imagery.manifest.scenes[0];
  const area = await page.getByTestId("area-card").innerText();
  check(area.includes(`${fmt(latest.change.drop_area_ha)} га`) && area.includes(`${fmt(latest.change.drop_share * 100)}%`), "Площадь и знаменатель не совпадают с картой");
  await mode("water");
  const agro = await api(`/api/polygon/${saved.uid}/agro?year=2025`);
  check((await page.getByTestId("weather-readout").innerText()).includes(`${fmt(agro.water.value.at(-1))} мм`), "Баланс ET0 неверен");
  await page.getByTestId("area-card").locator("summary").click();
  await page.getByRole("button", { name: "Открыть карту поля", exact: true }).click(); await loaded();
  check(await page.locator('[data-image="change"]').getAttribute("aria-pressed") === "true", "Кнопка площади открыла не изменение");
  const pngUrl = await page.getByTestId("imagery-map").getAttribute("data-image-url"), png = await page.request.get(base + pngUrl);
  check(png.ok() && png.headers()["content-type"].includes("image/png") && (await png.body()).length > 1000, "На карте нет реального PNG");
  await page.locator('[data-image="ndmi"]').click(); await loaded();
  check((await page.getByTestId("imagery-note").innerText()).includes("NDMI"), "Нет слоя NDMI");
  // Повтор после ошибки файла: манифест ещё не означает, что PNG отрисовался.
  const brokenPng = `**/imagery/2025/${first.date}/ndmi.png?*`;
  await page.route(brokenPng, route => route.fulfill({ status: 503, body: "test image outage" }));
  await page.getByLabel("Дата снимка").selectOption(first.date);
  await page.getByRole("button", { name: "Повторить загрузку карты" }).waitFor();
  check(await page.getByTestId("imagery-map").getAttribute("data-state") === "error", "Нет ошибки PNG");
  await page.unroute(brokenPng);
  await page.getByRole("button", { name: "Повторить загрузку карты" }).click(); await loaded();
  check(await page.getByRole("button", { name: "Повторить загрузку карты" }).count() === 0, "Повтор PNG не сработал");
  await page.getByLabel("Дата снимка").selectOption(first.date); await loaded();
  check(await page.locator('[data-image="change"]').isDisabled(), "Сравнение без предыдущего снимка");
  check((await page.getByTestId("area-card").innerText()).includes("Нужны два снимка"), "Отсутствие сравнения заменили нулём");
  await page.getByLabel("Дата снимка").selectOption("2025-06-08");
  await page.locator('[data-image="change"]').click(); await loaded();
  check((await page.getByTestId("imagery-note").innerText()).includes("3 июн → 8 июн"), "Неверный предыдущий снимок");
  await page.getByTestId("imagery-map").screenshot({ path: "artifacts/e2e/main-map.png" });
  await page.locator('.season-years [data-year="2024"]').click();
  check(await page.getByTestId("imagery-map").count() === 0, "Остался снимок прошлого сезона");
  await page.locator('.season-years [data-year="2025"]').click();
  await page.getByTestId("weather-chart").waitFor();
  await page.setViewportSize({ width: 390, height: 844 });
  await page.getByTestId("area-card").scrollIntoViewIfNeeded();
  check(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth), "Горизонтальная прокрутка на телефоне");
  await page.locator(".imagery-heading").click(); await loaded();
  check((await page.getByTestId("imagery-map").boundingBox()).width <= 390, "Карта шире телефона");
  await page.getByTestId("season-panel").screenshot({ path: "artifacts/e2e/main-mobile.png" });
  await page.setViewportSize({ width: 1440, height: 1050 });

  // Ошибка погоды и восстановление по кнопке без поломки NDVI.
  const outage = "**/api/polygon/AOI-0037/agro?*";
  await page.route(outage, route => route.fulfill({ status: 503, json: { detail: "test outage" } }));
  await open("AOI-0037", false);
  await page.getByRole("button", { name: "Повторить загрузку погоды" }).waitFor();
  check(await page.getByTestId("ndvi-chart").isVisible(), "Ошибка погоды ломает поле");
  await page.unroute(outage); await page.getByRole("button", { name: "Повторить загрузку погоды" }).click();
  await page.getByTestId("weather-chart").waitFor();
  check(await page.getByRole("button", { name: "Повторить загрузку погоды" }).count() === 0, "Повтор не восстановил погоду");

  // Холодный кэш и ошибка POST: реальная кнопка сбора, затем повтор с реальным манифестом.
  const pattern = `**/api/polygon/${saved.uid}/imagery?year=2025`;
  let posts = 0;
  await page.route(pattern, route => {
    if (route.request().method() === "GET") return route.fulfill({ json: { available: false, year: 2025 } });
    posts++; return posts === 1 ? route.fulfill({ status: 503, json: { detail: "Источник временно недоступен" } }) : route.fulfill({ json: imagery });
  });
  await open(saved.uid);
  await page.locator(".imagery-heading").click();
  await page.getByRole("button", { name: "Загрузить снимки сезона", exact: true }).click();
  await page.getByRole("alert").waitFor();
  await page.getByRole("button", { name: "Загрузить снимки сезона", exact: true }).click(); await loaded();
  check(posts === 2, "Кнопка не отправляет POST и повтор"); await page.unroute(pattern);
  // Запоздалый ответ за старый год не меняет карту текущего сезона.
  const delayed = `**/api/polygon/${saved.uid}/imagery?year=2024`;
  await page.route(delayed, async route => { await page.waitForTimeout(500); await route.fulfill({ json: { available: false, year: 2024 } }).catch(() => {}); });
  await page.locator('.season-years [data-year="2024"]').click();
  await page.locator('.season-years [data-year="2025"]').click();
  await page.waitForTimeout(800);
  check((await page.getByTestId("area-card").innerText()).includes(`${fmt(latest.change.drop_area_ha)} га`), "Поздний ответ стёр площадь нового сезона");
  await page.unroute(delayed);
  await page.getByTestId("season-panel").screenshot({ path: "artifacts/e2e/main-desktop.png" });
  // Все вкладки main доступны из той же шапки.
  for (const [name, url] of [["Аномалии", "/anomalies"], ["Как это работает", "/method"], ["Новая территория", "/explore"]]) {
    await page.getByRole("link", { name, exact: true }).click(); await page.waitForURL("**" + url);
    await page.getByRole("heading", { name: name === "Аномалии" ? "Периоды угнетения" : name, exact: true }).waitFor();
    check(await page.locator("h1").count() === 1, `Экран ${name} не открылся`);
  }
  check(errors.length === 0, "Ошибки JavaScript: " + errors.join("; "));
  page.off("pageerror", onError);
  return { status: "passed", checks, rain, heat, dry, realScenes: imagery.manifest.scenes.length, errors };
}
