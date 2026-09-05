// Отдельная проверка реального Nominatim через интерфейс и бэкенд, без сетевых подмен.
async (page, base) => {
  const errors = [];
  page.on("pageerror", error => errors.push(error.message));
  await page.setViewportSize({ width: 1440, height: 900 });
  await page.goto(`${base}/explore`);
  await page.locator('[data-testid="field-map"][data-state="ready"]').waitFor();
  await page.getByLabel("Найти место", { exact: true }).fill("Ростов-на-Дону, Большая Садовая улица");
  const responsePromise = page.waitForResponse(r => r.url().includes("/api/places?"));
  await page.getByRole("button", { name: "Найти", exact: true }).click();
  const response = await responsePromise;
  if (!response.ok()) throw new Error(`Поиск вернул ${response.status()}: ${await response.text()}`);
  const results = await response.json();
  if (!results.length || !results[0].label.includes("Большая Садовая")
      || results[0].address.state !== "Ростовская область") throw new Error("Не найден адрес и его регион");
  await page.locator(".place-results button").first().waitFor();
  await page.screenshot({ path: "artifacts/e2e/geocoding-live-results.png", timeout: 10000 });
  await page.locator(".place-results button").first().click();
  await page.waitForFunction(center => {
    const [w, s, e, n] = document.querySelector('[data-testid="field-map"]').dataset.bounds.split(",").map(Number);
    return Math.abs((w + e) / 2 - center[0]) < .01 && Math.abs((s + n) / 2 - center[1]) < .01 && e - w < .1;
  }, results[0].center);
  // Подложка загружается отдельно от камеры: даём тайлам отрисоваться перед снимком.
  await page.waitForTimeout(2500);
  await page.screenshot({ path: "artifacts/e2e/geocoding-live-map.png", timeout: 10000 });
  if (errors.length) throw new Error(errors.join("; "));
  return { source: "Nominatim через FastAPI", results: results.length, selected: results[0].label, errors };
}
