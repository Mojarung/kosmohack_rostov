// Регрессия: старый сервер/прокси возвращает HTML, затем повтор работает с настоящим API.
async (page, base) => {
  const errors = [];
  page.on("pageerror", error => errors.push(error.message));
  await page.goto(`${base}/explore`);
  await page.locator('[data-testid="field-map"][data-state="ready"]').waitFor();
  const missing = await page.request.get(`${base}/api/unknown-route`);
  if (missing.status() !== 404 || !(await missing.json()).detail) throw new Error("API вернул HTML вместо 404 JSON");
  const input = page.getByLabel("Найти место", { exact: true });
  const submit = page.getByRole("button", { name: "Найти", exact: true });
  await input.fill("47.22, 39.72");
  await page.route("**/api/places?**", route => route.fulfill({
    status: 200, contentType: "text/html", body: "<!doctype html><html>Старая версия сервера</html>",
  }));
  await submit.click();
  await page.getByText("Сервер вернул страницу вместо данных.", { exact: false }).waitFor();
  await page.unroute("**/api/places?**");
  await page.route("**/api/places?**", route => route.fulfill({
    status: 503, contentType: "text/plain", body: "Service Unavailable",
  }));
  await submit.click();
  await page.getByText(/неизвестном формате \(HTTP 503\)/).waitFor();
  await page.unroute("**/api/places?**");
  await submit.click();
  await page.locator(".place-results button").first().click();
  await page.getByText("На карте: 47.22, 39.72", { exact: true }).waitFor();
  if (errors.length) throw new Error(errors.join("; "));
  return { base, checks: ["404 JSON", "HTML 200", "text 503", "реальный API после повтора"], errors };
}
