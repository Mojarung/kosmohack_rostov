// Запуск сценария с установленным Playwright и Chrome; сервер запускается отдельно.
import { createRequire } from "node:module";
import { readFile, mkdir } from "node:fs/promises";
import vm from "node:vm";
const require = createRequire(import.meta.url);
const { chromium } = require(process.env.E2E_PLAYWRIGHT_MODULE || "playwright");
const browser = await chromium.launch({ channel: process.env.E2E_BROWSER || "chrome", headless: true });
const page = await browser.newPage();
try {
  await mkdir("artifacts/e2e", { recursive: true });
  page.setDefaultTimeout(20000);
  const scenarioFile = process.env.E2E_ANOMALIES === "1" ? "./anomalies.js" : process.env.E2E_RESPONSE === "1" ? "./api_response.js"
    : process.env.E2E_LIVE === "1" ? "./geocoding_live.js" : "./geocoding.js";
  const scenario = vm.runInThisContext(await readFile(new URL(scenarioFile, import.meta.url), "utf8"));
  console.log(JSON.stringify(await scenario(page, process.env.E2E_BASE || "http://127.0.0.1:8011")));
} catch (error) {
  console.error(error);
  await page.screenshot({ path: "artifacts/e2e/geocoding-failure.png", timeout: 5000 }).catch(() => {});
  throw error;
} finally {
  await browser.close();
}
