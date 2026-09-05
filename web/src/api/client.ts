/** Тонкая обёртка над fetch: единый разбор ошибок FastAPI (поле detail). */

import type {
  AskAnswer,
  CollectProgressState,
  Episode,
  Meta,
  OsmField,
  PolygonDetail,
  PolygonSummary,
  Summary,
  UserPolygon,
} from "./types";
import type { AgroContext, ImageryResponse } from "./analytics";

// Повторное открытие поля во время сбора присоединяется к уже запущенному запросу.
const collecting = new Map<string, Promise<ImageryResponse>>();
function collectImagery(pid: string, year: number) {
  const path = `/api/polygon/${encodeURIComponent(pid)}/imagery?year=${year}`;
  if (!collecting.has(path)) {
    collecting.set(path, request<ImageryResponse>(path, { method: "POST" }).finally(() => collecting.delete(path)));
  }
  return collecting.get(path)!;
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(path, {
    ...init,
    headers: init?.body ? { "Content-Type": "application/json", ...init?.headers } : init?.headers,
  });
  if (!response.ok) {
    let detail = response.statusText;
    try {
      const body = (await response.json()) as { detail?: string };
      detail = body.detail ?? detail;
    } catch {
      /* тело не JSON — оставляем статус */
    }
    throw new Error(detail);
  }
  return (await response.json()) as T;
}

export const api = {
  polygons: () => request<PolygonSummary[]>("/api/polygons"),
  polygon: (pid: string) => request<PolygonDetail>(`/api/polygon/${encodeURIComponent(pid)}`),
  agro: (pid: string, year: number, signal?: AbortSignal) =>
    request<AgroContext>(`/api/polygon/${encodeURIComponent(pid)}/agro?year=${year}`, { signal }),
  imagery: (pid: string, year: number, signal?: AbortSignal) =>
    request<ImageryResponse>(`/api/polygon/${encodeURIComponent(pid)}/imagery?year=${year}`, { signal }),
  collectImagery,
  episodes: (params: { year?: number; cause?: string; severity?: string } = {}) => {
    const query = new URLSearchParams();
    if (params.year) query.set("year", String(params.year));
    if (params.cause) query.set("cause", params.cause);
    if (params.severity) query.set("severity", params.severity);
    const suffix = query.toString();
    return request<Episode[]>(`/api/episodes${suffix ? `?${suffix}` : ""}`);
  },
  summary: () => request<Summary>("/api/summary"),
  meta: () => request<Meta>("/api/meta"),
  fields: (bbox: [number, number, number, number]) =>
    request<OsmField[]>(`/api/fields?bbox=${bbox.map((v) => v.toFixed(4)).join(",")}`),
  userPolygons: () => request<UserPolygon[]>("/api/user-polygons"),
  userPolygon: (uid: string) => request<PolygonDetail>(`/api/user-polygons/${uid}`),
  deleteUserPolygon: (uid: string) => request<{ deleted: string }>(`/api/user-polygons/${uid}`, { method: "DELETE" }),
  analyze: (body: { geometry: GeoJSON.Polygon; name: string; start_year?: number; end_year?: number; job?: string }) =>
    request<PolygonDetail>("/api/analyze", { method: "POST", body: JSON.stringify(body) }),
  analyzeProgress: (job: string) => request<CollectProgressState>(`/api/analyze/progress/${job}`),
  /** Вопрос о поле своими словами: с ключом отвечает модель, без ключа — разбор по правилам. */
  ask: (body: { pid: string; question: string; year?: number }) =>
    request<AskAnswer>("/api/ask", { method: "POST", body: JSON.stringify(body) }),
  /** Ссылка на отчёт по полю: самодостаточный HTML, печатается в PDF из браузера. */
  reportUrl: (pid: string, year?: number) =>
    `/api/report/${encodeURIComponent(pid)}${year ? `?year=${year}` : ""}`,
};
