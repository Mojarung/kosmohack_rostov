/** Тонкая обёртка над fetch: единый разбор ошибок FastAPI (поле detail). */

import type {
  Episode,
  Meta,
  OsmField,
  PolygonDetail,
  PolygonSummary,
  Summary,
  UserPolygon,
} from "./types";

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
  analyze: (body: { geometry: GeoJSON.Polygon; name: string; start_year?: number; end_year?: number }) =>
    request<PolygonDetail>("/api/analyze", { method: "POST", body: JSON.stringify(body) }),
};
