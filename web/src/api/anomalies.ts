import type { Episode } from "./types";
export type FieldSource = "mine" | "case" | "all";
export type AnomalyLevel = "high" | "medium" | "data" | "context" | "unknown" | "clear";
export type AnomalyEpisode = Pick<Episode, "year" | "start" | "end" | "days" | "severity" | "cause" | "text"> & {
  min_z: number | null; confidence: number | null; reasons: string | null;
};
export interface AnomalyField {
  key: string; pid: string; uid: string | null; name: string; source: "mine" | "case";
  years: number[]; level: AnomalyLevel; has_season: boolean; episodes: AnomalyEpisode[];
}
export interface AnomalyFeed { years: number[]; year: number | null; fields: AnomalyField[] }
