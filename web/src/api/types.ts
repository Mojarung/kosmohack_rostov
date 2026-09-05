/** Типы ответов API сервиса (service/app.py). */

export type Severity = "критическая" | "умеренная";

export interface PolygonSummary {
  pid: string;
  crop: string;
  kind: string;
  years: number[];
  n_obs: number;
  n_episodes: number;
  n_critical: number;
  has_weather: boolean;
  n_gaps: number;
}

export interface Observation {
  date: string;
  value: number;
  harmonized: number;
  sensor: "Sentinel-2" | "Landsat" | "MODIS";
  artifact: boolean;
}

export interface Point {
  date: string;
  value: number;
}

export interface SeasonYear {
  norm_source: string;
  observations: Observation[];
  restored: Point[];
  curve: Point[];
  norm_mean: Point[];
  norm_std: Point[];
  z: Point[];
}

export interface Episode {
  pid: string;
  year: number;
  start: string;
  end: string;
  days: number;
  n_obs: number;
  min_z: number;
  mean_z: number;
  critical_days: number;
  severity: Severity;
  cause: string;
  confidence: number;
  norm_source: string;
  weather_source?: string;
  region_z?: number | null;
  region_share_depressed?: number | null;
  reasons: string;
  text: string;
  ndwi_anomaly?: number | null;
}

export interface ShapeFlag {
  pid: string;
  year: number;
  shape_direction: string;
  shape_reasons: string;
}

export interface WeatherYear {
  date: string[];
  temp: number[];
  precip: number[];
}

export interface PolygonDetail {
  pid: string;
  crop: string;
  kind: string;
  years: Record<string, SeasonYear>;
  episodes: Episode[];
  shape: ShapeFlag[];
  weather: Record<string, WeatherYear>;
  /* только для собранных пользователем территорий */
  uid?: string;
  name?: string;
  collected?: string;
  geometry?: GeoJSON.Polygon;
  created_at?: string;
}

export interface UserPolygon {
  uid: string;
  name: string;
  created_at: string;
  geometry: GeoJSON.Polygon | null;
  years: number[];
  n_episodes: number;
  n_critical: number;
  last_year: number | null;
  last_year_status: "критическая" | "умеренная" | "норма";
}

export interface OsmField {
  id: string;
  name: string;
  geometry: GeoJSON.Polygon;
  area_ha?: number;
}

export interface Summary {
  by_cause: Record<string, number>;
  by_year: Record<string, number>;
  by_severity: Record<string, number>;
  n_polygons: number;
}

export interface Meta {
  task1: {
    n_gaps: number;
    rmse_val: number;
    rmse_testlike: number;
    gap_score: number;
    baseline_rmse: number;
    baseline_gap_score: number;
    models: string;
    train_points: number;
    rmse_model_only: number;
    gap_score_model_only: number;
    rmse_spread: number;
    val_seeds: number;
    status: string;
  };
  task2: {
    n_polygons: number;
    n_seasons: number;
    n_episodes: number;
    seasons_with_episode: number;
    precision_points: number;
    recall_critical: number;
    by_cause: Record<string, number>;
  };
  data: {
    test_file: string;
    test_rows: number;
    train_rows: number;
    extra_file: string | null;
  };
  sources: { name: string; detail: string }[];
}
