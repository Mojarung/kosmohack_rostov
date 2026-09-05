/** Погодные показатели и карты: значения рассчитывает тот же API, что и в /legacy. */
export interface WeatherMetric {
  available: boolean;
  date: string[];
  value: (number | null)[];
  mean: (number | null)[];
  low: (number | null)[];
  high: (number | null)[];
  n_years: number[];
  history_years: number[];
}

export interface AgroContext {
  rain: WeatherMetric;
  thermal: WeatherMetric;
  water: WeatherMetric;
  dry_spell: { available: boolean; days: number; start: string | null; end: string | null; complete: boolean };
  source: string;
}

export interface DeviationTrend {
  available: boolean;
  status: string;
  label: string;
  start?: string;
  end?: string;
  before_z?: number;
  after_z?: number;
  before_count: number;
  after_count: number;
}

export type ImageIndex = "ndvi" | "ndmi" | "change";
export interface ImageStats { mean: number | null; clear_share: number; area_ha: number }
export interface ImageScene {
  date: string;
  ndvi: ImageStats;
  ndmi: ImageStats;
  change?: ImageStats & { previous_date: string; drop_area_ha: number; drop_share: number | null; threshold: number };
}
export interface ImageManifest {
  pid: string;
  year: number;
  generation: string;
  source: string;
  geometry: GeoJSON.Polygon;
  bounds: [[number, number], [number, number]];
  area_ha: number;
  scenes: ImageScene[];
}
export type ImageryResponse = { available: false; year: number } | { available: true; manifest: ImageManifest };
