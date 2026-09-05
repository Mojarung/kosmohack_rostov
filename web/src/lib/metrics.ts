/** Форматирование неизвестных значений без подстановки нуля. */
import type { WeatherMetric } from "../api/analytics";
import { isoDay } from "./format";

export const finite = (value: unknown): value is number => typeof value === "number" && Number.isFinite(value);
export const number = (value: unknown, digits = 1) => finite(value)
  ? value.toLocaleString("ru-RU", { maximumFractionDigits: digits }) : "—";
export const dayAt = (time: number | null) => time === null ? null : isoDay(new Date(time));
export function metricPoint(metric: WeatherMetric, day: string | null) {
  const i = day ? metric.date.indexOf(day) : metric.value.findLastIndex(finite);
  return { date: metric.date[i], value: metric.value[i], mean: metric.mean[i] };
}
export function comparison(value: unknown, mean: unknown, units = "", digits = 1) {
  if (!finite(value) || !finite(mean)) return "Нет оценки для сравнения";
  const diff = number(value - mean, digits);
  if (diff === "0" || diff === "-0") return "на уровне среднего";   // «+0 к среднему» выглядит как ошибка
  return `${value - mean > 0 ? "+" : ""}${diff} ${units} к среднему`.replace("  ", " ");
}

/** Годы, по которым посчитан ориентир: « за 1996–2025»; пусто, если истории нет. */
export function historySpan(metric: WeatherMetric): string {
  const years = metric.history_years ?? [];
  if (!years.length) return "";
  const first = Math.min(...years), last = Math.max(...years);
  return first === last ? ` за ${first}` : ` за ${first}–${last}`;
}
