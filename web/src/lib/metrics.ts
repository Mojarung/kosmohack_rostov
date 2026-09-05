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
  return finite(value) && finite(mean)
    ? `${value - mean > 0 ? "+" : ""}${number(value - mean, digits)} ${units} к среднему`.replace("  ", " ")
    : "Нет оценки для сравнения";
}
