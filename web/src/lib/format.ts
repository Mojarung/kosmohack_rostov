/** Форматирование и словари, общие для всех экранов. */

import type { Severity } from "../api/types";

export const CAUSE_LABEL: Record<string, string> = {
  weather_drought: "погодный стресс",
  unsown_or_changed: "не засеяно или другая культура",
  early_decline: "ранний спад",
  weak_season: "ослабленный сезон",
  late_start: "поздний старт",
  crop_rotation: "севооборот, не угнетение",
  data_suspect: "вероятная ошибка данных",
};

export const SENSOR_COLOR: Record<string, string> = {
  "Sentinel-2": "#3a6ea5",
  Landsat: "#6b5b95",
  MODIS: "#c2703a",
};

export const SEVERITY_TONE: Record<string, "crit" | "warn" | "ok"> = {
  критическая: "crit",
  умеренная: "warn",
  норма: "ok",
};

export function severityTone(severity: Severity | string): "crit" | "warn" | "ok" {
  return SEVERITY_TONE[severity] ?? "ok";
}

const MONTHS = ["янв", "фев", "мар", "апр", "мая", "июн", "июл", "авг", "сен", "окт", "ноя", "дек"];

/** 2024-06-10 → «10 июн» */
export function shortDate(iso: string): string {
  const [, month, day] = iso.split("-");
  return `${Number(day)} ${MONTHS[Number(month) - 1]}`;
}

/** Диапазон дат эпизода одной строкой. */
export function dateRange(start: string, end: string): string {
  return `${shortDate(start)} — ${shortDate(end)}`;
}

export function plural(n: number, one: string, few: string, many: string): string {
  const mod10 = n % 10;
  const mod100 = n % 100;
  if (mod10 === 1 && mod100 !== 11) return one;
  if (mod10 >= 2 && mod10 <= 4 && (mod100 < 10 || mod100 >= 20)) return few;
  return many;
}

export function ms(iso: string): number {
  return new Date(`${iso}T00:00:00Z`).getTime();
}
