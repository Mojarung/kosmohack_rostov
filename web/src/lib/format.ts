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

/** ISO-дата «2024-06-10» → полночь этого дня по местному времени браузера.
 *
 *  Все даты в графиках создаются только так. Причина: временная шкала MUI X Charts (d3 scaleTime)
 *  расставляет деления по МЕСТНЫМ полуночам, поэтому и точки данных, и подписи должны жить
 *  в одном поясе. Если данные хранить в UTC, а подписи форматировать с timeZone: "UTC",
 *  то восточнее Гринвича 1 июля 00:00 МСК = 30 июня 21:00 UTC и месяц на оси уезжает назад. */
export function localDate(iso: string): Date {
  const [year, month, day] = iso.slice(0, 10).split("-").map(Number);
  return new Date(year, month - 1, day);
}

/** Дата в миллисекундах (местная полночь) — общая единица шкал и окна просмотра. */
export function ms(iso: string): number {
  return localDate(iso).getTime();
}

/** Обратное преобразование: Date → «2024-06-10» по местному времени (не через toISOString). */
export function isoDay(date: Date): string {
  const pad = (value: number) => String(value).padStart(2, "0");
  return `${date.getFullYear()}-${pad(date.getMonth() + 1)}-${pad(date.getDate())}`;
}

/** Подписи временной оси: на делениях месяц, во всплывающей подсказке день и месяц.
 *  Без timeZone — форматируем в том же поясе, в котором построены даты и деления. */
export function axisDate(date: Date, location: string): string {
  return location === "tick"
    ? date.toLocaleDateString("ru-RU", { month: "short" })
    : date.toLocaleDateString("ru-RU", { day: "numeric", month: "long" });
}
