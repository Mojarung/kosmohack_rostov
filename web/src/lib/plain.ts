/** Перевод показателей и эпизодов на бытовой язык для плашки-вердикта и карточек.
 *
 *  Все фразы строятся из данных, которые уже отдаёт API: кривая NDVI и ориентир сезона,
 *  эпизоды детектора, тренд отклонения и погодные показатели. Детектор и модель не трогаем —
 *  здесь только формулировки. Числа остаются доступны мелким шрифтом рядом с фразой. */

import type { AgroContext, DeviationTrend } from "../api/analytics";
import type { Episode, SeasonYear } from "../api/types";
import { isoDay, localDate, plural, shortDate } from "./format";
import { finite, metricPoint } from "./metrics";

export type Tone = "ok" | "warn" | "crit";

/** Причина эпизода — как её сказал бы агроном соседу, без внутренних кодов. */
export const CAUSE_PLAIN: Record<string, string> = {
  weather_drought: "из-за засухи или жары",
  unsown_or_changed: "поле не засеяно или на нём другая культура",
  early_decline: "растения увяли раньше срока: уборка, полегание или болезнь",
  weak_season: "сезон в целом слабее обычного",
  late_start: "всходы задержались",
  crop_rotation: "на поле другая культура, чем обычно, это не беда",
  data_suspect: "похоже на ошибку спутниковых данных, а не на проблему поля",
};

/** Совет по причине: что фермеру делать с этим эпизодом. */
export const CAUSE_ADVICE: Record<string, string> = {
  weather_drought: "Осмотрите поле: при засухе важно оценить, восстановятся ли растения после дождей.",
  unsown_or_changed: "Проверьте, что на этом контуре действительно посеяно; возможно, границы поля изменились.",
  early_decline: "Съездите на поле и проверьте растения на полегание, болезни и вредителей.",
  weak_season: "Сравните с соседними полями и урожайностью прошлых лет; возможно, дело в питании или семенах.",
  late_start: "Если всходы не догонят норму к середине сезона, стоит пересмотреть план подкормок.",
  crop_rotation: "Делать ничего не нужно: график просто сравнивает поле с другой культурой.",
  data_suspect: "Подождите следующий чистый снимок; если снижение сохранится, тогда проверьте поле.",
};

/** Уверенность детектора словом: 0.75 → «скорее всего». */
export function confidenceWord(confidence: number): string {
  if (confidence >= 0.8) return "почти наверняка";
  if (confidence >= 0.6) return "скорее всего";
  return "возможно";
}

/** Насколько эпизод серьёзен, простыми словами. */
export function severityPlain(episode: Episode): string {
  return episode.severity === "критическая" ? "сильно отстаёт" : "заметно отстаёт";
}

/** Одна фраза про эпизод: «Поле сильно отставало с 15 авг по 28 сен, скорее всего из-за засухи». */
export function episodeSentence(episode: Episode, past = true): string {
  const verb = past ? severityPlain(episode).replace("отстаёт", "отставало") : severityPlain(episode);
  const cause = CAUSE_PLAIN[episode.cause] ?? "причина неясна";
  return `Поле ${verb} с ${shortDate(episode.start)} по ${shortDate(episode.end)}: ${confidenceWord(episode.confidence)} ${cause}.`;
}

/** Процент отклонения зелени от ориентира: (0.42, 0.50) → −16. */
function percentOff(value: number, normal: number): number | null {
  if (!finite(value) || !finite(normal) || normal < 0.08) return null;
  return Math.round(((value - normal) / normal) * 100);
}

/** Фраза про зелень поля: «Зелени на 16 % меньше, чем обычно в это время». */
export function ndviPlain(value: number | undefined, normal: number | undefined): { text: string; tone: Tone } {
  const off = finite(value) && finite(normal) ? percentOff(value, normal) : null;
  if (off === null) return { text: "Не с чем сравнить: мало истории", tone: "ok" };
  if (Math.abs(off) <= 7) return { text: "Зелени столько же, сколько обычно в это время", tone: "ok" };
  const more = off > 0;
  const text = `Зелени на ${Math.abs(off)} % ${more ? "больше" : "меньше"}, чем обычно в это время`;
  return { text, tone: more ? "ok" : off < -25 ? "crit" : "warn" };
}

/** Осадки за 30 дней против нормы: «Дождей за месяц вдвое меньше нормы». */
export function rainPlain(value: number | null | undefined, mean: number | null | undefined): { text: string; tone: Tone } {
  if (!finite(value)) return { text: "Нет данных о дождях", tone: "ok" };
  if (!finite(mean) || mean <= 0) return { text: "Дожди есть в данных, но сравнить не с чем: мало истории", tone: "ok" };
  const ratio = value / mean;
  if (ratio < 0.35) return { text: "Дождей за месяц почти не было: меньше трети нормы", tone: "crit" };
  if (ratio < 0.6) return { text: "Дождей за месяц вдвое меньше нормы", tone: "warn" };
  if (ratio < 0.85) return { text: "Дождей за месяц немного меньше нормы", tone: "ok" };
  if (ratio <= 1.25) return { text: "Дождей за месяц выпало как обычно", tone: "ok" };
  if (ratio <= 1.8) return { text: "Дождей за месяц больше нормы", tone: "ok" };
  return { text: "Дождей за месяц почти вдвое больше нормы", tone: "warn" };
}

/** Накопленное тепло против нормы: «Сезон идёт с опережением на неделю». */
export function heatPlain(value: number | null | undefined, mean: number | null | undefined): { text: string; tone: Tone } {
  if (!finite(value)) return { text: "Нет данных о тепле", tone: "ok" };
  if (!finite(mean) || mean <= 0) return { text: "Тепло посчитано, но сравнить не с чем: мало истории", tone: "ok" };
  // В разгар сезона сутки дают около 18–22 °C·дней, поэтому разницу переводим в дни грубо, через 20.
  const days = Math.round((value - mean) / 20);
  if (Math.abs(days) < 3) return { text: "Тепла накоплено как обычно, сезон идёт по графику", tone: "ok" };
  const word = `${Math.abs(days)} ${plural(Math.abs(days), "день", "дня", "дней")}`;
  return days > 0
    ? { text: `Сезон опережает обычный примерно на ${word}: теплее, чем всегда`, tone: Math.abs(days) > 10 ? "warn" : "ok" }
    : { text: `Сезон отстаёт от обычного примерно на ${word}: прохладнее, чем всегда`, tone: Math.abs(days) > 10 ? "warn" : "ok" };
}

/** Самый длинный сухой период: «Три недели без дождя». */
export function dryPlain(days: number): { text: string; tone: Tone } {
  if (days < 7) return { text: "Долгих перерывов в дождях не было", tone: "ok" };
  if (days < 14) return { text: `Самая долгая сушь: ${days} ${plural(days, "день", "дня", "дней")} без дождя`, tone: "ok" };
  if (days < 21) return { text: `Две недели без дождя (${days} ${plural(days, "день", "дня", "дней")} подряд)`, tone: "warn" };
  return { text: `Больше трёх недель без дождя (${days} ${plural(days, "день", "дня", "дней")} подряд)`, tone: "crit" };
}

/** Тренд отклонения словами фермера. */
export function trendPlain(trend: DeviationTrend | undefined): string | null {
  if (!trend?.available) return null;
  switch (trend.status) {
    case "within": return "Последние две недели поле держится на уровне обычного.";
    case "recovered": return "Поле выправилось: за последние две недели вернулось к обычному уровню.";
    case "easing": return "Отставание сокращается: последние две недели поле догоняет норму.";
    case "worsening": return "Отставание растёт: последние две недели поле сдаёт.";
    case "stable": return "За последние две недели ничего не изменилось.";
    default: return null;
  }
}

export interface Verdict {
  tone: Tone;
  /** Короткий заголовок светофора: «Всё в норме», «Стоит присмотреться», «Поле отстаёт». */
  title: string;
  /** Одна фраза-вывод. */
  sentence: string;
  /** Ответ на «как поле сейчас?». */
  now: string;
  /** Ответ на «что было в сезоне?»: по одной строке на событие. */
  history: string[];
  /** Ответ на «что делать?». */
  advice: string;
}

/** Последнее реальное наблюдение сезона и ориентир на ту же дату. */
export function latestState(season: SeasonYear): { date: string; value: number; normal: number | undefined } | null {
  const latest = season.observations.filter(p => !p.artifact && finite(p.harmonized))
    .sort((a, b) => a.date.localeCompare(b.date)).at(-1);
  if (!latest) return null;
  return { date: latest.date, value: latest.harmonized, normal: season.norm_mean.find(p => p.date === latest.date)?.value };
}

/** Эпизод, который ещё идёт на дату последнего наблюдения (или закончился за неделю до неё). */
function ongoing(episodes: Episode[], lastDate: string | undefined): Episode | undefined {
  if (!lastDate) return undefined;
  const limit = localDate(lastDate); limit.setDate(limit.getDate() - 7);
  const edge = isoDay(limit);
  return [...episodes].filter(e => e.end >= edge).sort((a, b) => a.min_z - b.min_z)[0];
}

/** Светофор поля за сезон: тон, заголовок, фраза и ответы на три вопроса. */
export function seasonVerdict(season: SeasonYear, episodes: Episode[], trend?: DeviationTrend, weather?: AgroContext): Verdict {
  const state = latestState(season);
  const worst = [...episodes].sort((a, b) => a.min_z - b.min_z)[0];
  const real = episodes.filter(e => e.cause !== "crop_rotation" && e.cause !== "data_suspect");
  const current = ongoing(real, state?.date);
  const greens = ndviPlain(state?.value, state?.normal);
  const trendText = trendPlain(trend);

  const history = [...episodes].sort((a, b) => a.start.localeCompare(b.start)).map(e => episodeSentence(e));
  const rain = weather?.rain.available ? rainPlain(metricPoint(weather.rain, null).value, metricPoint(weather.rain, null).mean) : null;
  if (rain && rain.tone !== "ok") history.push(`${rain.text}.`);
  const dry = weather?.dry_spell?.available ? dryPlain(weather.dry_spell.days) : null;
  if (dry && dry.tone === "crit" && weather?.dry_spell.start && weather.dry_spell.end)
    history.push(`С ${shortDate(weather.dry_spell.start)} по ${shortDate(weather.dry_spell.end)} дождей не было вовсе.`);

  const now = state
    ? `${greens.text} (снимок ${shortDate(state.date)}).${trendText ? ` ${trendText}` : ""}`
    : "Реальных снимков за этот сезон нет.";

  if (current) {
    const crit = current.severity === "критическая";
    return {
      tone: crit ? "crit" : "warn",
      title: crit ? "Поле отстаёт" : "Стоит присмотреться",
      sentence: episodeSentence(current, false),
      now, history, advice: CAUSE_ADVICE[current.cause] ?? "Осмотрите поле лично.",
    };
  }
  if (real.length) {
    const past = [...real].sort((a, b) => a.min_z - b.min_z)[0];
    const recovered = trend?.status === "recovered" || trend?.status === "within" || greens.tone === "ok";
    return {
      tone: recovered ? "ok" : "warn",
      title: recovered ? "Сейчас в норме" : "Стоит присмотреться",
      sentence: recovered
        ? `В сезоне было отставание (${shortDate(past.start)} — ${shortDate(past.end)}), но сейчас поле на обычном уровне.`
        : `Поле отставало ${shortDate(past.start)} — ${shortDate(past.end)} и пока не догнало обычный уровень.`,
      now, history,
      advice: recovered ? "Ничего срочного. Отставание уже позади, следите за следующими снимками." : CAUSE_ADVICE[past.cause] ?? "Осмотрите поле лично.",
    };
  }
  if (worst) {
    return {
      tone: "ok", title: "Всё в норме", sentence: episodeSentence(worst),
      now, history, advice: CAUSE_ADVICE[worst.cause] ?? "Делать ничего не нужно.",
    };
  }
  const hasData = season.z.some(p => finite(p.value));
  return {
    tone: greens.tone === "crit" ? "warn" : "ok",
    title: hasData ? "Всё в норме" : "Мало данных",
    sentence: hasData ? "Поле развивается как обычно, отставаний за сезон не найдено." : "Наблюдений пока недостаточно, чтобы судить о поле.",
    now, history,
    advice: hasData ? "Делать ничего не нужно. Заглядывайте после новых снимков." : "Подождите новых чистых снимков со спутника.",
  };
}
