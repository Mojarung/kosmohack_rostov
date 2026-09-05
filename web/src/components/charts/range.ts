/** Общий диапазон дат для трёх графиков сезона: NDVI, отклонение и погода.
 *  Диапазон держится в миллисекундах, чтобы одинаково ложиться на временные шкалы всех графиков. */

import { useCallback, useMemo, useState } from "react";

export interface DateRange {
  from: number;
  to: number;
}

/** Минимальная ширина окна: глубже приближать бессмысленно, наблюдения реже. */
const MIN_SPAN_MS = 5 * 24 * 3600 * 1000;

export interface RangeControl {
  /** Текущее окно или null, если показан весь сезон. */
  range: DateRange | null;
  /** Границы всего сезона. */
  full: DateRange;
  /** Видимое окно с учётом сброса. */
  view: DateRange;
  zoomed: boolean;
  select: (from: number, to: number) => void;
  zoomAt: (center: number, factor: number) => void;
  reset: () => void;
}

/** Управление окном просмотра: выделение мышью, зум колесом, сброс. */
export function useDateRange(full: DateRange): RangeControl {
  const [range, setRange] = useState<DateRange | null>(null);

  const clamp = useCallback(
    (from: number, to: number): DateRange => {
      const span = Math.max(MIN_SPAN_MS, to - from);
      let start = Math.max(full.from, Math.min(from, full.to - span));
      let end = Math.min(full.to, start + span);
      if (end - start < span) start = Math.max(full.from, end - span);
      return { from: start, to: end };
    },
    [full.from, full.to],
  );

  const select = useCallback(
    (from: number, to: number) => {
      const [a, b] = from <= to ? [from, to] : [to, from];
      if (b - a < MIN_SPAN_MS) return;                 // случайный клик, а не выделение
      setRange(clamp(a, b));
    },
    [clamp],
  );

  const zoomAt = useCallback(
    (center: number, factor: number) => {
      setRange((current) => {
        const base = current ?? full;
        const span = (base.to - base.from) * factor;
        if (span >= full.to - full.from) return null;  // отдалили до целого сезона
        const share = (center - base.from) / (base.to - base.from);
        return clamp(center - span * share, center + span * (1 - share));
      });
    },
    [clamp, full],
  );

  const reset = useCallback(() => setRange(null), []);
  const view = useMemo(() => range ?? full, [range, full]);

  return { range, full, view, zoomed: range !== null, select, zoomAt, reset };
}
