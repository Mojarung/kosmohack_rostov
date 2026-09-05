/** Масштаб доступен кнопками, в том числе на телефоне и с клавиатуры. */
import { isoDay, shortDate } from "../../lib/format";
import type { RangeControl } from "./range";

export function RangeToolbar({ control }: { control: RangeControl }) {
  const center = (control.view.from + control.view.to) / 2;
  return <div className="range-toolbar" role="group" aria-label="Масштаб графика">
    <span className="meta" aria-live="polite">{shortDate(isoDay(new Date(control.view.from)))} — {shortDate(isoDay(new Date(control.view.to)))}</span>
    <span className="range-toolbar-actions">
    <button type="button" className="btn btn--sm btn--ghost" aria-label="Отдалить график"
      disabled={!control.zoomed} onClick={() => control.zoomAt(center, 1.7)}>−</button>
    <button type="button" className="btn btn--sm btn--ghost" aria-label="Приблизить график"
      disabled={control.view.to - control.view.from <= 5 * 86400000} onClick={() => control.zoomAt(center, 0.6)}>+</button>
    <button type="button" className="btn btn--sm btn--ghost" onClick={control.reset} disabled={!control.zoomed}>Весь сезон</button>
    </span>
  </div>;
}
