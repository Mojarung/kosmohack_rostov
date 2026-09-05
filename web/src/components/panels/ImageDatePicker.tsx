/** Календарь доступных снимков: пустые дни недоступны, стрелки листают реальные снимки. */
import { useId, useState } from "react";
import Popover from "@mui/material/Popover";
import { localDate, shortDate } from "../../lib/format";

export function ImageDatePicker({ dates, value, onChange }: {
  dates: string[]; value: string; onChange: (date: string) => void;
}) {
  const sorted = [...new Set(dates)].sort(), months = [...new Set(sorted.map(d => d.slice(0, 7)))];
  const selected = sorted.indexOf(value), available = new Set(sorted);
  const [anchor, setAnchor] = useState<HTMLButtonElement | null>(null);
  const [month, setMonth] = useState(value.slice(0, 7));
  const id = useId(), monthIndex = months.indexOf(month);
  const first = localDate(`${month}-01`), offset = (first.getDay() + 6) % 7;
  const days = new Date(first.getFullYear(), first.getMonth() + 1, 0).getDate();
  const monthLabel = first.toLocaleDateString("ru-RU", { month: "long", year: "numeric" });
  return <div className="image-date-picker" role="group" aria-label="Выбор снимка">
    <button type="button" className="image-date-step" aria-label="Предыдущий снимок" disabled={selected <= 0}
      onClick={() => onChange(sorted[selected - 1])}>‹</button>
    <button type="button" className="image-date-trigger" aria-label="Дата снимка" aria-haspopup="dialog"
      aria-expanded={Boolean(anchor)} aria-controls={anchor ? id : undefined}
      onClick={e => { setMonth(value.slice(0, 7)); setAnchor(e.currentTarget); }}>
      {shortDate(value)} {value.slice(0, 4)} <span aria-hidden>▾</span>
    </button>
    <button type="button" className="image-date-step" aria-label="Следующий снимок" disabled={selected < 0 || selected >= sorted.length - 1}
      onClick={() => onChange(sorted[selected + 1])}>›</button>
    <Popover open={Boolean(anchor)} anchorEl={anchor} onClose={() => setAnchor(null)}
      anchorOrigin={{ vertical: "bottom", horizontal: "left" }}
      slotProps={{ paper: { className: "field-popover image-calendar", role: "dialog", id, "aria-label": "Дата снимка" } }}>
      <div className="popover-heading"><strong>Дата снимка</strong>
        <button type="button" className="info-button" aria-label="Закрыть календарь" onClick={() => setAnchor(null)}>×</button></div>
      <div className="calendar-month">
        <button type="button" className="image-date-step" aria-label="Предыдущий месяц со снимками" disabled={monthIndex <= 0}
          onClick={() => setMonth(months[monthIndex - 1])}>‹</button>
        <span aria-live="polite">{monthLabel}</span>
        <button type="button" className="image-date-step" aria-label="Следующий месяц со снимками" disabled={monthIndex >= months.length - 1}
          onClick={() => setMonth(months[monthIndex + 1])}>›</button>
      </div>
      <div className="calendar-days" role="group" aria-label={monthLabel}>
        {["Пн", "Вт", "Ср", "Чт", "Пт", "Сб", "Вс"].map(d => <span className="calendar-weekday" key={d} aria-hidden>{d}</span>)}
        {Array.from({ length: offset }, (_, i) => <span key={`empty-${i}`} aria-hidden />)}
        {Array.from({ length: days }, (_, i) => {
          const day = i + 1, date = `${month}-${String(day).padStart(2, "0")}`;
          return <button type="button" key={date} disabled={!available.has(date)} aria-pressed={date === value}
            aria-label={`${shortDate(date)} ${date.slice(0, 4)}`}
            onClick={() => { onChange(date); setAnchor(null); }}>{day}</button>;
        })}
      </div>
      <p className="calendar-hint">Доступны только дни со снимками.</p>
    </Popover>
  </div>;
}
