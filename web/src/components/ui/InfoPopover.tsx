/** Короткая справка поверх страницы: не меняет высоту соседних карточек. */
import { useId, useState, type ReactNode } from "react";
import Popover from "@mui/material/Popover";

export function InfoPopover({ title, children }: { title: string; children: ReactNode }) {
  const [anchor, setAnchor] = useState<HTMLButtonElement | null>(null);
  const id = useId();
  return <>
    <button type="button" className="info-button" aria-label={`Справка: ${title}`}
      aria-haspopup="dialog" aria-expanded={Boolean(anchor)} aria-controls={anchor ? id : undefined}
      onClick={e => setAnchor(e.currentTarget)}>ⓘ</button>
    <Popover open={Boolean(anchor)} anchorEl={anchor} onClose={() => setAnchor(null)}
      anchorOrigin={{ vertical: "bottom", horizontal: "right" }} transformOrigin={{ vertical: "top", horizontal: "right" }}
      slotProps={{ paper: { className: "field-popover", role: "dialog", id, "aria-label": title } }}>
      <div className="popover-heading"><strong>{title}</strong>
        <button type="button" className="info-button" aria-label="Закрыть справку" onClick={() => setAnchor(null)}>×</button></div>
      <div className="popover-copy">{children}</div>
    </Popover>
  </>;
}
