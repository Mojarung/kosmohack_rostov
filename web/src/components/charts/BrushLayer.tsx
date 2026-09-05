/** Слой взаимодействия поверх графика: выделение диапазона мышью, зум колесом,
 *  двойной клик — сброс, общий вертикальный курсор для всех графиков сезона.
 *
 *  Лежит внутри SVG графика и пользуется его же шкалами, поэтому координаты совпадают
 *  с линиями без пересчёта. События всплывают дальше, так что подсказки MUI продолжают работать. */

import { useEffect, useRef, useState } from "react";
import { useDrawingArea, useXScale } from "@mui/x-charts/hooks";

import type { RangeControl } from "./range";

interface BrushLayerProps {
  axisId: string;
  control: RangeControl;
  /** Дата под курсором в миллисекундах; null — курсор вне графика. */
  onHover?: (value: number | null) => void;
  hover?: number | null;
}

export function BrushLayer({ axisId, control, onHover, hover }: BrushLayerProps) {
  const { left, top, width, height } = useDrawingArea();
  const scale = useXScale(axisId) as unknown as {
    (value: number | Date): number | undefined;
    invert?: (px: number) => Date;
  };
  const rectRef = useRef<SVGRectElement>(null);
  const [drag, setDrag] = useState<{ from: number; to: number } | null>(null);

  /** Пиксель курсора внутри области рисования. */
  const localX = (clientX: number) => {
    const svg = rectRef.current?.ownerSVGElement;
    if (!svg) return null;
    const box = svg.getBoundingClientRect();
    return Math.max(left, Math.min(left + width, clientX - box.left));
  };

  const toTime = (px: number) => scale.invert?.(px)?.getTime() ?? null;

  // React вешает onWheel пассивно, поэтому preventDefault там не работает: слушаем вручную
  useEffect(() => {
    const node = rectRef.current;
    if (!node) return;
    const onWheel = (event: WheelEvent) => {
      const svg = node.ownerSVGElement;
      if (!svg) return;
      const box = svg.getBoundingClientRect();
      const px = Math.max(left, Math.min(left + width, event.clientX - box.left));
      const center = scale.invert?.(px)?.getTime();
      if (center === undefined) return;
      event.preventDefault();
      control.zoomAt(center, event.deltaY < 0 ? 0.78 : 1 / 0.78);
    };
    node.addEventListener("wheel", onWheel, { passive: false });
    return () => node.removeEventListener("wheel", onWheel);
  }, [control, left, width, scale]);

  const hoverX = hover != null ? scale(hover) : undefined;

  return (
    <g>
      {/* выделяемый диапазон, пока тянут мышью */}
      {drag && (
        <rect
          x={Math.min(drag.from, drag.to)}
          y={top}
          width={Math.abs(drag.to - drag.from)}
          height={height}
          fill="#211d17"
          opacity={0.1}
          pointerEvents="none"
        />
      )}

      {/* общий курсор: та же дата на всех графиках сезона */}
      {hoverX !== undefined && !drag && (
        <line
          x1={hoverX}
          x2={hoverX}
          y1={top}
          y2={top + height}
          stroke="#211d17"
          strokeWidth={1}
          strokeDasharray="3 3"
          opacity={0.35}
          pointerEvents="none"
        />
      )}

      <rect
        ref={rectRef}
        x={left}
        y={top}
        width={width}
        height={height}
        fill="transparent"
        style={{ cursor: drag ? "ew-resize" : "crosshair", touchAction: "none" }}
        onPointerDown={(event) => {
          if (event.button !== 0) return;
          const x = localX(event.clientX);
          if (x === null) return;
          event.currentTarget.setPointerCapture(event.pointerId);
          setDrag({ from: x, to: x });
        }}
        onPointerMove={(event) => {
          const x = localX(event.clientX);
          if (x === null) return;
          onHover?.(toTime(x));
          setDrag((current) => (current ? { ...current, to: x } : null));
        }}
        onPointerUp={(event) => {
          const current = drag;
          setDrag(null);
          if (!current) return;
          if (Math.abs(current.to - current.from) < 6) return;   // клик, а не протяжка
          const a = toTime(current.from);
          const b = toTime(current.to);
          if (a !== null && b !== null) control.select(a, b);
          event.currentTarget.releasePointerCapture(event.pointerId);
        }}
        onPointerLeave={() => onHover?.(null)}
        onDoubleClick={() => control.reset()}
      />
    </g>
  );
}
