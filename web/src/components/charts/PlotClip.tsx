/** Обрезает линии и собственные SVG-слои по области данных при общем приближении. */
import { useId, type ReactNode } from "react";
import { useDrawingArea } from "@mui/x-charts/hooks";

export function PlotClip({ children }: { children: ReactNode }) {
  const id = useId(), { left, top, width, height } = useDrawingArea();
  return <g><defs><clipPath id={id}><rect x={left} y={top} width={width} height={height} /></clipPath></defs>
    <g clipPath={`url(#${id})`}>{children}</g></g>;
}
