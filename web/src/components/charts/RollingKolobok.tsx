/** Колобок катится по линии осадков. Физика — в kolobokPhysics.ts, здесь рельеф из графика,
 *  цикл кадров и отрисовка: спрайт с поворотом и тень на линии, которая отстаёт и бледнеет,
 *  когда колобок в воздухе. Докатившись до края, улетает, гаснет и начинает заново.
 *  После зума и ресайза продолжает с той же даты, если она осталась на экране. */

import { useEffect, useRef } from "react";
import { useDrawingArea, useXScale, useYScale } from "@mui/x-charts/hooks";
import type { WeatherMetric } from "../../api/analytics";
import { ms } from "../../lib/format";
import { finite } from "../../lib/metrics";
import {
  RADIUS, advance, buildSurface, heightAt, isGone, placeBall, timeAtX, xAtTime,
  type Ball, type Terrain,
} from "./kolobokPhysics";

const FADE_S = 0.35;          // появление и угасание
const REST_S = 0.9;           // пауза перед новым заходом
const SHADOW_REACH = 70;      // px высоты, на которой тень почти исчезает
const SHADOW_OPACITY = 0.22;

type Phase = "appear" | "roll" | "fade" | "rest";
type Scale = (value: number) => number | undefined;
interface Memory { time: number; v: number; angle: number }

/** Первый видимый непрерывный участок линии в пикселях, с датами для продолжения после зума. */
function visibleTerrain(metric: WeatherMetric, x: Scale, y: Scale, left: number, right: number): Terrain | null {
  const xs: number[] = [], ys: number[] = [], times: number[] = [];
  const push = (px: number, py: number, time: number) => { xs.push(px); ys.push(py); times.push(time); };
  for (let i = 1; i < metric.date.length; i += 1) {
    const a = metric.value[i - 1], b = metric.value[i];
    if (!finite(a) || !finite(b)) {
      if (xs.length) break;
      continue;
    }
    const t1 = ms(metric.date[i - 1]), t2 = ms(metric.date[i]);
    const x1 = x(t1), x2 = x(t2), y1 = y(a), y2 = y(b);
    if (x1 === undefined || x2 === undefined || y1 === undefined || y2 === undefined || x2 <= x1) continue;
    const start = Math.max(left, x1), end = Math.min(right, x2);
    if (end <= start) continue;
    const fraction = (px: number) => (px - x1) / (x2 - x1);
    if (!xs.length) push(start, y1 + (y2 - y1) * fraction(start), t1 + (t2 - t1) * fraction(start));
    push(end, y1 + (y2 - y1) * fraction(end), t1 + (t2 - t1) * fraction(end));
    if (end >= right) break;
  }
  return xs.length > 1 ? { xs, ys, times } : null;
}

function nextPhase(phase: Phase, elapsed: number, gone: boolean): Phase {
  if (gone && (phase === "roll" || phase === "appear")) return "fade";
  if (phase === "appear" && elapsed >= FADE_S) return "roll";
  if (phase === "fade" && elapsed >= FADE_S) return "rest";
  if (phase === "rest" && elapsed >= REST_S) return "appear";
  return phase;
}

function opacityFor(phase: Phase, elapsed: number): number {
  if (phase === "appear") return Math.min(1, elapsed / FADE_S);
  if (phase === "fade") return Math.max(0, 1 - elapsed / FADE_S);
  return phase === "rest" ? 0 : 1;
}

/** Спрайт и тень: тень лежит на линии графика под центром шара и бледнеет с высотой полёта. */
function draw(sprite: SVGGElement, shadow: SVGEllipseElement, terrain: Terrain, ball: Ball, opacity: number) {
  const degrees = ball.angle * 180 / Math.PI;
  sprite.setAttribute("transform", `translate(${ball.x.toFixed(2)},${ball.y.toFixed(2)}) rotate(${degrees.toFixed(1)})`);
  sprite.setAttribute("opacity", opacity.toFixed(3));
  const ground = heightAt(terrain, ball.x);
  if (Number.isNaN(ground)) {
    shadow.setAttribute("opacity", "0");
    return;
  }
  const lift = Math.max(0, ground - ball.y - RADIUS);
  const height = Math.min(1, lift / SHADOW_REACH);
  shadow.setAttribute("transform", `translate(${ball.x.toFixed(2)},${ground.toFixed(2)}) scale(${(1 + height * 0.6).toFixed(3)},1)`);
  shadow.setAttribute("opacity", (SHADOW_OPACITY * (1 - height) * opacity).toFixed(3));
}

export function RollingKolobok({ metric }: { metric: WeatherMetric }) {
  const x = useXScale("agro-x"), y = useYScale("agro-y");
  const { left, top, width, height } = useDrawingArea();
  const spriteRef = useRef<SVGGElement>(null);
  const shadowRef = useRef<SVGEllipseElement>(null);
  const memoryRef = useRef<Memory | null>(null);

  const terrain = visibleTerrain(metric, x as Scale, y as Scale, left, left + width);
  const key = terrain ? `${terrain.xs.join()}|${terrain.ys.join()}` : "";

  useEffect(() => {
    const sprite = spriteRef.current, shadow = shadowRef.current;
    if (!terrain || !sprite || !shadow) return;
    const surface = buildSurface(terrain);
    const right = left + width, bottom = top + height;
    const startX = terrain.xs[0];

    const saved = memoryRef.current;
    const restoredX = saved ? xAtTime(terrain, saved.time) : null;
    let ball = restoredX !== null && saved
      ? placeBall(surface, restoredX, saved.v, saved.angle)
      : placeBall(surface, startX);
    let phase: Phase = restoredX !== null ? "roll" : "appear";
    let phaseStart = performance.now(), previous = phaseStart, frame = 0;

    const tick = (now: number) => {
      const seconds = (now - previous) / 1000;
      previous = now;
      const moving = phase === "roll" || phase === "appear";
      if (moving) ball = advance(surface, ball, seconds);
      const phaseBefore = phase;
      phase = nextPhase(phase, (now - phaseStart) / 1000, moving && isGone(ball, right, bottom));
      if (phase !== phaseBefore) {
        phaseStart = now;
        if (phase === "appear") ball = placeBall(surface, startX);
      }
      memoryRef.current = ball.airborne || !moving ? null
        : { time: timeAtX(terrain, ball.x) ?? Number.NaN, v: ball.v, angle: ball.angle };
      draw(sprite, shadow, terrain, ball, opacityFor(phase, (now - phaseStart) / 1000));
      frame = requestAnimationFrame(tick);
    };
    frame = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(frame);
    // Рельеф пересобирается на каждом рендере; перезапуск нужен только когда он реально изменился.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [key]);

  if (!terrain) return null;
  return <g aria-hidden="true" pointerEvents="none">
    <ellipse ref={shadowRef} rx={RADIUS * 0.8} ry={2.6} fill="#211d17" opacity={0} />
    <g ref={spriteRef} opacity={0}>
      <image href={`${import.meta.env.BASE_URL}kolobok.png`} x={-RADIUS} y={-RADIUS}
        width={RADIUS * 2} height={RADIUS * 2} />
    </g>
  </g>;
}
