import { useEffect, useRef } from "react";
import { useDrawingArea, useXScale, useYScale } from "@mui/x-charts/hooks";
import type { WeatherMetric } from "../../api/analytics";
import { ms } from "../../lib/format";
import { finite } from "../../lib/metrics";

const RADIUS = 12;
const CRUISE_SPEED = 80;
const GRAVITY = 600;
const RESTART_DELAY = 2000;

export function RollingKolobok({ metric }: { metric: WeatherMetric }) {
  const x = useXScale("agro-x"), y = useYScale("agro-y");
  const { left, width } = useDrawingArea();
  const pathRef = useRef<SVGPathElement>(null);
  const spriteRef = useRef<SVGGElement>(null);

  // Первый видимый непрерывный участок той же линейной кривой, с учётом зума.
  const points: string[] = [];
  for (let i = 1; i < metric.date.length; i += 1) {
    const a = metric.value[i - 1], b = metric.value[i];
    if (!finite(a) || !finite(b)) {
      if (points.length) break;
      continue;
    }
    const x1 = x(ms(metric.date[i - 1])), x2 = x(ms(metric.date[i]));
    const y1 = y(a), y2 = y(b);
    if (x1 === undefined || x2 === undefined || y1 === undefined || y2 === undefined || x2 <= x1) continue;
    const start = Math.max(left, x1), end = Math.min(left + width, x2);
    if (end <= start) continue;
    const at = (px: number) => `${px},${y1 + (y2 - y1) * (px - x1) / (x2 - x1)}`;
    if (!points.length) points.push(at(start));
    points.push(at(end));
    if (end >= left + width) break;
  }
  const path = points.length > 1 ? `M${points.join(" L")}` : "";

  useEffect(() => {
    const track = pathRef.current, sprite = spriteRef.current;
    if (!track || !sprite || !path) return;
    const length = track.getTotalLength();
    if (!length) return;
    const poseAt = (distance: number) => {
      const point = track.getPointAtLength(distance);
      // Усредняем наклон на масштабе радиуса, чтобы не дёргаться на каждом узле линии.
      const before = track.getPointAtLength(Math.max(0, distance - RADIUS / 2));
      const after = track.getPointAtLength(Math.min(length, distance + RADIUS / 2));
      const span = Math.hypot(after.x - before.x, after.y - before.y) || 1;
      const tx = (after.x - before.x) / span, ty = (after.y - before.y) / span;
      // Центр отстоит от поверхности по нормали, а не просто вверх по экрану.
      return { x: point.x + ty * RADIUS, y: point.y - tx * RADIUS, tx, ty };
    };
    let distance = 0, speed = CRUISE_SPEED, rotation = 0;
    let pose = poseAt(0);
    let previous = performance.now();
    let flight: { started: number; x: number; y: number; vx: number; vy: number; spin: number; rotation: number } | null = null;
    let frame = 0;
    const animate = (now: number) => {
      // Малые шаги удерживают движение стабильным при разной частоте кадров.
      let remaining = Math.min((now - previous) / 1000, 0.05);
      previous = now;
      if (flight && now - flight.started >= RESTART_DELAY) {
        distance = 0;
        speed = CRUISE_SPEED;
        rotation = 0;
        pose = poseAt(0);
        flight = null;
        remaining = 0;
      }
      while (!flight && remaining > 0) {
        const dt = Math.min(remaining, 1 / 120);
        remaining -= dt;
        // Сила тяжести вдоль склона и инерция катящегося шара (5/7).
        // Лёгкая тяга и нижний предел скорости помогают колобку преодолеть любой подъём.
        const acceleration = GRAVITY * 5 / 7 * pose.ty + (CRUISE_SPEED - speed) * 0.8;
        speed = Math.max(28, Math.min(220, speed + acceleration * dt));
        const nextDistance = Math.min(length, distance + speed * dt);
        const nextPose = poseAt(nextDistance);
        rotation += Math.hypot(nextPose.x - pose.x, nextPose.y - pose.y) / RADIUS;
        distance = nextDistance;
        pose = nextPose;
        if (distance >= length) {
          flight = { started: now, x: pose.x, y: pose.y,
            vx: pose.tx * speed, vy: pose.ty * speed, spin: speed / RADIUS, rotation };
        }
      }
      let px = pose.x, py = pose.y, angle = rotation;
      if (flight) {
        const seconds = (now - flight.started) / 1000;
        // С края вылетаем по касательной: инерция и вращение сохраняются в воздухе.
        px = flight.x + flight.vx * seconds;
        py = flight.y + flight.vy * seconds + GRAVITY * seconds * seconds / 2;
        angle = flight.rotation + flight.spin * seconds;
      }
      sprite.setAttribute("transform",
        `translate(${px},${py}) rotate(${angle * 180 / Math.PI})`);
      sprite.setAttribute("visibility", "visible");
      frame = requestAnimationFrame(animate);
    };
    frame = requestAnimationFrame(animate);
    return () => cancelAnimationFrame(frame);
  }, [path]);

  if (!path) return null;
  return <g aria-hidden="true" pointerEvents="none">
    <path ref={pathRef} d={path} fill="none" stroke="none" />
    <g ref={spriteRef} visibility="hidden">
      <image href={`${import.meta.env.BASE_URL}kolobok.png`} x={-RADIUS} y={-RADIUS}
        width={RADIUS * 2} height={RADIUS * 2} />
    </g>
  </g>;
}
