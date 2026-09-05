/** Упрощённая физика колобка на графике. Шар радиуса RADIUS катится по ломаной линии показателя:
 *  под горку разгоняется, на подъёме теряет ход, на трамплине (гребень или обрыв) отрывается
 *  и летит по параболе, приземляется с небольшим отскоком. Всё в пикселях экрана, ось y вниз.
 *
 *  Линия графика превращается в «поверхность центра»: для каждого столбца x берётся самое
 *  высокое положение, в котором шар ещё касается ломаной (минимум по всем точкам касания).
 *  Так шар огибает гребни дугой, а не подпрыгивает на каждом узле линии. В контакте шар движется
 *  вдоль этой поверхности по длине дуги, поэтому крутую ступеньку он проходит, а не перескакивает.
 *
 *  Одно допущение сверх физики: у графика осадков бывают отвесные стены, на которые шар сам
 *  не заедет. Если он почти остановился на подъёме, включается «рука» — плавно доводит скорость
 *  до CLIMB_SPEED и не даёт оторваться от поверхности, пока склон не станет пологим.
 *  Чистые функции без DOM, состояние не мутируется — каждый шаг возвращает новый объект. */

export const RADIUS = 12;
export const GRAVITY = 720;                    // px/s², свободное падение
const ROLL_GRAVITY = GRAVITY * 5 / 7;          // сплошной шар: часть энергии уходит во вращение
const DRAG = 0.006;                            // квадратичное сопротивление: предел на ровном ≈ 90 px/s
const ROLLING_RESISTANCE = 12;                 // px/s², трение качения
const MOTOR = 60;                              // px/s², лёгкая постоянная тяга, иначе уснёт в первой ямке
const STALL_SPEED = 30;                        // px/s, медленнее на подъёме — нужна «рука»
const HOLD_DELAY = 0.1;                        // с, столько шар буксует, прежде чем его подтолкнут
const HOLD_RAMP = 0.2;                         // с, за сколько «рука» включается и отпускает
const HOLD_LOOK = 4;                           // px вперёд по дуге: «рука» смотрит на склон впереди, не под шаром
const CLIMB_SPEED = 75;                        // px/s, с такой скоростью шар доводят по склону
const SERVO = 60;                              // 1/с, жёсткость «руки»: хватает и на отвесную стену
const RESTITUTION = 0.3;                       // доля скорости, возвращаемая при ударе
const BOUNCE_MIN = 45;                         // px/s, слабее удар — просто прилипает к поверхности
const LOOKAHEAD = 1 / 60;                      // с, горизонт проверки отрыва от поверхности
const MAX_SPEED = 420;
const MAX_FRAME = 0.05;                        // с, длиннее кадр не догоняем: вкладка была неактивна
export const STEP = 1 / 120;

export interface Terrain { xs: number[]; ys: number[]; times: number[] }

/** Поверхность центра: высоты по столбцам шириной 1 px и накопленная длина дуги до каждого узла. */
export interface Surface { x0: number; samples: Float64Array; arc: Float64Array; terrain: Terrain }

export interface Ball {
  x: number; y: number;
  vx: number; vy: number;      // скорость в полёте
  s: number; v: number;        // положение по длине дуги и скорость вдоль поверхности (со знаком)
  airborne: boolean;
  angle: number; omega: number; // поворот спрайта и угловая скорость, рад
  slow: number;                // сколько секунд подряд буксует на подъёме
  hold: number;                // сила «руки» от 0 до 1
}

interface Pose { x: number; y: number; tx: number; ty: number }

const clamp = (value: number, low: number, high: number) => Math.min(high, Math.max(low, value));
const approach = (value: number, target: number, maxDelta: number) =>
  value + clamp(target - value, -maxDelta, maxDelta);

/** Индекс последнего узла монотонного ряда, не превышающего key. */
function lowerIndex(keys: ArrayLike<number>, key: number, last: number): number {
  let lo = 0, hi = last;
  while (hi - lo > 1) {
    const mid = (lo + hi) >> 1;
    if (keys[mid] <= key) lo = mid; else hi = mid;
  }
  return lo;
}

/** Линейная интерполяция по монотонному ряду ключей; вне диапазона — null. */
function interpolate(keys: number[], values: number[], key: number): number | null {
  const last = keys.length - 1;
  if (last < 0 || key < keys[0] || key > keys[last]) return null;
  const lo = lowerIndex(keys, key, last), hi = Math.min(last, lo + 1);
  const span = keys[hi] - keys[lo];
  const fraction = span > 0 ? (key - keys[lo]) / span : 0;
  return values[lo] + (values[hi] - values[lo]) * fraction;
}

export const timeAtX = (terrain: Terrain, x: number) => interpolate(terrain.xs, terrain.times, x);
export const xAtTime = (terrain: Terrain, time: number) => interpolate(terrain.times, terrain.xs, time);

/** Высота линии графика под точкой x. Слева от первой точки — ровная площадка старта,
 *  справа от последней — пусто (NaN): там колобок улетает. */
export function heightAt(terrain: Terrain, x: number): number {
  if (x <= terrain.xs[0]) return terrain.ys[0];
  return interpolate(terrain.xs, terrain.ys, x) ?? Number.NaN;
}

/** Поверхность центра шара: от первой точки линии до последней плюс радиус,
 *  чтобы за краем шар ещё огибал угол дугой, а потом срывался. */
export function buildSurface(terrain: Terrain): Surface {
  const x0 = terrain.xs[0];
  const end = terrain.xs[terrain.xs.length - 1] + RADIUS;
  const count = Math.max(2, Math.floor(end - x0) + 1);
  const samples = new Float64Array(count), arc = new Float64Array(count);
  for (let column = 0; column < count; column += 1) {
    const cx = x0 + column;
    let best = Number.POSITIVE_INFINITY;
    for (let dx = -RADIUS; dx <= RADIUS; dx += 1) {
      const ground = heightAt(terrain, cx + dx);
      if (Number.isNaN(ground)) continue;
      best = Math.min(best, ground - Math.sqrt(RADIUS * RADIUS - dx * dx));
    }
    samples[column] = best;
    if (column > 0) arc[column] = arc[column - 1] + Math.hypot(1, best - samples[column - 1]);
  }
  return { x0, samples, arc, terrain };
}

export function surfaceAt(surface: Surface, x: number): number {
  const offset = x - surface.x0;
  const last = surface.samples.length - 1;
  if (offset < 0 || offset > last) return Number.NaN;
  const index = Math.min(Math.floor(offset), last - 1);
  return surface.samples[index] + (surface.samples[index + 1] - surface.samples[index]) * (offset - index);
}

export const totalArc = (surface: Surface) => surface.arc[surface.arc.length - 1];

/** Длина дуги до точки поверхности над x. */
export function arcAtX(surface: Surface, x: number): number {
  const last = surface.samples.length - 1;
  const offset = clamp(x - surface.x0, 0, last);
  const index = Math.min(Math.floor(offset), last - 1);
  return surface.arc[index] + (surface.arc[index + 1] - surface.arc[index]) * (offset - index);
}

/** Точка и единичная касательная (вперёд по x) на расстоянии s вдоль поверхности. */
export function poseAt(surface: Surface, s: number): Pose {
  const last = surface.samples.length - 1;
  const index = Math.min(lowerIndex(surface.arc, s, last), last - 1);
  const length = surface.arc[index + 1] - surface.arc[index];
  const fraction = clamp((s - surface.arc[index]) / length, 0, 1);
  const rise = surface.samples[index + 1] - surface.samples[index];
  return {
    x: surface.x0 + index + fraction, y: surface.samples[index] + rise * fraction,
    tx: 1 / length, ty: rise / length,
  };
}

/** Шар на поверхности над точкой x, в контакте. */
export function placeBall(surface: Surface, x: number, v = 0, angle = 0): Ball {
  const s = arcAtX(surface, x), pose = poseAt(surface, s);
  return {
    x: pose.x, y: pose.y, vx: v * pose.tx, vy: v * pose.ty, s, v, airborne: false,
    angle, omega: v / RADIUS, slow: 0, hold: 0,
  };
}

/** Трамплин: если за короткий горизонт свободный полёт оказался бы выше поверхности,
 *  значит склон уходит вниз быстрее, чем тянет гравитация, и шар отрывается. */
function leavesSurface(surface: Surface, ball: Ball): boolean {
  const x = ball.x + ball.vx * LOOKAHEAD;
  const y = ball.y + ball.vy * LOOKAHEAD + GRAVITY * LOOKAHEAD * LOOKAHEAD / 2;
  const ground = surfaceAt(surface, x);
  return Number.isNaN(ground) || y < ground - 0.05;
}

/** «Рука»: включается после HOLD_DELAY буксования перед подъёмом, отпускает, когда склон впереди
 *  стал таким, что шар едет сам (тяга мотора перекрывает скат гравитации). Смотрим на склон чуть
 *  впереди: у подножия стены сам шар ещё стоит на ровном, а буксует уже об неё. */
function holdLevel(surface: Surface, ball: Ball, dt: number): { slow: number; hold: number } {
  const ahead = poseAt(surface, Math.min(ball.s + HOLD_LOOK, totalArc(surface)));
  const selfPropelled = ROLL_GRAVITY * ahead.ty + MOTOR > 0;
  const slow = !selfPropelled && ball.v < STALL_SPEED ? ball.slow + dt : 0;
  const target = selfPropelled ? 0 : slow >= HOLD_DELAY ? 1 : ball.hold;
  return { slow, hold: approach(ball.hold, target, dt / HOLD_RAMP) };
}

/** Шаг качения: гравитация вдоль склона, сопротивление, лёгкая тяга и «рука» на крутом подъёме. */
function stepContact(surface: Surface, ball: Ball, dt: number): Ball {
  const here = poseAt(surface, ball.s);
  const { slow, hold } = holdLevel(surface, ball, dt);
  const acceleration = ROLL_GRAVITY * here.ty - DRAG * ball.v * Math.abs(ball.v)
    - ROLLING_RESISTANCE * Math.tanh(ball.v / 5) + MOTOR + hold * SERVO * (CLIMB_SPEED - ball.v);
  let v = clamp(ball.v + acceleration * dt, -MAX_SPEED, MAX_SPEED);
  let s = ball.s + v * dt;
  if (s < 0) { s = 0; v = 0; }                                  // слева стенка: с площадки старта не скатываемся
  const angle = ball.angle + v * dt / RADIUS;
  if (s >= totalArc(surface)) {                                 // край линии: улетаем по касательной
    return { ...ball, x: here.x + v * here.tx * dt, y: here.y + v * here.ty * dt, vx: v * here.tx, vy: v * here.ty,
      s, v, angle, omega: v / RADIUS, airborne: true, slow: 0, hold: 0 };
  }
  // В яме или у подножия стены касательная ломается: в стену шар не пробивается, поэтому
  // остаётся только проекция скорости на новое направление (неупругий удар о склон).
  const next = poseAt(surface, s);
  v *= clamp(here.tx * next.tx + here.ty * next.ty, 0, 1);
  const moved: Ball = { ...ball, x: next.x, y: next.y, vx: v * next.tx, vy: v * next.ty,
    s, v, angle, omega: v / RADIUS, slow, hold };
  const held = hold > 0.01;
  return !held && leavesSurface(surface, moved) ? { ...moved, airborne: true, slow: 0, hold: 0 } : moved;
}

/** Приземление: скорость раскладывается на касательную и нормаль; сильный удар даёт отскок,
 *  слабый — шар прилипает и дальше катится. */
function land(surface: Surface, ball: Ball): Ball {
  const s = arcAtX(surface, ball.x), pose = poseAt(surface, s);
  const nx = pose.ty, ny = -pose.tx;                            // нормаль наружу, то есть вверх
  const vt = ball.vx * pose.tx + ball.vy * pose.ty;
  const vn = ball.vx * nx + ball.vy * ny;
  const grounded = { ...ball, x: pose.x, y: pose.y, s, v: vt, omega: vt / RADIUS };
  if (vn < -BOUNCE_MIN) {
    const bounced = -vn * RESTITUTION;
    return { ...grounded, vx: vt * pose.tx + bounced * nx, vy: vt * pose.ty + bounced * ny };
  }
  return { ...grounded, vx: vt * pose.tx, vy: vt * pose.ty, airborne: false };
}

/** Шаг полёта по параболе; вращение сохраняется по инерции. */
function stepFlight(surface: Surface, ball: Ball, dt: number): Ball {
  const vy = ball.vy + GRAVITY * dt;
  const x = ball.x + ball.vx * dt, y = ball.y + vy * dt;
  const angle = ball.angle + ball.omega * dt;
  const ground = surfaceAt(surface, x);
  if (Number.isNaN(ground) || y < ground) return { ...ball, x, y, vy, angle };
  return land(surface, { ...ball, x, y: ground, vy, angle });
}

export function step(surface: Surface, ball: Ball, dt: number): Ball {
  return ball.airborne ? stepFlight(surface, ball, dt) : stepContact(surface, ball, dt);
}

/** Продвигает шар на время кадра мелкими шагами: так движение одинаково при любой частоте кадров. */
export function advance(surface: Surface, ball: Ball, seconds: number): Ball {
  let current = ball;
  let remaining = Math.min(seconds, MAX_FRAME);
  while (remaining > 0) {
    const dt = Math.min(remaining, STEP);
    remaining -= dt;
    current = step(surface, current, dt);
  }
  return current;
}

/** Улетел ли шар за правый или нижний край области графика. */
export function isGone(ball: Ball, right: number, bottom: number): boolean {
  return ball.airborne && (ball.y - RADIUS > bottom + RADIUS * 2 || ball.x - RADIUS > right + RADIUS * 2);
}
