/** Крупная иллюстрация первого экрана: сезонная кривая с нормой, пролёты спутников,
 *  пропуск в наблюдениях и выделенный период угнетения. Рисуется одним SVG, без картинок и шрифтовых иконок. */

const CURVE = "M14 152 C 60 150, 92 78, 132 62 C 166 49, 196 52, 214 66 C 236 84, 244 140, 268 156 C 292 172, 330 168, 356 164";
const NORM = "M14 146 C 62 142, 96 92, 134 78 C 170 66, 200 70, 220 86 C 242 104, 250 128, 274 136 C 298 144, 332 142, 356 140";
const BAND =
  "M14 132 C 62 128, 96 78, 134 64 C 170 52, 200 56, 220 72 C 242 90, 250 114, 274 122 C 298 130, 332 128, 356 126 " +
  "L356 154 C 332 156, 298 158, 274 150 C 250 142, 242 118, 220 100 C 200 84, 170 80, 134 92 C 96 106, 62 156, 14 160 Z";

const SENSOR_DOTS: [number, number, string][] = [
  [30, 150, "#3a6ea5"],
  [58, 136, "#6b5b95"],
  [86, 108, "#3a6ea5"],
  [112, 78, "#3a6ea5"],
  [138, 62, "#c2703a"],
  [166, 55, "#3a6ea5"],
  [196, 57, "#6b5b95"],
  [284, 160, "#3a6ea5"],
  [312, 166, "#6b5b95"],
  [340, 165, "#3a6ea5"],
];

export function HeroArt({ className }: { className?: string }) {
  return (
    <svg
      viewBox="0 0 380 210"
      className={className}
      width="100%"
      style={{ maxWidth: 460, display: "block" }}
      role="img"
      aria-label="Схема: сезонная кривая NDVI, норма, пропуск в наблюдениях и период угнетения"
    >
      {/* период угнетения */}
      <rect x="214" y="18" width="76" height="150" fill="#c8423f" opacity="0.1" rx="3" />
      {/* норма ±1σ */}
      <path d={BAND} fill="#9aa091" opacity="0.16" />
      <path d={NORM} fill="none" stroke="#9aa091" strokeWidth="1.6" strokeDasharray="5 4" />
      {/* восстановленная кривая */}
      <path d={CURVE} fill="none" stroke="#1a1d16" strokeWidth="2.4" strokeLinecap="round" />
      {/* пропуск: участок кривой, восстановленный моделью */}
      <path
        d="M214 66 C 236 84, 244 140, 268 156"
        fill="none"
        stroke="#b03a76"
        strokeWidth="2.6"
        strokeLinecap="round"
        strokeDasharray="1 7"
      />
      {SENSOR_DOTS.map(([x, y, color]) => (
        <circle key={`${x}-${y}`} cx={x} cy={y} r="3.6" fill={color} stroke="#fff" strokeWidth="1.2" />
      ))}
      {[236, 252].map((x, index) => (
        <g key={x} transform={`translate(${x} ${index === 0 ? 104 : 130}) rotate(45)`}>
          <rect x="-4" y="-4" width="8" height="8" fill="#b03a76" stroke="#fff" strokeWidth="1.2" />
        </g>
      ))}
      {/* оси */}
      <path d="M14 12 V172 H366" fill="none" stroke="#d3cfc2" strokeWidth="1.2" />
      {/* спутник и трасса съёмки */}
      <path d="M40 24 C 120 6, 250 6, 350 26" fill="none" stroke="#d3cfc2" strokeWidth="1.2" strokeDasharray="3 5" />
      <g transform="translate(300 14) rotate(-12)" stroke="#1a1d16" strokeWidth="1.5" fill="none" strokeLinejoin="round">
        <path d="M-8 0 L0 -8 L8 0 L0 8 Z" fill="#f7f6f3" />
        <path d="M-18 -4 L-10 -12 L-4 -6 L-12 2 Z" />
        <path d="M4 -6 L12 -14 L18 -8 L10 0 Z" />
      </g>
      {/* подписи */}
      <text x="14" y="196" fontSize="9" fill="#7a7c71" fontFamily="ui-monospace, monospace">
        апрель
      </text>
      <text x="188" y="196" fontSize="9" fill="#7a7c71" fontFamily="ui-monospace, monospace">
        июль
      </text>
      <text x="326" y="196" fontSize="9" fill="#7a7c71" fontFamily="ui-monospace, monospace">
        октябрь
      </text>
      <text x="222" y="14" fontSize="8.5" fill="#9f2f2d" fontFamily="ui-monospace, monospace">
        угнетение
      </text>
    </svg>
  );
}
