/** Собственные SVG-иллюстрации вместо иконочного шрифта: непрерывная линия чернилами
 *  плюс одна смещённая геометрическая фигура пастельного цвета. Каждая рисуется inline,
 *  без сетевых запросов, наследует currentColor и масштабируется размером size. */

import type { CSSProperties } from "react";

interface ArtProps {
  size?: number;
  tone?: string;
  className?: string;
  style?: CSSProperties;
  title?: string;
}

const INK = "currentColor";

/** Смещённая фигура за линией. Без явного tone она берёт цвет текста с малой непрозрачностью,
 *  поэтому на любом фоне читается как лёгкая тень, а не как белое пятно. */
function toneFill(tone?: string) {
  return tone ? { fill: tone } : { fill: INK, opacity: 0.13 };
}

function frame(size: number, title?: string) {
  return {
    width: size,
    height: size,
    viewBox: "0 0 48 48",
    fill: "none",
    xmlns: "http://www.w3.org/2000/svg",
    role: title ? ("img" as const) : ("presentation" as const),
    "aria-hidden": title ? undefined : true,
  };
}

const stroke = {
  stroke: INK,
  strokeWidth: 1.6,
  strokeLinecap: "round" as const,
  strokeLinejoin: "round" as const,
};

/** Спутник над полем — сбор данных. */
export function SatelliteArt({ size = 48, tone, className, style, title }: ArtProps) {
  return (
    <svg {...frame(size, title)} className={className} style={style}>
      {title ? <title>{title}</title> : null}
      <circle cx="33" cy="14" r="9" {...toneFill(tone)} />
      <path {...stroke} d="M13 21.5 21.5 13l5 5-8.5 8.5z" />
      <path {...stroke} d="m6 28.5 5-5 5 5-5 5zM24 10.5l5-5 5 5-5 5z" />
      <path {...stroke} d="m17 24.5 5.5 5.5" />
      <path {...stroke} d="M27.5 30.5a10 10 0 0 0 5-8" opacity="0.65" />
      <path {...stroke} d="M31.5 35a15 15 0 0 0 7.5-12" opacity="0.4" />
      <path {...stroke} d="M4 43h40" />
    </svg>
  );
}

/** Ряды поля в перспективе — полигон, участок. */
export function FieldArt({ size = 48, tone, className, style, title }: ArtProps) {
  return (
    <svg {...frame(size, title)} className={className} style={style}>
      {title ? <title>{title}</title> : null}
      <path d="M30 8h12v13H30z" {...toneFill(tone)} />
      <path {...stroke} d="M4 38 17 12h14l13 26z" />
      <path {...stroke} d="M13 38 21 12M22 38l2-26M31 38l-2-26M40 38 35 20" opacity="0.55" />
      <path {...stroke} d="M8 31h32" opacity="0.35" />
    </svg>
  );
}

/** Кривая сезона с провалом — эпизод угнетения. */
export function CurveArt({ size = 48, tone, className, style, title }: ArtProps) {
  return (
    <svg {...frame(size, title)} className={className} style={style}>
      {title ? <title>{title}</title> : null}
      <rect x="26" y="14" width="12" height="22" rx="2" {...toneFill(tone)} />
      <path {...stroke} d="M5 40V7M5 40h38" />
      <path {...stroke} d="M8 34c5-1 8-16 13-16s6 8 9 12 8 3 12-14" />
      <circle cx="26" cy="30" r="2.4" fill={INK} />
    </svg>
  );
}

/** Солнце и трещины — засуха, погодный стресс. */
export function DroughtArt({ size = 48, tone, className, style, title }: ArtProps) {
  return (
    <svg {...frame(size, title)} className={className} style={style}>
      {title ? <title>{title}</title> : null}
      <circle cx="16" cy="15" r="9" {...toneFill(tone)} />
      <circle {...stroke} cx="20" cy="15" r="7" />
      <path {...stroke} d="M20 3v3M20 24v3M32 15h3M5 15h3M28.5 6.5l2 2M9.5 21.5l2 2M28.5 23.5l2-2M9.5 8.5l2-2" opacity="0.7" />
      <path {...stroke} d="M4 34h40" />
      <path {...stroke} d="M12 34v7M12 37h-4M24 34v9M24 39h5M36 34v6M36 37h4" opacity="0.6" />
    </svg>
  );
}

/** Росток — норма, здоровый сезон. */
export function SproutArt({ size = 48, tone, className, style, title }: ArtProps) {
  return (
    <svg {...frame(size, title)} className={className} style={style}>
      {title ? <title>{title}</title> : null}
      <circle cx="31" cy="17" r="9" {...toneFill(tone)} />
      <path {...stroke} d="M24 42V20" />
      <path {...stroke} d="M24 26c-8 0-12-4-12-11 7 0 12 4 12 11z" />
      <path {...stroke} d="M24 22c7-1 11-5 11-12-7 1-11 5-11 12z" />
      <path {...stroke} d="M13 42h22" />
    </svg>
  );
}

/** Облако и пропуск в ряде точек — пропущенные наблюдения. */
export function GapArt({ size = 48, tone, className, style, title }: ArtProps) {
  return (
    <svg {...frame(size, title)} className={className} style={style}>
      {title ? <title>{title}</title> : null}
      <circle cx="34" cy="30" r="8" {...toneFill(tone)} />
      <path
        {...stroke}
        d="M14 22a6 6 0 0 1 11.4-2.6A5 5 0 0 1 34 22a5 5 0 0 1-1 9.9H15A5 5 0 0 1 14 22z"
      />
      <circle cx="8" cy="40" r="2.2" fill={INK} />
      <circle cx="18" cy="40" r="2.2" fill={INK} />
      <path {...stroke} d="M26 40h6" strokeDasharray="1 4" />
      <circle cx="40" cy="40" r="2.2" fill={INK} />
    </svg>
  );
}

/** Слои данных — несколько источников. */
export function LayersArt({ size = 48, tone, className, style, title }: ArtProps) {
  return (
    <svg {...frame(size, title)} className={className} style={style}>
      {title ? <title>{title}</title> : null}
      <path d="M24 4 44 14 24 24 4 14z" {...toneFill(tone)} />
      <path {...stroke} d="M24 5 43 14.5 24 24 5 14.5z" />
      <path {...stroke} d="m8 21 16 8 16-8" opacity="0.7" />
      <path {...stroke} d="m8 29 16 8 16-8" opacity="0.45" />
    </svg>
  );
}

/** Лупа над сеткой — поиск контуров. */
export function SearchFieldsArt({ size = 48, tone, className, style, title }: ArtProps) {
  return (
    <svg {...frame(size, title)} className={className} style={style}>
      {title ? <title>{title}</title> : null}
      <circle cx="20" cy="20" r="10" {...toneFill(tone)} />
      <path {...stroke} d="M6 8h30v30H6z" opacity="0.5" />
      <path {...stroke} d="M16 8v30M26 8v30M6 18h30M6 28h30" opacity="0.3" />
      <circle {...stroke} cx="24" cy="24" r="11" />
      <path {...stroke} d="m32 32 10 10" />
    </svg>
  );
}
