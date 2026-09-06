/** Чип культуры с фотографией при наведении: карточка плавно «вырастает» из чипа.
 *  Фото — Wikimedia Commons, файлы лежат в public/crops (без внешних запросов),
 *  в подписи — автор и лицензия, как того требует CC BY-SA. Культура без фото — обычный чип. */

import type { CSSProperties } from "react";

interface CropPhoto {
  src: string;
  caption: string;
}

// Порядок важен: «пастбища/зерновые» должны попасть в пастбище раньше, чем в зерновые.
const PHOTOS: Array<{ test: RegExp; photo: CropPhoto }> = [
  { test: /пшениц/i, photo: { src: "/crops/wheat.webp", caption: "озимая пшеница · фото Michael Trolove, CC BY-SA 2.0" } },
  { test: /подсолн/i, photo: { src: "/crops/sunflower.webp", caption: "подсолнечник · фото Summer Stock, CC0" } },
  { test: /пастбищ/i, photo: { src: "/crops/pasture.webp", caption: "пастбище · фото Russel Wills, CC BY-SA 2.0" } },
  { test: /зернов/i, photo: { src: "/crops/cereals.webp", caption: "зерновые · фото Hugh Venables, CC BY-SA 2.0" } },
];

export function cropPhoto(crop?: string | null): CropPhoto | null {
  if (!crop) return null;
  return PHOTOS.find((item) => item.test.test(crop))?.photo ?? null;
}

export function CropChip({ crop, className = "tag", style }: { crop?: string | null; className?: string; style?: CSSProperties }) {
  const photo = cropPhoto(crop);
  if (!photo) return <span className={className} style={style}>{crop}</span>;
  return (
    <span className={`${className} crop-chip`} style={style}>
      {crop}
      <span className="crop-pop" aria-hidden>
        <img src={photo.src} alt="" loading="lazy" decoding="async" />
        <span className="crop-pop-caption">{photo.caption}</span>
      </span>
    </span>
  );
}
