/** Контур на подложке можно показать сразу, без ожидания пиксельных индексов. */
import { useMemo } from "react";
import type { ImageManifest } from "../../api/analytics";
import ImageryMap from "./ImageryMap";

export default function FieldOutlinePreview({ geometry }: { geometry: GeoJSON.Polygon }) {
  const area = useMemo(() => {
    const points = geometry.coordinates.flat();
    let west = Infinity, east = -Infinity, south = Infinity, north = -Infinity;
    for (const [x, y] of points) {
      west = Math.min(west, x); east = Math.max(east, x);
      south = Math.min(south, y); north = Math.max(north, y);
    }
    return { geometry, bounds: [[south, west], [north, east]] as ImageManifest["bounds"] };
  }, [geometry]);
  return <div className="imagery-outline-preview">
    <ImageryMap manifest={area} />
    <p className="meta">Контур выбранного поля на подложке карты. Снимки выбранного сезона ещё загружаются.</p>
  </div>;
}
