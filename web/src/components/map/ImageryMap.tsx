/** Карта индекса поверх той же подложки Esri, что используется на экране новой территории. */
import { useEffect, useRef, useState } from "react";
import { ImageLayer, LineLayer, RasterLayer, Scene, type ILayer } from "@antv/l7";
import { Map as L7Map } from "@antv/l7-maps";
import type { ImageManifest } from "../../api/analytics";

export default function ImageryMap({ manifest, url }: { manifest: ImageManifest; url: string }) {
  const container = useRef<HTMLDivElement>(null);
  const [scene, setScene] = useState<Scene | null>(null);
  const [loadedUrl, setLoadedUrl] = useState("");
  const [errorUrl, setErrorUrl] = useState("");
  const [attempt, setAttempt] = useState(0);
  const [[south, west], [north, east]] = manifest.bounds;
  useEffect(() => {
    if (!container.current) return;
    let disposed = false;
    const map = new Scene({ id: container.current, logoVisible: false,
      map: new L7Map({ center: [(west + east) / 2, (south + north) / 2], zoom: 14, maxZoom: 19, pitch: 0 }) });
    map.on("loaded", () => {
      if (disposed) return;
      const tiles = new RasterLayer({ zIndex: 0 }).source(
        "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        { parser: { type: "rasterTile", tileSize: 256, minZoom: 1, maxZoom: 18 } });
      map.addLayer(tiles);
      const line = new LineLayer({ zIndex: 3 }).source({ type: "FeatureCollection", features: [
        { type: "Feature", geometry: manifest.geometry, properties: {} },
      ] }).shape("line").size(2).color("#fffdf9");
      map.addLayer(line);
      map.fitBounds([[west, south], [east, north]], { padding: 28, animate: false });
      setScene(map);
    });
    return () => { disposed = true; map.destroy(); };
  }, [manifest, west, south, east, north]);

  useEffect(() => {
    if (!scene) return;
    setLoadedUrl(""); setErrorUrl("");
    let disposed = false;
    let layer: ILayer | undefined;
    const image = new Image();
    image.src = url;
    // Ошибку PNG обрабатываем до передачи в L7: загрузчик L7 иначе может ждать бесконечно.
    image.decode().then(() => {
      if (disposed) return;
      layer = new ImageLayer({ zIndex: 2 }).source(image, {
        parser: { type: "image", extent: [west, south, east, north] },
      });
      layer.on("inited", () => { if (!disposed) { scene.render(); setLoadedUrl(url); } });
      scene.addLayer(layer);
    }).catch(() => { if (!disposed) setErrorUrl(url); });
    return () => { disposed = true; if (layer) scene.removeLayer(layer); };
  }, [scene, url, west, south, east, north, attempt]);

  return <div className="field-map imagery-map" data-testid="imagery-map" data-image-url={url}
    data-state={errorUrl === url ? "error" : loadedUrl === url ? "ready" : "loading"}>
    <div ref={container} style={{ position: "absolute", inset: 0 }} />
    {loadedUrl !== url && <div className="imagery-map-status" role="status">
      {errorUrl === url ? "Снимок не загрузился." : "Загружаем снимок…"}
      {errorUrl === url && <button className="btn btn--sm btn--ghost" onClick={() => setAttempt(v => v + 1)}>Повторить загрузку карты</button>}
    </div>}
    <div className="map-switch">
      <button className="map-switch-item" aria-label="Приблизить карту поля" onClick={() => scene?.zoomIn()}>+</button>
      <button className="map-switch-item" aria-label="Отдалить карту поля" onClick={() => scene?.zoomOut()}>−</button>
      <button className="map-switch-item" onClick={() => scene?.fitBounds([[west, south], [east, north]], { padding: 28 })}>Всё поле</button>
    </div>
    <div className="map-credit">Фон © Esri, Maxar · индекс Sentinel-2</div>
  </div>;
}
