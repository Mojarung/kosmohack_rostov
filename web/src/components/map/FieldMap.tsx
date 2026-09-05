/** Карта на AntV L7 (WebGL): две подложки на выбор, контуры полей пользователя с окраской по
 *  состоянию последнего сезона, найденные контуры OSM и рисование произвольного полигона.
 *
 *  Подложки: схема OpenStreetMap (по умолчанию — на ней читаются дороги, границы и названия)
 *  и спутниковый снимок Esri World Imagery с прозрачным слоем названий от CARTO.
 *
 *  Компонент тяжёлый (WebGL + тайлы), поэтому подключается через React.lazy и монтируется
 *  только на экранах, где карта действительно нужна. */

import { useEffect, useRef, useState } from "react";
import { LineLayer, PointLayer, PolygonLayer, RasterLayer, Scene } from "@antv/l7";
import { Map as L7Map } from "@antv/l7-maps";
import { DrawEvent, DrawPolygon } from "@antv/l7-draw";

import type { OsmField, PlaceResult, UserPolygon } from "../../api/types";
import { PlaceSearch } from "./PlaceSearch";

type Basemap = "satellite" | "scheme";

const BASEMAPS: Record<Basemap, { label: string; url: string; labels?: string; credit: string; maxZoom: number }> = {
  scheme: {
    label: "Схема",
    url: "https://tile.openstreetmap.org/{z}/{x}/{y}.png",
    credit: "© OpenStreetMap contributors",
    maxZoom: 19,
  },
  satellite: {
    label: "Спутник",
    url: "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
    // подписи берём у CARTO: они по данным OSM, то есть на русском, у Esri — латиницей
    labels: "https://basemaps.cartocdn.com/rastertiles/light_only_labels/{z}/{x}/{y}.png",
    credit: "© Esri, Maxar · подписи © OpenStreetMap, CARTO",
    maxZoom: 18,
  },
};

/** Подсказки рисования на русском: библиотека по умолчанию показывает их по-китайски. */
const DRAW_HELPER = {
  draw: "кликните, чтобы поставить первую точку",
  drawContinue: "кликните, чтобы продолжить",
  drawFinish: "кликните, чтобы продолжить; двойной клик — завершить",
  pointHover: "точку можно перетащить",
  pointDrag: null,
  lineHover: "сторону можно перетащить",
  lineDrag: null,
  midPointHover: "кликните, чтобы добавить точку",
  polygonHover: "контур можно перетащить",
  polygonDrag: null,
};

// Слои рисовалки создаются раньше подложки, а L7 при равном zIndex рисует более поздние слои сверху.
// Без явного zIndex тайлы карты перекрывали точки и линии контура, и казалось, что рисование не работает.
const DRAW_LAYER_OPTIONS = { options: { zIndex: 10 } };
const DRAW_STYLE = {
  point: DRAW_LAYER_OPTIONS,
  line: DRAW_LAYER_OPTIONS,
  polygon: DRAW_LAYER_OPTIONS,
  midPoint: DRAW_LAYER_OPTIONS,
  dashLine: DRAW_LAYER_OPTIONS,
  text: DRAW_LAYER_OPTIONS,
};

const STATUS_COLOR: Record<string, string> = {
  критическая: "#c8423f",
  умеренная: "#e8a33d",
  норма: "#3f7d45",
};

export interface FieldMapProps {
  saved: UserPolygon[];
  osmFields: OsmField[];
  /** Полигон, выбранный сейчас (нарисованный или из OSM). */
  selection: GeoJSON.Polygon | null;
  drawing: boolean;
  onDrawn: (geometry: GeoJSON.Polygon) => void;
  onPickOsm: (field: OsmField) => void;
  onPickSaved: (uid: string) => void;
  onViewportChange?: (bbox: [number, number, number, number]) => void;
  center?: [number, number];
  height?: number | string;
}

interface Layers {
  savedFill?: PolygonLayer;
  savedLine?: LineLayer;
  osmFill?: PolygonLayer;
  osmLine?: LineLayer;
  selectionFill?: PolygonLayer;
  selectionLine?: LineLayer;
}

function featureCollection<T extends { geometry: GeoJSON.Polygon }>(items: T[], props: (item: T) => object) {
  return {
    type: "FeatureCollection" as const,
    features: items.map((item) => ({
      type: "Feature" as const,
      geometry: item.geometry,
      properties: props(item),
    })),
  };
}

export default function FieldMap({
  saved,
  osmFields,
  selection,
  drawing,
  onDrawn,
  onPickOsm,
  onPickSaved,
  onViewportChange,
  center = [39.72, 47.24],
  height = 460,
}: FieldMapProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const sceneRef = useRef<Scene | null>(null);
  const drawRef = useRef<DrawPolygon | null>(null);
  const layersRef = useRef<Layers>({});
  const callbacks = useRef({ onDrawn, onPickOsm, onPickSaved, onViewportChange });
  callbacks.current = { onDrawn, onPickOsm, onPickSaved, onViewportChange };
  // список контуров OSM нужен внутри обработчика клика, созданного один раз
  const osmFieldsRef = useRef(osmFields);
  osmFieldsRef.current = osmFields;
  const [ready, setReady] = useState(false);
  const [basemap, setBasemap] = useState<Basemap>("scheme");
  const baseRef = useRef<{ tiles?: RasterLayer; labels?: RasterLayer }>({});
  const [place, setPlace] = useState<PlaceResult | null>(null);
  const [viewport, setViewport] = useState<number[]>([]);

  // --- инициализация сцены (один раз) ---
  useEffect(() => {
    if (!containerRef.current) return;
    const scene = new Scene({
      id: containerRef.current,
      logoVisible: false,
      map: new L7Map({ center, zoom: 9, minZoom: 2, maxZoom: 17, pitch: 0 }),
    });
    sceneRef.current = scene;

    scene.on("loaded", () => {
      const draw = new DrawPolygon(scene, {
        liveUpdate: true,
        editable: true,
        multiple: false,
        helper: DRAW_HELPER,
        style: DRAW_STYLE,
      });
      draw.disable();
      draw.on(DrawEvent.Add, () => {
        const [feature] = draw.getData();
        if (feature?.geometry?.type === "Polygon") {
          callbacks.current.onDrawn(feature.geometry as GeoJSON.Polygon);
        }
      });
      drawRef.current = draw;

      const emitViewport = () => {
        const bounds = scene.getBounds() as number[][];
        if (!bounds) return;
        const [[west, south], [east, north]] = bounds;
        setViewport([west, south, east, north]);
        callbacks.current.onViewportChange?.([south, west, north, east]);
      };
      scene.on("mapmove", emitViewport);
      scene.on("zoomend", emitViewport);
      emitViewport();
      setReady(true);
    });

    return () => {
      drawRef.current?.destroy();
      drawRef.current = null;
      scene.destroy();
      sceneRef.current = null;
      layersRef.current = {};
    };
    // центр задаётся при монтировании: дальнейшее перемещение — это уже состояние карты
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // Поиск меняет камеру и ставит метку; контур поля пользователь выбирает отдельно.
  useEffect(() => {
    const scene = sceneRef.current;
    if (!scene || !ready || !place) return;
    const marker = new PointLayer({ zIndex: 8 });
    marker.source([{ lng: place.center[0], lat: place.center[1] }], {
      parser: { type: "json", x: "lng", y: "lat" },
    }).shape("circle").size(9).color("#b03a76").style({ stroke: "#fff", strokeWidth: 2 });
    scene.addLayer(marker);
    if (place.bbox && place.bbox[0] < place.bbox[2] && place.bbox[1] < place.bbox[3]) {
      const [west, south, east, north] = place.bbox;
      scene.fitBounds([[west, south], [east, north]], { padding: 60, maxZoom: 15, duration: 0 });
    } else {
      scene.setZoomAndCenter(15, place.center);
    }
    return () => { scene.removeLayer(marker); };
  }, [place, ready]);

  // --- подложка: пересобирается при переключении «Спутник» / «Схема» ---
  useEffect(() => {
    const scene = sceneRef.current;
    if (!scene || !ready) return;
    const config = BASEMAPS[basemap];
    const previous = baseRef.current;
    if (previous.tiles) scene.removeLayer(previous.tiles);
    if (previous.labels) scene.removeLayer(previous.labels);

    const tiles = new RasterLayer({ zIndex: 0 });
    tiles.source(config.url, {
      parser: { type: "rasterTile", tileSize: 256, minZoom: 1, maxZoom: config.maxZoom },
    });
    scene.addLayer(tiles);

    let labels: RasterLayer | undefined;
    if (config.labels) {
      labels = new RasterLayer({ zIndex: 0 });
      labels.source(config.labels, {
        parser: { type: "rasterTile", tileSize: 256, minZoom: 1, maxZoom: config.maxZoom },
      });
      scene.addLayer(labels);
    }
    baseRef.current = { tiles, labels };
  }, [basemap, ready]);

  // --- сохранённые поля пользователя ---
  useEffect(() => {
    const scene = sceneRef.current;
    if (!scene || !ready) return;
    const data = featureCollection(
      saved.filter((item): item is UserPolygon & { geometry: GeoJSON.Polygon } => Boolean(item.geometry)),
      (item) => ({ uid: item.uid, name: item.name, status: item.last_year_status, year: item.last_year }),
    );
    const { savedFill, savedLine } = layersRef.current;
    if (savedFill && savedLine) {
      savedFill.setData(data);
      savedLine.setData(data);
      scene.render();
      return;
    }
    const fill = new PolygonLayer({ zIndex: 2, name: "saved-fill" });
    fill
      .source(data)
      .shape("fill")
      .color("status", (status: string) => STATUS_COLOR[status] ?? "#3f7d45")
      .style({ opacity: 0.34 });
    const line = new LineLayer({ zIndex: 3, name: "saved-line" });
    line
      .source(data)
      .shape("line")
      .size(2)
      .color("status", (status: string) => STATUS_COLOR[status] ?? "#3f7d45")
      .style({ opacity: 0.95 });
    fill.on("click", (event: { feature?: { properties?: { uid?: string } } }) => {
      const uid = event.feature?.properties?.uid;
      if (uid) callbacks.current.onPickSaved(uid);
    });
    scene.addLayer(fill);
    scene.addLayer(line);
    layersRef.current.savedFill = fill;
    layersRef.current.savedLine = line;
  }, [saved, ready]);

  // --- контуры OSM ---
  useEffect(() => {
    const scene = sceneRef.current;
    if (!scene || !ready) return;
    const data = featureCollection(osmFields, (field) => ({ id: field.id, name: field.name }));
    const { osmFill, osmLine } = layersRef.current;
    if (osmFill && osmLine) {
      osmFill.setData(data);
      osmLine.setData(data);
      scene.render();
      return;
    }
    const fill = new PolygonLayer({ zIndex: 1, name: "osm-fill" });
    fill.source(data).shape("fill").color("#4c6b3c").style({ opacity: 0.14 });
    const line = new LineLayer({ zIndex: 1, name: "osm-line" });
    line.source(data).shape("line").size(1.1).color("#4c6b3c").style({ opacity: 0.7 });
    fill.on("click", (event: { feature?: { properties?: { id?: string } } }) => {
      const id = event.feature?.properties?.id;
      const picked = osmFieldsRef.current.find((f) => f.id === id);
      if (picked) callbacks.current.onPickOsm(picked);
    });
    scene.addLayer(fill);
    scene.addLayer(line);
    layersRef.current.osmFill = fill;
    layersRef.current.osmLine = line;
  }, [osmFields, ready]);

  // --- подсветка выбранного контура ---
  useEffect(() => {
    const scene = sceneRef.current;
    if (!scene || !ready) return;
    const data = {
      type: "FeatureCollection" as const,
      features: selection ? [{ type: "Feature" as const, geometry: selection, properties: {} }] : [],
    };
    const { selectionFill, selectionLine } = layersRef.current;
    if (selectionFill && selectionLine) {
      selectionFill.setData(data);
      selectionLine.setData(data);
      scene.render();
      return;
    }
    const fill = new PolygonLayer({ zIndex: 4, name: "sel-fill" });
    fill.source(data).shape("fill").color("#b03a76").style({ opacity: 0.22 });
    const line = new LineLayer({ zIndex: 5, name: "sel-line" });
    line.source(data).shape("line").size(2.6).color("#b03a76");
    scene.addLayer(fill);
    scene.addLayer(line);
    layersRef.current.selectionFill = fill;
    layersRef.current.selectionLine = line;
  }, [selection, ready]);

  // --- режим рисования ---
  useEffect(() => {
    const draw = drawRef.current;
    if (!draw) return;
    if (drawing) {
      draw.clear();
      draw.enable();
    } else {
      draw.disable();
    }
  }, [drawing, ready]);

  return (
    <div className="field-map" style={{ height }} data-testid="field-map"
      data-state={ready ? "ready" : "loading"} data-bounds={viewport.join(",")}>
      <div ref={containerRef} style={{ position: "absolute", inset: 0 }} />
      <PlaceSearch onSelect={setPlace} />

      <div className="map-switch" role="group" aria-label="Подложка карты">
        {(Object.keys(BASEMAPS) as Basemap[]).map((key) => (
          <button
            key={key}
            type="button"
            className={"map-switch-item" + (basemap === key ? " is-active" : "")}
            onClick={() => setBasemap(key)}
          >
            {BASEMAPS[key].label}
          </button>
        ))}
      </div>

      <div className="map-credit">{BASEMAPS[basemap].credit}</div>
    </div>
  );
}
