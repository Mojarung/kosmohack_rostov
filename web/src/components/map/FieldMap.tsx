/** Карта на AntV L7 (WebGL): подложка OpenStreetMap, контуры полей пользователя с окраской по
 *  состоянию последнего сезона, найденные контуры OSM и рисование произвольного полигона.
 *
 *  Компонент тяжёлый (WebGL + тайлы), поэтому подключается через React.lazy и монтируется
 *  только на экранах, где карта действительно нужна. */

import { useEffect, useRef, useState } from "react";
import { LineLayer, PolygonLayer, RasterLayer, Scene } from "@antv/l7";
import { Map as L7Map } from "@antv/l7-maps";
import { DrawEvent, DrawPolygon } from "@antv/l7-draw";

import type { OsmField, UserPolygon } from "../../api/types";

const OSM_TILES = "https://tile.openstreetmap.org/{z}/{x}/{y}.png";

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
      const basemap = new RasterLayer({ zIndex: 0 }).source(OSM_TILES, {
        parser: { type: "rasterTile", tileSize: 256, minZoom: 1, maxZoom: 19 },
      });
      scene.addLayer(basemap);

      const draw = new DrawPolygon(scene, {
        liveUpdate: true,
        editable: true,
        multiple: false,
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
    <div style={{ position: "relative", height, borderRadius: 12, overflow: "hidden", background: "#e8e6df" }}>
      <div ref={containerRef} style={{ position: "absolute", inset: 0 }} />
      <div
        style={{
          position: "absolute",
          right: 8,
          bottom: 6,
          fontSize: 10,
          background: "rgba(255,255,255,.78)",
          padding: "2px 6px",
          borderRadius: 4,
          color: "#5c5f54",
        }}
      >
        © OpenStreetMap contributors
      </div>
    </div>
  );
}
