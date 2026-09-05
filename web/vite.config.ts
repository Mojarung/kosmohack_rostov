import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// Сборка кладётся в web/dist, откуда её раздаёт FastAPI (service/app.py) в контейнере.
// В режиме разработки запросы /api проксируются на бэкенд (переменная API_ORIGIN).
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      "/api": { target: process.env.API_ORIGIN ?? "http://127.0.0.1:8000", changeOrigin: true },
    },
  },
  build: {
    target: "es2022",
    chunkSizeWarningLimit: 1600,
    rollupOptions: {
      output: {
        // Карта (L7 + WebGL) — самая тяжёлая часть; выносим её и графики в отдельные чанки,
        // чтобы первый экран грузился без них.
        manualChunks(id: string) {
          if (id.includes("node_modules")) {
            if (id.includes("@antv")) return "l7";
            if (id.includes("@mui") || id.includes("d3-")) return "charts";
            if (id.includes("react") || id.includes("@tanstack")) return "vendor";
          }
          return undefined;
        },
      },
    },
  },
});
