import { Suspense, lazy, useEffect } from "react";
import { Navigate, Route, Routes } from "react-router-dom";

import { AppShell } from "./components/layout/AppShell";
import { PagePending } from "./components/ui/PagePending";
import { LandingPage } from "./pages/LandingPage";

// Экраны с картой и графиками грузятся по требованию: первый экран остаётся лёгким.
const CHUNKS = {
  overview: () => import("./pages/OverviewPage"),
  field: () => import("./pages/FieldPage"),
  explore: () => import("./pages/ExplorePage"),
  anomalies: () => import("./pages/AnomaliesPage"),
  method: () => import("./pages/MethodPage"),
};

const OverviewPage = lazy(CHUNKS.overview);
const FieldPage = lazy(CHUNKS.field);
const ExplorePage = lazy(CHUNKS.explore);
const AnomaliesPage = lazy(CHUNKS.anomalies);
const MethodPage = lazy(CHUNKS.method);

/** Догружаем остальные экраны в простое браузера: переход по вкладке происходит без ожидания. */
function usePrefetchPages() {
  useEffect(() => {
    const load = () => Object.values(CHUNKS).forEach((chunk) => void chunk());
    const idle = window.requestIdleCallback;
    if (idle) {
      const id = idle(load, { timeout: 3000 });
      return () => window.cancelIdleCallback?.(id);
    }
    const timer = window.setTimeout(load, 1200);
    return () => window.clearTimeout(timer);
  }, []);
}

export function App() {
  usePrefetchPages();
  return (
    <AppShell>
      <Suspense fallback={<PagePending />}>
        <Routes>
          <Route path="/" element={<LandingPage />} />
          <Route path="/fields" element={<OverviewPage />} />
          <Route path="/field/:pid" element={<FieldPage />} />
          <Route path="/explore" element={<ExplorePage />} />
          <Route path="/explore/:uid" element={<ExplorePage />} />
          <Route path="/anomalies" element={<AnomaliesPage />} />
          <Route path="/method" element={<MethodPage />} />
          <Route path="*" element={<Navigate to="/" replace />} />
        </Routes>
      </Suspense>
    </AppShell>
  );
}
