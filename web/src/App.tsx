import { Suspense, lazy } from "react";
import { Navigate, Route, Routes } from "react-router-dom";

import { AppShell } from "./components/layout/AppShell";
import { Loader } from "./components/ui/Loader";
import { LandingPage } from "./pages/LandingPage";

// Экраны с картой и графиками грузятся по требованию: первый экран остаётся лёгким.
const OverviewPage = lazy(() => import("./pages/OverviewPage"));
const FieldPage = lazy(() => import("./pages/FieldPage"));
const ExplorePage = lazy(() => import("./pages/ExplorePage"));
const AnomaliesPage = lazy(() => import("./pages/AnomaliesPage"));
const MethodPage = lazy(() => import("./pages/MethodPage"));

export function App() {
  return (
    <AppShell>
      <Suspense fallback={<Loader label="Готовим экран" />}>
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
