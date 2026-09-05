/** Каркас приложения: шапка с навигацией, фоновый слой пузырьков, ограниченная по ширине рабочая область. */

import { useEffect, useRef, type ReactNode } from "react";
import { NavLink, useLocation } from "react-router-dom";

import { BubbleLayer, burstBubbles } from "../motion/Bubbles";
import { SproutArt } from "../art/Art";

const NAV = [
  { to: "/fields", label: "Поля кейса" },
  { to: "/explore", label: "Новая территория" },
  { to: "/anomalies", label: "Аномалии" },
  { to: "/method", label: "Как это работает" },
];

export function AppShell({ children }: { children: ReactNode }) {
  const { pathname } = useLocation();
  const previous = useRef(pathname);

  // при каждом переходе между экранами пускаем всплеск пузырьков
  useEffect(() => {
    if (previous.current !== pathname) {
      previous.current = pathname;
      burstBubbles({ count: 14 });
      window.scrollTo({ top: 0, behavior: "instant" as ScrollBehavior });
    }
  }, [pathname]);

  // главная сама рисует шапку поверх видео
  if (pathname === "/") {
    return <>{children}</>;
  }

  return (
    <div style={{ minHeight: "100%", position: "relative", isolation: "isolate" }}>
      <BubbleLayer count={14} opacity={0.5} intro={false} />
      <header
        style={{
          position: "sticky",
          top: 0,
          zIndex: 20,
          background: "rgba(247,246,243,.86)",
          backdropFilter: "blur(10px)",
          borderBottom: "1px solid var(--line)",
        }}
      >
        <div
          className="spread"
          style={{ maxWidth: "var(--shell-max)", margin: "0 auto", padding: "10px clamp(16px, 3vw, 34px)" }}
        >
          <NavLink to="/" style={{ textDecoration: "none" }} className="row">
            <span style={{ display: "flex", gap: 10, alignItems: "center" }}>
              <SproutArt size={26} />
              <span>
                <span style={{ fontFamily: "var(--font-serif)", fontSize: 18, letterSpacing: "-0.02em" }}>
                  Вегетация
                </span>
                <span className="meta" style={{ display: "block", marginTop: -3, fontSize: 11 }}>
                  спутниковый мониторинг полей
                </span>
              </span>
            </span>
          </NavLink>
          <nav className="row" style={{ gap: 4, flexWrap: "wrap" }}>
            {NAV.map((item) => (
              <NavLink
                key={item.to}
                to={item.to}
                style={({ isActive }) => ({
                  textDecoration: "none",
                  padding: "6px 12px",
                  borderRadius: 999,
                  fontSize: 13,
                  color: isActive ? "var(--ink)" : "var(--muted)",
                  background: isActive ? "var(--surface)" : "transparent",
                  border: `1px solid ${isActive ? "var(--line)" : "transparent"}`,
                  transition: "background .18s var(--ease), color .18s var(--ease)",
                })}
              >
                {item.label}
              </NavLink>
            ))}
          </nav>
        </div>
      </header>

      <main
        style={{
          position: "relative",
          zIndex: 1,
          maxWidth: "var(--shell-max)",
          margin: "0 auto",
          padding: "clamp(20px, 3vw, 40px) clamp(16px, 3vw, 34px) 72px",
        }}
      >
        {children}
      </main>

      <footer
        style={{
          borderTop: "1px solid var(--line)",
          position: "relative",
          zIndex: 1,
          background: "rgba(255,255,255,.6)",
        }}
      >
        <div
          className="spread"
          style={{
            maxWidth: "var(--shell-max)",
            margin: "0 auto",
            padding: "16px clamp(16px, 3vw, 34px)",
            flexWrap: "wrap",
            gap: 8,
          }}
        >
          <span className="meta">
            Данные: Sentinel-2 и Landsat (Earth Search, Planetary Computer), MODIS MOD13Q1, ERA5 через Open-Meteo,
            контуры полей OpenStreetMap.
          </span>
          <span className="mono meta">Космохакатон · Ростовская область</span>
        </div>
      </footer>
    </div>
  );
}
