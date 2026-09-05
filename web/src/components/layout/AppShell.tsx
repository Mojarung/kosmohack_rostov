/** Каркас внутренних экранов: шапка с вкладками, фоновый слой пузырьков, рабочая область.
 *
 *  Переключение вкладок без моргания: «пузырь» под активной вкладкой переезжает на новое место,
 *  содержимое всплывает, чанк экрана предзагружен (см. App.tsx), поэтому заглушка почти не появляется.
 *  Анимируются только transform и opacity. */

import { useEffect, useLayoutEffect, useRef, useTransition, type ReactNode } from "react";
import { NavLink, useLocation, useNavigate } from "react-router-dom";
import gsap from "gsap";

import { BubbleLayer, burstBubbles } from "../motion/Bubbles";
import { SproutArt } from "../art/Art";

const NAV = [
  { to: "/fields", label: "Поля кейса" },
  { to: "/explore", label: "Новая территория" },
  { to: "/anomalies", label: "Аномалии" },
  { to: "/method", label: "Как это работает" },
];

/** Оттенки пузырьков внутренних экранов: те же, что на главной. */
const PALETTE = ["#f6ead6", "#f0e2cc", "#e7eee0", "#e4ecf4", "#fbf4e8"];

/** Вкладка. Переход запускается через startTransition: React дорисовывает новый экран в фоне,
 *  старый остаётся на месте и кликабельным, поэтому смены вкладки не «проваливается» в пустоту. */
function Tab({ to, label, onNavigate }: { to: string; label: string; onNavigate: (to: string) => void }) {
  const { pathname } = useLocation();
  const active = pathname === to || pathname.startsWith(to + "/");
  return (
    <a
      href={to}
      className="shell-tab"
      aria-current={active ? "page" : undefined}
      onClick={(event) => {
        // средняя кнопка и клик с модификатором — открыть в новой вкладке, не перехватываем
        if (event.button !== 0 || event.metaKey || event.ctrlKey || event.shiftKey || event.altKey) return;
        event.preventDefault();
        if (!active) onNavigate(to);
      }}
    >
      {label}
    </a>
  );
}


/** Пузырь под активной вкладкой: переезжает и растягивается под её ширину. */
function useMovingPill(pathname: string) {
  const navRef = useRef<HTMLElement>(null);
  const pillRef = useRef<HTMLSpanElement>(null);
  const placed = useRef(false);

  useLayoutEffect(() => {
    const nav = navRef.current;
    const pill = pillRef.current;
    if (!nav || !pill) return;

    const move = () => {
      const active = nav.querySelector<HTMLElement>('a[aria-current="page"]');
      if (!active) {
        gsap.to(pill, { opacity: 0, duration: 0.2 });
        placed.current = false;
        return;
      }
      const box = active.getBoundingClientRect();
      const base = nav.getBoundingClientRect();
      const target = { x: box.left - base.left, y: box.top - base.top, width: box.width, opacity: 1 };
      // первый показ — без переезда через весь ряд
      const instant = !placed.current || window.matchMedia("(prefers-reduced-motion: reduce)").matches;
      placed.current = true;
      gsap.to(pill, { ...target, duration: instant ? 0 : 0.55, ease: "elastic.out(1, 0.78)" });
    };

    move();
    const observer = new ResizeObserver(move);
    observer.observe(nav);
    return () => observer.disconnect();
  }, [pathname]);

  return { navRef, pillRef };
}

export function AppShell({ children }: { children: ReactNode }) {
  const { pathname } = useLocation();
  const navigate = useNavigate();
  const [isPending, startTransition] = useTransition();
  const previous = useRef(pathname);
  const mainRef = useRef<HTMLElement>(null);
  const { navRef, pillRef } = useMovingPill(pathname);

  const go = (to: string) => startTransition(() => navigate(to));

  // всплеск пузырьков и подъём страницы при каждом переходе
  useEffect(() => {
    if (previous.current === pathname) return;
    previous.current = pathname;
    window.scrollTo({ top: 0, behavior: "instant" as ScrollBehavior });
    burstBubbles({ palette: PALETTE, count: 18 });
    if (mainRef.current && !window.matchMedia("(prefers-reduced-motion: reduce)").matches) {
      gsap.fromTo(mainRef.current, { opacity: 0, y: 18 }, { opacity: 1, y: 0, duration: 0.6, ease: "expo.out" });
    }
  }, [pathname]);

  // главная сама рисует шапку поверх видео
  if (pathname === "/") {
    return <>{children}</>;
  }

  return (
    <div className="shell">
      <BubbleLayer count={14} palette={PALETTE} opacity={0.45} intro={false} />

      {/* тонкая полоса сверху, пока новый экран дорисовывается */}
      <span className={"shell-progress" + (isPending ? " is-on" : "")} aria-hidden />

      <header className="shell-header">
        <div className="shell-header-inner">
          <NavLink to="/" className="shell-brand">
            <SproutArt size={26} />
            <span>
              <span className="shell-brand-name">Вегетация</span>
              <span className="shell-brand-sub">спутниковый мониторинг полей</span>
            </span>
          </NavLink>

          <nav ref={navRef} className="shell-nav">
            <span ref={pillRef} className="shell-pill" aria-hidden />
            {NAV.map((item) => (
              <Tab key={item.to} to={item.to} label={item.label} onNavigate={go} />
            ))}
          </nav>
        </div>
      </header>

      <main ref={mainRef} className="shell-main">
        {children}
      </main>

      <footer className="shell-footer">
        <div className="shell-footer-inner">
          <span className="meta" style={{ maxWidth: "72ch" }}>
            Данные: Sentinel-2 и Landsat (Earth Search, Planetary Computer), MODIS MOD13Q1, ERA5 через Open-Meteo,
            контуры полей OpenStreetMap.
          </span>
          <span className="mono meta">Космохакатон · Ростовская область</span>
        </div>
      </footer>
    </div>
  );
}
