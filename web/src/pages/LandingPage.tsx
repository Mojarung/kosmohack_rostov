/** Главный экран: слева текст, на фоне видео без рамки, ниже — разделы, которые появляются
 *  по ScrollTrigger в обе стороны (при прокрутке назад анимация отыгрывается обратно). */

import { useEffect, useLayoutEffect, useRef } from "react";
import { Link, useNavigate } from "react-router-dom";
import { useQuery } from "@tanstack/react-query";
import gsap from "gsap";
import { ScrollTrigger } from "gsap/ScrollTrigger";

import { api } from "../api/client";
import { CurveArt, DroughtArt, GapArt, LayersArt, SatelliteArt, SproutArt } from "../components/art/Art";
import { HeroVideo } from "../components/hero/HeroVideo";

gsap.registerPlugin(ScrollTrigger);

const STEPS = [
  {
    art: SatelliteArt,
    title: "Собираем",
    text: "Sentinel-2, Landsat и MODIS по контуру поля плюс погода ERA5. Ключей и ручной загрузки данных не нужно.",
  },
  {
    art: GapArt,
    title: "Восстанавливаем",
    text: "Облака и смена спутника оставляют дыры в ряде. Смесь бустинга и нейросети возвращает пропущенные даты.",
  },
  {
    art: CurveArt,
    title: "Сравниваем",
    text: "Сезон ложится на норму этого же поля по его прошлым годам, а не на средний график по региону.",
  },
  {
    art: DroughtArt,
    title: "Объясняем",
    text: "Каждый период угнетения получает причину: засуха, незасеянное поле, севооборот, ранний спад.",
  },
];

function Nav() {
  return (
    <nav className="landing-topbar">
      <span style={{ display: "flex", alignItems: "center", gap: 10 }}>
        <SproutArt size={26} />
        <span className="display" style={{ fontSize: 21, letterSpacing: "0.02em" }}>
          Вегетация
        </span>
      </span>
      <span className="row landing-nav" style={{ gap: 26 }}>
        <Link to="/fields">Поля кейса</Link>
        <Link to="/anomalies">Аномалии</Link>
        <Link to="/method">Метод</Link>
      </span>
      <Link to="/explore" className="btn btn--pill" style={{ textDecoration: "none" }}>
        Выбрать поле
      </Link>
    </nav>
  );
}

export function LandingPage() {
  const heroRef = useRef<HTMLDivElement>(null);
  const rootRef = useRef<HTMLDivElement>(null);
  const navigate = useNavigate();
  const meta = useQuery({ queryKey: ["meta"], queryFn: api.meta });

  useEffect(() => {
    document.body.dataset.landing = "1";
    return () => {
      delete document.body.dataset.landing;
    };
  }, []);

  useLayoutEffect(() => {
    const context = gsap.context(() => {
      if (window.matchMedia("(prefers-reduced-motion: reduce)").matches) return;

      gsap
        .timeline({ defaults: { ease: "expo.out" } })
        .from(".hero-line", { yPercent: 118, duration: 1.2, stagger: 0.1 })
        .from(".hero-eyebrow", { opacity: 0, y: 12, duration: 0.8 }, 0.1)
        .from(".hero-sub", { opacity: 0, y: 18, duration: 0.9 }, 0.45)
        .from(".hero-cta", { opacity: 0, y: 20, scale: 0.95, duration: 1, ease: "back.out(2)" }, 0.6)
        .from(".hero-feature", { opacity: 0, y: 24, duration: 0.9, stagger: 0.08, ease: "back.out(1.6)" }, 0.7);

      // параллакс первого экрана: текст и кадр расходятся с разной скоростью
      gsap.to(".hero-copy", {
        yPercent: -12,
        opacity: 0.3,
        ease: "none",
        scrollTrigger: { trigger: heroRef.current, start: "top top", end: "bottom top", scrub: 0.6 },
      });
      gsap.to(".hero-media-inner", {
        yPercent: 6,
        ease: "none",
        scrollTrigger: { trigger: heroRef.current, start: "top top", end: "bottom top", scrub: 0.8 },
      });

      // разделы: появление в обе стороны — назад анимация отыгрывается обратно
      gsap.utils.toArray<HTMLElement>(".spring-in").forEach((node) => {
        gsap.from(node, {
          y: 42,
          opacity: 0,
          duration: 1,
          ease: "back.out(1.5)",
          scrollTrigger: { trigger: node, start: "top 88%", end: "bottom 12%", toggleActions: "play none none reverse" },
        });
      });

      // числа считаются вверх, а при прокрутке назад — обратно к нулю
      gsap.utils.toArray<HTMLElement>(".count-up").forEach((node) => {
        const target = Number(node.dataset.value ?? "0");
        const digits = Number(node.dataset.digits ?? "0");
        const state = { value: 0 };
        gsap.to(state, {
          value: target,
          duration: 1.4,
          ease: "power2.out",
          scrollTrigger: { trigger: node, start: "top 90%", end: "bottom 10%", toggleActions: "play none none reverse" },
          onUpdate: () => {
            node.textContent = state.value.toLocaleString("ru-RU", {
              minimumFractionDigits: digits,
              maximumFractionDigits: digits,
            });
          },
        });
      });
    }, rootRef);

    return () => context.revert();
  }, [meta.data]);

  const task1 = meta.data?.task1;
  const task2 = meta.data?.task2;

  return (
    <div ref={rootRef} className="landing">

      <section ref={heroRef} className="hero">
        <HeroVideo src="/hero.mp4" poster="/hero-poster.jpg" />
        <Nav />

        <div className="hero-grid">
          <div className="hero-copy">
            <span className="hero-eyebrow eyebrow">спутниковые снимки · Ростовская область</span>
            <h1 className="display hero-title">
              <span className="hero-mask">
                <span className="hero-line">Считаем NDVI</span>
              </span>
              <span className="hero-mask">
                <span className="hero-line">и ищем аномалии.</span>
              </span>
            </h1>
            <p className="hero-sub">
              Сервис сам собирает снимки трёх спутников и погоду по контуру участка, заполняет пропуски в ряде
              и показывает периоды, когда поле развивалось хуже своей нормы, — с разбором причины.
            </p>
            <div className="hero-cta row" style={{ gap: 12, flexWrap: "wrap" }}>
              <button
                type="button"
                className="btn btn--pill btn--lg"
                onClick={() => navigate("/explore")}
              >
                Выбрать поле на карте
              </button>
              <Link to="/fields" className="btn btn--pill btn--lg btn--ghost" style={{ textDecoration: "none" }}>
                Смотреть готовые поля
              </Link>
            </div>
          </div>

        </div>

        <div className="hero-features">
          {[
            { art: GapArt, title: "Восстановление", text: "пропуски в ряде NDVI закрывает модель" },
            { art: CurveArt, title: "Аномалии", text: "периоды угнетения находятся сами" },
            { art: DroughtArt, title: "Причина", text: "погода, фенология и соседние поля" },
          ].map((item) => {
            const Art = item.art;
            return (
              <div key={item.title} className="hero-feature">
                <Art size={30} />
                <div>
                  <div className="hero-feature-title">{item.title}</div>
                  <div className="hero-feature-text">{item.text}</div>
                </div>
              </div>
            );
          })}
          <span className="eyebrow hero-exhale">( NDVI )</span>
        </div>
      </section>

      <section className="landing-section">
        <div className="landing-inner">
          <span className="eyebrow spring-in">Как это работает</span>
          <h2 className="display spring-in" style={{ fontSize: "clamp(28px, 3.4vw, 46px)", marginTop: 10 }}>
            Четыре шага от снимка до объяснения
          </h2>
          <div className="steps">
            {STEPS.map((step, index) => {
              const Art = step.art;
              return (
                <article key={step.title} className="step spring-in">
                  <span className="step-index mono">{String(index + 1).padStart(2, "0")}</span>
                  <Art size={40} />
                  <h3 className="display" style={{ fontSize: 22, marginTop: 10 }}>
                    {step.title}
                  </h3>
                  <p>{step.text}</p>
                </article>
              );
            })}
          </div>
        </div>
      </section>

      {task1 && task2 && (
        <section className="landing-section landing-section--tint">
          <div className="landing-inner">
            <span className="eyebrow spring-in">Результат на данных кейса</span>
            <div className="metrics">
              <div className="metric spring-in">
                <div className="metric-value">
                  <span className="count-up num" data-value={task1.rmse_val} data-digits="4">
                    0,0000
                  </span>
                </div>
                <div className="metric-label">RMSE восстановления</div>
                <p className="metric-note">
                  У простого способа «среднее двух соседних дат» — {task1.baseline_rmse.toFixed(3)}. Метрика
                  считается на отложенной выборке, которую модель не видела: среднее по {task1.val_seeds} разбиениям.
                </p>
              </div>
              <div className="metric spring-in">
                <div className="metric-value">
                  <span className="count-up num" data-value={task1.gap_score} data-digits="1">
                    0,0
                  </span>
                  <span className="metric-unit">из 30</span>
                </div>
                <div className="metric-label">GapScore</div>
                <p className="metric-note">
                  Оценка организаторов за восстановление пропусков: {task1.baseline_gap_score.toFixed(1)} у baseline.
                </p>
              </div>
              <div className="metric spring-in">
                <div className="metric-value">
                  <span className="count-up num" data-value={task2.n_episodes} data-digits="0">
                    0
                  </span>
                </div>
                <div className="metric-label">периодов угнетения</div>
                <p className="metric-note">
                  Найдено в {task2.n_seasons} сезонах {task2.n_polygons} полей, у каждого разобрана причина.
                </p>
              </div>
              <div className="metric spring-in">
                <div className="metric-value">
                  <span className="count-up num" data-value={task1.train_points} data-digits="0">
                    0
                  </span>
                </div>
                <div className="metric-label">наблюдений в обучении</div>
                <p className="metric-note">Только данные кейса: снимки трёх спутников за 2010–2025 годы.</p>
              </div>
            </div>
          </div>
        </section>
      )}

      <section className="landing-section">
        <div className="landing-inner landing-cta spring-in">
          <LayersArt size={54} />
          <h2 className="display" style={{ fontSize: "clamp(26px, 3vw, 40px)", marginTop: 14 }}>
            Проверьте на своём поле
          </h2>
          <p style={{ maxWidth: "56ch", margin: "10px auto 0" }}>
            Найдите контур в OpenStreetMap или обведите участок сами. Через пару минут появится ряд NDVI за
            несколько сезонов, восстановленная кривая и список периодов угнетения с объяснением.
          </p>
          <div className="row" style={{ gap: 12, justifyContent: "center", marginTop: 22, flexWrap: "wrap" }}>
            <Link to="/explore" className="btn btn--pill btn--lg" style={{ textDecoration: "none" }}>
              Открыть карту
            </Link>
            <Link to="/method" className="btn btn--pill btn--lg btn--ghost" style={{ textDecoration: "none" }}>
              Как устроено решение
            </Link>
          </div>
        </div>
      </section>
    </div>
  );
}
