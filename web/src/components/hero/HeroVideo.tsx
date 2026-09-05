/** Видео на фоне первого экрана. Кадр вписан в экран целиком без растяжения (max-width/max-height),
 *  размер элемента равен кадру, и маска растворяет его собственные края — рамки не видно,
 *  а градиент фона в тонах кадра выглядит его продолжением. Атрибуты width/height задают пропорции
 *  ещё до загрузки метаданных, чтобы кадр не прыгал.
 *  Ролик склеен из прямого и обратного проходов, поэтому петля идёт туда-обратно без рывка. */

import { useEffect, useRef } from "react";
import gsap from "gsap";

interface HeroVideoProps {
  src: string;
  poster: string;
}

export function HeroVideo({ src, poster }: HeroVideoProps) {
  const rootRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const root = rootRef.current;
    if (!root) return;
    if (window.matchMedia("(prefers-reduced-motion: reduce)").matches) return;

    const context = gsap.context(() => {
      gsap.fromTo(
        ".hero-media-inner",
        { scale: 1.06, opacity: 0 },
        { scale: 1, opacity: 1, duration: 1.8, ease: "expo.out" },
      );

      // кадр едва заметно сдвигается за курсором — глубина без «плавания»
      const moveX = gsap.quickTo(".hero-media-inner", "x", { duration: 1.2, ease: "power3.out" });
      const moveY = gsap.quickTo(".hero-media-inner", "y", { duration: 1.2, ease: "power3.out" });
      const onMove = (event: PointerEvent) => {
        const dx = event.clientX / window.innerWidth - 0.5;
        const dy = event.clientY / window.innerHeight - 0.5;
        moveX(dx * 12);
        moveY(dy * 9);
      };
      window.addEventListener("pointermove", onMove, { passive: true });
      return () => window.removeEventListener("pointermove", onMove);
    }, root);

    return () => context.revert();
  }, []);

  return (
    <div ref={rootRef} className="hero-media" aria-hidden>
      <div className="hero-media-inner">
        <video src={src} poster={poster} width={1500} height={1080} autoPlay muted loop playsInline preload="auto" />
      </div>
      <span className="hero-vignette" />
      <span className="hero-grain" />
    </div>
  );
}
