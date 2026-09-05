/** Свой курсор: точка и кольцо, которое догоняет её пружиной. Над ссылками и кнопками кольцо
 *  раскрывается, над видео — превращается в подпись. Работает только с мышью: на тач-устройствах
 *  и при prefers-reduced-motion компонент ничего не рисует. */

import { useEffect, useRef, useState } from "react";
import gsap from "gsap";

const INTERACTIVE = "a, button, [role='button'], input, select, textarea, .poly-card, .step";

export function Cursor() {
  const dotRef = useRef<HTMLDivElement>(null);
  const ringRef = useRef<HTMLDivElement>(null);
  const [enabled, setEnabled] = useState(false);

  useEffect(() => {
    const fine = window.matchMedia("(pointer: fine)").matches;
    const reduce = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    if (!fine || reduce) return;
    setEnabled(true);
  }, []);

  useEffect(() => {
    if (!enabled) return;
    const dot = dotRef.current;
    const ring = ringRef.current;
    if (!dot || !ring) return;

    document.body.classList.add("has-custom-cursor");
    gsap.set([dot, ring], { xPercent: -50, yPercent: -50, opacity: 0 });

    // точка следует почти мгновенно, кольцо — с задержкой: получается «пружина»
    const dotX = gsap.quickTo(dot, "x", { duration: 0.08, ease: "power3.out" });
    const dotY = gsap.quickTo(dot, "y", { duration: 0.08, ease: "power3.out" });
    const ringX = gsap.quickTo(ring, "x", { duration: 0.42, ease: "power3.out" });
    const ringY = gsap.quickTo(ring, "y", { duration: 0.42, ease: "power3.out" });

    let visible = false;
    const onMove = (event: PointerEvent) => {
      if (!visible) {
        visible = true;
        gsap.to([dot, ring], { opacity: 1, duration: 0.3 });
      }
      dotX(event.clientX);
      dotY(event.clientY);
      ringX(event.clientX);
      ringY(event.clientY);
    };

    const onOver = (event: PointerEvent) => {
      const target = (event.target as HTMLElement)?.closest?.(INTERACTIVE);
      gsap.to(ring, {
        scale: target ? 1.9 : 1,
        borderColor: target ? "rgba(33,29,23,.55)" : "rgba(33,29,23,.28)",
        duration: 0.45,
        ease: "elastic.out(1, 0.6)",
      });
      gsap.to(dot, { scale: target ? 0.45 : 1, duration: 0.35, ease: "power3.out" });
    };

    const onDown = () => gsap.to(ring, { scale: 0.85, duration: 0.18, ease: "power2.out" });
    const onUp = () => gsap.to(ring, { scale: 1, duration: 0.5, ease: "elastic.out(1, 0.5)" });
    const onLeave = () => {
      visible = false;
      gsap.to([dot, ring], { opacity: 0, duration: 0.25 });
    };

    window.addEventListener("pointermove", onMove, { passive: true });
    window.addEventListener("pointerover", onOver, { passive: true });
    window.addEventListener("pointerdown", onDown);
    window.addEventListener("pointerup", onUp);
    document.addEventListener("pointerleave", onLeave);

    return () => {
      document.body.classList.remove("has-custom-cursor");
      window.removeEventListener("pointermove", onMove);
      window.removeEventListener("pointerover", onOver);
      window.removeEventListener("pointerdown", onDown);
      window.removeEventListener("pointerup", onUp);
      document.removeEventListener("pointerleave", onLeave);
      gsap.killTweensOf([dot, ring]);
    };
  }, [enabled]);

  if (!enabled) return null;

  return (
    <>
      <div ref={ringRef} className="cursor-ring" aria-hidden />
      <div ref={dotRef} className="cursor-dot" aria-hidden />
    </>
  );
}
