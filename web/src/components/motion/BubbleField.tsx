/** Фоновый слой «пузырьков»: GSAP двигает только transform и opacity, слой не участвует
 *  в раскладке (position: fixed, pointer-events: none), поэтому кадр остаётся композиторным
 *  и на 120-герцовых экранах. При prefers-reduced-motion слой не монтируется. */

import { useEffect, useRef } from "react";
import gsap from "gsap";

interface BubbleFieldProps {
  /** Число пузырьков; больше 26 не нужно — визуально шумно и лишняя работа для GPU. */
  count?: number;
  /** Прозрачность слоя целиком. */
  opacity?: number;
}

interface Bubble {
  size: number;
  left: number;
  top: number;
  hue: string;
  drift: number;
  rise: number;
  delay: number;
  duration: number;
}

const HUES = ["#dfe7d8", "#e6efe1", "#f0e7d4", "#dfe9f2", "#eae4dc"];

function makeBubbles(count: number): Bubble[] {
  // Детерминированный псевдослучайный ряд: одинаковая композиция при каждом заходе.
  let seed = 7;
  const random = () => {
    seed = (seed * 1103515245 + 12345) % 2147483648;
    return seed / 2147483648;
  };
  return Array.from({ length: count }, () => {
    const size = 26 + random() * 190;
    return {
      size,
      left: random() * 100,
      top: random() * 100,
      hue: HUES[Math.floor(random() * HUES.length)],
      drift: (random() - 0.5) * 90,
      rise: 60 + random() * 150,
      delay: random() * 8,
      duration: 16 + random() * 20,
    };
  });
}

export function BubbleField({ count = 20, opacity = 0.5 }: BubbleFieldProps) {
  const rootRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const root = rootRef.current;
    if (!root) return;
    if (window.matchMedia("(prefers-reduced-motion: reduce)").matches) return;

    const nodes = Array.from(root.children) as HTMLElement[];
    const context = gsap.context(() => {
      nodes.forEach((node) => {
        const drift = Number(node.dataset.drift);
        const rise = Number(node.dataset.rise);
        const duration = Number(node.dataset.duration);
        const delay = Number(node.dataset.delay);
        gsap.to(node, {
          y: -rise,
          x: drift,
          duration,
          delay,
          ease: "sine.inOut",
          repeat: -1,
          yoyo: true,
        });
        gsap.to(node, {
          scale: 1.12,
          opacity: 0.85,
          duration: duration * 0.6,
          delay: delay * 0.5,
          ease: "sine.inOut",
          repeat: -1,
          yoyo: true,
        });
      });
    }, root);

    // Вкладка в фоне — глобально останавливаем тикер, чтобы не жечь батарею.
    const onVisibility = () => gsap.ticker.lagSmoothing(vis() ? 500 : 0);
    const vis = () => document.visibilityState === "visible";
    document.addEventListener("visibilitychange", onVisibility);
    return () => {
      document.removeEventListener("visibilitychange", onVisibility);
      context.revert();
    };
  }, []);

  const bubbles = makeBubbles(count);
  return (
    <div
      ref={rootRef}
      aria-hidden
      style={{
        position: "fixed",
        inset: 0,
        zIndex: 0,
        pointerEvents: "none",
        overflow: "hidden",
        opacity,
        contain: "strict",
      }}
    >
      {bubbles.map((bubble, index) => (
        <span
          key={index}
          data-drift={bubble.drift}
          data-rise={bubble.rise}
          data-duration={bubble.duration}
          data-delay={bubble.delay}
          style={{
            position: "absolute",
            left: `${bubble.left}%`,
            top: `${bubble.top}%`,
            width: bubble.size,
            height: bubble.size,
            borderRadius: "50%",
            background: `radial-gradient(circle at 32% 30%, ${bubble.hue}, transparent 68%)`,
            opacity: 0.55,
            willChange: "transform, opacity",
          }}
        />
      ))}
    </div>
  );
}
