/** Пузырьки на GSAP: постоянный фоновый слой и всплеск при переходе между экранами.
 *
 *  Анимируются только transform и opacity, слой лежит вне потока (position: fixed) и не принимает
 *  события мыши, поэтому композитор рисует его отдельно и кадры не проседают на 120 Гц.
 *  Появление — пружинное (elastic/back), движение вверх — длинная синусоида. */

import { useEffect, useRef } from "react";
import gsap from "gsap";

export interface BubbleLayerProps {
  count?: number;
  /** Оттенки пузырьков: подбираются под фон конкретного экрана. */
  palette?: string[];
  /** Насколько крупные пузырьки: множитель к базовому размеру. */
  scale?: number;
  opacity?: number;
  /** Появиться с пружиной при монтировании. */
  intro?: boolean;
}

const DEFAULT_PALETTE = ["#f3e7d2", "#e6efe1", "#dfe9f2", "#f6efe3", "#e9e2d6"];

function seeded(seed: number) {
  let value = seed;
  return () => {
    value = (value * 1103515245 + 12345) % 2147483648;
    return value / 2147483648;
  };
}

export function BubbleLayer({
  count = 16,
  palette = DEFAULT_PALETTE,
  scale = 1,
  opacity = 0.5,
  intro = true,
}: BubbleLayerProps) {
  const rootRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const root = rootRef.current;
    if (!root) return;
    if (window.matchMedia("(prefers-reduced-motion: reduce)").matches) return;

    const nodes = Array.from(root.children) as HTMLElement[];
    const context = gsap.context(() => {
      if (intro) {
        gsap.from(nodes, {
          scale: 0.2,
          opacity: 0,
          duration: 1.6,
          ease: "elastic.out(0.9, 0.55)",
          stagger: { each: 0.05, from: "random" },
        });
      }
      nodes.forEach((node) => {
        const rise = Number(node.dataset.rise);
        const drift = Number(node.dataset.drift);
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
          scale: 1.14,
          duration: duration * 0.55,
          delay: delay * 0.5,
          ease: "sine.inOut",
          repeat: -1,
          yoyo: true,
        });
      });
    }, root);

    return () => context.revert();
  }, [intro, count]);

  const random = seeded(11);
  const bubbles = Array.from({ length: count }, () => {
    const size = (30 + random() * 210) * scale;
    return {
      size,
      left: random() * 100,
      top: random() * 100,
      hue: palette[Math.floor(random() * palette.length)],
      drift: (random() - 0.5) * 110,
      rise: 70 + random() * 190,
      delay: random() * 6,
      duration: 15 + random() * 22,
    };
  });

  return (
    <div
      ref={rootRef}
      aria-hidden
      style={{ position: "fixed", inset: 0, zIndex: 0, pointerEvents: "none", overflow: "hidden", opacity }}
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
            background: `radial-gradient(circle at 32% 28%, ${bubble.hue}, transparent 70%)`,
            willChange: "transform, opacity",
          }}
        />
      ))}
    </div>
  );
}

/** Всплеск пузырьков при переходе между экранами: элементы создаются вне React и сами себя убирают. */
export function burstBubbles(options: { count?: number; palette?: string[] } = {}) {
  if (typeof window === "undefined") return;
  if (window.matchMedia("(prefers-reduced-motion: reduce)").matches) return;

  const { count = 18, palette = DEFAULT_PALETTE } = options;
  const host = document.createElement("div");
  host.setAttribute("aria-hidden", "true");
  Object.assign(host.style, {
    position: "fixed",
    inset: "0",
    zIndex: "60",
    pointerEvents: "none",
    overflow: "hidden",
  } satisfies Partial<CSSStyleDeclaration>);
  document.body.appendChild(host);

  const nodes: HTMLElement[] = [];
  for (let i = 0; i < count; i += 1) {
    const size = 16 + Math.random() * 120;
    const node = document.createElement("span");
    Object.assign(node.style, {
      position: "absolute",
      left: `${Math.random() * 100}%`,
      top: `${70 + Math.random() * 40}%`,
      width: `${size}px`,
      height: `${size}px`,
      borderRadius: "50%",
      background: `radial-gradient(circle at 34% 30%, ${palette[i % palette.length]}, transparent 72%)`,
      willChange: "transform, opacity",
    } satisfies Partial<CSSStyleDeclaration>);
    host.appendChild(node);
    nodes.push(node);
  }

  gsap
    .timeline({ onComplete: () => host.remove() })
    .fromTo(
      nodes,
      { scale: 0.3, opacity: 0, y: 60 },
      {
        scale: 1,
        opacity: 0.75,
        y: 0,
        duration: 0.75,
        ease: "back.out(2.2)",
        stagger: { each: 0.02, from: "random" },
      },
    )
    .to(
      nodes,
      {
        y: () => -260 - Math.random() * 320,
        x: () => (Math.random() - 0.5) * 160,
        opacity: 0,
        scale: 1.25,
        duration: 1.15,
        ease: "power2.in",
        stagger: { each: 0.015, from: "random" },
      },
      "-=0.35",
    );
}
