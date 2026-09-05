/** Ожидание: три точки, анимация только по opacity/transform. */

export function Loader({ label = "Загружаем" }: { label?: string }) {
  return (
    <div className="row" style={{ gap: 10, padding: "48px 0", color: "var(--muted)", justifyContent: "center" }}>
      <span className="meta">{label}</span>
      <span aria-hidden style={{ display: "inline-flex", gap: 4 }}>
        {[0, 1, 2].map((i) => (
          <span
            key={i}
            style={{
              width: 5,
              height: 5,
              borderRadius: "50%",
              background: "var(--line-strong)",
              animation: "pulse 1.1s infinite var(--ease)",
              animationDelay: `${i * 0.14}s`,
            }}
          />
        ))}
      </span>
      <style>{`@keyframes pulse { 0%,100% { opacity:.25; transform:scale(.8) } 50% { opacity:1; transform:scale(1) } }`}</style>
    </div>
  );
}

export function ErrorNote({ error }: { error: unknown }) {
  const message = error instanceof Error ? error.message : String(error);
  return (
    <div className="card" style={{ borderColor: "#f0d6d6", background: "var(--crit-bg)", color: "var(--crit-ink)" }}>
      <div className="eyebrow" style={{ color: "inherit" }}>
        Ошибка
      </div>
      <p style={{ marginTop: 6 }}>{message}</p>
    </div>
  );
}
