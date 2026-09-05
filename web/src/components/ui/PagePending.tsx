/** Заглушка на время загрузки чанка экрана. Шапка и фон остаются на месте,
 *  поэтому переход читается как ожидание внутри страницы, а не как её перезагрузка. */

export function PagePending({ label = "Готовим экран" }: { label?: string }) {
  return (
    <div className="shell-pending" role="status">
      <span className="meta">{label}</span>
      <span className="shell-pending-track" aria-hidden>
        <span className="shell-pending-bar" />
      </span>
    </div>
  );
}
