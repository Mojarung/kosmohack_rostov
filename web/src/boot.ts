/** Снятие загрузчика первого захода (разметка лежит в index.html).
 *  Ждём три условия: смонтирован интерфейс, готовы шрифты и загружен постер первого экрана.
 *  Дольше MAX_WAIT_MS не ждём ни при каких условиях — экран не должен залипнуть. */

const MAX_WAIT_MS = 4000;
const FADE_MS = 900;

/** Ждём загрузки картинки; ошибка сети тоже считается ответом. */
function waitImage(src: string): Promise<void> {
  return new Promise((resolve) => {
    const image = new Image();
    image.onload = () => resolve();
    image.onerror = () => resolve();
    image.src = src;
  });
}

function remove(node: HTMLElement): void {
  if (node.dataset.done) return;
  node.dataset.done = "1";
  node.addEventListener("transitionend", () => node.remove(), { once: true });
  window.setTimeout(() => node.remove(), FADE_MS);
}

export function hideBoot(): void {
  const node = document.getElementById("boot");
  if (!node) return;

  const fonts = document.fonts ? document.fonts.ready.then(() => undefined) : Promise.resolve();
  // постер нужен только главной: на других экранах первого кадра нет
  const poster = window.location.pathname === "/" ? waitImage("/hero-poster.jpg") : Promise.resolve();
  const timeout = new Promise<void>((resolve) => window.setTimeout(resolve, MAX_WAIT_MS));

  void Promise.race([Promise.all([fonts, poster]).then(() => undefined), timeout]).then(() => {
    // два кадра подряд: даём браузеру отрисовать смонтированный интерфейс до снятия шторки
    requestAnimationFrame(() => requestAnimationFrame(() => remove(node)));
  });
}
