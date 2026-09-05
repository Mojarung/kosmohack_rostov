/* Управление слайдами: стрелки, пробел, свайп, клик по правой/левой половине экрана.
   Слайд имеет фиксированный размер 1280x720 и вписывается в окно через transform: scale —
   так вёрстка не «плывёт» на проекторе с любым разрешением. */

const slides = Array.from(document.querySelectorAll('.slide'));
const scaler = document.getElementById('scaler');
const progressBar = document.getElementById('progress-bar');

let current = 0;

/** Показать слайд с индексом i (индекс приводится к допустимому диапазону). */
function show(i) {
  current = Math.max(0, Math.min(slides.length - 1, i));
  slides.forEach((slide, index) => slide.classList.toggle('is-active', index === current));
  progressBar.style.width = `${((current + 1) / slides.length) * 100}%`;
  // Номер слайда в адресной строке — можно дать ссылку сразу на нужный слайд.
  history.replaceState(null, '', `#${current + 1}`);
}

/** Вписать сцену 1280x720 в текущее окно, оставив небольшие поля. */
function fit() {
  const scale = Math.min(window.innerWidth / 1280, window.innerHeight / 720) * 0.96;
  scaler.style.transform = `scale(${scale})`;
}

window.addEventListener('resize', fit);

document.addEventListener('keydown', (event) => {
  if (event.key === 'ArrowRight' || event.key === 'PageDown' || event.key === ' ') {
    event.preventDefault();
    show(current + 1);
  } else if (event.key === 'ArrowLeft' || event.key === 'PageUp') {
    event.preventDefault();
    show(current - 1);
  } else if (event.key === 'Home') {
    show(0);
  } else if (event.key === 'End') {
    show(slides.length - 1);
  } else if (event.key === 'p' || event.key === 'P' || event.key === 'з' || event.key === 'З') {
    window.print();
  }
});

// Клик мышью: правая половина — вперёд, левая — назад.
document.addEventListener('click', (event) => {
  show(event.clientX > window.innerWidth / 2 ? current + 1 : current - 1);
});

// Свайп на планшете.
let touchStartX = null;
document.addEventListener('touchstart', (event) => {
  touchStartX = event.changedTouches[0].clientX;
}, { passive: true });
document.addEventListener('touchend', (event) => {
  if (touchStartX === null) return;
  const delta = event.changedTouches[0].clientX - touchStartX;
  if (Math.abs(delta) > 40) show(current + (delta < 0 ? 1 : -1));
  touchStartX = null;
});

fit();
show(Number(location.hash.replace('#', '')) - 1 || 0);
