/* Фронтенд сервиса NDVI-мониторинга: карта, запуск анализа, отрисовка результата. */
'use strict';

const $ = (s, r = document) => r.querySelector(s);
const css = (n) => getComputedStyle(document.documentElement).getPropertyValue(n).trim();
const SENSOR_COLOR = () => ({ s2: css('--s2'), landsat: css('--ls'), modis: css('--mo'), none: css('--ink-3') });
const SENSOR_NAME = { s2: 'Sentinel-2', landsat: 'Landsat', modis: 'MODIS', none: 'неизвестен' };

const state = { selected: null, map: null, drawn: null, layers: {}, running: false, saved: [] };

/* цвет контура по роли поля: на чём учимся, что предсказываем */
const ROLE_COLOR = () => ({ train: css('--accent'), both: css('--warn'), predict: css('--gap') });
const ROLE_RU = { train: 'обучение', both: 'обучение и предсказание', predict: 'предсказание' };

/* ------------------------------------------------------------- вкладки -- */
document.querySelectorAll('.tabs button').forEach(b => b.addEventListener('click', () => {
  document.querySelectorAll('.tabs button').forEach(x => x.setAttribute('aria-selected', String(x === b)));
  document.querySelectorAll('.view').forEach(v => v.classList.toggle('active', v.id === 'view-' + b.dataset.view));
  if (b.dataset.view === 'map' && state.map) setTimeout(() => state.map.invalidateSize(), 60);
  if (b.dataset.view === 'demo') loadDemoList();
}));

/* --------------------------------------------------------------- карта -- */
function initMap() {
  const map = L.map('map', { center: [46.85, 40.30], zoom: 11 });
  L.tileLayer('https://tile.openstreetmap.org/{z}/{x}/{y}.png', {
    maxZoom: 18, attribution: '© OpenStreetMap'
  }).addTo(map);
  const drawn = new L.FeatureGroup().addTo(map);
  map.addControl(new L.Control.Draw({
    edit: { featureGroup: drawn },
    draw: { polyline: false, circle: false, circlemarker: false, marker: false, rectangle: true, polygon: true }
  }));
  map.on(L.Draw.Event.CREATED, e => {
    drawn.clearLayers();
    drawn.addLayer(e.layer);
    select({ id: null, name: 'Нарисованное поле', geometry: e.layer.toGeoJSON().geometry }, 'draw');
  });
  state.map = map; state.drawn = drawn;
  // кандидаты из OSM — нейтральным серым, чтобы не спорить с цветами ролей
  state.layers.fields = L.geoJSON(null, {
    style: { color: css('--ink-3'), weight: 1.2, fillOpacity: .08 },
    onEachFeature: (f, l) => l.on('click', () => select(f.properties, 'osm', l))
  }).addTo(map);
  state.layers.saved = L.layerGroup().addTo(map);      // мои поля, цвет по роли
}

/* Перерисовывает сохранённые поля цветом по роли. */
function drawSaved(items) {
  const C = ROLE_COLOR();
  state.layers.saved.clearLayers();
  items.forEach(p => {
    const col = C[p.role] || C.predict;
    L.geoJSON(p.geometry, { style: { color: col, weight: 2.5, fillOpacity: .18 } })
      .bindTooltip(`${p.name} — ${ROLE_RU[p.role] || p.role}`)
      .addTo(state.layers.saved);
  });
}

function select(item, source, layer) {
  state.selected = { ...item, source };
  document.querySelectorAll('#fields .item').forEach(el =>
    el.setAttribute('aria-current', String(el.dataset.id === item.id)));
  if (state.layers.selected) state.map.removeLayer(state.layers.selected);
  if (source !== 'draw') {
    state.layers.selected = L.geoJSON(item.geometry,
      { style: { color: css('--crit'), weight: 3, fillOpacity: .2 } }).addTo(state.map);
    state.map.fitBounds(state.layers.selected.getBounds(), { maxZoom: 15 });
  }
  const btn = $('#btn-run');
  btn.disabled = false;
  btn.textContent = `Анализировать: ${item.name || 'поле'}`;
  const saved = state.saved.find(p => p.id === item.id);
  if (saved) $('#role').value = saved.role;
}

/* ------------------------------------------------------------- запросы -- */
async function api(url, opts) {
  const r = await fetch(url, opts);
  if (!r.ok) throw new Error(`${r.status} ${await r.text()}`);
  return r.json();
}

async function checkHealth() {
  const el = $('#health');
  try {
    const h = await api('/api/health');
    const bad = Object.entries(h.sources).filter(([, v]) => v !== true).map(([k]) => k);
    const modelBad = h.model !== true;
    el.innerHTML = `<span class="dot ${bad.length || modelBad ? 'off' : 'on'}"></span>` +
      (modelBad ? 'модель не обучена' : bad.length ? `недоступно: ${bad.join(', ')}` : 'источники доступны');
  } catch { el.innerHTML = '<span class="dot off"></span> сервис недоступен'; }
}

$('#btn-search').addEventListener('click', async () => {
  const box = $('#regions'); box.innerHTML = '<div class="item meta">ищу…</div>';
  try {
    const d = await api('/api/regions?q=' + encodeURIComponent($('#q').value));
    box.innerHTML = d.items.map((r, i) =>
      `<button class="item" data-i="${i}"><div>${r.name.split(',').slice(0, 3).join(',')}</div>
       <div class="meta">${r.type}</div></button>`).join('') || '<div class="item meta">ничего не найдено</div>';
    box.querySelectorAll('button').forEach(b => b.addEventListener('click', () => {
      const r = d.items[+b.dataset.i];
      state.map.fitBounds([[r.bbox[1], r.bbox[0]], [r.bbox[3], r.bbox[2]]]);
    }));
  } catch (e) { box.innerHTML = `<div class="item meta">ошибка: ${e.message}</div>`; }
});

$('#btn-fields').addEventListener('click', async () => {
  const box = $('#fields'); box.innerHTML = '<div class="item meta">запрашиваю OpenStreetMap…</div>';
  const b = state.map.getBounds();
  try {
    const d = await api(`/api/fields?minx=${b.getWest()}&miny=${b.getSouth()}&maxx=${b.getEast()}&maxy=${b.getNorth()}`);
    state.layers.fields.clearLayers();
    if (!d.items.length) {
      box.innerHTML = `<div class="item meta">${d.error ? 'источник недоступен' : 'в этой области нет размеченных полей — подвиньте карту или нарисуйте контур'}</div>`;
      return;
    }
    state.layers.fields.addData(d.items.map(f => ({
      type: 'Feature', geometry: f.geometry,
      properties: { id: f.id, name: f.name, geometry: f.geometry, area_ha: f.area_ha }
    })));
    box.innerHTML = d.items.map(f =>
      `<button class="item" data-id="${f.id}"><div>${f.name}</div>
       <div class="meta">${f.area_ha} га · ${f.landuse}</div></button>`).join('');
    box.querySelectorAll('button').forEach(btn => btn.addEventListener('click', () =>
      select(d.items.find(x => x.id === btn.dataset.id), 'osm')));
  } catch (e) { box.innerHTML = `<div class="item meta">ошибка: ${e.message}</div>`; }
});

$('#btn-run').addEventListener('click', async () => {
  if (!state.selected || state.running) return;
  const btn = $('#btn-run'); const target = $('#result-map');
  state.running = true;
  btn.disabled = true; btn.innerHTML = '<span class="spinner"></span> собираю данные…';
  target.innerHTML = '<div class="card"><div class="body">Идёт сбор спутниковых снимков по полигону. ' +
    'Для длинного периода это может занять до минуты.</div></div>';
  try {
    const out = await api('/api/analyze', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        geometry: state.selected.geometry, start: $('#start').value, end: $('#end').value,
        crop_type: $('#crop').value, name: state.selected.name,
        polygon_id: state.selected.id || undefined
      })
    });
    render(out, target);
    await api('/api/polygons', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        id: state.selected.id || out.polygon_id, name: state.selected.name,
        geometry: state.selected.geometry, crop_type: $('#crop').value,
        source: state.selected.source, role: $('#role').value
      })
    }).then(loadSaved).catch(() => {});
  } catch (e) {
    target.innerHTML = `<div class="card"><div class="body">Не получилось: ${e.message}</div></div>`;
  } finally {
    state.running = false; btn.disabled = false;
    btn.textContent = `Анализировать: ${state.selected.name || 'поле'}`;
  }
});

async function loadSaved() {
  try {
    const d = await api('/api/polygons');
    state.saved = d.items;
    drawSaved(d.items);
    $('#saved').innerHTML = d.items.map(p =>
      `<button class="item" data-id="${p.id}"><div>${p.name}
         <span class="rolechip ${p.role}">${ROLE_RU[p.role] || p.role}</span></div>
       <div class="meta">${p.crop_type} · ${(p.area_ha || 0).toFixed(0)} га · нажмите дважды, чтобы сменить роль</div></button>`).join('')
      || '<div class="item meta">пока пусто</div>';
    $('#saved').querySelectorAll('button').forEach(b => {
      const item = d.items.find(x => x.id === b.dataset.id);
      b.addEventListener('click', () => select(item, 'saved'));
      // двойной клик переключает роль по кругу: предсказание -> обучение -> обе
      b.addEventListener('dblclick', async () => {
        const next = { predict: 'train', train: 'both', both: 'predict' }[item.role] || 'train';
        await api(`/api/polygons/${item.id}/role`, {
          method: 'PATCH', headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ role: next })
        });
        $('#role').value = next;
        loadSaved();
      });
    });
  } catch { /* пустой список не критичен */ }
}

/* Демо-набор: реальные поля под Зерноградом, разложенные на эталонные и проверяемые. */
$('#btn-seed').addEventListener('click', async () => {
  const btn = $('#btn-seed');
  const label = btn.textContent;
  btn.disabled = true;
  btn.innerHTML = '<span class="spinner"></span> запрашиваю OpenStreetMap…';
  try {
    const d = await api('/api/demo/seed', { method: 'POST' });
    await loadSaved();
    fitSaved();
    btn.textContent = `Готово: ${d.n_train} эталонных, ${d.n_predict} проверяемых`;
  } catch (e) {
    btn.textContent = 'Не удалось: ' + e.message;
  } finally {
    btn.disabled = false;
    setTimeout(() => { btn.textContent = label; }, 6000);
  }
});

/* Показывает на карте все сохранённые поля целиком. */
function fitSaved() {
  const layers = state.layers.saved.getLayers();
  if (!layers.length) return;
  let b = null;
  layers.forEach(l => { b = b ? b.extend(l.getBounds()) : L.latLngBounds(l.getBounds()); });
  state.map.fitBounds(b.pad(0.25));
}

async function loadDemoList() {
  const box = $('#demo-list');
  if (box.dataset.loaded) return;
  box.innerHTML = '<div class="item meta">загружаю…</div>';
  try {
    const d = await api('/api/demo/polygons');
    box.dataset.loaded = '1';
    box.innerHTML = d.items.map(p =>
      `<button class="item" data-id="${p.id}"><div>${p.id}
         <span class="rolechip ${p.role}">${p.role_ru}</span></div>
       <div class="meta">${p.crop_type} · ${p.years[0]}–${p.years[1]} · ${p.n_observations} наблюдений</div></button>`).join('');
    box.querySelectorAll('button').forEach(b => b.addEventListener('click', async () => {
      box.querySelectorAll('.item').forEach(x => x.setAttribute('aria-current', String(x === b)));
      const t = $('#result-demo');
      t.innerHTML = '<div class="card"><div class="body">считаю…</div></div>';
      try { render(await api('/api/demo/' + b.dataset.id), t); }
      catch (e) { t.innerHTML = `<div class="card"><div class="body">ошибка: ${e.message}</div></div>`; }
    }));
  } catch (e) { box.innerHTML = `<div class="item meta">ошибка: ${e.message}</div>`; }
}

/* ---------------------------------------------------------- отрисовка -- */
function render(out, target) {
  if (!out.ok) {
    target.innerHTML = `<div class="card"><div class="body">${out.error || 'нет данных'}</div>
      ${srcHtml(out.sources)}</div>`;
    return;
  }
  const stress = out.episodes.filter(e => e.kind === 'стресс');
  const crit = stress.filter(e => e.severity === 'критическая');
  const ids = { plot: 'p' + Math.random().toString(36).slice(2), w: 'w' + Math.random().toString(36).slice(2) };

  target.innerHTML = `
    <div class="stats">
      <div class="stat"><div class="k">наблюдений</div><div class="v">${out.n_observations}</div></div>
      <div class="stat"><div class="k">восстановлено</div><div class="v">${out.n_filled}</div></div>
      <div class="stat"><div class="k">эпизодов угнетения</div><div class="v">${stress.length}</div></div>
      <div class="stat"><div class="k">из них критических</div><div class="v">${crit.length}</div></div>
      ${out.area_ha ? `<div class="stat"><div class="k">площадь</div><div class="v">${out.area_ha} га</div></div>` : ''}
      ${out.reference_fields ? `<div class="stat"><div class="k">эталонных полей в норме</div><div class="v">${out.reference_fields}</div></div>` : ''}
    </div>
    <div class="card" style="margin-bottom:14px">
      <h2>${out.name || out.polygon_id} · NDVI и климатическая норма</h2>
      <div class="body">
        <div class="chart" id="${ids.plot}"></div>
        <div class="chart small" id="${ids.w}"></div>
        <div class="srcs" style="margin-top:10px">${srcBadges(out.sources)}</div>
        <div class="hint">Смещения сенсоров к шкале Sentinel-2, оценённые по этому полю:
          Landsat ${fmt(out.sensor_offsets.landsat)}, MODIS ${fmt(out.sensor_offsets.modis)}.
          Все точки на графике показаны уже в приведённой шкале, исходное значение — в подсказке.
          <a href="/api/export/${out.polygon_id}.csv">Скачать ряд в CSV</a></div>
      </div>
    </div>
    <div class="card">
      <h2>Аномальные эпизоды</h2>
      <div class="body">${episodesHtml(out.episodes)}</div>
    </div>`;

  drawSeries(ids.plot, out);
  drawWeather(ids.w, out);
}

const fmt = v => (v == null ? '—' : (v > 0 ? '+' : '') + v.toFixed(3));

function drawSeries(id, out) {
  const S = SENSOR_COLOR();
  const s = out.series;
  const obs = s.filter(p => !p.filled && p.ndvi != null);
  const fil = s.filter(p => p.filled && p.ndvi != null);
  const norm = s.filter(p => p.norm != null);
  const traces = [];

  // коридор нормы ±1σ
  traces.push({
    x: norm.map(p => p.date).concat(norm.map(p => p.date).reverse()),
    y: norm.map(p => p.norm_hi).concat(norm.map(p => p.norm_lo).reverse()),
    fill: 'toself', fillcolor: css('--norm-fill'), line: { width: 0 },
    hoverinfo: 'skip', name: 'норма ±1σ', showlegend: true, type: 'scatter'
  });
  traces.push({
    x: norm.map(p => p.date), y: norm.map(p => p.norm), mode: 'lines',
    line: { color: css('--norm'), width: 1, dash: 'dot' }, name: 'норма поля', hoverinfo: 'skip'
  });

  // всё показываем в единой шкале Sentinel-2: иначе точки MODIS систематически
  // висят выше коридора нормы просто потому, что этот сенсор читает выше
  for (const k of ['s2', 'landsat', 'modis']) {
    const pts = obs.filter(p => p.sensor === k && p.ndvi_s2 != null);
    if (!pts.length) continue;
    traces.push({
      x: pts.map(p => p.date), y: pts.map(p => p.ndvi_s2), mode: 'markers',
      marker: { color: S[k], size: 6 }, name: SENSOR_NAME[k],
      customdata: pts.map(p => p.ndvi),
      hovertemplate: '%{x}<br>NDVI %{y:.3f} (в шкале S2)<br>исходное %{customdata:.3f}' +
        '<extra>' + SENSOR_NAME[k] + '</extra>'
    });
  }
  if (fil.length) traces.push({
    x: fil.map(p => p.date), y: fil.map(p => p.ndvi_s2), mode: 'markers',
    marker: { color: css('--gap'), size: 5, symbol: 'diamond-open', line: { width: 1.2 } },
    name: 'восстановлено моделью',
    hovertemplate: '%{x}<br>NDVI %{y:.3f} (оценка)<extra></extra>'
  });

  const shapes = out.episodes.map(e => ({
    type: 'rect', xref: 'x', yref: 'paper', x0: e.start, x1: e.end, y0: 0, y1: 1,
    fillcolor: e.kind !== 'стресс' ? css('--panel-2')
      : e.severity === 'критическая' ? css('--crit') : css('--warn'),
    opacity: e.kind !== 'стресс' ? .35 : .16, line: { width: 0 }, layer: 'below'
  }));

  Plotly.newPlot(id, traces, {
    margin: { l: 44, r: 12, t: 8, b: 34 }, shapes,
    paper_bgcolor: 'transparent', plot_bgcolor: 'transparent',
    font: { family: 'IBM Plex Sans, system-ui', color: css('--ink-2'), size: 11 },
    xaxis: { gridcolor: css('--line-soft'), zeroline: false },
    yaxis: { title: 'NDVI (шкала Sentinel-2)', gridcolor: css('--line-soft'), zeroline: false, range: [-0.25, 1.02] },
    legend: { orientation: 'h', y: 1.14, x: 0 }, hovermode: 'closest'
  }, { displayModeBar: false, responsive: true });
}

function drawWeather(id, out) {
  const w = out.weather;
  if (!w.length) { $('#' + id).innerHTML = '<div class="hint">погодные данные недоступны</div>'; return; }
  // помесячная агрегация: суточный ряд за несколько лет нечитаем
  const byMonth = {};
  for (const d of w) {
    const k = d.date.slice(0, 7);
    (byMonth[k] ||= { p: 0, t: [], n: 0 });
    byMonth[k].p += d.precip || 0;
    if (d.temp != null) byMonth[k].t.push(d.temp);
  }
  const keys = Object.keys(byMonth).sort();
  Plotly.newPlot(id, [
    { x: keys, y: keys.map(k => byMonth[k].p), type: 'bar', name: 'осадки, мм/мес',
      marker: { color: css('--s2'), opacity: .55 } },
    { x: keys, y: keys.map(k => byMonth[k].t.reduce((a, b) => a + b, 0) / (byMonth[k].t.length || 1)),
      type: 'scatter', mode: 'lines', name: 'температура, °C', yaxis: 'y2',
      line: { color: css('--ls'), width: 1.5 } }
  ], {
    margin: { l: 44, r: 44, t: 6, b: 30 }, paper_bgcolor: 'transparent', plot_bgcolor: 'transparent',
    font: { family: 'IBM Plex Sans, system-ui', color: css('--ink-2'), size: 11 },
    xaxis: { gridcolor: css('--line-soft') },
    yaxis: { title: 'мм', gridcolor: css('--line-soft') },
    yaxis2: { overlaying: 'y', side: 'right', title: '°C', showgrid: false },
    legend: { orientation: 'h', y: 1.2, x: 0 }, bargap: .2
  }, { displayModeBar: false, responsive: true });
}

function episodesHtml(eps) {
  if (!eps.length) return '<div class="hint">Устойчивых отклонений от нормы не найдено.</div>';
  const order = { 'стресс': 0, 'уборка': 1, 'низкое качество данных': 2 };
  return [...eps].sort((a, b) => (order[a.kind] - order[b.kind]) || (a.start < b.start ? 1 : -1))
    .map(e => {
      const cls = e.kind !== 'стресс' ? (e.kind === 'уборка' ? 'harvest' : 'noise')
        : e.severity === 'критическая' ? 'crit' : 'warn';
      const tag = e.kind !== 'стресс' ? `<span class="tag">${e.kind}</span>`
        : `<span class="tag ${e.severity === 'критическая' ? 'crit' : 'warn'}">${e.severity}</span>`;
      return `<div class="ep ${cls}">
        <div class="h"><b>${e.start} — ${e.end}</b>${tag}
          <span class="tag">уверенность ${e.confidence}</span>
          ${(e.drivers || []).map(d => `<span class="tag">${d}</span>`).join('')}</div>
        <p>${e.explanation}</p></div>`;
    }).join('');
}

function srcBadges(sources) {
  if (!sources) return '';
  return Object.entries(sources).map(([k, v]) =>
    `<span><span class="dot ${v.ok ? 'on' : 'off'}"></span>${k}${v.ok ? ` · ${v.n}${v.cached ? ' (кэш)' : ''}` : ` · ${v.error || 'нет данных'}`}</span>`).join('');
}
const srcHtml = s => `<div class="body"><div class="srcs">${srcBadges(s)}</div></div>`;

initMap();
checkHealth();
loadSaved().then(fitSaved);
