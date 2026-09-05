(() => {
  let pid = null, manifest = null, index = "ndvi";
  const $ = id => document.getElementById(id);
  const draw = () => {
    if (!manifest) return;
    const date = $("imagery-date").value;
    const scene = manifest.scenes.find(s => s.date === date);
    const image = $("imagery-image");
    image.src = `/api/polygon/${encodeURIComponent(pid)}/imagery/${manifest.year}/${date}/${index}.png`;
    image.hidden = false;
    $("imagery-status").textContent = `${date} · ${index === "ndmi" ? "влажность растительности" : index === "change" ? "разница с предыдущим снимком" : "состояние растительности"}`;
    $("imagery-note").textContent = scene ? `Чистое покрытие: ${Math.round(scene[index]?.clear_share * 100 || 0)}% · среднее ${scene[index]?.mean ?? "—"}` : "";
  };
  window.renderImagery = async (field, year) => {
    pid = field; manifest = null; $("imagery-card").hidden = true;
    try {
      const response = await fetch(`/api/polygon/${encodeURIComponent(field)}/imagery?year=${year}`);
      const data = await response.json();
      if (!data.available) return;
      manifest = data.manifest; manifest.year = year;
      const dates = manifest.scenes.filter(s => s.ndvi).map(s => s.date);
      $("imagery-date").innerHTML = dates.map(d => `<option value="${d}">${d}</option>`).join("");
      $("imagery-card").hidden = false; draw();
    } catch { /* карта остаётся скрытой при недоступном источнике */ }
  };
  $("imagery-date").onchange = draw;
  $("imagery-load").onclick = () => $("imagery-status").textContent = "Снимки загружаются при создании географического поля.";
  document.querySelectorAll("[data-image]").forEach(b => b.onclick = () => { index = b.dataset.image; document.querySelectorAll("[data-image]").forEach(x => x.setAttribute("aria-pressed", String(x === b))); draw(); });
})();
