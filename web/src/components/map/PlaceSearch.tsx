/** Ручной поиск места: отменяем запрос при изменении текста и не показываем старые ответы. */
import { useEffect, useRef, useState } from "react";
import { api } from "../../api/client";
import type { PlaceResult } from "../../api/types";
import "./place-search.css";

export function PlaceSearch({ onSelect }: { onSelect: (place: PlaceResult) => void }) {
  const [query, setQuery] = useState("");
  const [results, setResults] = useState<PlaceResult[]>([]);
  const [status, setStatus] = useState("");
  const [busy, setBusy] = useState(false);
  const controller = useRef<AbortController | null>(null);
  useEffect(() => () => controller.current?.abort(), []);

  function change(value: string) {
    controller.current?.abort();
    setQuery(value);
    setResults([]);
    setStatus("");
    setBusy(false);
  }

  async function search() {
    controller.current?.abort();
    const current = new AbortController();
    controller.current = current;
    setBusy(true);
    setResults([]);
    setStatus("Ищем место…");
    try {
      const places = await api.places(query.trim(), current.signal);
      if (current.signal.aborted) return;
      setResults(places);
      setStatus(places.length ? "Выберите место из списка" : "Ничего не найдено. Уточните город, улицу или адрес.");
    } catch (error) {
      if (!current.signal.aborted) setStatus(error instanceof Error ? error.message : "Не удалось найти место");
    } finally {
      if (!current.signal.aborted) setBusy(false);
    }
  }

  return <div className="place-search" role="search" aria-label="Поиск места на карте">
    <form onSubmit={(event) => { event.preventDefault(); if (query.trim().length >= 2 && !busy) void search(); }}>
      <label htmlFor="place-query">Найти место</label>
      <div className="place-search-row">
        <input id="place-query" value={query} onChange={(event) => change(event.target.value)}
          placeholder="Город, улица, адрес или координаты" maxLength={200} autoComplete="off" />
        <button type="submit" className="btn btn--sm" disabled={busy || query.trim().length < 2}>Найти</button>
        {query && <button type="button" className="place-clear" aria-label="Очистить поиск" onClick={() => change("")}>×</button>}
      </div>
    </form>
    <div className="place-search-feedback" role="status" aria-live="polite">{status}</div>
    {results.length > 0 && <ul className="place-results" aria-label="Найденные места">
      {results.map((place) => <li key={place.id}><button type="button" onClick={() => {
        onSelect(place); setResults([]); setStatus(`На карте: ${place.label}`);
      }}>{place.label}</button></li>)}
    </ul>}
    <div className="place-search-credit">Координаты: широта, долгота · <a href="https://www.openstreetmap.org/copyright"
      target="_blank" rel="noreferrer">© OpenStreetMap contributors</a></div>
  </div>;
}
