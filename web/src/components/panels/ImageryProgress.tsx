/** Этапы и реальные счётчики. Если сервер не сообщает прогресс, не имитируем загрузку. */
import { useEffect, useState } from "react";
import type { ImageryProgress as Progress } from "../../api/analytics";
import { shortDate } from "../../lib/format";

const STAGES = ["Поиск", "Проверка снимков", "Подготовка карты"];
const STEP = { queued: -1, catalog: 0, download: 1, render: 2, done: 3, error: -1 };
const elapsedLabel = (seconds: number) => `${Math.floor(seconds / 60)}:${String(seconds % 60).padStart(2, "0")}`;

export function ImageryProgress({ progress, year, requestedAt, supported, connectionError }: {
  progress?: Progress | null; year: number; requestedAt?: number; supported: boolean; connectionError: boolean;
}) {
  const [now, setNow] = useState(Date.now);
  useEffect(() => { const timer = window.setInterval(() => setNow(Date.now()), 1000); return () => clearInterval(timer); }, []);
  const started = progress ? Date.parse(progress.started) : requestedAt;
  const elapsed = started ? Math.max(0, Math.floor((now - started) / 1000)) : null;
  const quiet = progress ? Math.max(0, Math.floor((now - Date.parse(progress.updated)) / 1000)) : 0;
  const step = progress ? STEP[progress.stage] : -1;
  const counted = progress && ["download", "render"].includes(progress.stage) && progress.total > 0;
  const failed = progress?.stage === "error";
  return <div className="imagery-progress" data-testid="imagery-progress" data-stage={progress?.stage ?? "unknown"}>
    <div className="imagery-progress-heading"><strong>{failed ? "Не удалось собрать снимки" : `Снимки поля за ${year} год`}</strong>
      {elapsed !== null && !failed && <span className="meta">Прошло {elapsedLabel(elapsed)}</span>}</div>
    {!failed && <ol className="imagery-progress-steps" aria-label="Этапы сбора снимков">
      {STAGES.map((label, i) => <li key={label} data-state={i < step ? "done" : i === step ? "current" : "waiting"}
        aria-current={i === step ? "step" : undefined}><span aria-hidden>{i < step ? "✓" : i + 1}</span>{label}</li>)}
    </ol>}
    <div className="imagery-progress-detail" role={failed ? "alert" : "status"}>
      {failed ? <p>{progress.message || "Попробуйте запустить сбор ещё раз."}</p> : counted ? <>
        <p className="imagery-progress-count">{progress.stage === "download" ? "Проверено" : "Подготовлено"} <strong>{progress.processed} из {progress.total} дат</strong>
          {progress.stage === "download" && <span> · подходят снимки: <strong>{progress.suitable}</strong></span>}</p>
        <div className="imagery-progress-track" role="progressbar" aria-label={STAGES[step]} aria-valuemin={0}
          aria-valuemax={progress.total} aria-valuenow={progress.processed}>
          <span style={{ width: `${Math.min(100, Math.max(0, progress.processed / progress.total * 100))}%` }} />
        </div>
      </> : <p>{progress?.stage === "queued" ? "Ожидаем завершения предыдущей задачи." : progress?.stage === "catalog"
        ? "Ищем доступные даты в каталоге Sentinel-2." : progress?.stage === "done" ? "Снимки готовы. Открываем карту…"
          : supported ? "Ожидаем сведения о загрузке…" : "Сервер не передаёт подробный прогресс. Проверяем готовность снимков."}</p>}
    </div>
    {!failed && progress?.stage === "download" && progress.period_start && progress.period_end && <p className="meta">
      Сейчас: {shortDate(progress.period_start)} — {shortDate(progress.period_end)} · исключаем облачные снимки.</p>}
    {connectionError ? <p className="imagery-progress-notice">Не удалось обновить статус. Повторяем запрос к серверу.</p>
      : !failed && quiet >= 60 && <p className="imagery-progress-notice">{progress?.stage === "queued" ? "Задача пока в очереди."
        : progress?.stage === "render" ? "Подготовка карт занимает больше времени." : "Источник отвечает медленно."} Последнее обновление — {elapsedLabel(quiet)} назад.</p>}
  </div>;
}
