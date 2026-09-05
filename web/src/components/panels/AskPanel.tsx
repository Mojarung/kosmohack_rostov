/** Вопрос о поле своими словами. Ответ строится по тем же числам, что показаны на экране:
 *  с ключом модели — связным текстом, без ключа — разбором по правилам. */

import { useState } from "react";
import { useMutation } from "@tanstack/react-query";

import { api } from "../../api/client";
import type { AskAnswer } from "../../api/types";

/** Подсказки: показывают, о чём вообще можно спросить. */
const HINTS = ["Что самое плохое было с полем?", "Почему упала зелёность?", "Какая была погода?", "Можно ли верить цифрам?"];

export function AskPanel({ pid, year }: { pid: string; year?: number }) {
  const [question, setQuestion] = useState("");
  const [history, setHistory] = useState<{ question: string; answer: AskAnswer }[]>([]);

  const ask = useMutation({
    mutationFn: (text: string) => api.ask({ pid, question: text, year }),
    onSuccess: (answer, text) => {
      setHistory((items) => [...items, { question: text, answer }]);
      setQuestion("");
    },
  });

  const send = (text: string) => {
    const trimmed = text.trim();
    if (trimmed.length > 1 && !ask.isPending) ask.mutate(trimmed);
  };

  return (
    <section className="pane ask-pane">
      <div className="pane-head">
        <span className="pane-title">Спросить про поле</span>
        <a className="btn btn--sm btn--ghost" href={api.reportUrl(pid, year)} target="_blank" rel="noreferrer">
          Отчёт для печати
        </a>
      </div>

      <div className="pane-body pane-body--pad stack" style={{ gap: 10 }}>
        {history.length === 0 && !ask.isPending && (
          <div className="chips">
            {HINTS.map((hint) => (
              <button key={hint} type="button" className="chip" onClick={() => send(hint)}>
                {hint}
              </button>
            ))}
          </div>
        )}

        {history.map((item, index) => (
          <div key={index} className="ask-item">
            <p className="ask-question">{item.question}</p>
            <p className="ask-answer">{item.answer.answer}</p>
            <p className="meta ask-source">
              {item.answer.source === "llm" ? "ответ модели по фактам поля" : "ответ по правилам, без модели"}
              {item.answer.tools_used.length > 0 && ` · запрошено: ${item.answer.tools_used.join(", ")}`}
            </p>
          </div>
        ))}

        {ask.isPending && <p className="meta">Смотрю данные поля…</p>}
        {ask.isError && <p className="meta">Не удалось ответить: {(ask.error as Error).message}</p>}
      </div>

      <form
        className="ask-form"
        onSubmit={(event) => {
          event.preventDefault();
          send(question);
        }}
      >
        <input
          className="input"
          value={question}
          onChange={(event) => setQuestion(event.target.value)}
          placeholder="например: что было в 2020 году?"
          maxLength={500}
        />
        <button type="submit" className="btn btn--sm" disabled={question.trim().length < 2 || ask.isPending}>
          Спросить
        </button>
      </form>
    </section>
  );
}
