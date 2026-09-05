/** «Светофор поля»: цветная плашка с одной фразой-выводом и ответы на три вопроса фермера.
 *
 *  Как поле сейчас? Что с ним было в сезоне? Что делать? Все ответы собирает `seasonVerdict`
 *  из уже посчитанных данных; графики и таблицы остаются ниже для тех, кому нужны подробности. */
import type { AgroContext, DeviationTrend } from "../../api/analytics";
import type { Episode, SeasonYear } from "../../api/types";
import { seasonVerdict } from "../../lib/plain";

const TONE_ICON = { ok: "●", warn: "●", crit: "●" } as const;

export function FieldVerdict({ season, episodes, trend, weather }: {
  season: SeasonYear; episodes: Episode[]; trend?: DeviationTrend; weather?: AgroContext;
}) {
  const verdict = seasonVerdict(season, episodes, trend, weather);
  return <section className="verdict" data-tone={verdict.tone} data-testid="verdict" aria-label="Состояние поля">
    <div className="verdict-light">
      <span className="verdict-dot" aria-hidden>{TONE_ICON[verdict.tone]}</span>
      <div className="stack" style={{ gap: 2, minWidth: 0 }}>
        <strong className="verdict-title">{verdict.title}</strong>
        <p className="verdict-sentence">{verdict.sentence}</p>
      </div>
    </div>
    <dl className="verdict-questions">
      <div>
        <dt>Как поле сейчас?</dt>
        <dd>{verdict.now}</dd>
      </div>
      <div>
        <dt>Что было в этом сезоне?</dt>
        <dd>{verdict.history.length
          ? <ul>{verdict.history.map((line, index) => <li key={index}>{line}</li>)}</ul>
          : "Ничего необычного: поле шло по своему обычному графику."}</dd>
      </div>
      <div>
        <dt>Что делать?</dt>
        <dd>{verdict.advice}</dd>
      </div>
    </dl>
  </section>;
}
