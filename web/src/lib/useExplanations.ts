/** Объяснения периодов снижения от языковой модели.
 *
 *  Анализ отдаёт текст по правилам сразу, а модель пишет свой вариант десятки секунд.
 *  Поэтому текст догружается отдельно: пока статус pending, запрос повторяется,
 *  а как только ответ готов, карточки эпизодов подменяют текст.
 */

import { useQuery } from "@tanstack/react-query";

import { api } from "../api/client";
import type { Episode } from "../api/types";

/** Ключ эпизода такой же, как на сервере: год и границы периода. */
export function episodeKey(episode: Episode): string {
  return `${episode.year}:${episode.start}:${episode.end}`;
}

export function useExplanations(pid: string, year?: number) {
  const query = useQuery({
    // объясняем только тот сезон, который открыт: остальные пользователь сейчас не видит
    queryKey: ["explanations", pid, year],
    queryFn: () => api.explanations(pid, year),
    enabled: Boolean(pid),
    // пока модель пишет — переспрашиваем; как только готово, опрос прекращается
    refetchInterval: (query) => (query.state.data?.status === "pending" ? 4000 : false),
    staleTime: Infinity,
    retry: false,
  });

  return {
    pending: query.data?.status === "pending",
    /** Текст модели для эпизода или null, если его ещё нет. */
    textFor: (episode: Episode): string | null => query.data?.items?.[episodeKey(episode)]?.text ?? null,
  };
}
