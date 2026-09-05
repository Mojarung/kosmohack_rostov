/** Как устроено решение: пайплайн, метрики, источники данных. Экран для эксперта и для защиты. */

import { useQuery } from "@tanstack/react-query";

import { api } from "../api/client";
import { CAUSE_LABEL } from "../lib/format";
import { CurveArt, DroughtArt, GapArt, LayersArt, SatelliteArt, SproutArt } from "../components/art/Art";
import { ErrorNote } from "../components/ui/Loader";
import { PagePending } from "../components/ui/PagePending";

const STEPS = [
  {
    art: SatelliteArt,
    title: "Сбор",
    text: "Sentinel-2 L2A из Earth Search, Landsat 8/9 и MODIS MOD13Q1 из Planetary Computer, ERA5 из Open-Meteo. Снимки обрезаются по контуру, облака и тени убираются масками SCL и qa_pixel.",
  },
  {
    art: LayersArt,
    title: "Один ряд",
    text: "primary_ndvi собирается по приоритету Sentinel-2 → Landsat → MODIS. Сенсоры смещены между собой (Landsat +0.037, MODIS +0.083), поэтому перед анализом ряд приводится к шкале Sentinel-2.",
  },
  {
    art: GapArt,
    title: "Восстановление",
    text: "Пропуски заполняет смесь двух моделей: градиентный бустинг на 243 признаках ряда и нейросеть по сезонной сетке. Обе учатся на искусственных пропусках в известных точках.",
  },
  {
    art: CurveArt,
    title: "Норма и эпизоды",
    text: "Норма считается по другим годам того же поля (или по полям той же культуры). Эпизод — устойчивое или сильное отклонение ниже нормы, а не единичная точка.",
  },
  {
    art: DroughtArt,
    title: "Причина",
    text: "Правила сопоставляют эпизод с погодой тех же дат, фенологией сезона и поведением соседних полей: засуха, незасеянное поле, севооборот, ранний спад, ошибка данных.",
  },
  {
    art: SproutArt,
    title: "Проверка",
    text: "Экспертной разметки нет, поэтому качество проверяется косвенно: совпадение с точечными статусами организаторов, независимость от сенсора, ранжирование засушливых сезонов, устойчивость к скрытию наблюдений.",
  },
];

export default function MethodPage() {
  const meta = useQuery({ queryKey: ["meta"], queryFn: api.meta });

  if (meta.isLoading) return <PagePending label="Собираем сводку" />;
  if (meta.isError) return <ErrorNote error={meta.error} />;
  if (!meta.data) return null;
  const data = meta.data;

  const facts = [
    { label: "RMSE восстановления", value: data.task1.rmse_val.toFixed(3), note: `baseline ${data.task1.baseline_rmse.toFixed(3)}` },
    { label: "GapScore", value: data.task1.gap_score.toFixed(1), note: "из 30" },
    { label: "Контрольных точек", value: data.task1.n_gaps.toLocaleString("ru-RU"), note: data.data.test_file },
    { label: "Наблюдений в обучении", value: data.task1.train_points.toLocaleString("ru-RU"), note: "снимки трёх спутников" },
    { label: "Эпизодов найдено", value: data.task2.n_episodes.toLocaleString("ru-RU"), note: `в ${data.task2.n_seasons} сезонах` },
    { label: "Сезонов с эпизодом", value: `${Math.round(data.task2.seasons_with_episode * 100)} %`, note: `из ${data.task2.n_seasons} сезонов ${data.task2.n_polygons} полей` },
  ];

  return (
    <div className="workspace workspace--method">
      <section className="pane">
        <div className="pane-head">
          <span className="row" style={{ gap: 10 }}>
            <span className="pane-title">Как это работает</span>
            <span className="meta">
              пайплайн одинаков для полей кейса и для произвольной территории
            </span>
          </span>
        </div>

        <div className="pane-body">
          <div className="method-grid">
            {STEPS.map((step, index) => {
              const Art = step.art;
              return (
                <article key={step.title} className="method-card">
                  <Art size={32} />
                  <div className="row" style={{ gap: 8 }}>
                    <span className="mono meta" style={{ fontSize: 11 }}>
                      {String(index + 1).padStart(2, "0")}
                    </span>
                    <span className="pane-title" style={{ fontSize: 15 }}>
                      {step.title}
                    </span>
                  </div>
                  <p>{step.text}</p>
                </article>
              );
            })}
          </div>

          <div className="method-split">
            <div>
              <div className="eyebrow" style={{ marginBottom: 8 }}>Метрики решения</div>
              <div className="method-facts">
                {facts.map((fact) => (
                  <div key={fact.label} className="kpi">
                    <span className="eyebrow">{fact.label}</span>
                    <span className="kpi-value">{fact.value}</span>
                    <span className="kpi-note">{fact.note}</span>
                  </div>
                ))}
              </div>
            </div>

            <div>
              <div className="eyebrow" style={{ marginBottom: 8 }}>Причины эпизодов</div>
              <div className="stack" style={{ gap: 4 }}>
                {Object.entries(data.task2.by_cause)
                  .sort((a, b) => b[1] - a[1])
                  .map(([cause, value]) => (
                    <div key={cause} className="spread" style={{ fontSize: 12.5 }}>
                      <span style={{ color: "var(--ink-soft)" }}>{CAUSE_LABEL[cause] ?? cause}</span>
                      <span className="num meta">{value}</span>
                    </div>
                  ))}
              </div>

              <div className="eyebrow" style={{ margin: "16px 0 8px" }}>Источники данных</div>
              <div className="stack" style={{ gap: 5 }}>
                {data.sources.map((source) => (
                  <div key={source.name} style={{ fontSize: 12.5 }}>
                    <span style={{ color: "var(--ink-soft)" }}>{source.name}</span>{" "}
                    <span className="meta">{source.detail}</span>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </div>
      </section>
    </div>
  );
}
