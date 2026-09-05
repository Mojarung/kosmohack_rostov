/** Как устроено решение: пайплайн, метрики, источники данных. Экран для эксперта и для защиты. */

import { useQuery } from "@tanstack/react-query";

import { api } from "../api/client";
import { CAUSE_LABEL } from "../lib/format";
import { CurveArt, DroughtArt, GapArt, LayersArt, SatelliteArt, SproutArt } from "../components/art/Art";
import { Reveal } from "../components/motion/Reveal";
import { ErrorNote, Loader } from "../components/ui/Loader";

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

  if (meta.isLoading) return <Loader label="Собираем сводку" />;
  if (meta.isError) return <ErrorNote error={meta.error} />;
  if (!meta.data) return null;
  const data = meta.data;

  return (
    <div className="stack" style={{ gap: 22 }}>
      <div className="stack" style={{ gap: 6 }}>
        <span className="eyebrow">Метод</span>
        <h1 style={{ fontSize: 34 }}>Как это работает</h1>
        <p className="meta" style={{ maxWidth: "72ch" }}>
          Пайплайн одинаков для полей кейса и для произвольной территории: разница только в том, что для новых
          полей данные сначала собираются из открытых каталогов.
        </p>
      </div>

      <div style={{ display: "grid", gap: 14, gridTemplateColumns: "repeat(auto-fit, minmax(260px, 1fr))" }}>
        {STEPS.map((step, index) => {
          const Art = step.art;
          return (
            <Reveal key={step.title} index={index}>
              <div className="card stack" style={{ gap: 10, height: "100%" }}>
                <Art size={38} />
                <div className="row" style={{ gap: 8 }}>
                  <span className="mono meta">{String(index + 1).padStart(2, "0")}</span>
                  <h3>{step.title}</h3>
                </div>
                <p style={{ color: "var(--ink-soft)", fontSize: 13.5 }}>{step.text}</p>
              </div>
            </Reveal>
          );
        })}
      </div>

      <Reveal>
        <section className="card">
          <h2 style={{ marginBottom: 14 }}>Метрика восстановления пропусков</h2>
          <div style={{ display: "grid", gap: 18, gridTemplateColumns: "repeat(auto-fit, minmax(200px, 1fr))" }}>
            <div>
              <div className="eyebrow">RMSE на валидации</div>
              <div className="num" style={{ fontSize: 28 }}>
                {data.task1.rmse_val.toFixed(4)}
              </div>
              <p className="meta">15 % известных точек прячутся так же, как контрольные в тестовом файле.</p>
            </div>
            <div>
              <div className="eyebrow">RMSE на страте теста</div>
              <div className="num" style={{ fontSize: 28 }}>
                {data.task1.rmse_testlike.toFixed(4)}
              </div>
              <p className="meta">Взвешено под состав контрольных точек: новые поля, история без 2025 года.</p>
            </div>
            <div>
              <div className="eyebrow">GapScore</div>
              <div className="num" style={{ fontSize: 28 }}>
                {data.task1.gap_score.toFixed(1)}
                <span className="meta" style={{ fontSize: 14 }}>
                  {" "}
                  из 30
                </span>
              </div>
              <p className="meta">
                Baseline «среднее двух соседей»: RMSE {data.task1.baseline_rmse.toFixed(3)}, то есть{" "}
                {data.task1.baseline_gap_score.toFixed(1)} балла.
              </p>
            </div>
            <div>
              <div className="eyebrow">Контрольных точек</div>
              <div className="num" style={{ fontSize: 28 }}>
                {data.task1.n_gaps.toLocaleString("ru-RU")}
              </div>
              <p className="meta">
                Файл {data.data.test_file}, {data.data.test_rows.toLocaleString("ru-RU")} строк.
              </p>
            </div>
          </div>
          <hr className="divider" style={{ margin: "16px 0" }} />
          <p style={{ color: "var(--ink-soft)" }}>{data.task1.models}</p>
        </section>
      </Reveal>

      <Reveal>
        <section className="card">
          <h2 style={{ marginBottom: 14 }}>Детекция аномалий</h2>
          <div style={{ display: "grid", gap: 18, gridTemplateColumns: "repeat(auto-fit, minmax(200px, 1fr))" }}>
            <div>
              <div className="eyebrow">Сезонов разобрано</div>
              <div className="num" style={{ fontSize: 28 }}>
                {data.task2.n_seasons}
              </div>
              <p className="meta">{data.task2.n_polygons} полей кейса.</p>
            </div>
            <div>
              <div className="eyebrow">Сезонов с эпизодом</div>
              <div className="num" style={{ fontSize: 28 }}>
                {Math.round(data.task2.seasons_with_episode * 100)} %
              </div>
              <p className="meta">Без критерия устойчивости было бы 64 % — слишком много ложных срабатываний.</p>
            </div>
            <div>
              <div className="eyebrow">Точность к статусам</div>
              <div className="num" style={{ fontSize: 28 }}>
                {data.task2.precision_points.toFixed(2)}
              </div>
              <p className="meta">Доля точек ниже −1σ внутри найденных эпизодов при базовой доле 0.17.</p>
            </div>
            <div>
              <div className="eyebrow">Полнота по критическим</div>
              <div className="num" style={{ fontSize: 28 }}>
                {data.task2.recall_critical.toFixed(2)}
              </div>
              <p className="meta">Точки ниже −2σ, попавшие в эпизоды.</p>
            </div>
          </div>
          <hr className="divider" style={{ margin: "16px 0" }} />
          <div className="row" style={{ gap: 8, flexWrap: "wrap" }}>
            {Object.entries(data.task2.by_cause)
              .sort((a, b) => b[1] - a[1])
              .map(([cause, count]) => (
                <span key={cause} className="tag">
                  {CAUSE_LABEL[cause] ?? cause} · {count}
                </span>
              ))}
          </div>
        </section>
      </Reveal>

      <Reveal>
        <section className="card">
          <h2 style={{ marginBottom: 12 }}>Источники данных</h2>
          <div className="stack" style={{ gap: 0 }}>
            {data.sources.map((source, index) => (
              <div
                key={source.name}
                className="spread"
                style={{
                  padding: "11px 0",
                  borderTop: index === 0 ? "none" : "1px solid var(--line)",
                  gap: 16,
                  alignItems: "baseline",
                }}
              >
                <span style={{ fontSize: 14 }}>{source.name}</span>
                <span className="meta" style={{ textAlign: "right" }}>
                  {source.detail}
                </span>
              </div>
            ))}
          </div>
        </section>
      </Reveal>
    </div>
  );
}
