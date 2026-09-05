/** Поле кейса: сводка по всем сезонам и подробности выбранного года. */

import { useQuery } from "@tanstack/react-query";
import { Link, useParams } from "react-router-dom";

import { api } from "../api/client";
import { plural } from "../lib/format";
import { SeasonPanel } from "../components/panels/SeasonPanel";
import { ErrorNote, Loader } from "../components/ui/Loader";

export default function FieldPage() {
  const { pid = "" } = useParams();
  const detail = useQuery({ queryKey: ["polygon", pid], queryFn: () => api.polygon(pid), enabled: Boolean(pid) });

  if (detail.isLoading) return <Loader label={`Считаем кривые для ${pid}`} />;
  if (detail.isError) return <ErrorNote error={detail.error} />;
  if (!detail.data) return null;

  const data = detail.data;
  const criticalYears = [...new Set(data.episodes.filter((e) => e.severity === "критическая").map((e) => e.year))];
  const restored = Object.values(data.years).reduce((sum, year) => sum + year.restored.length, 0);

  return (
    <div className="stack" style={{ gap: 18 }}>
      <div className="spread" style={{ flexWrap: "wrap", gap: 12 }}>
        <div className="stack" style={{ gap: 6 }}>
          <Link to="/" className="meta" style={{ textDecoration: "none" }}>
            ← все поля
          </Link>
          <h1 style={{ fontSize: 34 }}>{data.pid}</h1>
          <div className="row" style={{ gap: 8, flexWrap: "wrap" }}>
            <span className="tag">{data.crop}</span>
            <span className="meta">{data.kind}</span>
          </div>
        </div>
        <div className="row" style={{ gap: 22, flexWrap: "wrap" }}>
          <div>
            <div className="eyebrow">Сезонов</div>
            <div className="num" style={{ fontSize: 22 }}>
              {Object.keys(data.years).length}
            </div>
          </div>
          <div>
            <div className="eyebrow">Эпизодов</div>
            <div className="num" style={{ fontSize: 22 }}>
              {data.episodes.length}
            </div>
          </div>
          <div>
            <div className="eyebrow">Критических лет</div>
            <div className="num" style={{ fontSize: 22 }}>
              {criticalYears.length}
            </div>
          </div>
          <div>
            <div className="eyebrow">Восстановлено точек</div>
            <div className="num" style={{ fontSize: 22 }}>
              {restored}
            </div>
          </div>
        </div>
      </div>

      {criticalYears.length > 0 && (
        <p className="meta">
          Критические сезоны: {criticalYears.sort().join(", ")} —{" "}
          {criticalYears.length} {plural(criticalYears.length, "год", "года", "лет")} из{" "}
          {Object.keys(data.years).length}.
        </p>
      )}

      <SeasonPanel detail={data} />
    </div>
  );
}
