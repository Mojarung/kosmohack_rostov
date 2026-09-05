/** Поле кейса: компактная шапка со сводкой и рабочая область сезона во всю высоту. */

import { useQuery } from "@tanstack/react-query";
import { Link, useParams } from "react-router-dom";

import { api } from "../api/client";
import { AskPanel } from "../components/panels/AskPanel";
import { SeasonPanel } from "../components/panels/SeasonPanel";
import { ErrorNote } from "../components/ui/Loader";
import { PagePending } from "../components/ui/PagePending";

export default function FieldPage() {
  const { pid = "" } = useParams();
  const detail = useQuery({ queryKey: ["polygon", pid], queryFn: () => api.polygon(pid), enabled: Boolean(pid) });

  if (detail.isLoading) return <PagePending label={`Считаем кривые для ${pid}`} />;
  if (detail.isError) return <ErrorNote error={detail.error} />;
  if (!detail.data) return null;

  const data = detail.data;
  const criticalYears = [...new Set(data.episodes.filter((e) => e.severity === "критическая").map((e) => e.year))];
  const restored = Object.values(data.years).reduce((sum, year) => sum + year.restored.length, 0);

  const facts = [
    { label: "сезонов", value: Object.keys(data.years).length },
    { label: "эпизодов", value: data.episodes.length },
    { label: "критических лет", value: criticalYears.length },
    { label: "восстановлено точек", value: restored },
  ];

  return (
    <div className="workspace workspace--field">
      <div className="screen-head">
        <Link to="/fields" className="back-link">
          ← поля кейса
        </Link>
        <span className="screen-title">{data.name || data.pid}</span>
        <span className="tag">{data.crop}</span>
        <span className="meta">{data.kind}</span>
        <span className="head-facts">
          {facts.map((fact) => (
            <span key={fact.label} className="head-fact">
              <span className="num">{fact.value}</span>
              <span className="eyebrow">{fact.label}</span>
            </span>
          ))}
        </span>
      </div>

      <SeasonPanel key={data.pid} detail={data} aside={(year) => <AskPanel pid={data.pid} year={year} />} />
    </div>
  );
}
