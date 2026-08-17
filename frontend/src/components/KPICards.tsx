export default function KPICards({ kpi }: { kpi:any }) {
  if (!kpi) return null
  const cards = [
    { label: "Rows", value: kpi.rows?.toLocaleString(), sub: kpi.full_rows ? `${kpi.rows_delta?.toFixed(1)}% vs all` : undefined },
    { label: "Columns", value: kpi.columns },
    { label: kpi.metric ? `${kpi.metric} total` : "Missing cells", value: kpi.metric ? kpi.metric_total?.toLocaleString(undefined,{maximumFractionDigits:1}) : `${kpi.missing_pct?.toFixed(1)}%` },
    { label: kpi.top_category ? `${kpi.top_category.column}` : "Duplicate rows", value: kpi.top_category ? `${kpi.top_category.value} (${kpi.top_category.share?.toFixed(1)}%)` : kpi.duplicate_rows },
  ]
  return (
    <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
      {cards.map((c,i)=>(
        <div key={i} className="bg-white rounded-xl shadow p-4">
          <div className="text-xs text-slate-500 uppercase">{c.label}</div>
          <div className="font-bold text-lg truncate">{c.value ?? '—'}</div>
          {c.sub && <div className="text-xs text-slate-400">{c.sub}</div>}
        </div>
      ))}
    </div>
  )
}
