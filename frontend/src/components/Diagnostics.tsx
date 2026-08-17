import { useState } from 'react'
import { getCorrelation, getGroupStats, getAnova, getChiSquare, getOutliers } from '../api/client'

export default function Diagnostics({ dataset_id, roles }: { dataset_id:string, roles:Record<string,string> }) {
  const [result, setResult] = useState<any>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')

  const numeric = Object.entries(roles).filter(([,r])=>r==='numeric').map(([c])=>c)
  const categorical = Object.entries(roles).filter(([,r])=>['category','binary','boolean'].includes(r)).map(([c])=>c)

  const run = async (fn:()=>Promise<any>) => {
    setLoading(true); setError('')
    try { const r = await fn(); setResult(r) }
    catch (e:any) { setError(e.response?.data?.detail || e.message) }
    finally { setLoading(false) }
  }

  return (
    <div className="bg-white rounded-2xl shadow p-6 space-y-4">
      <h3 className="font-semibold text-lg">🔎 Diagnostics</h3>

      <div className="flex flex-wrap gap-2">
        <button onClick={()=>run(()=>getCorrelation(dataset_id))} className="bg-brand-blue text-white px-3 py-1 rounded text-sm">Correlations</button>
        <button onClick={()=>run(()=>getGroupStats(dataset_id, numeric[0], categorical[0]))} className="bg-slate-800 text-white px-3 py-1 rounded text-sm">Group stats ({numeric[0]} by {categorical[0]})</button>
        <button onClick={()=>run(()=>getAnova(dataset_id, numeric[0], categorical[0]))} className="bg-slate-800 text-white px-3 py-1 rounded text-sm">ANOVA</button>
        <button onClick={()=>run(()=>getChiSquare(dataset_id, categorical[0], categorical[1]||categorical[0]))} className="bg-slate-800 text-white px-3 py-1 rounded text-sm">Chi-square</button>
        <button onClick={()=>run(()=>getOutliers(dataset_id))} className="bg-slate-800 text-white px-3 py-1 rounded text-sm">Outliers</button>
      </div>

      {loading && <div className="text-sm">⏳ Running…</div>}
      {error && <div className="text-sm text-red-600 bg-red-50 p-2 rounded">{error}</div>}

      {result && (
        <pre className="bg-slate-900 text-slate-100 text-xs p-4 rounded overflow-auto max-h-[400px]">{JSON.stringify(result, null, 2)}</pre>
      )}
    </div>
  )
}
