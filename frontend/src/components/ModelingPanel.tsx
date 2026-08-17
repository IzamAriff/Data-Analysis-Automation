import { useState } from 'react'
import { runRegression, runClassification, runClustering, runForecast } from '../api/client'

export default function ModelingPanel({ dataset_id, roles }: { dataset_id:string, roles:Record<string,string> }) {
  const [result, setResult] = useState<any>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')
  const [target, setTarget] = useState('')

  const numeric = Object.entries(roles).filter(([,r])=>r==='numeric').map(([c])=>c)
  const categorical = Object.entries(roles).filter(([,r])=>['category','binary','boolean'].includes(r)).map(([c])=>c)
  const dates = Object.entries(roles).filter(([,r])=>r==='date').map(([c])=>c)
  const allFeatures = [...numeric, ...categorical]

  const run = async (fn:()=>Promise<any>) => {
    setLoading(true); setError('')
    try { const r = await fn(); setResult(r) }
    catch (e:any) { setError(e.response?.data?.detail || e.message) }
    finally { setLoading(false) }
  }

  return (
    <div className="bg-white rounded-2xl shadow p-6 space-y-4">
      <h3 className="font-semibold text-lg">🤖 Predictive</h3>

      <div className="flex gap-2 items-end">
        <div>
          <label className="text-xs">Target column</label>
          <input value={target} onChange={e=>setTarget(e.target.value)} placeholder={numeric[0]||categorical[0]} className="block border rounded px-2 py-1 text-sm" />
        </div>
      </div>

      <div className="flex flex-wrap gap-2">
        <button onClick={()=>run(()=>runRegression(dataset_id, target || numeric[0], allFeatures.filter(c=>c!==target).slice(0,5)))} className="bg-brand-green text-white px-3 py-1 rounded text-sm">Regression</button>
        <button onClick={()=>run(()=>runClassification(dataset_id, target || categorical[0], allFeatures.filter(c=>c!==target).slice(0,5)))} className="bg-brand-green text-white px-3 py-1 rounded text-sm">Classification</button>
        <button onClick={()=>run(()=>runClustering(dataset_id, numeric.slice(0,4)))} className="bg-brand-green text-white px-3 py-1 rounded text-sm">Clustering</button>
        <button onClick={()=>run(()=>runForecast(dataset_id, dates[0], numeric[0], 12))} className="bg-brand-green text-white px-3 py-1 rounded text-sm">Forecast</button>
      </div>

      {loading && <div className="text-sm">⏳ Fitting models (5-fold CV)…</div>}
      {error && <div className="text-sm text-red-600 bg-red-50 p-2 rounded">{error}</div>}

      {result && (
        <pre className="bg-slate-900 text-slate-100 text-xs p-4 rounded overflow-auto max-h-[500px]">{JSON.stringify(result, null, 2)}</pre>
      )}
    </div>
  )
}
