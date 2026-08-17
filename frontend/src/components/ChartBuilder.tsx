import { useState } from 'react'
import { generatePlot } from '../api/client'
import Plot from 'react-plotly.js'

interface Props {
  dataset_id: string
  roles: Record<string,string>
}

export default function ChartBuilder({ dataset_id, roles }: Props) {
  const [type, setType] = useState('histogram')
  const [col, setCol] = useState('')
  const [groupCol, setGroupCol] = useState('')
  const [fig, setFig] = useState<any>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')

  const numeric = Object.entries(roles).filter(([,r])=>r==='numeric').map(([c])=>c)
  const categorical = Object.entries(roles).filter(([,r])=>['category','binary','boolean'].includes(r)).map(([c])=>c)
  const dates = Object.entries(roles).filter(([,r])=>r==='date').map(([c])=>c)

  const build = async () => {
    setLoading(true); setError('')
    try {
      let chartType = type
      let params:any = {}
      if (type==='histogram') { params = { col: col || numeric[0], group_col: groupCol || undefined } }
      if (type==='box') { params = { col: col || numeric[0], group_col: groupCol || categorical[0] } }
      if (type==='bar') { params = { cat_col: col || categorical[0], value_col: numeric[0], agg:'sum' } }
      if (type==='scatter') { params = { x: numeric[0], y: numeric[1]||numeric[0], color: categorical[0] } }
      if (type==='heatmap') { params = { method:'pearson' } }
      if (type==='trend') { params = { date_col: dates[0], value_col: numeric[0], freq:'M', agg:'sum' } }
      if (type==='missing') { params = {} }
      const json = await generatePlot(dataset_id, chartType as any, params)
      setFig(json)
    } catch (e:any) {
      setError(e.response?.data?.detail || e.message)
    } finally { setLoading(false) }
  }

  return (
    <div className="bg-white rounded-2xl shadow p-6 space-y-4">
      <h3 className="font-semibold text-lg">📈 Chart Builder</h3>

      <div className="flex flex-wrap gap-3 items-end">
        <div>
          <label className="text-xs">Chart type</label>
          <select value={type} onChange={e=>setType(e.target.value)} className="block border rounded px-2 py-1 text-sm">
            <option value="histogram">Distribution (histogram)</option>
            <option value="box">Box by category</option>
            <option value="bar">Bar comparison</option>
            <option value="scatter">Relationship (scatter)</option>
            <option value="heatmap">Correlation heatmap</option>
            <option value="trend">Trend</option>
            <option value="missing">Missing values</option>
            <option value="composition">Composition</option>
          </select>
        </div>
        <div>
          <label className="text-xs">Column</label>
          <input value={col} onChange={e=>setCol(e.target.value)} placeholder={numeric[0]||categorical[0]} className="block border rounded px-2 py-1 text-sm" />
        </div>
        <div>
          <label className="text-xs">Group / Color</label>
          <input value={groupCol} onChange={e=>setGroupCol(e.target.value)} placeholder={categorical[0]||''} className="block border rounded px-2 py-1 text-sm" />
        </div>
        <button onClick={build} disabled={loading} className="bg-slate-900 text-white px-4 py-2 rounded text-sm">Build</button>
      </div>

      {error && <div className="text-sm text-red-600 bg-red-50 p-2 rounded">{error}</div>}

      {fig && (
        <div className="border rounded overflow-hidden">
          {/* @ts-ignore */}
          <Plot data={fig.data} layout={{...fig.layout, height:420, autosize:true}} config={{responsive:true}} style={{width:'100%'}} />
        </div>
      )}
    </div>
  )
}
