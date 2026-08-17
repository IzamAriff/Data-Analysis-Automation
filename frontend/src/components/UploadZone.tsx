import { useState } from 'react'
import { uploadFile, loadUrl, listSamples, loadSample, UploadResponse, Sample } from '../api/client'

interface Props {
  onLoaded: (r: UploadResponse) => void
}

export default function UploadZone({ onLoaded }: Props) {
  const [url, setUrl] = useState('')
  const [samples, setSamples] = useState<Sample[]>([])
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string>('')

  const handleFile = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const f = e.target.files?.[0]
    if (!f) return
    setLoading(true); setError('')
    try {
      const r = await uploadFile(f)
      onLoaded(r)
    } catch (err:any) {
      setError(err.response?.data?.detail || err.message)
    } finally { setLoading(false) }
  }

  const handleUrl = async () => {
    if (!url.trim()) return
    setLoading(true); setError('')
    try {
      const r = await loadUrl(url)
      onLoaded(r)
    } catch (err:any) {
      setError(err.response?.data?.detail || err.message)
    } finally { setLoading(false) }
  }

  const fetchSamples = async () => {
    try {
      const list = await listSamples()
      setSamples(list)
    } catch (err:any) {
      setError(err.message)
    }
  }

  const handleSample = async (label: string) => {
    setLoading(true); setError('')
    try {
      const r = await loadSample(label)
      onLoaded(r)
    } catch (err:any) {
      setError(err.response?.data?.detail || err.message)
    } finally { setLoading(false)}
  }

  return (
    <div className="bg-white rounded-2xl shadow p-6 space-y-6">
      <h2 className="text-xl font-semibold">📁 Load your data</h2>

      <div className="grid md:grid-cols-3 gap-6">
        {/* Upload */}
        <div className="border-2 border-dashed rounded-xl p-4">
          <h3 className="font-medium mb-2">Upload file</h3>
          <p className="text-sm text-slate-500 mb-3">CSV, Excel, Parquet, JSON — 250 MB max</p>
          <input type="file" accept=".csv,.tsv,.txt,.xlsx,.xls,.parquet,.json" onChange={handleFile} className="block w-full text-sm" />
        </div>

        {/* URL */}
        <div className="border rounded-xl p-4">
          <h3 className="font-medium mb-2">From URL</h3>
          <input value={url} onChange={e=>setUrl(e.target.value)} placeholder="https://example.com/data.csv" className="w-full border rounded px-3 py-2 text-sm mb-2" />
          <button onClick={handleUrl} disabled={loading} className="bg-slate-900 text-white px-4 py-2 rounded text-sm disabled:opacity-50">⬇️ Download & load</button>
        </div>

        {/* Samples */}
        <div className="border rounded-xl p-4">
          <h3 className="font-medium mb-2">Sample dataset</h3>
          <button onClick={fetchSamples} className="text-sm bg-brand-blue text-white px-3 py-1 rounded mb-2">🎁 Browse samples</button>
          <div className="space-y-2 max-h-48 overflow-auto">
            {samples.map(s=>(
              <button key={s.label} onClick={()=>handleSample(s.label)} className="block w-full text-left text-sm border rounded px-2 py-1 hover:bg-slate-50">
                {s.label} <span className="text-slate-400">{s.rows_hint}</span>
              </button>
            ))}
          </div>
        </div>
      </div>

      {loading && <div className="text-sm text-slate-600">⏳ Loading…</div>}
      {error && <div className="text-sm text-red-600 bg-red-50 p-2 rounded">{error}</div>}
    </div>
  )
}
