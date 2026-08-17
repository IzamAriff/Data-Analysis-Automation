import { useState } from 'react'
import Landing from './pages/Landing'
import Dashboard from './pages/Dashboard'
import { UploadResponse, prepareDataset, ProfileResponse } from './api/client'

export default function App() {
  const [dataset, setDataset] = useState<UploadResponse | null>(null)
  const [profile, setProfile] = useState<ProfileResponse | null>(null)
  const [error, setError] = useState('')

  const handleLoaded = async (r: UploadResponse) => {
    setDataset(r)
    setError('')
    try {
      const p = await prepareDataset(r.dataset_id, r.sheets?.[0])
      setProfile(p)
    } catch (e:any) {
      setError(e.response?.data?.detail || e.message)
    }
  }

  if (!dataset) {
    return <Landing onLoaded={handleLoaded} />
  }

  if (error) {
    return <div className="p-6"><div className="bg-red-50 text-red-700 p-4 rounded">{error}</div><button onClick={()=>{setDataset(null); setProfile(null)}} className="mt-4 bg-slate-900 text-white px-4 py-2 rounded">← Back</button></div>
  }

  if (!profile) {
    return <div className="p-6">⏳ Preparing dataset {dataset.name} ({dataset.rows} rows)…</div>
  }

  return (
    <div className="min-h-screen">
      <div className="flex justify-between items-center max-w-[1600px] mx-auto p-4">
        <div className="text-sm text-slate-500">Dataset: <span className="font-medium text-slate-900">{dataset.name}</span> • {dataset.rows}×{dataset.cols} • ID: {dataset.dataset_id}</div>
        <button onClick={()=>{setDataset(null); setProfile(null)}} className="text-sm bg-white border px-3 py-1 rounded">← New dataset</button>
      </div>
      <Dashboard datasetId={dataset.dataset_id} initialProfile={profile} />
    </div>
  )
}
