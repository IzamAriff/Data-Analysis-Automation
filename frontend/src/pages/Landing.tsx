import UploadZone from '../components/UploadZone'
import { UploadResponse } from '../api/client'

export default function Landing({ onLoaded }: { onLoaded: (r: UploadResponse)=>void }) {
  return (
    <div className="max-w-6xl mx-auto p-6 space-y-6">
      <header className="space-y-2">
        <h1 className="text-4xl font-bold tracking-tight">📊 DataPilot</h1>
        <p className="text-lg text-slate-600">Automated Data Analysis Studio — Fullstack Rebuild (FastAPI + React)</p>
        <p className="text-sm text-slate-500">Upload any tabular dataset and the platform profiles, visualises, diagnoses and models it — no hard-coded columns, no code.</p>
      </header>

      <div className="grid md:grid-cols-3 gap-4 text-sm">
        <div className="bg-white rounded-xl shadow p-4"><div className="font-medium">1️⃣ Load</div><div className="text-slate-500">CSV/Excel/Parquet/JSON, URL, or bundled sample</div></div>
        <div className="bg-white rounded-xl shadow p-4"><div className="font-medium">2️⃣ Profile</div><div className="text-slate-500">Auto-detect column roles, review, override</div></div>
        <div className="bg-white rounded-xl shadow p-4"><div className="font-medium">3️⃣ Explore</div><div className="text-slate-500">KPIs, charts, diagnostics, models — all live via API</div></div>
      </div>

      <UploadZone onLoaded={onLoaded} />

      <div className="text-xs text-slate-400">
        Backend: <code>http://localhost:8000/api/v1</code> • Frontend: React + Vite + Tailwind + Plotly.js • Core: <code>src/</code> reused 100% • Dockerized • Legacy Streamlit still at <code>app.py</code>
      </div>
    </div>
  )
}
