import { useEffect, useState } from 'react'
import { ProfileResponse, getKpi, prepareDataset } from '../api/client'
import ProfileTable from '../components/ProfileTable'
import KPICards from '../components/KPICards'
import ChartBuilder from '../components/ChartBuilder'
import Diagnostics from '../components/Diagnostics'
import ModelingPanel from '../components/ModelingPanel'

interface Props {
  datasetId: string
  initialProfile?: ProfileResponse
}

export default function Dashboard({ datasetId, initialProfile }: Props) {
  const [profile, setProfile] = useState<ProfileResponse | undefined>(initialProfile)
  const [kpi, setKpi] = useState<any>(null)
  const [tab, setTab] = useState<'overview'|'charts'|'diagnostics'|'modeling'>('overview')

  useEffect(() => {
    if (!profile) {
      prepareDataset(datasetId).then(setProfile).catch(console.error)
    }
  }, [datasetId])

  useEffect(() => {
    if (profile) {
      const metric = Object.entries(profile.roles).find(([,r])=>r==='numeric')?.[0]
      const dateCol = Object.entries(profile.roles).find(([,r])=>r==='date')?.[0]
      getKpi(datasetId, metric, dateCol).then(setKpi).catch(()=>{})
    }
  }, [profile, datasetId])

  if (!profile) return <div className="p-6">⏳ Profiling dataset…</div>

  return (
    <div className="max-w-[1600px] mx-auto p-4 space-y-4">
      <header className="flex justify-between items-center bg-white rounded-2xl shadow p-4">
        <div>
          <h1 className="font-bold">📊 DataPilot Dashboard</h1>
          <div className="text-xs text-slate-500">{profile.structure_hint} • {Object.keys(profile.roles).length} columns</div>
        </div>
        <div className="flex gap-2">
          {(['overview','charts','diagnostics','modeling'] as const).map(t=>(
            <button key={t} onClick={()=>setTab(t)} className={`px-3 py-1 rounded text-sm capitalize ${tab===t ? 'bg-slate-900 text-white':'bg-slate-100'}`}>{t}</button>
          ))}
        </div>
      </header>

      {tab==='overview' && (
        <div className="space-y-4">
          <KPICards kpi={kpi} />
          <ProfileTable profile={profile} onRoleChange={async (roles)=>{
            const { overrideRoles } = await import('../api/client')
            const updated = await overrideRoles(datasetId, roles)
            setProfile(updated)
          }} onContinue={()=>setTab('charts')} />
        </div>
      )}

      {tab==='charts' && (
        <ChartBuilder dataset_id={datasetId} roles={profile.roles} />
      )}

      {tab==='diagnostics' && (
        <Diagnostics dataset_id={datasetId} roles={profile.roles} />
      )}

      {tab==='modeling' && (
        <ModelingPanel dataset_id={datasetId} roles={profile.roles} />
      )}
    </div>
  )
}
