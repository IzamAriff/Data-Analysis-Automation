import { useState } from 'react'

interface Props {
  profile: any
  onRoleChange: (roles: Record<string,string>) => void
  onContinue: () => void
}

const ROLES = ["date","year","numeric","binary","boolean","category","text","id"]

export default function ProfileTable({ profile, onRoleChange, onContinue }: Props) {
  const [roles, setRoles] = useState<Record<string,string>>(profile.roles)

  const handleChange = (col: string, role: string) => {
    const next = {...roles, [col]: role}
    setRoles(next)
  }

  const save = () => {
    onRoleChange(roles)
  }

  return (
    <div className="bg-white rounded-2xl shadow p-6 space-y-4">
      <div className="flex justify-between items-center">
        <h2 className="text-xl font-semibold">🧬 Profile</h2>
        <div className="text-sm text-slate-500">{profile.structure_hint}</div>
      </div>

      {profile.summary && (
        <div className="grid grid-cols-2 md:grid-cols-4 gap-3 text-sm">
          <div className="bg-slate-50 rounded p-3"><div className="text-slate-500">Rows</div><div className="font-bold text-lg">{profile.summary.rows?.toLocaleString()}</div></div>
          <div className="bg-slate-50 rounded p-3"><div className="text-slate-500">Columns</div><div className="font-bold text-lg">{profile.summary.columns}</div></div>
          <div className="bg-slate-50 rounded p-3"><div className="text-slate-500">Missing</div><div className="font-bold text-lg">{profile.summary.missing_pct?.toFixed(1)}%</div></div>
          <div className="bg-slate-50 rounded p-3"><div className="text-slate-500">Memory</div><div className="font-bold text-lg">{profile.summary.memory_mb?.toFixed(1)} MB</div></div>
        </div>
      )}

      <div className="overflow-auto max-h-[420px] border rounded">
        <table className="w-full text-sm">
          <thead className="bg-slate-100 sticky top-0">
            <tr>
              <th className="p-2 text-left">Column</th>
              <th className="p-2 text-left">Role</th>
              <th className="p-2 text-left">Type</th>
              <th className="p-2 text-left">Missing %</th>
              <th className="p-2 text-left">Unique</th>
            </tr>
          </thead>
          <tbody>
            {profile.column_profile.map((row:any)=>(
              <tr key={row.Column} className="border-t">
                <td className="p-2 font-medium">{row.Column}</td>
                <td className="p-2">
                  <select value={roles[row.Column] || row.Role} onChange={e=>handleChange(row.Column, e.target.value)} className="border rounded px-1 py-1">
                    {ROLES.map(r=> <option key={r} value={r}>{r}</option>)}
                  </select>
                </td>
                <td className="p-2 text-slate-500">{row.Type}</td>
                <td className="p-2">{row["Missing %"] ?? row.MissingPct}</td>
                <td className="p-2">{row.Unique}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {profile.prep_notes?.length>0 && (
        <div className="text-xs bg-amber-50 border border-amber-200 rounded p-3">
          <div className="font-medium mb-1">Preprocessing log</div>
          <ul className="list-disc ml-4">{profile.prep_notes.map((n:string,i:number)=><li key={i}>{n}</li>)}</ul>
        </div>
      )}

      <div className="flex gap-3">
        <button onClick={save} className="bg-brand-blue text-white px-4 py-2 rounded">💾 Save roles</button>
        <button onClick={onContinue} className="bg-slate-900 text-white px-4 py-2 rounded">🚀 Continue to dashboard</button>
      </div>
    </div>
  )
}
