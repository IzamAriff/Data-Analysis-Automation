export default function FilterSidebar() {
  return (
    <div className="text-xs text-slate-400 p-4">
      <div className="font-medium mb-2">🎚️ Filters (backend-driven)</div>
      <p>Dynamic filters are generated from column roles on the backend. Pass <code>filters</code> to any analysis/modeling request to filter server-side.</p>
      <p className="mt-2">In the full app, this panel builds a FilterStateSchema and sends it with each API call.</p>
    </div>
  )
}
