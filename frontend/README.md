# Frontend — DataPilot React

React + Vite + Tailwind + Plotly.js frontend for the fullstack rebuild.

## Setup
```bash
npm ci
npm run dev
# http://localhost:5173
```

## Env
- `VITE_API_URL` — leave empty to use Vite proxy (`/api` → `http://localhost:8000`), or set to `http://localhost:8000` for direct.

## Structure
- `src/api/client.ts` — axios client for backend.
- `src/components/` — UploadZone, ProfileTable, KPICards, ChartBuilder, Diagnostics, ModelingPanel.
- `src/pages/` — Landing, Dashboard.
- `src/App.tsx` — simple routing via state.
- `vite.config.ts` — proxy config for `/api`.

## Build
```bash
npm run build
npm run preview
```

Production is served via Nginx (`Dockerfile.frontend`) with `/api/` proxied to backend.

