# Architecture — Fullstack Rebuild (3e2ebb4)

## Goals
- Decouple UI from analytics: `src/` becomes a pure domain library, usable by any frontend.
- Provide a modern, typed HTTP API (FastAPI + Pydantic v2) with OpenAPI docs.
- Ship a modern React frontend (Vite + Tailwind + Plotly) while keeping legacy Streamlit.
- Keep deployment simple: Docker Compose with 2 services + optional legacy profile.

## Layers

### 1. Core `src/` (domain)
- `loader.py`: ingestion (encoding, delimiter, date/numeric parsing), URL safety, sample registry. Returns `DataBundle`.
- `profile.py`: role inference (`date|year|numeric|binary|boolean|category|text|id`), profiling, data dictionary, descriptive stats.
- `analysis.py`: KPI, correlation (r + p via t-dist), top pairs, group stats, ANOVA/Kruskal-Wallis (Levene gate), chi-square + Cramér's V, outlier (IQR), time aggregation.
- `modeling.py`: `build_model_matrix` (impute median/mean/drop, one-hot top 20, date → days), 5-fold pooled CV, regression (Linear + RF), classification (RF), clustering (k-means + elbow/silhouette, PCA 2D), forecasting (Holt-Winters with linear fallback + holdout MAPE).
- `plots.py`: colorblind-safe Plotly builders (Okabe-Ito/Cividis/BrBG), style helper, trend/grouped trend, histogram, box, bar, scatter, heatmap, composition (treemap/pie), missing, forecast, elbow, cluster scatter, importance, confusion, residuals.
- `ui.py`: `FilterState` dataclass + `apply_filters` (vectorised), sidebar widget helpers for Streamlit legacy.

### 2. Backend `backend/app/`
- `config.py`: 12-factor settings (`MAX_FILE_MB`, `CORS_ORIGINS`, TTL).
- `services/session_store.py`: thread-safe dict `dataset_id → DatasetSession` (raw df, prepared df, roles, notes, sheets, TTL).
- `services/data_service.py`: thin wrappers translating Pydantic schemas to `src/` calls, filter conversion.
- `models/schemas.py`: request/response models.
- `routers/data.py`: upload (multipart), url, sample, prepare, info, delete.
- `routers/profile.py`: role override, dictionary, roles.
- `routers/analysis.py`: kpi, correlation, group-stats, anova, chi-square, outliers, trend.
- `routers/modeling.py`: regression, classification, clustering, forecasting — JSON-safe serialization.
- `routers/plots.py`: `generate` → Plotly JSON (fig.to_dict) for frontend rendering.
- `main.py`: FastAPI app, CORS, health, OpenAPI, global error handler.

Flow:
```
Client POST /data/upload (file)
  → loader.load_from_bytes → SessionStore.create → dataset_id
Client POST /data/prepare {dataset_id, sheet?}
  → prepare_dataset (loader.prepare_dataframe + profile.infer_roles)
  → store prepared + roles → ProfileResponse
Client POST /analysis/* or /modeling/* or /plots/*
  → SessionStore.get → apply_user_filters (if filters) → src.analysis/modeling/plots → JSON
```

### 3. Frontend `frontend/src/`
- `api/client.ts`: axios instance (proxy /api → backend:8000 in dev, Nginx proxy in prod), typed helpers.
- `components/UploadZone.tsx`: file input, URL input, sample browser (calls listSamples/loadSample).
- `components/ProfileTable.tsx`: editable role dropdowns, summary cards, prep log, Save + Continue.
- `components/KPICards.tsx`: KPI snapshot.
- `components/ChartBuilder.tsx`: chart type selector, column pickers, calls /plots/generate, renders Plotly via react-plotly.js.
- `components/Diagnostics.tsx` & `ModelingPanel.tsx`: buttons → /analysis/* and /modeling/*, shows raw JSON (can be upgraded to rich visuals).
- `pages/Landing.tsx` & `Dashboard.tsx`: simple client-side routing via state (dataset_id + profile), tab navigation.
- `App.tsx`: orchestrates Landing → Dashboard after upload+prepare.
- Tooling: Vite (HMR, proxy), Tailwind (utility-first), TypeScript, ESLint.

### 4. Legacy `app.py`
- Still works: Streamlit monolith importing src/* directly. No backend dependency. Kept for Streamlit Cloud / HF Spaces deployments.

## Data flow & state
- Backend is stateless except for in-memory session store (TTL eviction, touch on get). Could be swapped for Redis in prod.
- Frontend holds only dataset_id + profile; each analysis call is stateless aside from dataset_id.
- Filters: `FilterStateSchema` (JSON) → `_filters_from_schema` → `FilterState` dataclass → `apply_filters(df, state)` → filtered df for downstream.

## Deployment
- `docker-compose.yml`: backend (FastAPI uvicorn), frontend (Node build → Nginx, /api proxy to backend), streamlit-legacy (profile legacy).
- `Dockerfile.backend`: Python 3.11-slim + requirements + src + backend, uvicorn.
- `Dockerfile.frontend`: Node 20 build stage → Nginx with static + /api proxy.
- `Dockerfile`: legacy Streamlit.

## Future improvements
- Replace in-memory store with Redis + serialized Parquet for persistence & multi-worker.
- Add auth, rate limiting, background tasks (Celery) for heavy modeling.
- Expand frontend: rich filter sidebar (date pickers, multi-select), proper chart gallery, data table virtualization, Markdown report export from backend.
- Add backend tests: pytest-asyncio + TestClient covering each router.
- Add e2e: Playwright against frontend + backend.
