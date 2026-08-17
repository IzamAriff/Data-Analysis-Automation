# 📊 DataPilot — Automated Data Analysis Studio (Fullstack Rebuild)

> **Fullstack rebuild — commit `3e2ebb4`**  
> A self-serve, interactive platform that **profiles, visualises, diagnoses and models any tabular dataset** — no hard-coded column names, no code required.

Upload a CSV/Excel/Parquet/JSON file (or paste a URL), and DataPilot automatically:
1. **Detects everything it can about the data** — encoding, delimiter, date columns, numeric columns stored as formatted strings (`$1,234.50`), categorical vs. free-text vs. ID columns.
2. **Profiles data quality** — row/column counts, duplicate rows, missing-value %, per-column stats and a data dictionary.
3. **Builds an interactive dashboard** — KPI cards, time trends, distributions, comparisons, relationships, compositions, correlation heatmaps — all driven by backend filters.
4. **Runs diagnostics** — correlation with p-values, ANOVA/Kruskal–Wallis, chi-square + Cramér's V, IQR outlier detection.
5. **Fits predictive models on demand** — regression, classification, k-means clustering, and time-series forecasting, each with cross-validated metrics, baselines and stated assumptions.
6. **Exports everything** — filtered CSV, chart PNG/HTML, data dictionary, reproducible Markdown report.

---

## 🏗️ Architecture — Fullstack Rebuild (3e2ebb4)

This branch refactors the original Streamlit monolith (`app.py`) into a clean **FastAPI backend + React frontend** while keeping 100% of the battle-tested core logic in `src/`.

```
Data-Analysis-Automation/
├── backend/                    # FastAPI service (new)
│   ├── app/
│   │   ├── main.py             # FastAPI entry, CORS, health
│   │   ├── config.py           # 12-factor settings
│   │   ├── models/schemas.py   # Pydantic v2 schemas
│   │   ├── routers/
│   │   │   ├── data.py         # upload / url / samples / prepare
│   │   │   ├── profile.py      # role overrides, data dictionary
│   │   │   ├── analysis.py     # KPI, correlations, ANOVA, chi2, outliers, trend
│   │   │   ├── modeling.py     # regression, classification, clustering, forecast
│   │   │   └── plots.py        # Plotly JSON for frontend
│   │   └── services/
│   │       ├── session_store.py # in-memory TTL store (dataset_id → df)
│   │       └── data_service.py  # wrappers around src/* modules
│   └── requirements.txt
│
├── frontend/                   # React + Vite + Tailwind + Plotly.js (new)
│   ├── src/
│   │   ├── api/client.ts       # axios client for /api/v1
│   │   ├── components/
│   │   │   ├── UploadZone.tsx
│   │   │   ├── ProfileTable.tsx
│   │   │   ├── KPICards.tsx
│   │   │   ├── ChartBuilder.tsx
│   │   │   ├── Diagnostics.tsx
│   │   │   ├── ModelingPanel.tsx
│   │   │   └── FilterSidebar.tsx
│   │   ├── pages/
│   │   │   ├── Landing.tsx
│   │   │   └── Dashboard.tsx
│   │   ├── App.tsx
│   │   └── main.tsx
│   ├── vite.config.ts
│   ├── tailwind.config.js
│   └── package.json
│
├── src/                        # Core domain logic (unchanged, reused 100%)
│   ├── loader.py
│   ├── profile.py
│   ├── analysis.py
│   ├── modeling.py
│   ├── plots.py
│   └── ui.py (legacy helpers)
│
├── app.py                      # Legacy Streamlit monolith (still works)
├── data/                       # Bundled samples
├── tests/                      # pytest
├── docker-compose.yml          # backend + frontend + legacy
├── Dockerfile.backend
├── Dockerfile.frontend
├── Dockerfile                  # legacy Streamlit
└── Makefile
```

### Why this split?
- **Backend** is stateless-ish: `dataset_id` in-memory store (TTL 1h), idempotent preparation, all heavy lifting in `src/`. Easy to scale, test, document via `/docs`.
- **Frontend** is modern React: file drop, sample browser, editable role table, dynamic chart builder (Plotly), diagnostics & modeling panels — all talking to `/api/v1/*` via axios.
- **Core `src/`** remains pure functions, no Streamlit dependency, so both frontends reuse it.

---

## 🚀 Quick start

### Option A — Docker (recommended)
```bash
docker-compose up --build
# frontend → http://localhost:5173
# backend  → http://localhost:8000/docs
# legacy Streamlit (optional) → docker-compose --profile legacy up
```

### Option B — Local dev (two terminals)
```bash
# terminal 1 — backend
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
uvicorn backend.app.main:app --host 0.0.0.0 --port 8000 --reload

# terminal 2 — frontend
cd frontend
npm ci
npm run dev
# open http://localhost:5173
```

### Option C — Legacy Streamlit (still supported)
```bash
streamlit run app.py
# http://localhost:8501
```

> **Try instantly:** on the React UI click **“🎁 Browse samples”** → pick Superstore (9,994 rows) or vgsales (16,595 rows).

---

## 🔌 API Reference

Base: `/api/v1`

| Group | Method | Endpoint | Description |
|-------|--------|----------|-------------|
| data | GET | `/data/samples` | List bundled samples |
| data | POST | `/data/upload` | Upload file → `dataset_id` |
| data | POST | `/data/url` | Load from URL |
| data | POST | `/data/sample/{label}` | Load bundled sample |
| data | POST | `/data/prepare` | Prepare + infer roles |
| profile | POST | `/profile/override` | Override column roles |
| profile | GET | `/profile/dictionary/{id}` | Data dictionary |
| analysis | POST | `/analysis/kpi` | KPI snapshot |
| analysis | POST | `/analysis/correlation` | Correlation matrix + top pairs |
| analysis | POST | `/analysis/group-stats` | Per-group stats |
| analysis | POST | `/analysis/anova` | ANOVA/Kruskal-Wallis |
| analysis | POST | `/analysis/chi-square` | Chi-square + Cramér's V |
| analysis | POST | `/analysis/outliers` | IQR outlier summary |
| analysis | POST | `/analysis/trend` | Time aggregation |
| modeling | POST | `/modeling/regression` | 5-fold CV regression |
| modeling | POST | `/modeling/classification` | RF classification |
| modeling | POST | `/modeling/clustering` | k-means + elbow/silhouette |
| modeling | POST | `/modeling/forecast` | Holt-Winters forecast |
| plots | POST | `/plots/generate` | Plotly JSON for any chart |

Full OpenAPI docs at `http://localhost:8000/docs`.

---

## 🧭 How to use (React frontend)

### 1. Load data
Upload, URL, or sample. Backend returns `dataset_id` + notes.

### 2. Review profile
See row/column counts, duplicates, missing %, memory, **inferred roles** (`date`, `year`, `numeric`, `binary`, `boolean`, `category`, `text`, `id`). Override any role → Save.

### 3. Explore dashboard
- **Overview** — KPI cards, data-quality, column roles, descriptive stats, preprocessing log.
- **Charts** — builder (trend, distribution, comparison, relationship, composition, correlation) — Plotly JSON from backend rendered with `react-plotly.js`.
- **Diagnostics** — correlation matrix + strongest pairs, ANOVA box prep, chi-square contingency, outlier detection.
- **Predictive** — target + Run: regression & classification (5-fold CV + baselines), k-means, forecasting with holdout MAPE.

Filters are sent as `FilterStateSchema` with every request — backend applies them server-side before analysis.

---

## 🧹 Preprocessing (documented & reproducible)

1. Column-name cleanup — whitespace trimmed, duplicate names suffixed.
2. Fully-empty rows removed; duplicate rows dropped by default.
3. Datetime parsing — ≥90 % of sampled values parse as dates → datetime (pure 4-digit years kept as years).
4. Numeric-string parsing — `$1,234.50`, `12,5 %` → numbers.
5. Missing values for models — `median`/`mean`/`drop`; categoricals get `(missing)` level.
6. Model encoding — one-hot top 20 levels per column → `(other)` for rest; dates → days since earliest; ID/text excluded (logged).

---

## 🧠 Methods & honest reporting

- 5-fold CV: every row scored by model that never saw it. Naive baseline always shown.
- Hypothesis tests: Levene → ANOVA or Kruskal-Wallis (η²), chi-square + Cramér's V, correlation p-values from t-dist.
- Assumptions stated under every model; association ≠ causation; app reports when model fails to beat baseline.

### Findings on bundled Sample Superstore (same as legacy)

| Analysis | Result |
|---|---|

---

## 🧪 Testing

```bash
pytest tests/ -v
cd frontend && npm run build  # type-check + production build
```

- Unit tests cover parsing edge cases, role inference, stats, every model family, chart builders, filter logic.
- AppTest E2E drives legacy `app.py`: load both samples, profile, filters, all four model flows.
- New backend can be tested via `/docs` → Try it out, or via `curl`/axios.

---

## 📦 Project structure (full)

```
Data-Analysis-Automation/
├── backend/app/            # FastAPI
├── frontend/src/           # React
├── src/                    # Core analytics (shared)
├── app.py                  # Legacy Streamlit entry
├── data/                   # Demo datasets
├── tests/
├── docker-compose.yml
├── Dockerfile.backend
├── Dockerfile.frontend
├── Dockerfile (legacy)
├── Makefile
└── README.md
```

---

## ☁️ Deployment

### Docker Compose (any VM)
```bash
docker-compose up --build -d
```

### Streamlit Cloud (legacy)
Push repo, pick `app.py`.

### Hugging Face Spaces (legacy)
New Space → Streamlit SDK → push repo.

### Internal server — backend only
```bash
docker build -f Dockerfile.backend -t datapilot-backend .
docker run -p 8000:8000 datapilot-backend
```

---

## 🔒 Security & robustness

- URL loading restricted to `http(s)`, timeout 30s, 250 MB cap; credentials stripped.
- Uploads parsed only by format-specific readers; never pickled/executed; Excel macros not run.
- Text search literal (`regex=False`).
- No `eval`/`exec`; column names sanitised.
- Per-section error handling in backend → 400 with safe message; 500 logged.
- CORS configurable via `CORS_ORIGINS`.

## 🎨 Accessibility

- Okabe-Ito categorical palette, Cividis sequential, BrBG diverging.
- Plotly hover tooltips, labelled axes, readable fonts, legends.

## 📚 Dataset provenance

- `sample_superstore.csv` — Tableau “Sample – Superstore” (public, learning purposes).
- `sample_video_game_sales.csv` — public “Video Game Sales (vgsales)” from VGChartz.

Both bundled only as clearly-labelled demo data — your own files never leave your session (or your deployment).

---

## 📝 Commit history

- `3e2ebb4` — **Fullstack rebuild**: FastAPI backend + React frontend + Docker Compose, 100% reuse of `src/` core, legacy Streamlit kept, updated docs/tests.
