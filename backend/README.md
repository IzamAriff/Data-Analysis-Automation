# Backend — DataPilot API

FastAPI service exposing the full analytical power of `src/` as a typed HTTP API.

## Run
```bash
pip install -r ../requirements.txt
# or pip install -r requirements.txt
uvicorn backend.app.main:app --host 0.0.0.0 --port 8000 --reload
# docs: http://localhost:8000/docs
```

## Endpoints
- `GET /health` + `/api/v1/health`
- `GET /api/v1/data/samples`
- `POST /api/v1/data/upload`
- `POST /api/v1/data/url`
- `POST /api/v1/data/sample/{label}`
- `POST /api/v1/data/prepare`
- `POST /api/v1/profile/override`
- `GET /api/v1/profile/dictionary/{id}`
- `POST /api/v1/analysis/kpi`
- `POST /api/v1/analysis/correlation`
- `POST /api/v1/analysis/group-stats`
- `POST /api/v1/analysis/anova`
- `POST /api/v1/analysis/chi-square`
- `POST /api/v1/analysis/outliers`
- `POST /api/v1/analysis/trend`
- `POST /api/v1/modeling/regression`
- `POST /api/v1/modeling/classification`
- `POST /api/v1/modeling/clustering`
- `POST /api/v1/modeling/forecast`
- `POST /api/v1/plots/generate`

## Config (env)
- `MAX_FILE_MB=250`
- `CORS_ORIGINS=*`
- `LOG_LEVEL=INFO`
- `SESSION_TTL=3600`
- `MAX_MODEL_ROWS=100000`
