"""FastAPI entry point — DataPilot fullstack backend."""

from __future__ import annotations

import logging
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from .config import get_settings
from .models.schemas import HealthResponse
from .routers import data, profile, analysis, modeling, plots

settings = get_settings()

logging.basicConfig(level=getattr(logging, settings.log_level.upper(), logging.INFO))
logger = logging.getLogger("datapilot.api")

app = FastAPI(
    title=settings.app_name,
    version=settings.version,
    description="""
# 📊 DataPilot API — Automated Data Analysis Studio (Fullstack Rebuild)

Backend for the fullstack rebuild (commit 3e2ebb4):

- Upload any tabular dataset (CSV/Excel/Parquet/JSON/URL or bundled samples)
- Prepare & profile (encoding, date, numeric parsing, role inference)
- Analyse (KPI, correlation, ANOVA, chi-square, outliers, trends)
- Model (regression, classification, clustering, forecasting)
- Plots (Plotly JSON for frontend rendering)

All computation reuses the battle-tested `src/` core library.

**Frontend**: React + Vite + Tailwind + Plotly.js (see `/frontend`)

**Legacy**: Streamlit monolith still available via `app.py` for backward compatibility.
""",
)

# CORS — allow all by default for dev; restrict via env CORS_ORIGINS in prod
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(data.router, prefix=settings.api_prefix)
app.include_router(profile.router, prefix=settings.api_prefix)
app.include_router(analysis.router, prefix=settings.api_prefix)
app.include_router(modeling.router, prefix=settings.api_prefix)
app.include_router(plots.router, prefix=settings.api_prefix)


@app.get("/", response_model=HealthResponse)
def root():
    return HealthResponse(version=settings.version, service=settings.app_name)


@app.get("/health", response_model=HealthResponse)
def health():
    return HealthResponse(version=settings.version, service=settings.app_name)


@app.get(f"{settings.api_prefix}/health", response_model=HealthResponse)
def api_health():
    return HealthResponse(version=settings.version, service=settings.app_name)


@app.exception_handler(Exception)
def global_exception_handler(request, exc):
    logger.exception("Unhandled error: %s", exc)
    return JSONResponse(status_code=500, content={"detail": "Internal server error", "error": str(exc)})


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.app.main:app", host="0.0.0.0", port=8000, reload=True)
