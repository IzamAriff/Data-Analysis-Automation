"""DataPilot — Automated Data Analysis Studio.

A self-serve Streamlit application that profiles, visualises, diagnoses and
models *any* tabular dataset the user provides (upload, URL or bundled sample).

Sub-modules:
    loader     — file/URL/sample ingestion with encoding, delimiter and date detection
    profile    — column-role inference and dataset profiling
    analysis   — descriptive statistics, correlations and hypothesis tests
    modeling   — regression, classification, clustering and time-series forecasting
    plots      — colorblind-safe Plotly chart builders
    ui         — shared UI helpers (filters, KPI cards, download buttons)
"""

__version__ = "1.0.0"
APP_NAME = "DataPilot"
