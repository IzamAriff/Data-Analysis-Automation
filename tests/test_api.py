"""Backend API tests — fullstack rebuild."""

from __future__ import annotations

import io
from pathlib import Path

import pandas as pd
import pytest
from fastapi.testclient import TestClient

import sys
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from backend.app.main import app
from src import loader

client = TestClient(app)

def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"

def test_api_health():
    r = client.get("/api/v1/health")
    assert r.status_code == 200

def test_samples():
    r = client.get("/api/v1/data/samples")
    assert r.status_code == 200
    assert len(r.json()["samples"]) >= 1

def _load_sample_via_api(label: str) -> str:
    r = client.post(f"/api/v1/data/sample/{label}")
    assert r.status_code == 200, r.text
    return r.json()["dataset_id"]

def test_upload_and_prepare_flow():
    # Create small CSV in memory
    df = pd.DataFrame({
        "date": ["2024-01-01", "2024-02-01", "2024-03-01"],
        "sales": [100, 200, 300],
        "region": ["A", "B", "A"]
    })
    csv_bytes = df.to_csv(index=False).encode()
    r = client.post("/api/v1/data/upload", files={"file": ("test.csv", csv_bytes, "text/csv")})
    assert r.status_code == 200, r.text
    dataset_id = r.json()["dataset_id"]

    # prepare
    r2 = client.post("/api/v1/data/prepare", json={"dataset_id": dataset_id})
    assert r2.status_code == 200, r2.text
    assert "roles" in r2.json()
    roles = r2.json()["roles"]
    assert roles["sales"] == "numeric"

    # kpi
    r3 = client.post("/api/v1/analysis/kpi", json={"dataset_id": dataset_id, "metric": "sales", "date_col": "date"})
    assert r3.status_code == 200

    # group stats
    r4 = client.post("/api/v1/analysis/group-stats", json={"dataset_id": dataset_id, "metric": "sales", "group_col": "region"})
    assert r4.status_code == 200

def test_full_superstore_flow():
    # Load bundled sample via API
    label = "Retail orders — 'Sample Superstore' (9,994 rows)"
    dataset_id = _load_sample_via_api(label)

    r = client.post("/api/v1/data/prepare", json={"dataset_id": dataset_id})
    assert r.status_code == 200
    roles = r.json()["roles"]
    assert "Sales" in roles

    # correlation
    r2 = client.post("/api/v1/analysis/correlation", json={"dataset_id": dataset_id})
    assert r2.status_code == 200
    assert "top_pairs" in r2.json()

    # outliers
    r3 = client.post("/api/v1/analysis/outliers", json={"dataset_id": dataset_id})
    assert r3.status_code == 200

    # trend
    date_cols = [c for c, r in roles.items() if r == "date"]
    numeric_cols = [c for c, r in roles.items() if r == "numeric"]
    if date_cols and numeric_cols:
        r4 = client.post("/api/v1/analysis/trend", json={
            "dataset_id": dataset_id,
            "date_col": date_cols[0],
            "value_col": numeric_cols[0],
            "freq": "M",
            "agg": "sum"
        })
        assert r4.status_code == 200

    # forecast
    if date_cols and numeric_cols:
        r5 = client.post("/api/v1/modeling/forecast", json={
            "dataset_id": dataset_id,
            "date_col": date_cols[0],
            "value_col": numeric_cols[0],
            "freq": "M",
            "periods": 6
        })
        # may succeed or fail depending on data, but should not 500
        assert r5.status_code in (200, 400)

def test_modeling_regression():
    label = "Retail orders — 'Sample Superstore' (9,994 rows)"
    dataset_id = _load_sample_via_api(label)
    client.post("/api/v1/data/prepare", json={"dataset_id": dataset_id})

    # get roles first
    r = client.get(f"/api/v1/data/{dataset_id}")
    assert r.status_code == 200

    # try regression Profit ~ Sales
    r2 = client.post("/api/v1/modeling/regression", json={
        "dataset_id": dataset_id,
        "target": "Profit",
        "features": ["Sales", "Quantity", "Discount"]
    })
    assert r2.status_code in (200, 400)
    if r2.status_code == 200:
        assert "r2" in str(r2.json()) or "linear" in r2.json()

def test_plots():
    df = pd.DataFrame({
        "date": pd.date_range("2024-01-01", periods=10),
        "sales": range(10),
        "region": ["A","B"]*5
    })
    csv_bytes = df.to_csv(index=False).encode()
    r = client.post("/api/v1/data/upload", files={"file": ("test.csv", csv_bytes, "text/csv")})
    dataset_id = r.json()["dataset_id"]
    client.post("/api/v1/data/prepare", json={"dataset_id": dataset_id})

    r2 = client.post("/api/v1/plots/generate", json={
        "dataset_id": dataset_id,
        "chart_type": "histogram",
        "params": {"col": "sales"}
    })
    assert r2.status_code == 200
    assert "data" in r2.json()
