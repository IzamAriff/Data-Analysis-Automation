"""Predictive modeling endpoints."""

from __future__ import annotations

from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from fastapi import APIRouter, HTTPException, Depends
import numpy as np

from src import modeling as modeling_mod
from src.modeling import ModelingError

from ..config import get_settings
from ..models.schemas import (
    RegressionRequest,
    ClassificationRequest,
    ClusteringRequest,
    ForecastRequest,
)
from ..services.session_store import get_store, SessionStore
from ..services.data_service import apply_user_filters
from ..utils import clean_nans, df_to_dict

router = APIRouter(prefix="/modeling", tags=["modeling"])

def _store_dep() -> SessionStore:
    return get_store(ttl_seconds=get_settings().session_ttl_seconds)

def _get_prepared(dataset_id: str, store: SessionStore):
    sess = store.get(dataset_id)
    if not sess or sess.df_prepared is None or sess.roles is None:
        raise HTTPException(status_code=404, detail="Dataset not prepared")
    return sess


@router.post("/regression")
def regression(req: RegressionRequest, store: SessionStore = Depends(_store_dep)):
    sess = _get_prepared(req.dataset_id, store)
    df = apply_user_filters(sess.df_prepared, req.filters) if req.filters else sess.df_prepared
    try:
        result = modeling_mod.run_regression(
            df, req.target, req.features, sess.roles, req.missing_strategy, req.with_random_forest
        )
    except ModelingError as e:
        raise HTTPException(status_code=400, detail=str(e))
    linear = result["linear"]
    out = clean_nans({
        "kind": "regression",
        "target": result["target"],
        "n_features": result["n_features"],
        "n_rows": result["n_rows"],
        "baseline_rmse": result["baseline_rmse"],
        "linear": {
            "r2": linear["r2"],
            "rmse": linear["rmse"],
            "mae": linear["mae"],
            "fold_scores": linear["fold_scores"],
        },
        "linear_coefs": {k: float(v) for k, v in result.get("linear_coefs", {}).items()},
        "notes": result["notes"],
    })
    if result.get("random_forest"):
        rf = result["random_forest"]
        out["random_forest"] = clean_nans({
            "r2": rf["r2"],
            "rmse": rf["rmse"],
            "mae": rf["mae"],
            "fold_scores": rf["fold_scores"],
        })
        if result.get("rf_importance"):
            out["rf_importance"] = clean_nans([{"feature": str(k), "importance": float(v)} for k, v in result["rf_importance"]])
    return out


@router.post("/classification")
def classification(req: ClassificationRequest, store: SessionStore = Depends(_store_dep)):
    sess = _get_prepared(req.dataset_id, store)
    df = apply_user_filters(sess.df_prepared, req.filters) if req.filters else sess.df_prepared
    try:
        result = modeling_mod.run_classification(
            df, req.target, req.features, sess.roles, req.missing_strategy
        )
    except ModelingError as e:
        raise HTTPException(status_code=400, detail=str(e))

    cm = result["confusion_matrix"]
    if hasattr(cm, "tolist"):
        cm = cm.tolist()

    return clean_nans({
        "kind": "classification",
        "target": result["target"],
        "n_features": result["n_features"],
        "n_rows": result["n_rows"],
        "n_classes": result["n_classes"],
        "baseline_accuracy": result["baseline_accuracy"],
        "accuracy": result["accuracy"],
        "macro_f1": result["macro_f1"],
        "confusion_matrix": cm,
        "class_labels": result["class_labels"],
        "importance": [{"feature": str(k), "importance": float(v)} for k, v in result["importance"]],
        "notes": result["notes"],
    })


@router.post("/clustering")
def clustering(req: ClusteringRequest, store: SessionStore = Depends(_store_dep)):
    sess = _get_prepared(req.dataset_id, store)
    df = apply_user_filters(sess.df_prepared, req.filters) if req.filters else sess.df_prepared
    try:
        result = modeling_mod.run_clustering(df, req.features, sess.roles, k_min=req.k_min, k_max=req.k_max)
    except ModelingError as e:
        raise HTTPException(status_code=400, detail=str(e))

    return clean_nans({
        "kind": "clustering",
        "k_range": result["k_range"],
        "inertias": result["inertias"],
        "silhouettes": result["silhouettes"],
        "best_k": result["best_k"],
        "explained_variance": result["explained_variance"],
        "cluster_sizes": result["cluster_sizes"],
        "cluster_means": df_to_dict(result["cluster_means"]),
        "n_rows": result["n_rows"],
        "features_used": result["features_used"],
        "pca_x": result["pca_x"][:1000].tolist(),
        "pca_y": result["pca_y"][:1000].tolist(),
        "labels": result["labels"][:1000],
    })


@router.post("/forecast")
def forecast(req: ForecastRequest, store: SessionStore = Depends(_store_dep)):
    sess = _get_prepared(req.dataset_id, store)
    df = apply_user_filters(sess.df_prepared, req.filters) if req.filters else sess.df_prepared
    try:
        result = modeling_mod.run_forecast(
            df, req.date_col, req.value_col, req.agg, req.freq, req.periods
        )
    except ModelingError as e:
        raise HTTPException(status_code=400, detail=str(e))

    def series_to_points(s):
        return [{"date": str(idx), "value": float(v) if not (v != v) else None} for idx, v in s.items()]

    return clean_nans({
        "kind": "forecast",
        "history": series_to_points(result["history"]),
        "fitted": series_to_points(result["fitted"]) if result.get("fitted") is not None else [],
        "forecast": series_to_points(result["forecast"]),
        "ci_lower": series_to_points(result["ci_lower"]) if result.get("ci_lower") is not None else [],
        "ci_upper": series_to_points(result["ci_upper"]) if result.get("ci_upper") is not None else [],
        "mape_model": result.get("mape_model"),
        "mape_naive": result.get("mape_naive"),
        "freq": result["freq"],
        "agg": result["agg"],
        "method": result["method"],
        "periods": result["periods"],
    })
