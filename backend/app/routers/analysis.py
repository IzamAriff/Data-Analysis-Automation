"""Diagnostics & descriptive analytics."""

from __future__ import annotations

from pathlib import Path
import sys
REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from fastapi import APIRouter, HTTPException, Depends
import pandas as pd

from src import analysis as analysis_mod

from ..config import get_settings
from ..models.schemas import (
    AnalysisRequest,
    CorrelationRequest,
    GroupStatsRequest,
    AnovaRequest,
    ChiSquareRequest,
    OutlierRequest,
    TrendRequest,
)
from ..services.session_store import get_store, SessionStore
from ..services.data_service import apply_user_filters
from ..utils import df_to_records, df_to_dict, clean_nans

router = APIRouter(prefix="/analysis", tags=["analysis"])

def _store_dep() -> SessionStore:
    return get_store(ttl_seconds=get_settings().session_ttl_seconds)

def _get_prepared(dataset_id: str, store: SessionStore):
    sess = store.get(dataset_id)
    if not sess or sess.df_prepared is None or sess.roles is None:
        raise HTTPException(status_code=404, detail="Dataset not prepared")
    return sess


@router.post("/kpi")
def kpi(req: AnalysisRequest, store: SessionStore = Depends(_store_dep)):
    sess = _get_prepared(req.dataset_id, store)
    df = apply_user_filters(sess.df_prepared, req.filters) if req.filters else sess.df_prepared
    try:
        snap = analysis_mod.kpi_snapshot(df, sess.roles, req.metric, req.date_col, full_rows=len(sess.df_prepared))
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    # serialize dates and clean NaN
    if "date_span" in snap and snap["date_span"]:
        lo, hi = snap["date_span"]
        snap["date_span"] = [str(lo), str(hi)]
    return clean_nans(snap)


@router.post("/correlation")
def correlation(req: CorrelationRequest, store: SessionStore = Depends(_store_dep)):
    sess = _get_prepared(req.dataset_id, store)
    df = apply_user_filters(sess.df_prepared, req.filters) if req.filters else sess.df_prepared
    numeric_cols = [c for c, r in sess.roles.items() if r == "numeric"]
    if len(numeric_cols) < 2:
        raise HTTPException(status_code=400, detail="Need at least 2 numeric columns")
    try:
        r_mat, p_mat = analysis_mod.correlation_matrix(df, numeric_cols, method=req.method, max_cols=req.max_cols)
        pairs = analysis_mod.top_correlated_pairs(r_mat, p_mat)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    return clean_nans({
        "r": r_mat.round(3).to_dict(),
        "p": p_mat.to_dict(),
        "top_pairs": df_to_records(pairs),
        "columns": list(r_mat.columns),
    })


@router.post("/group-stats")
def group_stats(req: GroupStatsRequest, store: SessionStore = Depends(_store_dep)):
    sess = _get_prepared(req.dataset_id, store)
    df = apply_user_filters(sess.df_prepared, req.filters) if req.filters else sess.df_prepared
    try:
        table = analysis_mod.group_stats(df, req.metric, req.group_col, max_groups=req.max_groups)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    return {"table": df_to_records(table)}


@router.post("/anova")
def anova(req: AnovaRequest, store: SessionStore = Depends(_store_dep)):
    sess = _get_prepared(req.dataset_id, store)
    df = apply_user_filters(sess.df_prepared, req.filters) if req.filters else sess.df_prepared
    try:
        res = analysis_mod.anova_test(df, req.metric, req.group_col)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    return clean_nans(res)


@router.post("/chi-square")
def chi_square(req: ChiSquareRequest, store: SessionStore = Depends(_store_dep)):
    sess = _get_prepared(req.dataset_id, store)
    df = apply_user_filters(sess.df_prepared, req.filters) if req.filters else sess.df_prepared
    try:
        res = analysis_mod.chi_square_test(df, req.col_a, req.col_b)
        cont = analysis_mod.contingency_table(df, req.col_a, req.col_b)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    return clean_nans({
        "chi2": res["chi2"],
        "p_value": res["p_value"],
        "dof": res["dof"],
        "n": res["n"],
        "cramers_v": res["cramers_v"],
        "verdict": res["verdict"],
        "contingency": df_to_dict(cont),
    })


@router.post("/outliers")
def outliers(req: OutlierRequest, store: SessionStore = Depends(_store_dep)):
    sess = _get_prepared(req.dataset_id, store)
    df = apply_user_filters(sess.df_prepared, req.filters) if req.filters else sess.df_prepared
    numeric_cols = [c for c, r in sess.roles.items() if r == "numeric"]
    try:
        table = analysis_mod.outlier_summary(df, numeric_cols)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    return {"table": df_to_records(table)}


@router.post("/trend")
def trend(req: TrendRequest, store: SessionStore = Depends(_store_dep)):
    sess = _get_prepared(req.dataset_id, store)
    df = apply_user_filters(sess.df_prepared, req.filters) if req.filters else sess.df_prepared
    try:
        if req.group_col:
            wide = analysis_mod.trend_by_group(df, req.date_col, req.value_col, req.group_col, agg=req.agg, freq=req.freq)
            wide_json = {}
            for col in wide.columns:
                series = wide[col].dropna()
                wide_json[str(col)] = [{"date": str(idx), "value": float(val)} for idx, val in series.items()]
            return {"type": "grouped", "data": wide_json}
        else:
            ts = analysis_mod.trend_series(df, req.date_col, req.value_col, agg=req.agg, freq=req.freq)
            points = [{"date": str(idx), "value": float(val)} for idx, val in ts.items()]
            return {"type": "single", "data": points}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
