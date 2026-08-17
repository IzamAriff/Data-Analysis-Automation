"""Plot generation — returns Plotly JSON."""

from __future__ import annotations

from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from fastapi import APIRouter, HTTPException, Depends

from src import plots as plots_mod, analysis as analysis_mod
from ..config import get_settings
from ..models.schemas import PlotRequest
from ..services.session_store import get_store, SessionStore
from ..services.data_service import apply_user_filters
from ..utils import clean_nans

router = APIRouter(prefix="/plots", tags=["plots"])

def _store_dep() -> SessionStore:
    return get_store(ttl_seconds=get_settings().session_ttl_seconds)

def _get_prepared(dataset_id: str, store: SessionStore):
    sess = store.get(dataset_id)
    if not sess or sess.df_prepared is None or sess.roles is None:
        raise HTTPException(status_code=404, detail="Dataset not prepared")
    return sess


@router.post("/generate")
def generate_plot(req: PlotRequest, store: SessionStore = Depends(_store_dep)):
    sess = _get_prepared(req.dataset_id, store)
    df = apply_user_filters(sess.df_prepared, req.filters) if req.filters else sess.df_prepared
    p = req.params

    try:
        if req.chart_type == "trend":
            ts = analysis_mod.trend_series(df, p["date_col"], p["value_col"], agg=p.get("agg","sum"), freq=p.get("freq","M"))
            fig = plots_mod.trend_chart(ts, p["value_col"], title=p.get("title",""))
        elif req.chart_type == "grouped_trend":
            wide = analysis_mod.trend_by_group(df, p["date_col"], p["value_col"], p["group_col"], agg=p.get("agg","sum"), freq=p.get("freq","M"))
            fig = plots_mod.grouped_trend_chart(wide, p["value_col"], p["group_col"], title=p.get("title",""))
        elif req.chart_type == "histogram":
            fig = plots_mod.histogram_chart(df, p["col"], group_col=p.get("group_col"), bins=p.get("bins",40), title=p.get("title",""))
        elif req.chart_type == "box":
            fig = plots_mod.box_chart(df, p["col"], p["group_col"], title=p.get("title",""))
        elif req.chart_type == "bar":
            cat_col = p["cat_col"]
            value_col = p.get("value_col")
            agg = p.get("agg","sum")
            if value_col:
                agg_df = df.groupby(cat_col, observed=True)[value_col].agg(agg).reset_index()
            else:
                agg_df = df[cat_col].value_counts().reset_index()
                agg_df.columns = [cat_col, "count"]
                value_col = "count"
            fig = plots_mod.bar_chart(agg_df, cat_col, value_col, title=p.get("title",""), horizontal=p.get("horizontal", False))
        elif req.chart_type == "scatter":
            fig = plots_mod.scatter_chart(df, p["x"], p["y"], color=p.get("color"), size=p.get("size"), trendline=p.get("trendline", False), title=p.get("title",""))
        elif req.chart_type == "heatmap":
            numeric_cols = [c for c, r in sess.roles.items() if r == "numeric"]
            r_mat, _ = analysis_mod.correlation_matrix(df, numeric_cols, method=p.get("method","pearson"))
            fig = plots_mod.correlation_heatmap(r_mat, method_label=p.get("method","Pearson"), title=p.get("title",""))
        elif req.chart_type == "composition":
            cat_col = p["cat_col"]
            value_col = p["value_col"]
            agg_df = df.groupby(cat_col, observed=True)[value_col].sum().reset_index()
            fig = plots_mod.composition_chart(agg_df, cat_col, value_col, kind=p.get("kind","treemap"), title=p.get("title",""))
        elif req.chart_type == "missing":
            missing_pct = df.isna().mean()*100
            fig = plots_mod.missing_chart(missing_pct, title=p.get("title",""))
        elif req.chart_type == "elbow":
            from src import modeling as modeling_mod
            cl_res = modeling_mod.run_clustering(df, p["features"], sess.roles, k_min=p.get("k_min",2), k_max=p.get("k_max",8))
            fig = plots_mod.elbow_chart(cl_res["k_range"], cl_res["inertias"], cl_res["silhouettes"])
        else:
            raise HTTPException(status_code=400, detail=f"Chart type {req.chart_type} not implemented in this endpoint (use dedicated modeling endpoints for forecast/cluster/confusion)")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

    return clean_nans(fig.to_dict())
