"""DataPilot — Automated Data Analysis Studio (Streamlit entry point).

The app adapts to *any* tabular dataset the user provides:

  1. Load    — upload a file, fetch a URL, or use a bundled public sample.
  2. Profile — the app infers column roles (date / numeric / category / id / …),
               shows a data-quality summary, and lets the user override roles.
  3. Explore — KPI cards, interactive charts, diagnostics (correlations,
               ANOVA, chi-square), predictive models (regression,
               classification, clustering, forecasting), a data dictionary,
               raw data browsing and a downloadable auto-report.

Everything responds to the dynamic filters in the sidebar. All computation
is cached where expensive, errors are caught per section with friendly
messages, and activity is logged to logs/app.log.

Run locally with:  streamlit run app.py
"""

from __future__ import annotations

import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import pandas as pd
import streamlit as st

from src import analysis, loader, modeling, plots, profile, ui
from src.loader import DataBundle, LoaderError
from src.modeling import ModelingError

# --------------------------------------------------------------------------- #
# Logging — console + file (relative path, never absolute in the UI)
# --------------------------------------------------------------------------- #
REPO_ROOT = Path(__file__).resolve().parent
LOG_DIR = REPO_ROOT / "logs"
LOG_DIR.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOG_DIR / "app.log", encoding="utf-8"),
    ],
)
logger = logging.getLogger("datapilot")

st.set_page_config(
    page_title="DataPilot — Automated Data Analysis Studio",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --------------------------------------------------------------------------- #
# Cached, expensive steps
# --------------------------------------------------------------------------- #
@st.cache_data(show_spinner=False)
def cached_load_bytes(data: bytes, name: str, suffix: str) -> DataBundle:
    return loader.load_from_bytes(data, name, suffix)


@st.cache_data(show_spinner=False)
def cached_load_url(url: str) -> DataBundle:
    return loader.load_from_url(url)


@st.cache_data(show_spinner=False)
def cached_load_sample(label: str) -> DataBundle:
    return loader.load_bundled_sample(label)


@st.cache_data(show_spinner=False)
def cached_prepare(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    return loader.prepare_dataframe(df)


@st.cache_data(show_spinner="Fitting regression models…")
def cached_regression(df, target, features, roles, missing, rf) -> Dict:
    return modeling.run_regression(df, target, list(features), dict(roles), missing, rf)


@st.cache_data(show_spinner="Fitting the classifier…")
def cached_classification(df, target, features, roles, missing) -> Dict:
    return modeling.run_classification(df, target, list(features), dict(roles), missing)


@st.cache_data(show_spinner="Clustering…")
def cached_clustering(df, features, roles, k_max) -> Dict:
    return modeling.run_clustering(df, list(features), dict(roles), k_min=2, k_max=k_max)


@st.cache_data(show_spinner="Forecasting…")
def cached_forecast(df, date_col, value_col, agg, freq, periods) -> Dict:
    return modeling.run_forecast(df, date_col, value_col, agg, freq, periods)


# --------------------------------------------------------------------------- #
# Session-state helpers
# --------------------------------------------------------------------------- #
_TRANSIENT_PREFIXES = ("f_", "c_", "diag_", "pred_", "role_editor", "chart_")
_CORE_KEYS = {
    "bundle", "df", "roles", "prep_notes", "explored", "source_key",
    "active_sheet", "last_results", "load_notes",
}


def reset_everything() -> None:
    """Drop dataset + all widget state so a fresh dataset can be loaded."""
    for key in list(st.session_state.keys()):
        if key in _CORE_KEYS or key.startswith(_TRANSIENT_PREFIXES):
            del st.session_state[key]
    st.session_state["filter_epoch"] = 0
    logger.info("Session reset — loading screen shown.")


def numeric_columns(roles: Dict[str, str]) -> List[str]:
    return [c for c, r in roles.items() if r == "numeric"]


def categorical_columns(roles: Dict[str, str]) -> List[str]:
    return [c for c, r in roles.items() if r in ("category", "binary", "boolean")]


def time_columns(roles: Dict[str, str]) -> List[str]:
    return [c for c, r in roles.items() if r in ("date", "year")]


# --------------------------------------------------------------------------- #
# Landing screen
# --------------------------------------------------------------------------- #
def render_landing() -> None:
    st.markdown(
        """
        # 📊 DataPilot
        ### Automated Data Analysis Studio

        Upload **any tabular dataset** and DataPilot will profile it, infer the
        roles of every column, and build an interactive dashboard with KPIs,
        charts, diagnostics and predictive models — no hard-coded column names,
        no setup, no code.
        """
    )
    left, mid, right = st.columns(3)
    left.markdown("**1️⃣ Load**\n\nUpload CSV/Excel/Parquet/JSON, paste a URL, or explore a bundled public sample.")
    mid.markdown("**2️⃣ Profile**\n\nReview the auto-detected data types & column roles, then adjust anything before you start.")
    right.markdown("**3️⃣ Explore**\n\nFilter, visualise, diagnose and model — every view updates live with your filters.")

    st.divider()
    mode = st.segmented_control(
        "Choose a data source",
        ["📁 Upload a file", "🌐 From a URL", "🎁 Sample dataset"],
        default="🎁 Sample dataset",
        key="load_mode",
    )

    if mode == "📁 Upload a file":
        uploaded = st.file_uploader(
            "Upload your data file",
            type=["csv", "tsv", "txt", "xlsx", "xls", "parquet", "json"],
            accept_multiple_files=False,
            help=f"Maximum {loader.MAX_FILE_BYTES // 1_000_000} MB. The file never leaves your session.",
        )
        if uploaded is not None:
            try:
                with st.spinner(f"Reading {uploaded.name}…"):
                    bundle = cached_load_bytes(uploaded.getvalue(), uploaded.name, Path(uploaded.name).suffix)
                _activate_bundle(bundle)
                st.rerun()
            except LoaderError as exc:
                st.error(f"Could not load the file: {exc}")
                logger.warning("Upload failed for '%s': %s", uploaded.name, exc)

    elif mode == "🌐 From a URL":
        url = st.text_input(
            "Public URL of the data file",
            placeholder="https://example.com/data.csv",
            key="load_url",
            help="Only http(s) URLs are allowed. The file is downloaded with a 30-second timeout.",
        )
        if st.button("⬇️ Download & load", key="load_url_btn"):
            if not url.strip():
                st.warning("Please paste a URL first.")
            else:
                try:
                    with st.spinner("Downloading…"):
                        bundle = cached_load_url(url)
                    _activate_bundle(bundle)
                    st.rerun()
                except LoaderError as exc:
                    st.error(f"Could not download the file: {exc}")
                    logger.warning("URL load failed: %s", exc)

    else:
        try:
            samples = loader.bundled_sample_names()
        except LoaderError as exc:
            st.error(str(exc))
            return
        label = st.selectbox("Bundled public dataset", list(samples), key="sample_select")
        st.caption(
            "These are real public datasets shipped with the app for instant "
            "exploration (see the README for provenance). Upload your own data "
            "to analyse it instead."
        )
        if st.button("🚀 Load sample data", key="load_sample", type="primary"):
            with st.spinner("Loading…"):
                bundle = cached_load_sample(label)
            _activate_bundle(bundle)
            st.rerun()


def _activate_bundle(bundle: DataBundle) -> None:
    """Store a freshly loaded bundle and clear any stale widget state."""
    for key in list(st.session_state.keys()):
        if key in _CORE_KEYS or key.startswith(_TRANSIENT_PREFIXES):
            del st.session_state[key]
    st.session_state["bundle"] = bundle
    st.session_state["active_sheet"] = None
    st.session_state["load_notes"] = list(bundle.notes)
    st.session_state["explored"] = False
    logger.info("Activated dataset '%s' (%d×%d) from %s", bundle.name, *bundle.df.shape, bundle.source)


# --------------------------------------------------------------------------- #
# Profile & confirm step
# --------------------------------------------------------------------------- #
def render_confirm_step(bundle: DataBundle) -> None:
    st.markdown(f"## ✅ Data loaded: `{bundle.name}`")

    if bundle.sheets and len(bundle.sheets) > 1:
        sheet_names = list(bundle.sheets)
        chosen = st.selectbox(
            f"This workbook has {len(bundle.sheets)} sheets — pick one to analyse",
            sheet_names,
            key="sheet_select",
        )
        raw_df = bundle.sheets[chosen].copy()
    else:
        chosen = None
        raw_df = bundle.df.copy()
    raw_df.columns = loader.sanitize_column_names(raw_df.columns)

    for note in bundle.notes:
        st.caption(f"ℹ️ {note}")

    # Data-quality summary before any user decision.
    prepared, prep_notes = cached_prepare(raw_df)
    inferred_roles = profile.infer_roles(prepared)
    summary = profile.profile_summary(prepared, inferred_roles)

    cols = st.columns(6)
    cols[0].metric("Rows", f"{summary['rows']:,}")
    cols[1].metric("Columns", f"{summary['columns']:,}")
    cols[2].metric("Duplicate rows", f"{summary['duplicate_rows']:,}")
    cols[3].metric("Missing cells", f"{summary['missing_pct']:.2f}%")
    if summary["date_columns"]:
        date_col = summary["date_columns"][0]
        lo, hi = summary["date_spans"][date_col]
        cols[4].metric("Date span", f"{lo:%Y-%m-%d} → {hi:%Y-%m-%d}")
        cols[5].metric("Memory", f"{summary['memory_mb']:.1f} MB")
    else:
        cols[4].metric("Date span", "—")
        cols[5].metric("Memory", f"{summary['memory_mb']:.1f} MB")

    st.markdown("### 🧠 Detected column roles")
    st.caption(
        "Roles drive the whole app (filters, charts, models). "
        "Override anything that looks wrong — e.g. a code column used as a measure. "
        + profile.dataset_structure_hint(inferred_roles)
    )
    role_options = sorted(set(profile.ROLE_ORDER) | set(inferred_roles.values()))
    roles_table = pd.DataFrame(
        {
            "Column": list(prepared.columns),
            "Role": [inferred_roles.get(c, "text") for c in prepared.columns],
        }
    )
    edited_roles = st.data_editor(
        roles_table,
        column_config={
            "Column": st.column_config.TextColumn(disabled=True),
            "Role": st.column_config.SelectboxColumn(options=role_options, required=True, width="small"),
        },
        hide_index=True,
        key="role_editor",
        width="stretch",
        num_rows="fixed",
    )
    final_roles: Dict[str, str] = dict(zip(edited_roles["Column"], edited_roles["Role"]))

    with st.expander("👀 Preview of the first 100 rows"):
        st.dataframe(prepared.head(100), width="stretch", height=320)

    st.markdown("### 🧹 Preprocessing choices")
    opt1, opt2 = st.columns(2)
    with opt1:
        st.selectbox(
            "Missing values for **models**",
            ["median", "mean", "drop"],
            index=0,
            key="missing_strategy",
            help=(
                "Charts and descriptive stats always skip missing values per column. "
                "This choice only controls how predictive models handle them: "
                "impute numeric features with the median/mean (categoricals get a "
                "'(missing)' level), or drop incomplete rows."
            ),
        )
    with opt2:
        drop_duplicates = st.checkbox(
            "Drop exact duplicate rows", value=True, key="drop_duplicates",
            help=f"The dataset contains {summary['duplicate_rows']:,} exact duplicate row(s).",
        )

    st.info(
        "Already applied automatically (reversible, logged below): column-name "
        "cleanup, datetime parsing and numeric-string parsing (e.g. currency). "
        "Every step is recorded in the preprocessing log shown on the Overview tab."
    )
    if st.button("✅ Confirm & open the dashboard", key="confirm_explore", type="primary"):
        chosen_strategy = st.session_state.get("missing_strategy", "median")
        final_df = prepared.drop_duplicates() if drop_duplicates else prepared
        notes = list(prep_notes)
        notes.append(
            f"Missing-value strategy for models: '{chosen_strategy}'. "
            f"Exact duplicate rows: {'dropped' if drop_duplicates else 'kept'}."
        )
        st.session_state["df"] = final_df
        st.session_state["roles"] = final_roles
        st.session_state["prep_notes"] = notes
        st.session_state["explored"] = True
        st.session_state["active_sheet"] = chosen
        st.session_state["source_key"] = loader.data_hash(final_df)
        logger.info(
            "Confirmed dataset: %d rows × %d cols; roles: %s",
            *final_df.shape,
            {r: sum(1 for v in final_roles.values() if v == r) for r in profile.ROLE_ORDER},
        )
        st.rerun()


# --------------------------------------------------------------------------- #
# Main dashboard
# --------------------------------------------------------------------------- #
def render_main_app() -> None:
    bundle: DataBundle = st.session_state["bundle"]
    df: pd.DataFrame = st.session_state["df"]
    roles: Dict[str, str] = st.session_state["roles"]
    prep_notes: List[str] = st.session_state.get("prep_notes", [])
    missing_strategy: str = st.session_state.get("missing_strategy", "median")
    metric = profile.primary_metric(df, roles)

    # ------------------------- Sidebar -------------------------------- #
    with st.sidebar:
        st.title("📊 DataPilot")
        st.caption(f"**Dataset:** {bundle.name}")
        st.caption(f"{len(df):,} rows × {df.shape[1]} columns")
        if st.button("🔄 Load different data", key="load_other", width="stretch"):
            reset_everything()
            st.rerun()
        st.divider()
        filter_state = ui.render_filter_sidebar(df, roles)
        if filter_state.is_active():
            with st.container(border=True):
                st.caption("**Active filters**")
                for chip in filter_state.descriptions():
                    st.caption(f"• {chip}")
                if st.button("🧹 Reset filters", key="reset_filters", width="stretch"):
                    ui.reset_filter_widgets()
                    st.rerun()
        st.divider()
        st.markdown(
            "<small>Data stays in your browser session — nothing is uploaded anywhere.</small>",
            unsafe_allow_html=True,
        )

    # ------------------------- Filtering ------------------------------ #
    fdf = ui.apply_filters(df, filter_state)
    if fdf.empty:
        st.warning("⚠️ No rows match the current filter combination.")
        st.caption("Adjust or reset the filters in the sidebar to continue.")
        logger.info("Filters left 0 rows: %s", filter_state.to_key())
        return
    logger.info("Filters applied: %d -> %d rows (%s)", len(df), len(fdf), filter_state.to_key()[:200])

    st.markdown(f"### 📊 {bundle.name}")
    st.caption(
        f"Showing **{len(fdf):,}** of {len(df):,} rows • "
        f"{profile.dataset_structure_hint(roles)}"
    )

    tab_overview, tab_charts, tab_diagnostics, tab_predictive, tab_dictionary, tab_raw, tab_report = st.tabs(
        ["📊 Overview", "📈 Charts", "🔎 Diagnostics", "🤖 Predictive", "📚 Data Dictionary", "🗂️ Raw Data", "📄 Report"]
    )

    with tab_overview:
        _guarded("overview", lambda: render_overview(fdf, df, roles, prep_notes, metric))
    with tab_charts:
        _guarded("charts", lambda: render_charts(fdf, roles, metric))
    with tab_diagnostics:
        _guarded("diagnostics", lambda: render_diagnostics(fdf, roles))
    with tab_predictive:
        _guarded("predictive", lambda: render_predictive(fdf, roles, missing_strategy))
    with tab_dictionary:
        _guarded("dictionary", lambda: render_dictionary(df, roles))
    with tab_raw:
        _guarded("raw data", lambda: render_raw_data(fdf))
    with tab_report:
        _guarded("report", lambda: render_report(fdf, df, roles, filter_state, prep_notes, metric))


def _guarded(section: str, render_fn) -> None:
    """Run a tab's renderer, converting unexpected errors into friendly UI."""
    try:
        render_fn()
    except (LoaderError, ModelingError) as exc:
        st.warning(str(exc))
    except Exception as exc:  # noqa: BLE001 — top-level guard per tab
        logger.exception("Error rendering %s", section)
        st.error(
            f"Something went wrong in the **{section}** section: `{exc}`. "
            "Check logs/app.log for details and adjust the inputs."
        )


# --------------------------------------------------------------------------- #
# Overview tab
# --------------------------------------------------------------------------- #
def render_overview(
    fdf: pd.DataFrame, full_df: pd.DataFrame, roles: Dict[str, str],
    prep_notes: List[str], metric: Optional[str],
) -> None:
    date_cols = [c for c, r in roles.items() if r == "date"]
    date_col = date_cols[0] if date_cols else None
    snapshot = analysis.kpi_snapshot(fdf, roles, metric, date_col, full_rows=len(full_df))

    cards: List[Tuple[str, str, Optional[float]]] = [
        ("Rows", f"{snapshot['rows']:,}", snapshot.get("rows_delta")),
        ("Columns", f"{snapshot['columns']:,}", None),
        ("Duplicate rows", f"{snapshot['duplicate_rows']:,}", None),
        ("Missing cells", f"{snapshot['missing_pct']:.2f}%", None),
    ]
    if snapshot.get("date_span"):
        lo, hi = snapshot["date_span"]
        cards.append(("Date span", f"{lo:%d %b %Y} → {hi:%d %b %Y}", None))
    else:
        cards.append(("Date span", "—", None))
    if snapshot.get("metric"):
        cards.append(("Total " + snapshot["metric"], plots.fmt(snapshot["metric_total"], 2), snapshot.get("metric_delta")))
        cards.append(("Avg " + snapshot["metric"], plots.fmt(snapshot["metric_mean"], 2), None))
    if snapshot.get("top_category"):
        top = snapshot["top_category"]
        cards.append(("Top " + top["column"], str(top["value"]), None))
    ui.kpi_cards(cards, columns=4)
    if snapshot.get("top_category"):
        top = snapshot["top_category"]
        st.caption(f"Top **{top['column']}**: '{top['value']}' with {top['count']:,} rows ({top['share']:.1f}% of the filtered data).")

    left, right = st.columns([1, 1])
    with left:
        st.markdown("#### Data quality — missing values")
        missing_pct = 100.0 * full_df.isna().sum() / max(len(full_df), 1)
        st.plotly_chart(plots.missing_chart(missing_pct), width="stretch", key="ov_missing_chart")
    with right:
        st.markdown("#### Column roles")
        st.dataframe(
            profile.column_profile(full_df, roles),
            width="stretch",
            height=340,
            column_config={
                "Missing %": st.column_config.ProgressColumn(min_value=0.0, max_value=100.0, format="%.1f%%"),
            },
        )

    st.markdown("#### Descriptive statistics (numeric columns)")
    stats_table = profile.numeric_describe(fdf, numeric_columns(roles))
    if stats_table.empty:
        st.info("No numeric columns detected — descriptive statistics are not available.")
    else:
        st.dataframe(
            stats_table.style.format(
                {c: "{:,.2f}" for c in ["Mean", "Std dev", "Min", "Q1", "Median", "Q3", "Max"]}
            ),
            width="stretch",
        )

    with st.expander("🧾 Preprocessing log (reproducibility)"):
        st.markdown("Every transformation applied to the dataset, in order:")
        for i, note in enumerate([*st.session_state.get("load_notes", []), *prep_notes], start=1):
            st.caption(f"{i}. {note}")


# --------------------------------------------------------------------------- #
# Charts tab
# --------------------------------------------------------------------------- #
def render_charts(fdf: pd.DataFrame, roles: Dict[str, str], metric: Optional[str]) -> None:
    st.markdown("#### Build an interactive chart")
    numerics = numeric_columns(roles)
    times = time_columns(roles)
    cats = categorical_columns(roles)

    if not numerics and not times and not cats:
        st.info("This dataset has no numeric, time or categorical columns to chart.")
        return

    chart_type = st.segmented_control(
        "Chart type",
        ["Auto", "Trend", "Distribution", "Comparison", "Relationship", "Composition", "Correlation"],
        default="Auto",
        key="chart_type",
    )
    if chart_type == "Auto":
        if times and numerics:
            chart_type = "Trend"
        elif len(numerics) >= 2:
            chart_type = "Relationship"
        elif cats and numerics:
            chart_type = "Comparison"
        elif numerics:
            chart_type = "Distribution"
        else:
            chart_type = "Comparison"

    fig, title, name = None, "", "chart"
    try:
        if chart_type == "Trend":
            fig, title, name = _chart_trend(fdf, roles, times, numerics, cats, metric)
        elif chart_type == "Distribution":
            fig, title, name = _chart_distribution(fdf, numerics, cats)
        elif chart_type == "Comparison":
            fig, title, name = _chart_comparison(fdf, numerics, cats)
        elif chart_type == "Relationship":
            fig, title, name = _chart_relationship(fdf, numerics, cats, metric)
        elif chart_type == "Composition":
            fig, title, name = _chart_composition(fdf, numerics, cats)
        elif chart_type == "Correlation":
            fig, title, name = _chart_correlation(fdf, numerics)
    except (ValueError, KeyError, TypeError) as exc:
        st.warning(f"Could not build this chart with the current selections: {exc}")
        return

    if fig is None:
        st.info("Not enough data or columns for this chart type — try another combination.")
        return

    st.plotly_chart(fig, width="stretch", key=f"chart_fig_{name}")
    col1, col2 = st.columns([1, 3])
    with col1:
        ui.download_figure_button(
            fig, f"{ui.safe_filename(name)}.png", "💾 Download chart", key=f"dl_{name}"
        )
    if fdf is not None:
        ui.download_csv_button(
            fdf, f"{ui.safe_filename(name)}_filtered_data.csv",
            "💾 Download filtered data (CSV)", key=f"dlcsv_{name}",
        )


def _chart_trend(fdf, roles, times, numerics, cats, metric):
    if not times:
        raise ValueError("No time column detected (need a date or year column).")
    if not numerics:
        raise ValueError("No numeric column available to plot over time.")
    c1, c2, c3, c4 = st.columns(4)
    time_col = c1.selectbox("Time column", times, key="c_trend_time")
    value_col = c2.selectbox(
        "Value column", numerics,
        index=numerics.index(metric) if metric in numerics else 0,
        key="c_trend_value",
    )
    group_options = ["(none)"] + cats
    group_col = c3.selectbox("Group by (optional)", group_options, key="c_trend_group")
    agg = c4.selectbox("Aggregation", ["sum", "mean", "median", "count"], key="c_trend_agg")
    freq = st.selectbox("Frequency", ["Auto", "D", "W", "M", "Q", "Y"], key="c_trend_freq")

    if time_col in [c for c, r in roles.items() if r == "year"]:
        # Year columns: aggregate per year directly (no resampling needed).
        if agg == "count":
            ts = fdf[time_col].dropna().astype(int).value_counts().sort_index().astype(float)
        else:
            frame = fdf[[time_col, value_col]].dropna()
            frame = frame.assign(_v=pd.to_numeric(frame[value_col], errors="coerce")).dropna(subset=["_v"])
            ts = frame.groupby(frame[time_col].astype(int), observed=True)["_v"].agg(agg)
        fig = plots.trend_chart(ts, value_col, title=f"{value_col} ({agg}) by {time_col}",
                                agg_label=agg.capitalize(), freq_label="year")
        return fig, f"Trend of {value_col} by {time_col}", f"trend_{value_col}_{time_col}"

    if freq == "Auto":
        freq = analysis.suggest_frequency(analysis.trend_series(fdf, time_col, value_col, agg, "D"))
    if group_col == "(none)":
        ts = analysis.trend_series(fdf, time_col, value_col, agg, freq)
        if len(ts) < 2:
            raise ValueError(f"Only {len(ts)} period(s) after aggregation — pick a different frequency or columns.")
        fig = plots.trend_chart(
            ts, value_col,
            title=f"{value_col} over time ({agg} per {analysis.FREQ_LABELS.get(freq, freq).lower()})",
            agg_label=agg.capitalize(), freq_label=analysis.FREQ_LABELS.get(freq, freq),
        )
        return fig, f"{value_col} over time", f"trend_{value_col}"
    wide = analysis.trend_by_group(fdf, time_col, value_col, group_col, agg, freq)
    if wide.empty or len(wide) < 2:
        raise ValueError("Not enough data after grouping — try a different group or frequency.")
    fig = plots.grouped_trend_chart(
        wide, value_col, group_col,
        title=f"{value_col} over time by {group_col} ({agg} per {analysis.FREQ_LABELS.get(freq, freq).lower()})",
    )
    return fig, f"{value_col} over time by {group_col}", f"trend_{value_col}_by_{group_col}"


def _chart_distribution(fdf, numerics, cats):
    if not numerics:
        raise ValueError("No numeric column available for a distribution.")
    c1, c2, c3 = st.columns([2, 1, 2])
    col = c1.selectbox("Numeric column", numerics, key="c_dist_col")
    bins = c2.slider("Bins", 5, 100, 40, key="c_dist_bins")
    group = c3.selectbox("Split by (optional)", ["(none)"] + cats, key="c_dist_group")
    fig = plots.histogram_chart(
        fdf, col, None if group == "(none)" else group, bins,
        title=f"Distribution of {col}" + (f" by {group}" if group != "(none)" else ""),
    )
    return fig, f"Distribution of {col}", f"hist_{col}"


def _chart_comparison(fdf, numerics, cats):
    if not cats:
        raise ValueError("No categorical column available to compare.")
    c1, c2, c3, c4 = st.columns(4)
    cat = c1.selectbox("Category", cats, key="c_cmp_cat")
    if numerics:
        value = c2.selectbox("Measure", numerics, key="c_cmp_value")
        agg = c3.selectbox("Aggregation", ["sum", "mean", "median", "count"], key="c_cmp_agg")
    else:
        value, agg = None, "count"
        c2.caption("No numeric measure — counting rows.")
    top_n = c4.slider("Top N categories", 5, 30, 10, key="c_cmp_top")
    horizontal = st.checkbox("Horizontal bars", value=False, key="c_cmp_h")

    if agg == "count" or value is None:
        table = fdf[cat].dropna().value_counts().head(top_n).rename("count").reset_index()
        table.columns = [cat, "count"]
        value_label = "count"
    else:
        table = (
            fdf.groupby(cat, observed=True)[value].agg(agg).nlargest(top_n).reset_index()
        )
        value_label = f"{agg} of {value}"
    if table.empty:
        raise ValueError("No rows for the selected category.")
    fig = plots.bar_chart(table, cat, value_label, title=f"{value_label} by {cat} (top {top_n})", horizontal=horizontal)
    return fig, f"{value_label} by {cat}", f"bar_{cat}"


def _chart_relationship(fdf, numerics, cats, metric):
    if len(numerics) < 2:
        raise ValueError("At least two numeric columns are required for a relationship chart.")
    c1, c2, c3 = st.columns(3)
    default_y = numerics.index(metric) if metric in numerics else 0
    x = c1.selectbox("X (horizontal)", numerics, index=min(len(numerics) - 1, 1), key="c_rel_x")
    y = c2.selectbox("Y (vertical)", numerics, index=default_y, key="c_rel_y")
    color = c3.selectbox("Colour by (optional)", ["(none)"] + cats, key="c_rel_color")
    trendline = st.checkbox("Add OLS trendline", value=True, key="c_rel_trend")
    if x == y:
        raise ValueError("Pick two different numeric columns.")
    fig = plots.scatter_chart(
        fdf, x, y, None if color == "(none)" else color, trendline=trendline,
        title=f"{y} vs {x}" + (f" coloured by {color}" if color != "(none)" else ""),
    )
    return fig, f"{y} vs {x}", f"scatter_{y}_{x}"


def _chart_composition(fdf, numerics, cats):
    if not cats:
        raise ValueError("No categorical column available for composition.")
    c1, c2, c3 = st.columns(3)
    cat = c1.selectbox("Category", cats, key="c_comp_cat")
    if numerics:
        value = c2.selectbox("Measure", numerics, key="c_comp_value")
    else:
        value = None
        c2.caption("No numeric measure — counting rows.")
    kind = c3.radio("Style", ["treemap", "pie"], horizontal=True, key="c_comp_kind")
    top_n = st.slider("Top N categories", 5, 30, 12, key="c_comp_top")
    if value is None:
        table = fdf[cat].dropna().value_counts().head(top_n).rename("count").reset_index()
        table.columns = [cat, "count"]
        value_label = "count"
    else:
        table = fdf.groupby(cat, observed=True)[value].agg("sum").nlargest(top_n).reset_index()
        value_label = f"sum of {value}"
    if table.empty:
        raise ValueError("No rows for the selected category.")
    fig = plots.composition_chart(table, cat, value_label, kind=kind, title=f"Composition of {value_label} by {cat}")
    return fig, f"Composition of {value_label} by {cat}", f"comp_{cat}"


def _chart_correlation(fdf, numerics):
    if len(numerics) < 2:
        raise ValueError("At least two numeric columns are required for a correlation matrix.")
    method = st.radio("Method", ["pearson", "spearman"], horizontal=True, key="c_corr_method")
    r_values, _ = analysis.correlation_matrix(fdf, numerics, method=method)
    fig = plots.correlation_heatmap(r_values, method_label=method.capitalize())
    return fig, f"{method.capitalize()} correlation matrix", f"corr_{method}"


# --------------------------------------------------------------------------- #
# Diagnostics tab
# --------------------------------------------------------------------------- #
def render_diagnostics(fdf: pd.DataFrame, roles: Dict[str, str]) -> None:
    numerics = numeric_columns(roles)
    cats = categorical_columns(roles)

    st.markdown("#### 🔗 Correlation analysis")
    if len(numerics) < 2:
        st.info("Correlations need at least two numeric columns.")
    else:
        method = st.radio("Correlation method", ["pearson", "spearman"], horizontal=True, key="diag_corr_method")
        r_values, p_values = analysis.correlation_matrix(fdf, numerics, method=method)
        left, right = st.columns([3, 2])
        with left:
            st.plotly_chart(plots.correlation_heatmap(r_values, method_label=method.capitalize()), width="stretch", key="diag_corr_heat")
        with right:
            st.markdown("**Strongest relationships**")
            st.dataframe(analysis.top_correlated_pairs(r_values, p_values, n=8), width="stretch")
            st.caption("r = correlation strength (−1…+1); p < 0.05 suggests the relationship is statistically significant.")

    st.divider()
    st.markdown("#### ⚖️ Group comparison (ANOVA)")
    if not numerics or not cats:
        st.info("Group comparisons need at least one numeric measure and one categorical column.")
    else:
        c1, c2 = st.columns(2)
        metric = c1.selectbox("Measure", numerics, key="diag_anova_metric")
        group = c2.selectbox("Group by", cats, key="diag_anova_group")
        try:
            result = analysis.anova_test(fdf, metric, group)
            stats_table = analysis.group_stats(fdf, metric, group)
            left, right = st.columns([3, 2])
            with left:
                st.plotly_chart(plots.box_chart(fdf, metric, group, title=f"{metric} by {group}"), width="stretch", key="diag_anova_box")
            with right:
                st.markdown(f"**{result['test']}** — {result['groups']} groups, {result['observations']:,} observations")
                st.dataframe(stats_table, width="stretch")
                st.success(result["verdict"])
                st.caption("Assumption: observations are independent; normality matters for small samples. Welch/Kruskal–Wallis variants are used automatically when variances differ.")
        except ValueError as exc:
            st.warning(str(exc))

    st.divider()
    st.markdown("#### 🔗 Categorical association (chi-square)")
    if len(cats) < 2:
        st.info("This test needs at least two categorical columns.")
    else:
        c1, c2 = st.columns(2)
        col_a = c1.selectbox("Column A", cats, key="diag_chi_a")
        col_b = c2.selectbox("Column B", [c for c in cats if c != col_a], key="diag_chi_b")
        try:
            result = analysis.chi_square_test(fdf, col_a, col_b)
            st.success(result["verdict"])
            st.caption(f"χ² = {result['chi2']:.2f}, df = {result['dof']}, n = {result['n']:,}. Cramér's V measures association strength (0 = none, 1 = perfect).")
            st.dataframe(analysis.contingency_table(fdf, col_a, col_b), width="stretch")
        except ValueError as exc:
            st.warning(str(exc))

    st.divider()
    st.markdown("#### 🚨 Outlier detection (IQR method)")
    if not numerics:
        st.info("Outlier detection needs numeric columns.")
    else:
        try:
            table = analysis.outlier_summary(fdf, numerics)
            left, right = st.columns([2, 3])
            with left:
                st.dataframe(table, width="stretch")
            with right:
                worst = table.iloc[0]["Column"]
                st.plotly_chart(
                    plots.histogram_chart(fdf, worst, title=f"Distribution of {worst} (outliers visible in tails)"),
                    width="stretch", key="diag_out_hist",
                )
        except ValueError as exc:
            st.warning(str(exc))


# --------------------------------------------------------------------------- #
# Predictive tab
# --------------------------------------------------------------------------- #
def render_predictive(fdf: pd.DataFrame, roles: Dict[str, str], missing_strategy: str) -> None:
    numerics = numeric_columns(roles)
    cats = categorical_columns(roles)
    st.markdown(
        "Predictive models run **on demand** — pick a target and press Run. "
        "Metrics come from **5-fold cross-validation** (every row is scored by a model "
        "that never saw it), so they reflect unseen data. Check the assumptions "
        "expanders before trusting the results."
    )
    mode = st.radio(
        "Model family",
        ["📉 Regression", "🏷️ Classification", "🧩 Clustering", "🔮 Forecasting"],
        horizontal=True,
        key="pred_mode",
    )

    if mode == "📉 Regression":
        _render_regression(fdf, roles, numerics, cats, missing_strategy)
    elif mode == "🏷️ Classification":
        _render_classification(fdf, roles, numerics, cats, missing_strategy)
    elif mode == "🧩 Clustering":
        _render_clustering(fdf, roles, numerics)
    else:
        _render_forecasting(fdf, roles, numerics)


def _default_features(df, roles, target, numerics, cats, cap=10):
    """Sensible default feature set: related numerics + low-cardinality categories."""
    picks = [c for c in numerics if c != target][:8]
    picks += [c for c in cats if c != target and df[c].nunique(dropna=True) <= 20][:6]
    return picks[:cap]


def _render_regression(fdf, roles, numerics, cats, missing_strategy):
    if not numerics:
        st.info("Regression needs at least one numeric column to predict.")
        return
    c1, c2 = st.columns(2)
    target = c1.selectbox("Target (what to predict)", numerics, key="pred_reg_target")
    defaults = _default_features(fdf, roles, target, numerics, cats)
    features = c2.multiselect("Features", [c for c in numerics + cats if c != target], default=defaults, key="pred_reg_feat")
    with st.expander("⚙️ Options"):
        with_rf = st.checkbox("Also fit a Random Forest", value=True, key="pred_reg_rf")
    if not st.button("▶️ Run regression", key="pred_reg_run", type="primary"):
        st.caption(
            "Choose a target and features, then press Run. Numeric, date and categorical "
            "features are handled automatically; metrics come from 5-fold cross-validation."
        )
        return
    try:
        outcome = cached_regression(fdf, target, tuple(features), roles, missing_strategy, with_rf)
    except ModelingError as exc:
        st.warning(str(exc))
        return
    _store_result("regression", outcome)
    st.success(
        f"Fitted on {outcome['n_rows']:,} rows ({outcome['n_features']} encoded features) with "
        f"5-fold cross-validation. Baseline (predict the mean) RMSE: {plots.fmt(outcome['baseline_rmse'], 2)}."
    )
    cards = []
    lin = outcome["linear"]
    cards.append(("Linear — R²", f"{lin['r2']:.3f}", None))
    cards.append(("Linear — RMSE", plots.fmt(lin["rmse"], 2), None))
    cards.append(("Linear — MAE", plots.fmt(lin["mae"], 2), None))
    if outcome["random_forest"]:
        rf = outcome["random_forest"]
        cards.append(("Forest — R²", f"{rf['r2']:.3f}", None))
        cards.append(("Forest — RMSE", plots.fmt(rf["rmse"], 2), None))
        cards.append(("Forest — MAE", plots.fmt(rf["mae"], 2), None))
    ui.kpi_cards(cards, columns=3)
    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(
            plots.residuals_chart(pd.Series(lin["test_actuals"]), pd.Series(lin["test_predictions"]),
                                  title=f"Linear model — actual vs predicted ({target})"),
            width="stretch", key="pred_reg_scatter",
        )
    with col2:
        if outcome["rf_importance"]:
            names = [n for n, _ in outcome["rf_importance"]][:15]
            values = [v for _, v in outcome["rf_importance"]][:15]
            st.plotly_chart(plots.importance_chart(names, values, title="Random Forest — feature importance (top 15)"),
                            width="stretch", key="pred_reg_imp")
    with st.expander("🧠 Assumptions & limitations"):
        st.markdown(
            "- Linear regression assumes a roughly linear relationship, independent errors and no severe outliers. "
            "R² is computed on cross-validated (out-of-fold) predictions, not the training set.\n"
            "- Random Forest relaxes linearity but can overfit noisy data; importances reflect the model, not causal effects.\n"
            "- Heavy-tailed targets (like profit) can still pull R² down — check RMSE against the baseline too.\n"
            "- Imputation and one-hot encoding were applied as documented in the preprocessing log. "
            f"{'; '.join(outcome['notes'])}"
        )


def _render_classification(fdf, roles, numerics, cats, missing_strategy):
    candidate_targets = cats + [c for c in numerics if fdf[c].nunique(dropna=True) <= modeling.MAX_CLASSES]
    if not candidate_targets:
        st.info("No categorical (or low-cardinality) column available as a classification target.")
        return
    c1, c2 = st.columns(2)
    target = c1.selectbox("Target (class to predict)", candidate_targets, key="pred_cls_target")
    defaults = _default_features(fdf, roles, target, numerics, cats)
    features = c2.multiselect("Features", [c for c in numerics + cats if c != target], default=defaults, key="pred_cls_feat")
    if not st.button("▶️ Run classification", key="pred_cls_run", type="primary"):
        st.caption(
            "Choose a target and features, then press Run. "
            "Metrics come from stratified 5-fold cross-validation."
        )
        return
    try:
        outcome = cached_classification(fdf, target, tuple(features), roles, missing_strategy)
    except ModelingError as exc:
        st.warning(str(exc))
        return
    _store_result("classification", outcome)
    st.success(
        f"{outcome['n_classes']} classes, {outcome['n_rows']:,} rows, 5-fold cross-validation. "
        f"Baseline (predict the majority class) accuracy: {outcome['baseline_accuracy']:.3f}."
    )
    ui.kpi_cards(
        [
            ("Accuracy (holdout)", f"{outcome['accuracy']:.3f}", None),
            ("Macro F1 (holdout)", f"{outcome['macro_f1']:.3f}", None),
            ("Baseline accuracy", f"{outcome['baseline_accuracy']:.3f}", None),
        ],
        columns=3,
    )
    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(
            plots.confusion_heatmap(outcome["confusion_matrix"], outcome["class_labels"]),
            width="stretch", key="pred_cls_cm",
        )
    with col2:
        names = [n for n, _ in outcome["importance"]][:15]
        values = [v for _, v in outcome["importance"]][:15]
        st.plotly_chart(plots.importance_chart(names, values, title="Feature importance (top 15)"),
                        width="stretch", key="pred_cls_imp")
    with st.expander("🧠 Assumptions & limitations"):
        st.markdown(
            "- A random forest was evaluated with stratified 5-fold cross-validation; accuracy/F1 are "
            "**out-of-fold** figures (every row is scored by a model that never saw it).\n"
            "- Class imbalance is common — compare accuracy against the majority-class baseline; "
            "if they are close, the model adds little value.\n"
            "- The model captures association, not causation. "
            f"{'; '.join(outcome['notes'])}"
        )


def _render_clustering(fdf, roles, numerics):
    if len(numerics) < 2:
        st.info("Clustering needs at least two numeric columns.")
        return
    defaults = numerics[:3]
    features = st.multiselect("Numeric features to cluster", numerics, default=defaults, key="pred_clu_feat")
    k_max = st.slider("Maximum k to evaluate", 3, 10, 8, key="pred_clu_kmax")
    if not st.button("▶️ Run clustering", key="pred_clu_run", type="primary"):
        st.caption("Features are standardised automatically; k-means is evaluated from k = 2 up to your maximum.")
        return
    try:
        outcome = cached_clustering(fdf, tuple(features), roles, k_max)
    except ModelingError as exc:
        st.warning(str(exc))
        return
    _store_result("clustering", outcome)
    st.success(f"Best k = {outcome['best_k']} (highest silhouette score). {outcome['n_rows']:,} rows clustered on {len(outcome['features_used'])} features.")
    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(plots.elbow_chart(outcome["k_range"], outcome["inertias"], outcome["silhouettes"]),
                        width="stretch", key="pred_clu_elbow")
    with col2:
        pca_df = pd.DataFrame({"PCA 1": outcome["pca_x"], "PCA 2": outcome["pca_y"], "Cluster": outcome["labels"]})
        st.plotly_chart(
            plots.cluster_scatter(pca_df, "PCA 1", "PCA 2", "Cluster",
                                  title=f"Clusters projected to 2-D PCA ({100 * outcome['explained_variance']:.0f}% variance)"),
            width="stretch", key="pred_clu_scatter",
        )
    st.markdown("**Cluster sizes & average profile**")
    st.dataframe(
        pd.concat(
            [
                pd.DataFrame({"Cluster": list(outcome["cluster_sizes"]), "Rows": list(outcome["cluster_sizes"].values())}).set_index("Cluster"),
                outcome["cluster_means"],
            ],
            axis=1,
        ),
        width="stretch",
    )
    with st.expander("🧠 Assumptions & limitations"):
        st.markdown(
            "- k-means finds spherical, similarly-sized clusters; scaling is applied, outliers influence centres.\n"
            "- The elbow/silhouette plot is a guide, not a proof — domain knowledge should confirm the chosen k.\n"
            "- PCA projection is a 2-D approximation of the true distances."
        )


def _render_forecasting(fdf, roles, numerics):
    date_cols = [c for c, r in roles.items() if r == "date"]
    if not date_cols or not numerics:
        st.info("Forecasting needs at least one datetime column and one numeric column.")
        return
    c1, c2, c3, c4 = st.columns(4)
    date_col = c1.selectbox("Date column", date_cols, key="pred_fc_date")
    value_col = c2.selectbox("Value column", numerics, key="pred_fc_value")
    agg = c3.selectbox("Aggregation", ["sum", "mean", "count"], key="pred_fc_agg")
    freq = c4.selectbox("Frequency", ["Auto", "D", "W", "M", "Q", "Y"], key="pred_fc_freq")
    periods = st.slider("Periods to forecast", 1, 36, 12, key="pred_fc_periods")
    if not st.button("▶️ Run forecast", key="pred_fc_run", type="primary"):
        st.caption("The series is aggregated to a regular frequency, split into training/holdout, and scored with MAPE against a naive baseline.")
        return
    if freq == "Auto":
        freq = analysis.suggest_frequency(analysis.trend_series(fdf, date_col, value_col, agg, "D"))
    try:
        outcome = cached_forecast(fdf, date_col, value_col, agg, freq, periods)
    except ModelingError as exc:
        st.warning(str(exc))
        return
    _store_result("forecast", outcome)
    st.caption(f"Method: {outcome['method']} • frequency: {outcome['freq']}")
    ui.kpi_cards(
        [
            ("Model MAPE (holdout)", f"{outcome['mape_model']:.1f}%" if outcome["mape_model"] is not None else "n/a (zeros in data)", None),
            ("Naive MAPE (holdout)", f"{outcome['mape_naive']:.1f}%" if outcome["mape_naive"] is not None else "n/a (zeros in data)", None),
            ("Forecast periods", str(outcome["periods"]), None),
        ],
        columns=3,
    )
    st.plotly_chart(
        plots.forecast_chart(
            outcome["history"], outcome["fitted"], outcome["forecast"],
            outcome["ci_lower"], outcome["ci_upper"],
            title=f"Forecast of {value_col} ({agg} per {analysis.FREQ_LABELS.get(freq, freq).lower()})",
            value_col=value_col,
        ),
        width="stretch", key="pred_fc_chart",
    )
    with st.expander("🧠 Assumptions & limitations"):
        st.markdown(
            "- Exponential smoothing extrapolates historical patterns; it cannot foresee shocks, policy changes or seasonality shifts.\n"
            "- MAPE is computed on a held-out tail; if the model does not beat the naive baseline, treat the forecast as indicative only.\n"
            "- The shaded band is a rough 95% interval based on in-sample residual spread."
        )


def _store_result(kind: str, outcome: Dict) -> None:
    results = st.session_state.setdefault("last_results", {})
    results[kind] = outcome
    st.session_state["last_results"] = results


# --------------------------------------------------------------------------- #
# Data Dictionary & Raw Data tabs
# --------------------------------------------------------------------------- #
def render_dictionary(df: pd.DataFrame, roles: Dict[str, str]) -> None:
    st.markdown(
        "The dictionary below is **generated automatically** from the data. "
        "Descriptions reflect the inferred role of each column; edit roles on the "
        "loading screen if needed."
    )
    dictionary = profile.build_data_dictionary(df, roles)
    st.dataframe(dictionary, width="stretch", height=420)
    ui.download_csv_button(dictionary, "data_dictionary.csv", "💾 Download dictionary (CSV)", key="dl_dict")


def render_raw_data(fdf: pd.DataFrame) -> None:
    st.markdown("Browse the **filtered** rows. Large tables are paginated by Streamlit automatically.")
    shown = fdf
    note = ""
    if len(fdf) > 100_000:
        shown = fdf.head(100_000)
        note = f" Showing the first 100,000 of {len(fdf):,} rows on screen — downloads always contain the full filtered set."
    columns = st.multiselect("Columns to show", list(fdf.columns), default=list(fdf.columns)[:30], key="raw_cols")
    if not columns:
        st.info("Select at least one column.")
        return
    st.dataframe(shown[columns], width="stretch", height=480)
    if note:
        st.caption(note)
    ui.download_csv_button(fdf[columns], "filtered_data.csv", "💾 Download filtered data (CSV)", key="dl_raw")


# --------------------------------------------------------------------------- #
# Report tab
# --------------------------------------------------------------------------- #
def render_report(
    fdf: pd.DataFrame, full_df: pd.DataFrame, roles: Dict[str, str],
    filter_state, prep_notes: List[str], metric: Optional[str],
) -> None:
    st.markdown("A reproducible summary of the current session — download it and share it alongside your data.")
    report_md = build_report_markdown(fdf, full_df, roles, filter_state, prep_notes, metric)
    st.download_button(
        "💾 Download report (Markdown)", data=report_md.encode("utf-8"),
        file_name="analysis_report.md", mime="text/markdown", key="dl_report",
    )
    with st.expander("👁️ Preview report"):
        st.markdown(report_md)


def build_report_markdown(
    fdf: pd.DataFrame, full_df: pd.DataFrame, roles: Dict[str, str],
    filter_state, prep_notes: List[str], metric: Optional[str],
) -> str:
    bundle: DataBundle = st.session_state["bundle"]
    summary = profile.profile_summary(full_df, roles)
    date_col = next((c for c, r in roles.items() if r == "date"), None)
    snapshot = analysis.kpi_snapshot(fdf, roles, metric, date_col, full_rows=len(full_df))
    lines = [
        f"# DataPilot — analysis report",
        f"- Dataset: **{bundle.name}**",
        f"- Generated: {datetime.now():%Y-%m-%d %H:%M}",
        f"- Source: {bundle.source}",
        "",
        "## 1. Data profile",
        f"- Rows: {summary['rows']:,} • Columns: {summary['columns']:,}",
        f"- Duplicate rows: {summary['duplicate_rows']:,} • Missing cells: {summary['missing_pct']:.2f}%",
        f"- Structure: {profile.dataset_structure_hint(roles)}",
        "",
        "## 2. Preprocessing log",
    ]
    lines += [f"{i}. {n}" for i, n in enumerate([*st.session_state.get("load_notes", []), *prep_notes], 1)]
    lines += ["", "## 3. Active filters"]
    lines += ([f"- {chip}" for chip in filter_state.descriptions()] if filter_state.is_active() else ["- None (all rows shown)"])
    lines += ["", f"## 4. KPIs (filtered to {len(fdf):,} rows)"]
    if snapshot.get("metric"):
        lines += [
            f"- Total {snapshot['metric']}: {snapshot['metric_total']:,.2f}",
            f"- Average {snapshot['metric']}: {snapshot['metric_mean']:,.2f}",
        ]
    if snapshot.get("top_category"):
        top = snapshot["top_category"]
        lines += [f"- Top {top['column']}: '{top['value']}' ({top['count']:,} rows, {top['share']:.1f}%)"]
    numerics = numeric_columns(roles)
    if len(numerics) >= 2:
        try:
            r_values, p_values = analysis.correlation_matrix(fdf, numerics, method="pearson")
            lines += ["", "## 5. Strongest correlations"]
            for _, row in analysis.top_correlated_pairs(r_values, p_values, n=5).iterrows():
                lines.append(f"- {row['Variable A']} ↔ {row['Variable B']}: r = {row['r']}, p = {row['p-value']} ({row['Interpretation']})")
        except ValueError:
            pass
    results = st.session_state.get("last_results", {})
    if results:
        lines += ["", "## 6. Model results"]
        for kind, outcome in results.items():
            if kind == "regression":
                lin = outcome["linear"]
                lines.append(f"- Regression on '{outcome['target']}': R² = {lin['r2']:.3f}, RMSE = {lin['rmse']:.2f}, MAE = {lin['mae']:.2f}")
            elif kind == "classification":
                lines.append(f"- Classification on '{outcome['target']}': accuracy = {outcome['accuracy']:.3f}, macro F1 = {outcome['macro_f1']:.3f}")
            elif kind == "clustering":
                lines.append(f"- Clustering: best k = {outcome['best_k']} (silhouette = {max(outcome['silhouettes']):.3f})")
            elif kind == "forecast":
                lines.append(f"- Forecast of '{outcome.get('value_col', 'value')}': {outcome['method']}; holdout MAPE = {outcome['mape_model']:.1f}% vs naive {outcome['mape_naive']:.1f}%")
    lines += [
        "",
        "## Caveats",
        "- Correlations and model metrics describe association, not causation.",
        "- Predictive metrics are computed on a held-out sample, but results may not generalise to other periods or populations.",
        "- Missing values were handled as documented in the preprocessing log.",
    ]
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #
def main() -> None:
    if "bundle" not in st.session_state:
        render_landing()
    elif not st.session_state.get("explored", False):
        render_confirm_step(st.session_state["bundle"])
    else:
        render_main_app()


if __name__ == "__main__":
    main()
