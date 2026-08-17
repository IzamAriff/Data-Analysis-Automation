"""Shared UI helpers: dynamic filters, KPI cards, download buttons.

Filter design
-------------
* Every filter widget is generated from the inferred column roles, so the
  sidebar adapts to *any* dataset without hard-coded column names.
* An empty multi-select means "no filter" (all values), which keeps the
  default state = full data and makes the app safe to test headlessly.
* High-cardinality categories are capped in the widget (top 100 by frequency);
  the text-search filter covers the long tail.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pandas as pd
import streamlit as st

from . import plots

logger = logging.getLogger("datapilot.ui")

MAX_FILTER_OPTIONS = 100    # per multi-select
MAX_SLIDER_COLUMNS = 10     # numeric range sliders shown
MAX_SEARCH_COLUMNS = 8      # text columns offered for search


# --------------------------------------------------------------------------- #
# Filter state
# --------------------------------------------------------------------------- #
@dataclass
class FilterState:
    """JSON-serialisable description of the user's active filters."""

    date_ranges: Dict[str, Tuple[str, str]] = field(default_factory=dict)
    year_ranges: Dict[str, Tuple[int, int]] = field(default_factory=dict)
    category_picks: Dict[str, List[str]] = field(default_factory=dict)
    numeric_ranges: Dict[str, Tuple[float, float]] = field(default_factory=dict)
    search_col: Optional[str] = None
    search_text: str = ""

    def is_active(self) -> bool:
        return bool(
            self.date_ranges or self.year_ranges or self.category_picks
            or self.numeric_ranges or (self.search_text.strip() != "")
        )

    def to_key(self) -> str:
        return json.dumps(self.__dict__, sort_keys=True, default=str)

    def descriptions(self) -> List[str]:
        """Short human-readable chips summarising active filters."""
        chips: List[str] = []
        for col, (lo, hi) in self.date_ranges.items():
            chips.append(f"{col}: {lo} → {hi}")
        for col, (lo, hi) in self.year_ranges.items():
            chips.append(f"{col}: {lo} → {hi}")
        for col, picks in self.category_picks.items():
            chips.append(f"{col}: {len(picks)} value(s)")
        for col, (lo, hi) in self.numeric_ranges.items():
            chips.append(f"{col}: {plots.fmt(lo)} – {plots.fmt(hi)}")
        if self.search_text.strip():
            chips.append(f"'{self.search_text.strip()}' in {self.search_col}")
        return chips


def apply_filters(df: pd.DataFrame, state: FilterState) -> pd.DataFrame:
    """Apply a :class:`FilterState` to a DataFrame (vectorised, no mutation)."""
    mask = pd.Series(True, index=df.index)
    for col, (lo, hi) in state.date_ranges.items():
        if col in df.columns:
            values = pd.to_datetime(df[col], errors="coerce")
            mask &= (values >= pd.Timestamp(lo)) & (values <= pd.Timestamp(hi))
    for col, (lo, hi) in state.year_ranges.items():
        if col in df.columns:
            values = pd.to_numeric(df[col], errors="coerce")
            mask &= (values >= lo) & (values <= hi)
    for col, picks in state.category_picks.items():
        if col in df.columns and picks:
            mask &= df[col].astype(str).isin(picks)
    for col, (lo, hi) in state.numeric_ranges.items():
        if col in df.columns:
            values = pd.to_numeric(df[col], errors="coerce")
            mask &= (values >= lo) & (values <= hi)
    if state.search_col and state.search_text.strip():
        query = state.search_text.strip()
        series = df[state.search_col].astype(str)
        mask &= series.str.contains(query, case=False, na=False, regex=False)
    return df.loc[mask]


# --------------------------------------------------------------------------- #
# Sidebar widgets
# --------------------------------------------------------------------------- #
def _safe_options(series: pd.Series, cap: int = MAX_FILTER_OPTIONS) -> List[str]:
    """Distinct string options for a multi-select, most-frequent first."""
    counts = series.dropna().astype(str).value_counts()
    return counts.head(cap).index.tolist()


def render_filter_sidebar(
    df: pd.DataFrame, roles: Dict[str, str]
) -> FilterState:
    """Build the dynamic filter widgets and return the resulting state."""
    state = FilterState()
    date_cols = [c for c, r in roles.items() if r == "date"]
    year_cols = [c for c, r in roles.items() if r == "year"]
    category_cols = [c for c, r in roles.items() if r in ("category", "binary", "boolean")]
    numeric_cols = [c for c, r in roles.items() if r == "numeric"]
    text_cols = [c for c, r in roles.items() if r in ("text", "category")]

    st.markdown("### 🎚️ Filters")
    st.caption("Empty selections mean “no filter” — all data is shown.")

    # --- Date ranges ----------------------------------------------------- #
    if date_cols:
        with st.expander("📅 Date range", expanded=True):
            date_col = st.selectbox("Date column", date_cols, key="f_date_col")
            values = pd.to_datetime(df[date_col], errors="coerce").dropna()
            if len(values) > 1:
                lo, hi = values.min(), values.max()
                if lo != hi:
                    picked = st.slider(
                        "Range", min_value=lo.to_pydatetime(), max_value=hi.to_pydatetime(),
                        value=(lo.to_pydatetime(), hi.to_pydatetime()), key="f_date_range",
                        format="YYYY-MM-DD",
                    )
                    full = (picked[0] == lo.to_pydatetime() and picked[1] == hi.to_pydatetime())
                    if not full:
                        state.date_ranges[date_col] = (
                            pd.Timestamp(picked[0]).strftime("%Y-%m-%d"),
                            pd.Timestamp(picked[1]).strftime("%Y-%m-%d"),
                        )

    # --- Year ranges ----------------------------------------------------- #
    if year_cols:
        with st.expander("🗓️ Year", expanded=False):
            for col in year_cols[:3]:
                values = pd.to_numeric(df[col], errors="coerce").dropna()
                if values.nunique() <= 1:
                    continue
                lo, hi = int(values.min()), int(values.max())
                picked = st.slider(f"{col} range", lo, hi, (lo, hi), key=f"f_year_{col}")
                if picked != (lo, hi):
                    state.year_ranges[col] = (int(picked[0]), int(picked[1]))

    # --- Category multi-selects ------------------------------------------ #
    if category_cols:
        with st.expander("🏷️ Categories", expanded=False):
            for col in category_cols:
                options = _safe_options(df[col])
                if not options:
                    continue
                picks = st.multiselect(col, options, default=options, key=f"f_cat_{col}")
                if picks and set(picks) != set(options):
                    state.category_picks[col] = picks

    # --- Numeric range sliders -------------------------------------------- #
    if numeric_cols:
        with st.expander("🔢 Numeric ranges", expanded=False):
            for col in numeric_cols[:MAX_SLIDER_COLUMNS]:
                values = pd.to_numeric(df[col], errors="coerce").dropna()
                if values.nunique() <= 1:
                    continue
                lo, hi = float(values.min()), float(values.max())
                picked = st.slider(
                    col, lo, hi, (lo, hi), key=f"f_num_{col}",
                    format="%.2f" if abs(hi - lo) < 10 else "%.0f",
                )
                if picked != (lo, hi):
                    state.numeric_ranges[col] = (float(picked[0]), float(picked[1]))

    # --- Text search ------------------------------------------------------- #
    searchable = [c for c in text_cols if c in df.columns][:MAX_SEARCH_COLUMNS]
    if searchable:
        with st.expander("🔍 Text search", expanded=False):
            state.search_col = st.selectbox("Search within", searchable, key="f_search_col")
            state.search_text = st.text_input(
                "Contains (case-insensitive)", key="f_search_text", placeholder="e.g. copier"
            )

    return state


def reset_filter_widgets() -> None:
    """Clear all filter widget values so the next run shows full data again."""
    for key in list(st.session_state.keys()):
        if key.startswith("f_"):
            del st.session_state[key]


# --------------------------------------------------------------------------- #
# KPI cards & downloads
# --------------------------------------------------------------------------- #
def kpi_cards(cards: Sequence[Tuple[str, str, Optional[float]]], columns: int = 6) -> None:
    """Render `(label, value, delta_pct)` tuples as metric cards."""
    cols = st.columns(columns)
    for col, (label, value, delta) in zip(cols, cards):
        with col:
            st.metric(label, value, delta=f"{delta:+.1f}% vs all data" if delta is not None else None)


def download_csv_button(df: pd.DataFrame, filename: str, label: str, key: str) -> None:
    """Streamlit CSV download button for a DataFrame."""
    if df is None or df.empty:
        st.caption("No rows to download for the current selection.")
        return
    csv_bytes = df.to_csv(index=False).encode("utf-8-sig")
    st.download_button(
        label, data=csv_bytes, file_name=filename, mime="text/csv", key=key,
        help="Download the currently filtered data as CSV (opens cleanly in Excel).",
    )


def download_figure_button(fig, filename: str, label: str, key: str) -> None:
    """PNG download via Kaleido, with an HTML fallback when unavailable."""
    png_bytes: Optional[bytes] = None
    try:
        png_bytes = fig.to_image(format="png", width=1200, height=675, scale=1.5)
    except Exception as exc:  # kaleido/Chrome missing on some hosts
        logger.debug("PNG export unavailable: %s", exc)
        st.caption(
            "Server-side PNG export needs Chrome (run `kaleido_get_chrome` once on headless "
            "servers). Use the HTML download instead — it opens interactively and its toolbar "
            "can save the chart as PNG."
        )
    if png_bytes:
        st.download_button(f"{label} (PNG)", data=png_bytes, file_name=filename, mime="image/png", key=f"{key}_png")
    html_bytes = fig.to_html(include_plotlyjs="cdn", full_html=True).encode()
    st.download_button(f"{label} (HTML)", data=html_bytes, file_name=filename.replace(".png", ".html"),
                       mime="text/html", key=f"{key}_html",
                       help="Interactive standalone chart — open in any browser (toolbar includes a PNG export).")


def safe_filename(name: str) -> str:
    """Filesystem-safe filename from a user-provided dataset name."""
    cleaned = "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in name)
    return cleaned[:60] or "dataset"
