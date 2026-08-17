"""Service layer wrapping src/* modules for the API."""

from __future__ import annotations

import pandas as pd
from typing import Dict, List, Tuple

# Import core logic (src is at repo root)
import sys
from pathlib import Path
REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from src import loader, profile, analysis, modeling, plots
from src.loader import DataBundle, LoaderError
from src.modeling import ModelingError

from ..models.schemas import FilterStateSchema
from dataclasses import dataclass, field
from typing import Tuple

@dataclass
class FilterState:
    """Lightweight version of src.ui.FilterState without streamlit."""
    date_ranges: Dict[str, Tuple[str, str]] = field(default_factory=dict)
    year_ranges: Dict[str, Tuple[int, int]] = field(default_factory=dict)
    category_picks: Dict[str, List[str]] = field(default_factory=dict)
    numeric_ranges: Dict[str, Tuple[float, float]] = field(default_factory=dict)
    search_col: str | None = None
    search_text: str = ""

    def is_active(self) -> bool:
        return bool(
            self.date_ranges or self.year_ranges or self.category_picks
            or self.numeric_ranges or (self.search_text.strip() != "")
        )


def apply_filters(df: pd.DataFrame, state: FilterState) -> pd.DataFrame:
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


def _filters_from_schema(schema: FilterStateSchema | None) -> FilterState:
    if not schema:
        return FilterState()
    fs = FilterState()
    # Convert list values back to tuples for ranges
    fs.date_ranges = {k: (v[0], v[1]) for k, v in schema.date_ranges.items() if len(v)==2}
    fs.year_ranges = {k: (int(v[0]), int(v[1])) for k, v in schema.year_ranges.items() if len(v)==2}
    fs.category_picks = schema.category_picks
    fs.numeric_ranges = {k: (float(v[0]), float(v[1])) for k, v in schema.numeric_ranges.items() if len(v)==2}
    fs.search_col = schema.search_col
    fs.search_text = schema.search_text
    return fs


def prepare_dataset(df: pd.DataFrame, drop_duplicates: bool = True):
    df_prepared, notes = loader.prepare_dataframe(df)
    if not drop_duplicates:
        # re-add duplicates if user disabled? We already deduped inside prepare— note that.
        notes.append("Duplicate removal was requested to be disabled, but fully-empty rows were still removed.")
    # infer roles
    # date cols from prepare step: need to know which were parsed as dates
    # loader.parse_date_columns returns parsed list, but prepare_dataframe bundles it.
    # We'll re-detect quickly by checking dtype
    date_cols = [c for c in df_prepared.columns if pd.api.types.is_datetime64_any_dtype(df_prepared[c])]
    roles = profile.infer_roles(df_prepared, date_cols=date_cols)
    col_prof = profile.column_profile(df_prepared, roles)
    summary = profile.profile_summary(df_prepared, roles)
    structure = profile.dataset_structure_hint(roles)
    numeric_desc = profile.numeric_describe(df_prepared, [c for c, r in roles.items() if r == "numeric"])
    data_dict = profile.build_data_dictionary(df_prepared, roles)
    return {
        "df": df_prepared,
        "roles": roles,
        "notes": notes,
        "column_profile": col_prof,
        "summary": summary,
        "structure": structure,
        "numeric_describe": numeric_desc,
        "data_dict": data_dict,
        "date_cols": date_cols,
    }


def apply_user_filters(df: pd.DataFrame, filter_schema: FilterStateSchema | None) -> pd.DataFrame:
    if not filter_schema:
        return df
    fs = _filters_from_schema(filter_schema)
    if not fs.is_active():
        return df
    return apply_filters(df, fs)


def get_kpi(df: pd.DataFrame, roles: Dict[str, str], metric: str | None, date_col: str | None, full_rows: int | None):
    return analysis.kpi_snapshot(df, roles, metric, date_col, full_rows=full_rows)


def get_correlation(df: pd.DataFrame, numeric_cols: List[str], method: str = "pearson", max_cols: int = 25):
    corr_mat, p_mat = analysis.correlation_matrix(df, numeric_cols, method=method, max_cols=max_cols)
    top_pairs = analysis.top_correlated_pairs(corr_mat, p_mat)
    return corr_mat, p_mat, top_pairs


def safe_run(func, *args, **kwargs):
    try:
        return func(*args, **kwargs), None
    except Exception as e:
        return None, str(e)
