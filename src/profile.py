"""Column-role inference and dataset profiling.

Roles drive every part of the app:
  date      — usable as a time axis / trend x-axis
  year      — ordinal time (e.g. integer year columns)
  numeric   — continuous measure (chart values, correlations, models)
  binary    — 0/1 style numeric flag (groupable, rarely a measure)
  boolean   — true/false column (groupable)
  category  — low-cardinality labels (filters, group-by, colours)
  text      — free text (searchable only)
  id        — row identifier (excluded from analysis)

The heuristics are deliberately conservative and documented; the user can
override any role from the confirmation screen.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd

# Column-name hints used to improve inference (all matching is lowercase).
_ID_HINTS = ("id", "key", "number", "ref", "code", "sku", "serial", "uuid")
_TEXT_HINTS = ("comment", "note", "description", "review", "feedback", "address", "email")
_YEAR_HINTS = ("year", "tahun")  # "tahun" = "year" in Malay/Indonesian
_POSTAL_HINTS = ("postal", "zip", "poskod")  # numeric codes that are really labels

ROLE_ORDER = ["date", "year", "numeric", "binary", "boolean", "category", "text", "id"]


def _name_has(name: str, hints: tuple[str, ...]) -> bool:
    lowered = name.lower()
    return any(h in lowered for h in hints)


def _looks_like_year(series: pd.Series, name: str) -> bool:
    """Whole-number column whose values sit in 1900..2100 and is year-ish by name."""
    if not pd.api.types.is_numeric_dtype(series):
        return False
    values = series.dropna()
    if values.empty or values.nunique() < 3:
        return False
    if not (values == values.round()).all():  # years must be whole numbers
        return False
    in_range = values.between(1900, 2100).mean()
    return in_range >= 0.95 and _name_has(name, _YEAR_HINTS)


def _string_looks_like_year(series: pd.Series, name: str) -> bool:
    """String column whose (sampled) values are 4-digit years in 1900..2100."""
    if not _name_has(name, _YEAR_HINTS):
        return False
    sample = series.dropna().astype(str).head(2000)
    if sample.empty or not sample.str.fullmatch(r"\d{4}").all():
        return False
    numeric = pd.to_numeric(sample, errors="coerce")
    return bool(numeric.between(1900, 2100).all()) and numeric.nunique() >= 3


def infer_roles(df: pd.DataFrame, date_cols: Optional[List[str]] = None) -> Dict[str, str]:
    """Infer a role for every column.

    Parameters
    ----------
    df : DataFrame
        The prepared (date-parsed, sanitised) DataFrame.
    date_cols : list of str, optional
        Columns already parsed as datetime.
    """
    date_cols = set(date_cols or [])
    roles: Dict[str, str] = {}
    n_rows = len(df)

    for col in df.columns:
        series = df[col]
        name = col
        nunique = series.nunique(dropna=True)

        if col in date_cols or pd.api.types.is_datetime64_any_dtype(series):
            roles[col] = "date"
        elif pd.api.types.is_bool_dtype(series):
            roles[col] = "boolean"
        elif pd.api.types.is_numeric_dtype(series):
            is_int = pd.api.types.is_integer_dtype(series)
            high_card = nunique > max(50, 0.9 * n_rows)
            if _name_has(name, _POSTAL_HINTS) and nunique <= max(500, 0.5 * n_rows):
                roles[col] = "category"
            elif (is_int and high_card) or (_name_has(name, _ID_HINTS) and nunique > 20 and is_int):
                roles[col] = "id"
            elif _looks_like_year(series, name):
                roles[col] = "year"
            elif nunique <= 2:
                roles[col] = "binary"
            else:
                roles[col] = "numeric"
        else:  # string-like
            if nunique == n_rows and n_rows > 200:
                roles[col] = "id"
            elif _name_has(name, _ID_HINTS) and nunique > 20:
                roles[col] = "id"
            elif _name_has(name, _TEXT_HINTS):
                roles[col] = "text"
            elif _string_looks_like_year(series, name):
                # string column containing year labels -> treat as year
                roles[col] = "year"
            elif nunique <= 500 or nunique / max(n_rows, 1) <= 0.5:
                roles[col] = "category"
            else:
                roles[col] = "text"
    return roles


def column_profile(df: pd.DataFrame, roles: Dict[str, str]) -> pd.DataFrame:
    """One row per column: role, dtype, missing, unique counts, top value."""
    rows = []
    n_rows = max(len(df), 1)
    for col in df.columns:
        series = df[col]
        missing = int(series.isna().sum())
        nunique = int(series.nunique(dropna=True))
        top = series.dropna().astype(str).mode()
        top_value = top.iloc[0] if len(top) else ""
        rows.append(
            {
                "Column": col,
                "Role": roles.get(col, "text"),
                "Type": str(series.dtype),
                "Missing": missing,
                "Missing %": round(100 * missing / n_rows, 2),
                "Unique": nunique,
                "Top value": (top_value[:40] + "…") if len(str(top_value)) > 40 else top_value,
            }
        )
    return pd.DataFrame(rows)


def profile_summary(df: pd.DataFrame, roles: Dict[str, str]) -> Dict:
    """Aggregate summary used by the Overview tab and the auto-report."""
    n_rows, n_cols = df.shape
    missing_cells = int(df.isna().sum().sum())
    dup_rows = int(df.duplicated().sum())
    date_cols = [c for c, r in roles.items() if r == "date"]
    summary: Dict = {
        "rows": n_rows,
        "columns": n_cols,
        "duplicate_rows": dup_rows,
        "missing_cells": missing_cells,
        "missing_pct": 100 * missing_cells / max(n_rows * n_cols, 1),
        "memory_mb": df.memory_usage(deep=True).sum() / 1e6,
        "date_columns": date_cols,
        "numeric_columns": [c for c, r in roles.items() if r in ("numeric", "binary")],
        "category_columns": [c for c, r in roles.items() if r in ("category", "binary", "boolean")],
        "text_columns": [c for c, r in roles.items() if r == "text"],
        "id_columns": [c for c, r in roles.items() if r == "id"],
    }
    if date_cols:
        spans = {c: (df[c].min(), df[c].max()) for c in date_cols if df[c].notna().any()}
        summary["date_spans"] = spans
    return summary


def numeric_describe(df: pd.DataFrame, numeric_cols: List[str], max_cols: int = 12) -> pd.DataFrame:
    """Transposed descriptive statistics for numeric columns (colorbar-ready)."""
    if not numeric_cols:
        return pd.DataFrame()
    cols = numeric_cols[:max_cols]
    stats = df[cols].describe().T
    stats.insert(0, "Column", stats.index)
    stats = stats.round(3)
    stats["Missing"] = [int(df[c].isna().sum()) for c in cols]
    stats = stats[
        ["Column", "count", "Missing", "mean", "std", "min", "25%", "50%", "75%", "max"]
    ].rename(
        columns={
            "count": "Non-null",
            "mean": "Mean",
            "std": "Std dev",
            "min": "Min",
            "25%": "Q1",
            "50%": "Median",
            "75%": "Q3",
            "max": "Max",
        }
    )
    return stats.reset_index(drop=True)


def primary_metric(df: pd.DataFrame, roles: Dict[str, str]) -> Optional[str]:
    """Pick the most likely 'headline' measure column.

    Preference for name hints (sales/revenue/profit/...), then the numeric
    column with the largest spread. Never returns id/date columns.
    """
    candidates = [c for c, r in roles.items() if r == "numeric"]
    if not candidates:
        return None
    hints = ("sales", "revenue", "amount", "value", "profit", "total", "price",
             "cost", "quantity", "income", "spend", "volume", "units", "qty", "gross")
    for hint in hints:
        for col in candidates:
            if hint in col.lower():
                return col
    spread = {c: float(df[c].nunique(dropna=True)) for c in candidates}
    return max(spread, key=spread.get)


def top_category(df: pd.DataFrame, category_cols: List[str]) -> Optional[Dict]:
    """Most frequent category value across the category columns (for KPIs)."""
    best: Optional[Dict] = None
    for col in category_cols[:10]:
        counts = df[col].dropna().value_counts(dropna=True)
        if counts.empty:
            continue
        share = counts.iloc[0] / len(df)
        if best is None or share > best["share"]:
            best = {"column": col, "value": str(counts.index[0])[:30], "share": share}
    return best


def dataset_structure_hint(roles: Dict[str, str]) -> str:
    """A one-line, human-readable summary of the data shape (used in UI copy)."""
    counts = {role: sum(1 for r in roles.values() if r == role) for role in ROLE_ORDER}
    parts = [f"{counts['numeric']} numeric", f"{counts['category']} categorical",
             f"{counts['date'] + counts['year']} time", f"{counts['text']} text",
             f"{counts['id']} id"]
    parts = [p for p in parts if not p.startswith("0 ")]
    return ", ".join(parts) if parts else "no structured columns detected"


def describe_column(series: pd.Series, role: str) -> str:
    """Auto-generated, plain-language description for the Data Dictionary."""
    nunique = series.nunique(dropna=True)
    missing = int(series.isna().sum())
    base = {
        "date": "Datetime column. Ideal for time-series trends and date filters.",
        "year": "Ordinal year column. Usable as a time axis for trends.",
        "numeric": "Continuous numeric measure. Usable for sums, averages, correlations and models.",
        "binary": "0/1 style flag. Usable for grouping and filtering.",
        "boolean": "True/false flag. Usable for grouping and filtering.",
        "category": f"Categorical label with {nunique:,} distinct value(s). Good for grouping, filtering and colouring charts.",
        "text": "Free-form text. Searchable but not aggregated.",
        "id": "Row identifier. Excluded from analysis to avoid meaningless aggregates.",
    }[role]
    if missing:
        base += f" {missing:,} missing value(s) ({100 * missing / max(len(series), 1):.1f}%)."
    else:
        base += " No missing values."
    return base


def example_values(series: pd.Series, n: int = 3) -> str:
    values = series.dropna().astype(str).drop_duplicates().head(n).tolist()
    if len(values) < series.nunique(dropna=True):
        values.append("…")
    return ", ".join(v[:25] for v in values)


def build_data_dictionary(df: pd.DataFrame, roles: Dict[str, str]) -> pd.DataFrame:
    """Auto-generated data dictionary (name, role, type, description, stats)."""
    n_rows = max(len(df), 1)
    rows = []
    for col in df.columns:
        series = df[col]
        rows.append(
            {
                "Column": col,
                "Role": roles.get(col, "text"),
                "Type": str(series.dtype),
                "Description": describe_column(series, roles.get(col, "text")),
                "Missing": int(series.isna().sum()),
                "Missing %": round(100 * series.isna().sum() / n_rows, 2),
                "Unique": int(series.nunique(dropna=True)),
                "Example values": example_values(series),
            }
        )
    return pd.DataFrame(rows)
