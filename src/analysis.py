"""Descriptive and diagnostic analytics.

Pure functions (no Streamlit dependencies) so they are easy to unit-test
and reuse in the auto-generated report.

Statistical tests included:
  * Pearson / Spearman correlation with p-values
  * One-way ANOVA (falls back to Kruskal–Wallis when assumptions are violated)
  * Chi-square test of independence with Cramér's V effect size
  * IQR-based outlier detection
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger("datapilot.analysis")

MAX_GROUPS = 12  # group comparisons cap the number of categories


# --------------------------------------------------------------------------- #
# Descriptive statistics & KPIs
# --------------------------------------------------------------------------- #
def period_change(current: float, previous: float) -> Optional[float]:
    """Relative change vs. a previous period (None when undefined)."""
    if previous in (None, 0) or pd.isna(current) or pd.isna(previous):
        return None
    return 100.0 * (current - previous) / previous


def kpi_snapshot(
    df: pd.DataFrame,
    roles: Dict[str, str],
    metric: Optional[str],
    date_col: Optional[str],
    full_rows: Optional[int] = None,
) -> Dict:
    """Compute the headline KPI values shown on the Overview tab."""
    date_cols = [c for c, r in roles.items() if r == "date"]
    active_date = date_col if date_col in date_cols else (date_cols[0] if date_cols else None)

    snapshot: Dict = {
        "rows": len(df),
        "columns": df.shape[1],
        "duplicate_rows": int(df.duplicated().sum()),
        "missing_cells": int(df.isna().sum().sum()),
        "missing_pct": 100 * df.isna().sum().sum() / max(df.shape[0] * df.shape[1], 1),
    }
    if full_rows:
        snapshot["rows_delta"] = period_change(len(df), full_rows) if full_rows else None
        snapshot["full_rows"] = full_rows

    if active_date and df[active_date].notna().any():
        snapshot["date_span"] = (df[active_date].min(), df[active_date].max())

    if metric and metric in df.columns:
        values = pd.to_numeric(df[metric], errors="coerce")
        snapshot["metric"] = metric
        snapshot["metric_total"] = float(values.sum())
        snapshot["metric_mean"] = float(values.mean())
        if active_date:
            # Current vs. previous period for the delta arrow.
            series = pd.Series(values.to_numpy(), index=df[active_date].to_numpy())
            series = series[~series.index.isna()]
            if len(series) >= 10:
                try:
                    series.index = pd.to_datetime(series.index)
                    halves = np.array_split(series.sort_index(), 2)
                    snapshot["metric_delta"] = period_change(halves[1].sum(), halves[0].sum())
                except Exception:  # pragma: no cover - defensive
                    logger.debug("Could not compute period delta for '%s'", metric)

    category_cols = [c for c, r in roles.items() if r in ("category", "binary", "boolean")]
    if category_cols:
        top = df[category_cols[0]].dropna().value_counts()
        if not top.empty:
            snapshot["top_category"] = {
                "column": category_cols[0],
                "value": str(top.index[0])[:30],
                "count": int(top.iloc[0]),
                "share": 100.0 * top.iloc[0] / max(len(df), 1),
            }
    return snapshot


# --------------------------------------------------------------------------- #
# Correlations
# --------------------------------------------------------------------------- #
def correlation_matrix(
    df: pd.DataFrame, numeric_cols: Sequence[str], method: str = "pearson", max_cols: int = 25
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Pearson or Spearman correlation matrix (r and p-value).

    Vectorised: r comes from ``DataFrame.corr`` (C-optimised) and p-values are
    derived from the t-distribution, so large datasets stay responsive.
    """
    cols = [c for c in numeric_cols if c in df.columns][:max_cols]
    if len(cols) < 2:
        raise ValueError("At least two numeric columns are required for a correlation matrix.")
    data = df[cols].apply(pd.to_numeric, errors="coerce")
    if method == "spearman":
        r_values = data.rank().corr(method="pearson")
    else:
        r_values = data.corr()

    # Pairwise-complete counts (n per pair) for the p-value degrees of freedom.
    presence = data.notna().astype(np.int64)
    counts = (presence.T @ presence).to_numpy()
    r = r_values.to_numpy()
    with np.errstate(divide="ignore", invalid="ignore"):
        t_stat = r * np.sqrt(np.maximum(counts - 2, 1) / np.maximum(1.0 - r**2, 1e-12))
    p_values = 2.0 * (1.0 - stats.t.cdf(np.abs(t_stat), np.maximum(counts - 2, 1)))
    p_values = pd.DataFrame(np.clip(p_values, 0.0, 1.0), index=cols, columns=cols)
    return r_values.astype(float), p_values.astype(float)


def top_correlated_pairs(
    r_values: pd.DataFrame, p_values: pd.DataFrame, n: int = 10
) -> pd.DataFrame:
    """The n strongest pairwise correlations, ranked by |r| (duplicates removed)."""
    rows = []
    cols = list(r_values.columns)
    for i, col_a in enumerate(cols):
        for col_b in cols[i + 1:]:
            r = float(r_values.loc[col_a, col_b])
            p = float(p_values.loc[col_a, col_b])
            rows.append(
                {
                    "Variable A": col_a,
                    "Variable B": col_b,
                    "r": round(r, 3),
                    "|r|": abs(r),
                    "p-value": f"{p:.3g}",
                    "Interpretation": correlation_label(r),
                }
            )
    frame = pd.DataFrame(rows).sort_values("|r|", ascending=False).head(n)
    return frame.drop(columns=["|r|"], errors="ignore").reset_index(drop=True)


def correlation_label(r: float) -> str:
    abs_r = abs(r)
    if abs_r >= 0.7:
        return "Strong"
    if abs_r >= 0.4:
        return "Moderate"
    if abs_r >= 0.2:
        return "Weak"
    return "Negligible"


# --------------------------------------------------------------------------- #
# Group comparisons (ANOVA / Kruskal–Wallis)
# --------------------------------------------------------------------------- #
def group_stats(
    df: pd.DataFrame, metric: str, group_col: str, max_groups: int = MAX_GROUPS
) -> pd.DataFrame:
    """Per-group n / mean / median / std / min / max for a metric."""
    data = df[[group_col, metric]].dropna()
    if data.empty:
        raise ValueError("No non-missing rows for this combination.")
    counts = data[group_col].value_counts()
    top_groups = counts.head(max_groups).index.tolist()
    if len(counts) > max_groups:
        data = data[data[group_col].isin(top_groups)]
    table = (
        data.groupby(group_col, observed=True)[metric]
        .agg(n="count", mean="mean", median="median", std="std", min="min", max="max")
        .round(3)
        .sort_values("mean", ascending=False)
        .reset_index()
    )
    table.columns = [group_col, "n", "Mean", "Median", "Std dev", "Min", "Max"]
    return table


def anova_test(df: pd.DataFrame, metric: str, group_col: str) -> Dict:
    """One-way ANOVA, falling back to Kruskal–Wallis when variances differ.

    Returns a dict of test statistics plus a plain-language interpretation.
    """
    data = df[[group_col, metric]].dropna()
    groups = [g[metric].to_numpy(dtype=float) for _, g in data.groupby(group_col, observed=True)]
    groups = [g for g in groups if len(g) >= 2]
    if len(groups) < 2:
        raise ValueError("At least two groups with 2+ observations are needed for the test.")

    test_name = "One-way ANOVA (Welch)"
    statistic = p_value = np.nan
    equal_var = True
    try:
        # Levene's test for homogeneity of variance (sampled to keep it fast).
        samples = [g if len(g) <= 1000 else np.random.default_rng(42).choice(g, 1000, replace=False) for g in groups]
        levene = stats.levene(*samples, center="median")
        equal_var = bool(levene.pvalue >= 0.05)
    except (ValueError, FloatingPointError):
        pass

    try:
        if equal_var:
            result = stats.f_oneway(*groups)
            test_name = "One-way ANOVA"
        else:
            result = stats.kruskal(*groups)
            test_name = "Kruskal–Wallis H test"
        statistic, p_value = float(result.statistic), float(result.pvalue)
    except (ValueError, FloatingPointError):
        raise ValueError("The groups could not be tested (constant values or too few observations).")

    # Effect size: eta-squared for ANOVA.
    grand_mean = np.concatenate(groups).mean()
    ss_between = sum(len(g) * (g.mean() - grand_mean) ** 2 for g in groups)
    ss_total = ((np.concatenate(groups) - grand_mean) ** 2).sum()
    eta_sq = float(ss_between / ss_total) if ss_total > 0 else 0.0

    if p_value < 0.001:
        verdict = f"Highly significant difference in '{metric}' across '{group_col}' (p < 0.001)."
    elif p_value < 0.05:
        verdict = f"Significant difference in '{metric}' across '{group_col}' (p = {p_value:.3f})."
    else:
        verdict = (
            f"No statistically significant difference in '{metric}' across '{group_col}' "
            f"(p = {p_value:.3f}) — differences may be due to chance."
        )
    if eta_sq >= 0.14:
        verdict += f" Effect size η² = {eta_sq:.3f} (large)."
    elif eta_sq >= 0.06:
        verdict += f" Effect size η² = {eta_sq:.3f} (medium)."
    elif eta_sq > 0:
        verdict += f" Effect size η² = {eta_sq:.3f} (small)."

    return {
        "test": test_name,
        "statistic": statistic,
        "p_value": p_value,
        "eta_sq": eta_sq,
        "groups": len(groups),
        "observations": int(data.shape[0]),
        "verdict": verdict,
        "equal_variances": equal_var,
    }


# --------------------------------------------------------------------------- #
# Categorical association (chi-square / Cramér's V)
# --------------------------------------------------------------------------- #
def chi_square_test(df: pd.DataFrame, col_a: str, col_b: str) -> Dict:
    """Chi-square test of independence between two categorical columns."""
    data = df[[col_a, col_b]].dropna().astype(str)
    if data.shape[0] < 10:
        raise ValueError("At least 10 non-missing rows are needed for this test.")
    table = pd.crosstab(data[col_a], data[col_b])
    if min(table.shape) < 2:
        raise ValueError("Both columns need at least 2 distinct values.")
    try:
        chi2, p, dof, _ = stats.chi2_contingency(table.to_numpy(), correction=True)
    except ValueError as exc:
        raise ValueError(f"Chi-square could not be computed: {exc}") from exc
    n = table.to_numpy().sum()
    cramers_v = float(np.sqrt(chi2 / (n * (min(table.shape) - 1))) if n > 0 else 0.0)

    if p < 0.05:
        verdict = f"The two columns are statistically associated (p = {p:.3g})."
    else:
        verdict = f"No evidence of association (p = {p:.3g})."
    if cramers_v >= 0.5:
        verdict += f" Cramér's V = {cramers_v:.3f} (strong association)."
    elif cramers_v >= 0.3:
        verdict += f" Cramér's V = {cramers_v:.3f} (moderate association)."
    elif cramers_v > 0:
        verdict += f" Cramér's V = {cramers_v:.3f} (weak association)."

    return {
        "chi2": float(chi2),
        "p_value": float(p),
        "dof": int(dof),
        "n": int(n),
        "cramers_v": cramers_v,
        "verdict": verdict,
        "table": table,
    }


def contingency_table(df: pd.DataFrame, col_a: str, col_b: str, max_cells: int = 10) -> pd.DataFrame:
    """Top-N × top-N contingency table with a totals column."""
    top_a = df[col_a].value_counts().head(max_cells).index
    top_b = df[col_b].value_counts().head(max_cells).index
    data = df[df[col_a].isin(top_a) & df[col_b].isin(top_b)]
    table = pd.crosstab(data[col_a], data[col_b]).astype(int)
    table["Total"] = table.sum(axis=1)
    return table


# --------------------------------------------------------------------------- #
# Outliers
# --------------------------------------------------------------------------- #
def outlier_summary(df: pd.DataFrame, numeric_cols: Sequence[str]) -> pd.DataFrame:
    """IQR-based outlier counts and bounds for numeric columns."""
    rows = []
    for col in numeric_cols:
        values = pd.to_numeric(df[col], errors="coerce").dropna()
        if values.empty:
            continue
        q1, q3 = values.quantile([0.25, 0.75])
        iqr = q3 - q1
        if iqr == 0:
            continue
        lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        count = int(((values < lower) | (values > upper)).sum())
        rows.append(
            {
                "Column": col,
                "Outliers": count,
                "% of values": round(100 * count / len(values), 2),
                "Lower bound": round(float(lower), 3),
                "Upper bound": round(float(upper), 3),
            }
        )
    if not rows:
        raise ValueError("No numeric columns with enough variation for outlier detection.")
    return pd.DataFrame(rows).sort_values("Outliers", ascending=False).reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Time-series aggregation
# --------------------------------------------------------------------------- #
FREQ_LABELS = {"D": "Day", "W": "Week", "M": "Month", "Q": "Quarter", "Y": "Year"}

# pandas >= 3.0 dropped the old period aliases ('M' -> month-end is now 'ME').
_FREQ_ALIASES = {"M": "ME", "Q": "QE", "Y": "YE", "A": "YE"}


def normalize_freq(freq: str) -> str:
    """Translate UI-friendly frequency labels into pandas 3.x aliases."""
    return _FREQ_ALIASES.get(freq, freq)


def aggregate_time_series(
    df: pd.DataFrame, date_col: str, value_col: str, agg: str = "sum", freq: str = "M"
) -> pd.Series:
    """Resample a date + value pair to a regular frequency."""
    series = pd.to_numeric(df[value_col], errors="coerce")
    frame = pd.DataFrame({"date": pd.to_datetime(df[date_col], errors="coerce"), "value": series})
    frame = frame.dropna()
    if frame.empty:
        raise ValueError("No non-missing (date, value) rows to aggregate.")
    frame = frame.set_index("date")
    ts = frame["value"].resample(normalize_freq(freq)).agg(agg)
    ts = ts[~ts.index.duplicated()]
    return ts.dropna()


def trend_series(
    df: pd.DataFrame, date_col: str, value_col: str, agg: str = "sum", freq: str = "M"
) -> pd.Series:
    """Aggregated time series; ``agg='count'`` counts rows instead of summing."""
    if agg == "count":
        index = pd.to_datetime(df[date_col], errors="coerce")
        series = pd.Series(1.0, index=index).dropna()
        return series.resample(normalize_freq(freq)).count().dropna()
    return aggregate_time_series(df, date_col, value_col, agg, freq)


def trend_by_group(
    df: pd.DataFrame,
    date_col: str,
    value_col: str,
    group_col: str,
    agg: str = "sum",
    freq: str = "M",
    top_n: int = 8,
) -> pd.DataFrame:
    """Time × group matrix (one series per group, capped at the biggest groups)."""
    frame = df[[date_col, value_col, group_col]].dropna()
    frame = frame.assign(_t=pd.to_datetime(frame[date_col], errors="coerce")).dropna(subset=["_t"])
    if frame.empty:
        raise ValueError("No non-missing rows for the selected columns.")
    freq = normalize_freq(freq)
    if agg == "count":
        wide = frame.set_index("_t").groupby(group_col, observed=True).resample(freq).size().unstack(level=0)
    else:
        frame = frame.assign(_v=pd.to_numeric(frame[value_col], errors="coerce")).dropna(subset=["_v"])
        wide = frame.set_index("_t").groupby(group_col, observed=True)["_v"].resample(freq).agg(agg).unstack(level=0)
    wide = wide.dropna(how="all")
    if wide.empty:
        raise ValueError("Not enough data after grouping — try a different group or frequency.")
    biggest = wide.sum().sort_values(ascending=False).head(top_n).index
    return wide[biggest]


def suggest_frequency(series: pd.Series) -> str:
    """Pick a sensible resampling frequency from the series' span."""
    if len(series) < 2:
        return "M"
    span = series.index.max() - series.index.min()
    days = span / pd.Timedelta(days=1)
    if days > 3 * 365:
        return "M"
    if days > 120:
        return "W"
    return "D"
