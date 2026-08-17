"""Colorblind-safe Plotly chart builders.

Discrete palette: Okabe–Ito (safe for deuteranopia/protanopia).
Sequential scales: Cividis (perceptually uniform, colorblind-safe).
Diverging scale: ColorBrewer BrBG (brown–teal, colorblind-safe).

All figures use a clean white template, descriptive axis titles and a
consistent layout so charts stay accessible and readable at any size.
"""

from __future__ import annotations

import logging
from typing import List, Optional, Sequence

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

logger = logging.getLogger("datapilot.plots")

# Okabe–Ito colorblind-safe qualitative palette.
PALETTE: List[str] = [
    "#0072B2",  # blue
    "#E69F00",  # orange
    "#009E73",  # bluish green
    "#D55E00",  # vermilion
    "#CC79A7",  # reddish purple
    "#56B4E9",  # sky blue
    "#F0E442",  # yellow
    "#999999",  # grey
]

SEQUENTIAL = px.colors.sequential.Cividis
DIVERGING = ["#8C510A", "#D8B365", "#F6E8C3", "#F5F5F5", "#C7EAE5", "#5AB4AC", "#01665E"]  # BrBG

MAX_SCATTER_POINTS = 20_000  # down-sample very large scatter plots
MAX_BOX_POINTS = 50_000


def style(
    fig: go.Figure,
    title: str = "",
    height: int = 420,
    xlabel: str = "",
    ylabel: str = "",
) -> go.Figure:
    """Apply the shared layout: clean template, grid, fonts and hover mode."""
    fig.update_layout(
        template="plotly_white",
        title={"text": title, "x": 0.02, "xanchor": "left", "font": {"size": 17}},
        height=height,
        margin={"l": 10, "r": 10, "t": 50 if title else 30, "b": 10},
        hoverlabel={"font_size": 13},
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "right", "x": 1},
        font={"family": "Segoe UI, Helvetica, Arial, sans-serif", "size": 13, "color": "#1A1C23"},
        colorway=PALETTE,
    )
    if xlabel:
        fig.update_xaxes(title_text=xlabel, title_font={"size": 13})
    if ylabel:
        fig.update_yaxes(title_text=ylabel, title_font={"size": 13})
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(gridcolor="#E4E7EC", gridwidth=1)
    return fig


def fmt(value: float, digits: int = 1) -> str:
    """Human-friendly number formatting (1.2M, 34.5K)."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "—"
    abs_value = abs(value)
    if abs_value >= 1e9:
        return f"{value / 1e9:.{digits}f}B"
    if abs_value >= 1e6:
        return f"{value / 1e6:.{digits}f}M"
    if abs_value >= 1e3:
        return f"{value / 1e3:.{digits}f}K"
    return f"{value:,.{digits}f}"


# --------------------------------------------------------------------------- #
# Chart builders
# --------------------------------------------------------------------------- #
def trend_chart(
    ts: pd.Series,
    value_col: str,
    group_col: Optional[str] = None,
    title: str = "",
    agg_label: str = "Sum",
    freq_label: str = "Month",
) -> go.Figure:
    """Time-series line/area chart for a single aggregated series."""
    fig = go.Figure()
    is_datetime = isinstance(ts.index, pd.DatetimeIndex)
    x_template = "%{x|%d %b %Y}" if is_datetime else "%{x}"
    fig.add_scatter(
        x=ts.index,
        y=ts.values,
        mode="lines",
        name=value_col if not group_col else f"{value_col} (all)",
        line={"color": PALETTE[0], "width": 2.5},
        hovertemplate=x_template + "<br>" + value_col + ": %{y:,.2f}<extra></extra>",
    )
    fig = style(fig, title=title, ylabel=f"{value_col} — {agg_label} per {freq_label}", xlabel="Date")
    return fig


def grouped_trend_chart(
    ts_by_group: pd.DataFrame, value_col: str, group_col: str, title: str = ""
) -> go.Figure:
    """Multi-line trend with one series per group (capped at 8 groups)."""
    fig = go.Figure()
    for i, group in enumerate(ts_by_group.columns[:8]):
        fig.add_scatter(
            x=ts_by_group.index,
            y=ts_by_group[group].values,
            mode="lines",
            name=str(group)[:25],
            line={"color": PALETTE[i % len(PALETTE)], "width": 2},
        )
    fig = style(fig, title=title, ylabel=value_col, xlabel="Date")
    return fig


def histogram_chart(
    df: pd.DataFrame, col: str, group_col: Optional[str] = None, bins: int = 40, title: str = ""
) -> go.Figure:
    """Histogram (optionally overlaid per group, capped at 6 groups)."""
    if group_col:
        data = df[[col, group_col]].copy()
        data[col] = pd.to_numeric(data[col], errors="coerce")
        data = data.dropna()
        top = data[group_col].value_counts().head(6).index
        data = data[data[group_col].isin(top)]
        fig = px.histogram(
            data, x=col, color=group_col, nbins=bins, opacity=0.65, barmode="overlay",
            color_discrete_sequence=PALETTE,
        )
        fig.update_xaxes(title_text=col)
        fig.update_yaxes(title_text="Count")
    else:
        numeric = pd.to_numeric(df[col], errors="coerce").dropna()
        fig = px.histogram(x=numeric, nbins=bins, color_discrete_sequence=[PALETTE[0]])
        fig.update_xaxes(title_text=col)
        fig.update_yaxes(title_text="Count")
    fig = style(fig, title=title or f"Distribution of {col}")
    fig.update_traces(marker_line_width=0.5, marker_line_color="#FFFFFF")
    return fig


def box_chart(
    df: pd.DataFrame, col: str, group_col: str, title: str = "", max_groups: int = 12
) -> go.Figure:
    """Box plot of a numeric column by category."""
    top = df[group_col].value_counts().head(max_groups).index
    data = df[df[group_col].isin(top)][[group_col, col]].dropna()
    if len(data) > MAX_BOX_POINTS:
        data = data.sample(MAX_BOX_POINTS, random_state=42)
    fig = px.box(
        data, x=group_col, y=col, color=group_col, points=False,
        color_discrete_sequence=PALETTE,
    )
    fig = style(fig, title=title or f"{col} by {group_col}", ylabel=col, xlabel=group_col)
    fig.update_xaxes(tickangle=25)
    return fig


def bar_chart(
    data: pd.DataFrame, cat_col: str, value_col: str, title: str = "", horizontal: bool = False
) -> go.Figure:
    """Aggregated bar chart for a categorical column (data pre-aggregated)."""
    data = data.sort_values(value_col, ascending=not horizontal)
    orientation = "h" if horizontal else "v"
    fig = go.Figure(
        go.Bar(
            x=data[value_col] if horizontal else data[cat_col],
            y=data[cat_col] if horizontal else data[value_col],
            orientation=orientation,
            marker={"color": PALETTE[0], "line": {"width": 0}},
            hovertemplate=(
                f"{cat_col}: %{{" + ("y" if horizontal else "x") + "}}<br>"
                f"{value_col}: %{{" + ("x" if horizontal else "y") + "}:,.2f}<extra></extra>"
            ),
        )
    )
    fig = style(fig, title=title, xlabel=value_col if horizontal else cat_col,
                ylabel=cat_col if horizontal else value_col)
    if not horizontal:
        fig.update_xaxes(tickangle=25)
    return fig


def scatter_chart(
    df: pd.DataFrame,
    x: str,
    y: str,
    color: Optional[str] = None,
    size: Optional[str] = None,
    trendline: bool = False,
    title: str = "",
) -> go.Figure:
    """Scatter plot with optional category colour, size and OLS trendline."""
    cols = [x, y] + ([color] if color else []) + ([size] if size else [])
    data = df[cols].dropna()
    if color:
        top = data[color].value_counts().head(8).index
        data = data[data[color].isin(top)]
    if len(data) > MAX_SCATTER_POINTS:
        data = data.sample(MAX_SCATTER_POINTS, random_state=42)
    trend = "ols" if trendline and not color else None
    fig = px.scatter(
        data, x=x, y=y, color=color, size=size, trendline=trend,
        color_discrete_sequence=PALETTE, opacity=0.75,
        marginal_x=None, marginal_y=None,
    )
    if trendline and not color:
        # Keep the trendline visually distinct (Okabe-Ito orange).
        fig.data[-1].line.color = "#D55E00"
        fig.data[-1].line.width = 3
    fig = style(fig, title=title or f"{y} vs {x}", xlabel=x, ylabel=y)
    return fig


def correlation_heatmap(
    r_values: pd.DataFrame, method_label: str = "Pearson", title: str = ""
) -> go.Figure:
    """Diverging heatmap of a correlation matrix with annotated values."""
    fig = go.Figure(
        go.Heatmap(
            z=r_values.values,
            x=list(r_values.columns),
            y=list(r_values.index),
            zmin=-1, zmax=1,
            colorscale=DIVERGING,
            colorbar={"title": "r", "thickness": 14},
            text=np.round(r_values.values, 2),
            texttemplate="%{text}",
            textfont={"size": 11, "color": "#1A1C23"},
            hovertemplate="%{y} ↔ %{x}<br>r = %{z:.3f}<extra></extra>",
        )
    )
    fig = style(fig, title=title or f"{method_label} correlation matrix", height=560)
    fig.update_layout(margin={"l": 10, "r": 10, "t": 60, "b": 10})
    return fig


def composition_chart(
    data: pd.DataFrame, cat_col: str, value_col: str, kind: str = "treemap", title: str = ""
) -> go.Figure:
    """Treemap or pie of a category's share of a measure."""
    if kind == "pie":
        fig = px.pie(
            data, names=cat_col, values=value_col, hole=0.45,
            color_discrete_sequence=PALETTE,
        )
        fig.update_traces(textposition="inside", textinfo="percent+label", hovertemplate=(
            f"{cat_col}: %{{label}}<br>{value_col}: %{{value:,.2f}} (%{{percent}})<extra></extra>"
        ))
    else:
        fig = px.treemap(
            data, path=[px.Constant("all"), cat_col], values=value_col,
            color_continuous_scale=SEQUENTIAL,
        )
        fig.update_traces(hovertemplate=(
            f"{cat_col}: %{{label}}<br>{value_col}: %{{value:,.2f}}<extra></extra>"
        ))
    fig = style(fig, title=title or f"Composition of {value_col} by {cat_col}", height=460)
    fig.update_layout(coloraxis={"colorbar": {"thickness": 14}} if kind == "treemap" else {})
    return fig


def missing_chart(missing_pct: pd.Series, title: str = "") -> go.Figure:
    """Horizontal bar of % missing values per column (only columns with gaps)."""
    data = missing_pct[missing_pct > 0].sort_values()
    if data.empty:
        data = pd.Series({"No missing values": 0.0})
    fig = go.Figure(
        go.Bar(
            x=data.values, y=data.index, orientation="h",
            marker={"color": "#D55E00", "line": {"width": 0}},
            hovertemplate="%{y}: %{x:.2f}% missing<extra></extra>",
        )
    )
    fig = style(fig, title=title or "Missing values by column (%)", height=max(260, 40 + 28 * len(data)),
                xlabel="% missing")
    return fig


def forecast_chart(
    history: pd.Series,
    fitted: pd.Series,
    forecast: pd.Series,
    ci_lower: Optional[pd.Series] = None,
    ci_upper: Optional[pd.Series] = None,
    title: str = "",
    value_col: str = "value",
) -> go.Figure:
    """History + fitted values + forecast band."""
    fig = go.Figure()
    fig.add_scatter(
        x=history.index, y=history.values, mode="lines", name="Actual",
        line={"color": PALETTE[0], "width": 2.2},
    )
    if fitted is not None and len(fitted):
        fig.add_scatter(
            x=fitted.index, y=fitted.values, mode="lines", name="Fitted (in-sample)",
            line={"color": PALETTE[1], "width": 2, "dash": "dot"},
        )
    if ci_lower is not None and ci_upper is not None:
        band_x = list(forecast.index)
        fig.add_trace(
            go.Scatter(
                x=band_x, y=ci_upper.values, mode="lines", name="95% band (approx.)",
                line={"width": 0}, showlegend=False, hoverinfo="skip",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=band_x, y=ci_lower.values, mode="lines", name="95% band (approx.)",
                line={"width": 0}, fill="tonexty", fillcolor="rgba(0,114,178,0.15)",
                showlegend=False, hoverinfo="skip",
            )
        )
    fig.add_scatter(
        x=forecast.index, y=forecast.values, mode="lines", name="Forecast",
        line={"color": "#D55E00", "width": 2.5, "dash": "dash"},
        hovertemplate="%{x|%d %b %Y}<br>forecast: %{y:,.2f}<extra></extra>",
    )
    fig = style(fig, title=title, ylabel=value_col, xlabel="Date")
    return fig


def elbow_chart(ks: List[int], inertias: List[float], silhouettes: List[float]) -> go.Figure:
    """Dual-axis plot: within-cluster inertia (elbow) and silhouette score."""
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=ks, y=inertias, mode="lines+markers", name="Inertia (lower is better)",
            line={"color": PALETTE[0], "width": 2.5}, marker={"size": 8},
        )
    )
    fig.add_trace(
        go.Scatter(
            x=ks, y=silhouettes, mode="lines+markers", name="Silhouette (higher is better)",
            line={"color": PALETTE[2], "width": 2.5, "dash": "dash"}, marker={"size": 8}, yaxis="y2",
        )
    )
    fig.update_layout(
        yaxis={"title": "Inertia", "gridcolor": "#E4E7EC"},
        yaxis2={"title": "Silhouette", "overlaying": "y", "side": "right"},
    )
    fig = style(fig, title="Choosing k: elbow & silhouette", xlabel="Number of clusters (k)")
    fig.update_xaxes(dtick=1)
    return fig


def cluster_scatter(pca_df: pd.DataFrame, x: str, y: str, label_col: str, title: str = "") -> go.Figure:
    """2-D PCA projection of clustered data."""
    if len(pca_df) > MAX_SCATTER_POINTS:
        pca_df = pca_df.sample(MAX_SCATTER_POINTS, random_state=42)
    fig = px.scatter(
        pca_df, x=x, y=y, color=pca_df[label_col].astype(str),
        color_discrete_sequence=PALETTE, opacity=0.7,
        category_orders={label_col: sorted(pca_df[label_col].astype(str).unique())},
    )
    fig = style(fig, title=title, xlabel=x, ylabel=y)
    fig.update_layout(legend_title_text=label_col)
    return fig


def importance_chart(names: Sequence[str], values: Sequence[float], title: str = "") -> go.Figure:
    """Horizontal feature-importance bar chart."""
    order = np.argsort(values)
    fig = go.Figure(
        go.Bar(
            x=[float(values[i]) for i in order],
            y=[str(names[i])[:35] for i in order],
            orientation="h",
            marker={"color": PALETTE[0], "line": {"width": 0}},
            hovertemplate="%{y}: %{x:.3f}<extra></extra>",
        )
    )
    fig = style(fig, title=title, xlabel="Importance", height=max(300, 80 + 24 * len(names)))
    return fig


def confusion_heatmap(cm: np.ndarray, labels: Sequence[str]) -> go.Figure:
    """Confusion matrix heatmap for classification results."""
    fig = go.Figure(
        go.Heatmap(
            z=cm, x=[str(l)[:15] for l in labels], y=[str(l)[:15] for l in labels],
            colorscale=SEQUENTIAL, colorbar={"title": "Count", "thickness": 14},
            text=cm.astype(int).astype(str), texttemplate="%{text}",
            textfont={"size": 12, "color": "#1A1C23"},
            hovertemplate="Actual %{y}<br>Predicted %{x}<br>%{z}<extra></extra>",
        )
    )
    fig = style(fig, title="Confusion matrix (rows = actual)", height=420,
                xlabel="Predicted", ylabel="Actual")
    return fig


def residuals_chart(actual: pd.Series, predicted: pd.Series, title: str = "") -> go.Figure:
    """Actual vs predicted scatter with the ideal 45° line."""
    lim = [min(actual.min(), predicted.min()), max(actual.max(), predicted.max())]
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=actual, y=predicted, mode="markers", name="Holdout predictions",
            marker={"color": PALETTE[0], "size": 7, "opacity": 0.55},
            hovertemplate="Actual: %{x:,.2f}<br>Predicted: %{y:,.2f}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=lim, y=lim, mode="lines", name="Perfect fit",
            line={"color": "#D55E00", "width": 2, "dash": "dash"},
        )
    )
    fig = style(fig, title=title, xlabel="Actual", ylabel="Predicted")
    return fig
