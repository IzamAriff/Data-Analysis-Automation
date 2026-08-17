"""Unit tests for the DataPilot analytics modules (no Streamlit runtime).

Run with:  pytest tests/test_modules.py -v
"""

from __future__ import annotations

import io

import numpy as np
import pandas as pd
import pytest

from src import analysis, loader, modeling, plots, profile, ui

REPO_ROOT = loader.REPO_ROOT


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def superstore():
    bundle = loader.load_bundled_sample("Retail orders — 'Sample Superstore' (9,994 rows)")
    df, notes = loader.prepare_dataframe(bundle.df)
    roles = profile.infer_roles(df)
    return df, roles


@pytest.fixture(scope="module")
def vgsales():
    bundle = loader.load_bundled_sample("Video game sales — vgsales (16,595 rows)")
    df, notes = loader.prepare_dataframe(bundle.df)
    roles = profile.infer_roles(df)
    return df, roles


# --------------------------------------------------------------------------- #
# Loader
# --------------------------------------------------------------------------- #
def test_bundled_samples_load():
    for label in loader.bundled_sample_names():
        bundle = loader.load_bundled_sample(label)
        assert not bundle.df.empty


def test_sanitize_column_names():
    assert loader.sanitize_column_names(["  A  B ", "a", "a", "\n"]) == ["A B", "a", "a_2", "Unnamed"]


def test_read_text_detects_delimiter_and_encoding():
    csv_semicolon = "name;value;date\nfoo;1,5;2024-01-01\nbar;2,5;2024-02-01\n".encode("latin-1")
    df = loader._read_text(csv_semicolon)
    assert list(df.columns) == ["name", "value", "date"]
    assert len(df) == 2


def test_read_json_records_and_lines():
    df_records = loader._read_json(b'[{"a": 1, "b": "x"}, {"a": 2, "b": "y"}]')
    assert df_records.shape == (2, 2)
    df_lines = loader._read_json(b'{"a": 1}\n{"a": 2}\n')
    assert df_lines.shape == (2, 1)


def test_parse_date_columns():
    df = pd.DataFrame({"d": ["2024-01-01", "2024-06-15", None], "n": [1, 2, 3]})
    df, parsed = loader.parse_date_columns(df)
    assert parsed == ["d"]
    assert pd.api.types.is_datetime64_any_dtype(df["d"])


def test_parse_numeric_strings():
    df = pd.DataFrame({"money": ["$1,234.50", "$2,000", None], "other": ["a", "b", "c"]})
    df, parsed = loader.parse_numeric_strings(df)
    assert parsed == ["money"]
    assert df["money"].dtype.kind == "f"


def test_url_scheme_validation():
    with pytest.raises(loader.LoaderError, match="http"):
        loader.load_from_url("file:///etc/passwd")


def test_size_limit_enforced():
    with pytest.raises(loader.LoaderError, match="limit"):
        loader.load_from_bytes(b"x" * (loader.MAX_FILE_BYTES + 1), "big.csv", ".csv")


def test_data_hash_stability():
    df = pd.DataFrame({"a": [1, 2, 3]})
    assert loader.data_hash(df) == loader.data_hash(df.copy())


# --------------------------------------------------------------------------- #
# Profile / role inference
# --------------------------------------------------------------------------- #
def test_superstore_roles(superstore):
    _, roles = superstore
    assert roles["Order Date"] == "date"
    assert roles["Sales"] == "numeric"
    assert roles["Region"] == "category"
    assert roles["Row ID"] == "id"
    assert roles["Customer ID"] == "id"
    assert roles["Postal Code"] == "category"


def test_vgsales_roles(vgsales):
    _, roles = vgsales
    assert roles["year"] == "year"
    assert roles["global_sales"] == "numeric"
    assert roles["genre"] == "category"
    assert roles["rank"] == "id"
    assert roles["name"] == "text"


def test_primary_metric(superstore, vgsales):
    df, roles = superstore
    assert profile.primary_metric(df, roles) == "Sales"
    vdf, vroles = vgsales
    assert "sales" in profile.primary_metric(vdf, vroles)


def test_profile_summary_and_dictionary(superstore):
    df, roles = superstore
    summary = profile.profile_summary(df, roles)
    assert summary["rows"] == len(df)
    assert summary["columns"] == 21
    dictionary = profile.build_data_dictionary(df, roles)
    assert list(dictionary["Column"]) == list(df.columns)
    assert dictionary["Description"].notna().all()


# --------------------------------------------------------------------------- #
# Analysis
# --------------------------------------------------------------------------- #
def test_correlation_matrix(superstore):
    df, _ = superstore
    r, p = analysis.correlation_matrix(df, ["Sales", "Quantity", "Discount", "Profit"])
    # Diagonal must be exactly 1.
    assert np.allclose(np.diag(r.values), 1.0)
    # Known relationship direction in Superstore.
    assert r.loc["Sales", "Quantity"] > 0
    assert p.loc["Sales", "Quantity"] < 0.05
    # Spearman path works too.
    rs, _ = analysis.correlation_matrix(df, ["Sales", "Quantity"], method="spearman")
    assert -1 <= rs.loc["Sales", "Quantity"] <= 1


def test_top_correlated_pairs(superstore):
    df, _ = superstore
    r, p = analysis.correlation_matrix(df, ["Sales", "Quantity", "Discount", "Profit"])
    pairs = analysis.top_correlated_pairs(r, p, n=3)
    assert len(pairs) <= 3
    assert abs(pairs.iloc[0]["r"]) >= abs(pairs.iloc[-1]["r"])


def test_anova_and_group_stats(superstore):
    df, _ = superstore
    result = analysis.anova_test(df, "Profit", "Region")
    assert result["p_value"] >= 0
    assert "test" in result and result["groups"] == 4
    table = analysis.group_stats(df, "Profit", "Region")
    assert len(table) == 4 and {"n", "Mean", "Median"} <= set(table.columns)


def test_chi_square(superstore):
    df, _ = superstore
    result = analysis.chi_square_test(df, "Segment", "Ship Mode")
    assert result["p_value"] < 0.05  # statistically associated in this dataset
    assert 0 <= result["cramers_v"] <= 1


def test_outlier_summary(superstore):
    df, _ = superstore
    table = analysis.outlier_summary(df, ["Sales", "Profit"])
    assert {"Column", "Outliers", "% of values"} <= set(table.columns)


def test_trend_aggregation(superstore):
    df, _ = superstore
    ts = analysis.trend_series(df, "Order Date", "Sales", "sum", "M")
    assert len(ts) == 48  # 4 years of monthly data
    assert ts.index.is_monotonic_increasing
    count_ts = analysis.trend_series(df, "Order Date", "Sales", "count", "M")
    assert count_ts.iloc[0] > 0
    wide = analysis.trend_by_group(df, "Order Date", "Sales", "Region", "sum", "M")
    assert list(wide.columns) == ["West", "East", "Central", "South"]


def test_kpi_snapshot(superstore):
    df, roles = superstore
    snap = analysis.kpi_snapshot(df, roles, "Sales", "Order Date")
    assert snap["metric_total"] > 2_000_000
    assert snap["date_span"][0] < snap["date_span"][1]
    assert snap["top_category"]["column"] == "Ship Mode"


# --------------------------------------------------------------------------- #
# Modeling
# --------------------------------------------------------------------------- #
def test_regression_cv(superstore):
    df, roles = superstore
    outcome = modeling.run_regression(df, "Profit", ["Sales", "Quantity", "Discount", "Category"], roles)
    assert outcome["kind"] == "regression"
    assert outcome["random_forest"]["r2"] > 0.3  # cross-validated improvement
    assert outcome["random_forest"]["rmse"] < outcome["baseline_rmse"]
    assert len(outcome["rf_importance"]) > 0


def test_classification_cv(superstore):
    df, roles = superstore
    outcome = modeling.run_classification(df, "Category", ["Sales", "Profit", "Quantity", "Segment"], roles)
    assert outcome["accuracy"] > outcome["baseline_accuracy"]
    assert outcome["confusion_matrix"].shape == (3, 3)
    assert outcome["class_labels"] == ["Furniture", "Office Supplies", "Technology"]


def test_classification_too_many_classes(superstore):
    df, roles = superstore
    with pytest.raises(modeling.ModelingError):
        modeling.run_classification(df, "City", ["Sales"], roles)  # 531 cities > MAX_CLASSES


def test_clustering(superstore):
    df, roles = superstore
    outcome = modeling.run_clustering(df, ["Sales", "Profit", "Quantity", "Discount"], roles)
    assert outcome["best_k"] >= 2
    assert len(outcome["pca_x"]) == outcome["n_rows"]
    assert set(outcome["cluster_means"].columns) >= {"Sales", "Profit"}


def test_forecast_beats_naive(superstore):
    df, _ = superstore
    outcome = modeling.run_forecast(df, "Order Date", "Sales", "sum", "M", 12)
    assert len(outcome["forecast"]) == 12
    assert outcome["mape_model"] < outcome["mape_naive"]
    assert len(outcome["ci_lower"]) == 12


def test_forecast_rejects_short_series(superstore):
    df, _ = superstore
    tiny = df.head(5).copy()
    with pytest.raises(modeling.ModelingError):
        modeling.run_forecast(tiny, "Order Date", "Sales", "sum", "M", 4)


def test_missing_strategy_drop(superstore):
    df, roles = superstore
    outcome = modeling.run_regression(
        df, "Profit", ["Sales", "Quantity"], roles, missing_strategy="drop", with_random_forest=False
    )
    assert outcome["linear"]["r2"] > 0


# --------------------------------------------------------------------------- #
# Plots (figure-level smoke tests)
# --------------------------------------------------------------------------- #
def test_figures_build(superstore):
    df, roles = superstore
    ts = analysis.trend_series(df, "Order Date", "Sales", "sum", "M")
    figs = {
        "trend": plots.trend_chart(ts, "Sales", title="t"),
        "grouped": plots.grouped_trend_chart(analysis.trend_by_group(df, "Order Date", "Sales", "Region", "sum", "M"), "Sales", "Region"),
        "hist": plots.histogram_chart(df, "Sales", "Region"),
        "box": plots.box_chart(df, "Sales", "Region"),
        "scatter": plots.scatter_chart(df, "Sales", "Profit", color="Region", trendline=False),
        "heat": plots.correlation_heatmap(analysis.correlation_matrix(df, ["Sales", "Profit", "Quantity"])[0]),
        "comp": plots.composition_chart(df.groupby("Category")["Sales"].sum().reset_index(), "Category", "Sales", kind="treemap"),
        "missing": plots.missing_chart(100 * df.isna().sum() / len(df)),
        "importance": plots.importance_chart(["Sales", "Discount"], [0.8, 0.2]),
    }
    for name, fig in figs.items():
        assert fig is not None, name
        assert len(fig.data) >= 1, name


def _chrome_available() -> bool:
    """Kaleido needs a Chrome/Chromium binary; skip the PNG test without it."""
    try:
        from plotly.io._kaleido import _get_chrome  # noqa: F401
        import kaleido
        from kaleido.scopes.plotly import PlotlyScope

        PlotlyScope()._ensure_chrome()
        return True
    except Exception:
        return False


@pytest.mark.skipif(not _chrome_available(), reason="Chrome not installed — PNG export falls back to HTML")
def test_kaleido_png_export(superstore):
    df, _ = superstore
    fig = plots.histogram_chart(df, "Sales")
    png = fig.to_image(format="png", width=400, height=300)
    assert png.startswith(b"\x89PNG")


def test_html_export_always_available(superstore):
    """The HTML fallback (used when kaleido/Chrome is missing) must always work."""
    df, _ = superstore
    fig = plots.histogram_chart(df, "Sales")
    html = fig.to_html(include_plotlyjs="cdn", full_html=True)
    assert "plotly" in html.lower()


# --------------------------------------------------------------------------- #
# UI filter logic (pure parts)
# --------------------------------------------------------------------------- #
def test_apply_filters(superstore):
    df, roles = superstore
    state = ui.FilterState(
        date_ranges={"Order Date": ("2017-01-01", "2017-12-31")},
        category_picks={"Region": ["West", "East"]},
        numeric_ranges={"Sales": (0.0, 500.0)},
        search_col="Sub-Category",
        search_text="chair",
    )
    filtered = ui.apply_filters(df, state)
    assert 0 < len(filtered) < len(df)
    assert set(filtered["Region"].unique()) <= {"West", "East"}
    assert filtered["Sales"].between(0, 500).all()
    assert filtered["Sub-Category"].str.lower().str.contains("chair", regex=False).all()
    # Filter state round-trips through its cache key.
    assert ui.FilterState().to_key() == ui.FilterState().to_key()


def test_apply_filters_empty_search(superstore):
    df, _ = superstore
    state = ui.FilterState(search_col="Sub-Category", search_text="zzzz-no-match")
    assert ui.apply_filters(df, state).empty


def test_safe_filename():
    assert ui.safe_filename("my data (final).csv") == "my_data__final_.csv"
    assert ui.safe_filename("") == "dataset"
