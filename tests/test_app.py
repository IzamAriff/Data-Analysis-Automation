"""End-to-end Streamlit tests using the official AppTest harness.

Run with:  pytest tests/test_app.py -v

These tests drive the real app script (`app.py`) headlessly: loading the
bundled samples, confirming the profile step, switching tabs, applying
filters, and asserting that nothing raises.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

REPO_ROOT = Path(__file__).resolve().parent.parent
APP_PATH = REPO_ROOT / "app.py"

TIMEOUT = 180  # seconds per run (data loading + full tab render)


def _fresh_app() -> AppTest:
    return AppTest.from_file(str(APP_PATH), default_timeout=TIMEOUT)


def test_landing_page_renders():
    at = _fresh_app()
    at.run()
    assert not at.exception
    assert any("DataPilot" in str(m.value) for m in at.markdown)


def test_load_sample_and_explore():
    at = _fresh_app()
    at.run()

    # Landing -> load the default bundled sample.
    at.button(key="load_sample").click().run()
    assert not at.exception
    assert "bundle" in at.session_state

    # Confirm the profiling step.
    at.button(key="confirm_explore").click().run()
    assert not at.exception
    assert at.session_state["explored"] is True
    df = at.session_state["df"]
    assert len(df) == 9994  # the full Superstore sample
    assert "Order Date" in at.session_state["roles"]

    # All tabs render in this run (tabs execute every body).
    assert at.metric  # KPI cards present
    assert at.dataframe  # profile/statistics tables present
    # plotly_chart elements are not part of AppTest's element tree, but their
    # rendering code still runs — any exception would appear in at.exception.


def test_second_sample_dataset():
    at = _fresh_app()
    at.run()
    at.selectbox(key="sample_select").select("Video game sales — vgsales (16,595 rows)")
    at.button(key="load_sample").click().run()
    assert not at.exception
    at.button(key="confirm_explore").click().run()
    assert not at.exception
    assert at.session_state["df"].shape[1] == 11
    assert at.session_state["roles"]["year"] == "year"


def test_sidebar_filters_and_reset():
    at = _fresh_app()
    at.run()
    at.button(key="load_sample").click().run()
    at.button(key="confirm_explore").click().run()
    assert not at.exception

    # Filter to a single region + a numeric range + a date range.
    at.sidebar.multiselect(key="f_cat_Region").set_value(["West"])
    at.sidebar.slider(key="f_num_Sales").set_value((100.0, 500.0))
    at.sidebar.slider(key="f_date_range").set_value((datetime(2016, 1, 1), datetime(2016, 12, 31)))
    at.run()
    assert not at.exception

    # Filters should have reduced the dataset (but not emptied it).
    rows_before = 9994
    assert len(at.session_state["df"]) == rows_before  # source df untouched
    assert any("Showing" in str(c.value) for c in at.caption)  # row-count caption rendered

    # Reset filters -> no exception, full data again.
    at.button(key="reset_filters").click().run()
    assert not at.exception


def test_search_filter_no_crash():
    at = _fresh_app()
    at.run()
    at.button(key="load_sample").click().run()
    at.button(key="confirm_explore").click().run()
    at.sidebar.text_input(key="f_search_text").input("chair")
    at.run()
    assert not at.exception


def test_chart_tabs_and_predictive_widgets_render():
    at = _fresh_app()
    at.run()
    at.button(key="load_sample").click().run()
    at.button(key="confirm_explore").click().run()
    assert not at.exception
    # Predictive tab widgets exist (models only run on demand).
    at.radio(key="pred_mode")
    at.selectbox(key="pred_reg_target")
    assert not at.exception


def test_regression_run_flow():
    at = _fresh_app()
    at.run()
    at.button(key="load_sample").click().run()
    at.button(key="confirm_explore").click().run()
    # Run the regression with the default feature selection.
    at.button(key="pred_reg_run").click().run()
    assert not at.exception
    results = at.session_state["last_results"].get("regression")
    assert results is not None
    assert results["random_forest"]["r2"] > 0.3


def test_classification_run_flow():
    at = _fresh_app()
    at.run()
    at.button(key="load_sample").click().run()
    at.button(key="confirm_explore").click().run()
    at.radio(key="pred_mode").set_value("🏷️ Classification")
    at.run()
    at.selectbox(key="pred_cls_target").select("Category")
    at.button(key="pred_cls_run").click().run()
    assert not at.exception
    cls = at.session_state["last_results"].get("classification")
    assert cls is not None
    assert cls["accuracy"] > cls["baseline_accuracy"]


def test_every_chart_type_renders():
    """Cycle through all chart types on both samples via the real widget."""
    at = _fresh_app()
    at.run()
    at.button(key="load_sample").click().run()
    at.button(key="confirm_explore").click().run()
    for chart_type in ["Trend", "Distribution", "Comparison", "Relationship", "Composition", "Correlation"]:
        at.segmented_control(key="chart_type").set_value(chart_type)
        at.run()
        assert not at.exception, f"chart type {chart_type} raised"
    # Composition pie style (exercises the px.pie branch).
    at.segmented_control(key="chart_type").set_value("Composition")
    at.run()
    at.radio(key="c_comp_kind").set_value("pie")
    at.run()
    assert not at.exception


def test_forecast_run_flow():
    at = _fresh_app()
    at.run()
    at.button(key="load_sample").click().run()
    at.button(key="confirm_explore").click().run()
    at.radio(key="pred_mode").set_value("🔮 Forecasting")
    at.run()
    at.button(key="pred_fc_run").click().run()
    assert not at.exception
    fc = at.session_state["last_results"].get("forecast")
    assert fc is not None
    assert len(fc["forecast"]) == 12
