"""Predictive modelling: regression, classification, clustering, forecasting.

Design principles
-----------------
* Every function is defensive: bad inputs raise :class:`ModelingError` with a
  message that is safe to show in the UI.
* Data prep is deterministic and documented (imputation + one-hot encoding).
* Results are returned as plain dicts so the UI can render them and the
  auto-report can re-state them.
* Results are *honest*: caveats, baseline comparisons and holdout metrics are
  always included so non-technical users do not over-interpret the output.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    silhouette_score,
)
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .analysis import normalize_freq

logger = logging.getLogger("datapilot.modeling")

RANDOM_STATE = 42
MAX_MODEL_ROWS = 100_000   # cap training size for responsiveness
MAX_CLASSES = 20           # classification target cardinality cap
MAX_CATEGORY_LEVELS = 20   # top-N levels kept per one-hot column


class ModelingError(Exception):
    """Raised when a model cannot be fitted; message is UI-safe."""


# --------------------------------------------------------------------------- #
# Data preparation shared by all models
# --------------------------------------------------------------------------- #
def build_model_matrix(
    df: pd.DataFrame,
    target: str,
    features: Sequence[str],
    roles: Dict[str, str],
    missing_strategy: str = "median",
    target_is_categorical: bool = False,
) -> Tuple[pd.DataFrame, pd.Series, List[str], Optional[Dict[int, str]]]:
    """Build a numeric design matrix + target vector.

    * numeric / binary features  -> imputed (median or mean)
    * categorical features       -> one-hot of the top-20 levels per column
    * date features              -> converted to days since the earliest date
    * id / text features         -> dropped with a note

    When ``target_is_categorical``, the target is label-encoded and the
    integer-code -> original-label mapping is returned as the 4th element.
    Returns (X, y, notes, class_map). X and y only contain complete rows.
    """
    notes: List[str] = []
    usable = [f for f in features if f in df.columns and f != target]
    if not usable:
        raise ModelingError(
            "Select at least one feature column that is not the target. "
            "Numeric, categorical and date columns are all supported."
        )

    frame = df[[target] + usable].copy().reset_index(drop=True)
    frame = frame.dropna(subset=[target])
    if frame.empty:
        raise ModelingError("The target column has no non-missing values.")

    if len(frame) > MAX_MODEL_ROWS:
        frame = frame.sample(MAX_MODEL_ROWS, random_state=RANDOM_STATE)
        notes.append(f"Training capped at a random {MAX_MODEL_ROWS:,}-row sample.")

    if missing_strategy == "drop":
        frame = frame.dropna(subset=usable)
        if frame.empty:
            raise ModelingError("Dropping incomplete rows removed all of the data.")

    numeric_cols = [c for c in usable if roles.get(c) in ("numeric", "binary")]
    date_cols = [c for c in usable if roles.get(c) == "date"]
    category_cols = [
        c for c in usable
        if roles.get(c) in ("category", "boolean") and frame[c].nunique(dropna=True) >= 2
    ]
    dropped = [c for c in usable if c not in numeric_cols + date_cols + category_cols]
    if dropped:
        notes.append(f"Dropped id/text feature(s) not usable by the model: {', '.join(dropped)}.")

    # Numeric imputation.
    num_strategy = missing_strategy if missing_strategy in ("median", "mean") else "median"
    numeric_imputer = SimpleImputer(strategy=num_strategy)
    X_parts = []
    if numeric_cols:
        X_parts.append(pd.DataFrame(numeric_imputer.fit_transform(frame[numeric_cols]), columns=numeric_cols))
    if date_cols:
        dated = frame[date_cols].apply(pd.to_datetime, errors="coerce")
        for col in date_cols:
            dated[col] = (dated[col] - dated[col].min()).dt.total_seconds() / 86_400.0
        X_parts.append(pd.DataFrame(SimpleImputer(strategy="median").fit_transform(dated), columns=date_cols))
    if category_cols:
        cat_frame = frame[category_cols].astype(str)
        cat_frame = cat_frame.fillna("(missing)")
        for col in category_cols:
            top = cat_frame[col].value_counts().head(MAX_CATEGORY_LEVELS).index
            cat_frame[col] = cat_frame[col].where(cat_frame[col].isin(top), "(other)")
        encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
        encoded = encoder.fit_transform(cat_frame)
        feature_names = [f"{col}={lvl}" for col, levels in zip(category_cols, encoder.categories_) for lvl in levels]
        X_parts.append(pd.DataFrame(encoded, columns=feature_names))
    if not X_parts:
        raise ModelingError(
            "None of the selected features could be used by the model "
            "(no numeric, date or categorical columns among them)."
        )

    X = pd.concat(X_parts, axis=1).astype(float)

    # Target preparation.
    class_map: Optional[Dict[int, str]] = None
    if target_is_categorical:
        codes, uniques = pd.factorize(frame[target].astype(str), sort=True)
        y = pd.Series(codes, index=frame.index)
        class_map = {code: label for code, label in enumerate(uniques)}
    else:
        y = pd.to_numeric(frame[target], errors="coerce")

    valid = y.notna() & X.notna().all(axis=1)
    X, y = X[valid], y[valid]
    if len(X) < 50:
        raise ModelingError("Fewer than 50 complete rows — too little data to fit a reliable model.")
    return X, y, notes, class_map


def _cross_validate(
    model, X: pd.DataFrame, y: pd.Series, is_classifier: bool, n_splits: int = 5
) -> Dict:
    """5-fold cross-validation with pooled out-of-fold predictions.

    Pooled predictions (every row predicted once by a model that never saw it)
    drive both the metrics and the residual/confusion charts, which makes the
    results far more stable than a single lucky/unlucky train-test split.
    """
    y_values = y.to_numpy()
    if is_classifier:
        counts = pd.Series(y_values).value_counts()
        if counts.min() >= n_splits:
            splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
        else:
            splitter = KFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    else:
        splitter = KFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)

    X_np = X.to_numpy(dtype=float)
    predictions = np.empty(len(y), dtype=float)
    fold_metrics: Dict[str, List[float]] = {"regression": [], "classification": []}

    for train_idx, test_idx in splitter.split(X_np, y_values):
        model.fit(X_np[train_idx], y_values[train_idx])
        preds = model.predict(X_np[test_idx])
        predictions[test_idx] = preds
        if is_classifier:
            fold_metrics["classification"].append(float(accuracy_score(y_values[test_idx], preds)))
        else:
            fold_metrics["regression"].append(float(r2_score(y_values[test_idx], preds)))

    results: Dict = {
        "n_train": int(len(X)),
        "n_test": int(len(X)),
        "test_predictions": predictions,
        "test_actuals": y_values,
        "fold_scores": fold_metrics["classification"] if is_classifier else fold_metrics["regression"],
    }
    if is_classifier:
        results["accuracy"] = float(np.mean(fold_metrics["classification"]))
        results["macro_f1"] = float(f1_score(y_values, predictions, average="macro"))
    else:
        results["r2"] = float(r2_score(y_values, predictions))
        results["rmse"] = float(np.sqrt(mean_squared_error(y_values, predictions)))
        results["mae"] = float(mean_absolute_error(y_values, predictions))
    return results


# --------------------------------------------------------------------------- #
# Regression
# --------------------------------------------------------------------------- #
def run_regression(
    df: pd.DataFrame,
    target: str,
    features: Sequence[str],
    roles: Dict[str, str],
    missing_strategy: str = "median",
    with_random_forest: bool = True,
) -> Dict:
    """Linear regression (+ optional random forest) with 5-fold CV metrics."""
    X, y, notes, _ = build_model_matrix(df, target, features, roles, missing_strategy)
    if X.shape[1] > 300:
        raise ModelingError("Too many features after encoding — deselect some categorical columns.")

    baseline = DummyRegressor(strategy="mean")
    base_results = _cross_validate(baseline, X, y, is_classifier=False)

    lr = LinearRegression()
    lr_results = _cross_validate(lr, X, y, is_classifier=False)
    try:
        lr.fit(X, y)
        lr_coefs = dict(zip(X.columns, lr.coef_))
    except Exception:
        lr_coefs = {}

    rf_results = None
    rf_importance = None
    if with_random_forest:
        rf = RandomForestRegressor(n_estimators=200, max_depth=20, n_jobs=-1, random_state=RANDOM_STATE)
        rf_results = _cross_validate(rf, X, y, is_classifier=False)
        rf.fit(X, y)
        rf_importance = list(zip(X.columns, rf.feature_importances_))

    outcome = {
        "kind": "regression",
        "target": target,
        "n_features": X.shape[1],
        "n_rows": int(len(X)),
        "baseline_rmse": base_results["rmse"],
        "linear": lr_results,
        "linear_coefs": lr_coefs,
        "random_forest": rf_results,
        "rf_importance": rf_importance,
        "notes": notes,
    }
    logger.info("Regression on '%s': R2=%.3f RMSE=%.3f", target, lr_results["r2"], lr_results["rmse"])
    return outcome


# --------------------------------------------------------------------------- #
# Classification
# --------------------------------------------------------------------------- #
def run_classification(
    df: pd.DataFrame,
    target: str,
    features: Sequence[str],
    roles: Dict[str, str],
    missing_strategy: str = "median",
) -> Dict:
    """Random-forest classification with 5-fold CV and a majority-class baseline."""
    X, y, notes, class_map = build_model_matrix(
        df, target, features, roles, missing_strategy, target_is_categorical=True
    )
    if y.nunique() < 2:
        raise ModelingError("The target has only one distinct value — nothing to classify.")
    if y.nunique() > MAX_CLASSES:
        raise ModelingError(
            f"The target has {y.nunique()} classes (limit: {MAX_CLASSES}). "
            "Pick a column with fewer categories, or aggregate rare categories first."
        )
    y = y.astype(int)  # label-encoded codes (0..k-1)

    baseline = DummyClassifier(strategy="most_frequent")
    base_results = _cross_validate(baseline, X, y, is_classifier=True)

    rf = RandomForestClassifier(n_estimators=200, max_depth=20, n_jobs=-1, random_state=RANDOM_STATE)
    rf_results = _cross_validate(rf, X, y, is_classifier=True)
    rf.fit(X, y)

    if class_map is not None:
        labels = [str(class_map[i]) for i in range(len(class_map))]
    else:
        labels = [str(c) for c in sorted(np.unique(y.astype(float)))]
    cm = confusion_matrix(rf_results["test_actuals"], rf_results["test_predictions"])
    outcome = {
        "kind": "classification",
        "target": target,
        "n_features": X.shape[1],
        "n_rows": int(len(X)),
        "n_classes": int(y.nunique()),
        "baseline_accuracy": base_results["accuracy"],
        "accuracy": rf_results["accuracy"],
        "macro_f1": rf_results["macro_f1"],
        "confusion_matrix": cm,
        "class_labels": labels,
        "importance": list(zip(X.columns, rf.feature_importances_)),
        "notes": notes,
    }
    logger.info("Classification on '%s': acc=%.3f f1=%.3f", target, rf_results["accuracy"], rf_results["macro_f1"])
    return outcome


# --------------------------------------------------------------------------- #
# Clustering
# --------------------------------------------------------------------------- #
def run_clustering(
    df: pd.DataFrame,
    features: Sequence[str],
    roles: Dict[str, str],
    k_min: int = 2,
    k_max: int = 8,
) -> Dict:
    """K-means over standardized numeric features, with elbow/silhouette data."""
    numeric = [c for c in features if roles.get(c) in ("numeric", "binary") and c in df.columns]
    if len(numeric) < 2:
        raise ModelingError("Select at least two numeric columns for clustering.")
    frame = df[numeric].apply(pd.to_numeric, errors="coerce").dropna()
    if len(frame) < 100:
        raise ModelingError("Fewer than 100 complete rows — not enough data for clustering.")
    if len(frame) > MAX_MODEL_ROWS:
        frame = frame.sample(MAX_MODEL_ROWS, random_state=RANDOM_STATE)

    X = StandardScaler().fit_transform(frame.to_numpy(dtype=float))
    ks = list(range(k_min, min(k_max, 10) + 1))
    inertias, silhouettes = [], []
    for k in ks:
        km = KMeans(n_clusters=k, n_init=10, random_state=RANDOM_STATE)
        labels = km.fit_predict(X)
        inertias.append(float(km.inertia_))
        silhouettes.append(float(silhouette_score(X, labels, sample_size=min(10_000, len(X)), random_state=RANDOM_STATE)))

    best_k = max(ks, key=lambda k: silhouettes[ks.index(k)])
    km = KMeans(n_clusters=best_k, n_init=10, random_state=RANDOM_STATE)
    labels = km.fit_predict(X)

    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    coords = pca.fit_transform(X)
    explained = float(pca.explained_variance_ratio_.sum())

    cluster_df = frame.copy()
    cluster_df["Cluster"] = [f"Cluster {c}" for c in labels]
    cluster_means = cluster_df.groupby("Cluster")[numeric].mean().round(3)

    sizes = pd.Series(labels).value_counts().sort_index()
    outcome = {
        "kind": "clustering",
        "k_range": ks,
        "inertias": inertias,
        "silhouettes": silhouettes,
        "best_k": best_k,
        "pca_x": coords[:, 0],
        "pca_y": coords[:, 1],
        "labels": [f"Cluster {c}" for c in labels],
        "explained_variance": explained,
        "cluster_sizes": {f"Cluster {i}": int(sizes.get(i, 0)) for i in range(best_k)},
        "cluster_means": cluster_means,
        "n_rows": int(len(frame)),
        "features_used": numeric,
    }
    logger.info("Clustering on %s: best k=%d (silhouette %.3f)", numeric, best_k, max(silhouettes))
    return outcome


# --------------------------------------------------------------------------- #
# Forecasting
# --------------------------------------------------------------------------- #
_SEASONAL_PERIODS = {"D": 7, "W": 52, "M": 12, "Q": 4, "Y": 1}


def run_forecast(
    df: pd.DataFrame,
    date_col: str,
    value_col: str,
    agg: str = "sum",
    freq: str = "M",
    periods: int = 12,
) -> Dict:
    """ETS (Holt–Winters) forecast with a holdout MAPE vs. a naive baseline."""
    series = pd.to_numeric(df[value_col], errors="coerce")
    frame = pd.DataFrame({"date": pd.to_datetime(df[date_col], errors="coerce"), "value": series}).dropna()
    if len(frame) < 20:
        raise ModelingError("At least 20 non-missing (date, value) rows are required for forecasting.")
    ts = frame.set_index("date")["value"].resample(normalize_freq(freq)).agg(agg)
    ts = ts[~ts.index.duplicated()].dropna()
    if len(ts) < 10:
        raise ModelingError(f"After aggregating to '{freq}' frequency only {len(ts)} periods remain — choose a coarser frequency.")

    holdout = min(max(periods, 2), len(ts) // 4)
    train, test = ts.iloc[:-holdout], ts.iloc[-holdout:]
    seasonal = _SEASONAL_PERIODS.get(freq)
    if seasonal and len(train) < 2 * seasonal:
        seasonal = None

    try:
        from statsmodels.tsa.holtwinters import ExponentialSmoothing

        model = ExponentialSmoothing(
            train.astype(float),
            trend="add",
            seasonal="add" if seasonal else None,
            seasonal_periods=seasonal,
            initialization_method="estimated",
        ).fit()
        fitted = model.fittedvalues
        forecast = model.forecast(periods)
        test_forecast = model.forecast(holdout)
        rmse = float(np.sqrt(mean_squared_error(test, test_forecast))) if len(test) else 0.0
        ci_lower = forecast - 1.96 * rmse
        ci_upper = forecast + 1.96 * rmse
        method = "Holt–Winters exponential smoothing (additive trend" + (f", seasonal period {seasonal})" if seasonal else ")")
        holdout_forecast = test_forecast
    except Exception as exc:  # statsmodels can fail on degenerate series
        logger.warning("ETS failed (%s); falling back to linear trend.", exc)
        fitted, forecast, rmse, ci_lower, ci_upper, holdout_forecast = _linear_trend_fallback(train, periods, test, freq)
        method = "Linear trend (fallback — exponential smoothing failed on this series)"

    # Baselines for honesty.
    naive_forecast = pd.Series([train.iloc[-1]] * len(test), index=test.index)
    mape_model = _mape(test, holdout_forecast) if len(test) else None
    mape_naive = _mape(test, naive_forecast) if len(test) else None

    return {
        "kind": "forecast",
        "history": ts,
        "fitted": fitted,
        "forecast": forecast,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "holdout": test,
        "mape_model": mape_model,
        "mape_naive": mape_naive,
        "freq": freq,
        "agg": agg,
        "value_col": value_col,
        "method": method,
        "periods": periods,
    }


def _mape(actual: pd.Series, predicted: pd.Series) -> Optional[float]:
    """Mean absolute percentage error (None when actuals contain zeros)."""
    if actual is None or len(actual) == 0 or (actual == 0).any():
        return None
    return float(np.mean(np.abs((actual - predicted) / actual)) * 100)


def _linear_trend_fallback(
    train: pd.Series, periods: int, test: pd.Series, freq: str
) -> Tuple[pd.Series, pd.Series, float, pd.Series, pd.Series, pd.Series]:
    """Least-squares line fit as a fallback forecaster."""
    t = np.arange(len(train), dtype=float)
    slope, intercept = np.polyfit(t, train.values.astype(float), 1)
    fitted = pd.Series(intercept + slope * t, index=train.index)
    future_t = np.arange(len(train), len(train) + periods, dtype=float)
    future_index = pd.date_range(
        start=train.index[-1], periods=periods + 1, freq=train.index.freq or normalize_freq(freq)
    )[1:]
    forecast = pd.Series(intercept + slope * future_t, index=future_index)
    rmse = float(np.sqrt(mean_squared_error(test, intercept + slope * np.arange(len(train) - len(test), len(train))))) if len(test) else 0.0
    holdout_forecast = pd.Series(intercept + slope * np.arange(len(train) - len(test), len(train)), index=test.index)
    return fitted, forecast, rmse, forecast - 1.96 * rmse, forecast + 1.96 * rmse, holdout_forecast
