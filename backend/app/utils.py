"""Utilities — JSON-safe conversions."""

from __future__ import annotations

import math
from typing import Any
import pandas as pd
import numpy as np

def _clean_val(v: Any):
    if isinstance(v, float):
        if math.isnan(v) or math.isinf(v):
            return None
        return v
    if isinstance(v, (np.floating,)):
        fv = float(v)
        if math.isnan(fv) or math.isinf(fv):
            return None
        return fv
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, pd.Timestamp):
        return str(v)
    return v

def clean_nans(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: clean_nans(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [clean_nans(x) for x in obj]
    if isinstance(obj, float) or isinstance(obj, np.floating):
        fv = float(obj)
        if math.isnan(fv) or math.isinf(fv):
            return None
        return fv
    return obj

def df_to_records(df: pd.DataFrame):
    if df.empty:
        return []
    # replace NaN with None
    df = df.replace({np.nan: None})
    # also replace inf
    df = df.replace({np.inf: None, -np.inf: None})
    records = df.to_dict(orient="records")
    # clean any remaining np types
    cleaned = []
    for rec in records:
        cleaned_rec = {}
        for k, v in rec.items():
            cleaned_rec[k] = _clean_val(v)
        cleaned.append(cleaned_rec)
    return cleaned

def df_to_dict(df: pd.DataFrame):
    if df.empty:
        return {}
    df = df.replace({np.nan: None, np.inf: None, -np.inf: None})
    raw = df.to_dict()
    return clean_nans(raw)
