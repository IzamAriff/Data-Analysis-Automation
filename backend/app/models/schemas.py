"""Pydantic schemas for the fullstack API."""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional, Union
from pydantic import BaseModel, Field, HttpUrl


class HealthResponse(BaseModel):
    status: str = "ok"
    version: str
    service: str


class SampleInfo(BaseModel):
    label: str
    filename: str
    rows_hint: Optional[str] = None


class SampleListResponse(BaseModel):
    samples: List[SampleInfo]


class DatasetUploadResponse(BaseModel):
    dataset_id: str
    name: str
    rows: int
    cols: int
    notes: List[str]
    sheets: Optional[List[str]] = None
    source: str


class DatasetPrepareRequest(BaseModel):
    dataset_id: str
    sheet: Optional[str] = None
    drop_duplicates: bool = True


class ColumnProfileRow(BaseModel):
    Column: str
    Role: str
    Type: str
    Missing: int
    MissingPct: float = Field(alias="Missing %")
    Unique: int
    TopValue: str = Field(alias="Top value")

    class Config:
        populate_by_name = True


class ProfileResponse(BaseModel):
    dataset_id: str
    roles: Dict[str, str]
    summary: Dict[str, Any]
    column_profile: List[Dict[str, Any]]
    structure_hint: str
    prep_notes: List[str]
    numeric_describe: List[Dict[str, Any]] = []


class RoleOverrideRequest(BaseModel):
    dataset_id: str
    roles: Dict[str, str]


class FilterStateSchema(BaseModel):
    date_ranges: Dict[str, List[str]] = {}
    year_ranges: Dict[str, List[int]] = {}
    category_picks: Dict[str, List[str]] = {}
    numeric_ranges: Dict[str, List[float]] = {}
    search_col: Optional[str] = None
    search_text: str = ""


class AnalysisRequest(BaseModel):
    dataset_id: str
    filters: Optional[FilterStateSchema] = None
    metric: Optional[str] = None
    group_col: Optional[str] = None
    date_col: Optional[str] = None


class CorrelationRequest(BaseModel):
    dataset_id: str
    method: Literal["pearson", "spearman"] = "pearson"
    filters: Optional[FilterStateSchema] = None
    max_cols: int = 25


class GroupStatsRequest(BaseModel):
    dataset_id: str
    metric: str
    group_col: str
    filters: Optional[FilterStateSchema] = None
    max_groups: int = 12


class AnovaRequest(BaseModel):
    dataset_id: str
    metric: str
    group_col: str
    filters: Optional[FilterStateSchema] = None


class ChiSquareRequest(BaseModel):
    dataset_id: str
    col_a: str
    col_b: str
    filters: Optional[FilterStateSchema] = None


class OutlierRequest(BaseModel):
    dataset_id: str
    filters: Optional[FilterStateSchema] = None


class TrendRequest(BaseModel):
    dataset_id: str
    date_col: str
    value_col: str
    group_col: Optional[str] = None
    agg: Literal["sum", "mean", "count", "min", "max"] = "sum"
    freq: Literal["D", "W", "M", "Q", "Y"] = "M"
    filters: Optional[FilterStateSchema] = None


class RegressionRequest(BaseModel):
    dataset_id: str
    target: str
    features: List[str]
    missing_strategy: Literal["median", "mean", "drop"] = "median"
    with_random_forest: bool = True
    filters: Optional[FilterStateSchema] = None


class ClassificationRequest(BaseModel):
    dataset_id: str
    target: str
    features: List[str]
    missing_strategy: Literal["median", "mean", "drop"] = "median"
    filters: Optional[FilterStateSchema] = None


class ClusteringRequest(BaseModel):
    dataset_id: str
    features: List[str]
    k_min: int = 2
    k_max: int = 8
    filters: Optional[FilterStateSchema] = None


class ForecastRequest(BaseModel):
    dataset_id: str
    date_col: str
    value_col: str
    agg: Literal["sum", "mean", "count"] = "sum"
    freq: Literal["D", "W", "M", "Q", "Y"] = "M"
    periods: int = 12
    filters: Optional[FilterStateSchema] = None


class PlotRequest(BaseModel):
    dataset_id: str
    chart_type: Literal[
        "trend", "grouped_trend", "histogram", "box", "bar", "scatter", "heatmap", 
        "composition", "missing", "forecast", "elbow", "cluster", "importance", "confusion"
    ]
    params: Dict[str, Any] = {}
    filters: Optional[FilterStateSchema] = None


class UrlLoadRequest(BaseModel):
    url: HttpUrl


class DataDictionaryResponse(BaseModel):
    dataset_id: str
    dictionary: List[Dict[str, Any]]
