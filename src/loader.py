"""Data ingestion: uploads, URLs and bundled samples.

The module applies conservative, reversible cleaning at load time:
  1. column-name sanitisation (whitespace, duplicates, control characters)
  2. encoding + delimiter detection for text files
  3. detection & parsing of datetime columns
  4. detection & parsing of numeric values stored as formatted strings
     (e.g. "$1,234.50" or "12,5 %")

Everything is recorded in :class:`DataBundle.notes` so the user can trace
exactly what happened to their data (reproducibility requirement).

Security notes
--------------
* Only http/https URLs are fetched, with a timeout and a hard size cap.
* Uploads are size-capped and only parsed with format-specific readers
  (never pickled/executed). Excel readers do not execute macros.
* Column names are sanitised and user-provided text is never evaluated.
"""

from __future__ import annotations

import hashlib
import io
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import pandas as pd
import requests

logger = logging.getLogger("datapilot.loader")

MAX_FILE_BYTES = 250 * 1024 * 1024  # 250 MB hard cap (uploads & URLs)
URL_TIMEOUT_SECONDS = 30

TEXT_SUFFIXES = {".csv", ".tsv", ".txt", ".dat"}
EXCEL_SUFFIXES = {".xlsx", ".xlsm", ".xls"}
PARQUET_SUFFIXES = {".parquet", ".pq"}
JSON_SUFFIXES = {".json"}
SUPPORTED_SUFFIXES = TEXT_SUFFIXES | EXCEL_SUFFIXES | PARQUET_SUFFIXES | JSON_SUFFIXES

_ENCODINGS_TO_TRY = ("utf-8-sig", "utf-8", "latin-1", "cp1252")

# Path to the bundled demo datasets (kept relative to the repository root).
REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"

_BUNDLED_SAMPLES = {
    "Retail orders — 'Sample Superstore' (9,994 rows)": "sample_superstore.csv",
    "Video game sales — vgsales (16,595 rows)": "sample_video_game_sales.csv",
}


class LoaderError(Exception):
    """Raised when a dataset cannot be read; message is safe to show users."""


@dataclass
class DataBundle:
    """A raw loaded dataset plus provenance metadata."""

    name: str
    df: pd.DataFrame
    source: str = ""
    notes: List[str] = field(default_factory=list)
    sheets: Optional[Dict[str, pd.DataFrame]] = None  # multi-sheet Excel only


# --------------------------------------------------------------------------- #
# Column-name sanitisation
# --------------------------------------------------------------------------- #
def sanitize_column_names(columns: Sequence[str]) -> List[str]:
    """Return clean, unique column names.

    Strips whitespace, collapses runs of inner whitespace, removes newlines
    and de-duplicates repeated names with numeric suffixes (``Name_2``).
    """
    cleaned: List[str] = []
    for raw in columns:
        name = str(raw).strip()
        name = re.sub(r"\s+", " ", name)
        name = re.sub(r"[\r\n\t]+", " ", name)
        base = name or "Unnamed"
        candidate, counter = base, 2
        while candidate in cleaned:
            candidate = f"{base}_{counter}"
            counter += 1
        cleaned.append(candidate)
    return cleaned


# --------------------------------------------------------------------------- #
# Format-specific readers
# --------------------------------------------------------------------------- #
def _read_text(data: bytes) -> pd.DataFrame:
    """Read CSV/TSV/TXT with automatic delimiter and encoding detection."""
    errors: List[str] = []
    for encoding in _ENCODINGS_TO_TRY:
        try:
            df = pd.read_csv(io.BytesIO(data), sep=None, engine="python", encoding=encoding)
            if df.shape[1] >= 2 or df.shape[0] > 0:
                logger.info("Text file parsed: encoding=%s shape=%s", encoding, df.shape)
                return df
            errors.append(f"{encoding}: single-column parse")
        except (UnicodeDecodeError, pd.errors.ParserError) as exc:
            errors.append(f"{encoding}: {exc}")
    raise LoaderError(
        "Could not parse this file as delimited text. Tried encodings "
        f"{', '.join(_ENCODINGS_TO_TRY)}. Details: {'; '.join(errors)}"
    )


def _read_excel(data: bytes) -> Tuple[pd.DataFrame, Optional[Dict[str, pd.DataFrame]]]:
    try:
        sheets = pd.read_excel(io.BytesIO(data), sheet_name=None)
    except Exception as exc:  # xlrd/openpyxl raise assorted errors
        raise LoaderError(f"Could not read this Excel workbook: {exc}") from exc
    if not sheets:
        raise LoaderError("The Excel workbook contains no readable sheets.")
    first_name, first_df = next(iter(sheets.items()))
    if len(sheets) == 1:
        return first_df, None
    return first_df, sheets


def _read_parquet(data: bytes) -> pd.DataFrame:
    try:
        return pd.read_parquet(io.BytesIO(data))
    except Exception as exc:
        raise LoaderError(f"Could not read this Parquet file: {exc}") from exc


def _read_json(data: bytes) -> pd.DataFrame:
    for lines in (False, True):
        try:
            df = pd.read_json(io.BytesIO(data), lines=lines)
            if not df.empty:
                return df
        except ValueError:
            continue
    raise LoaderError("Could not parse this JSON file as records or JSON-lines.")


def _read_any(data: bytes, suffix: str) -> Tuple[pd.DataFrame, Optional[Dict[str, pd.DataFrame]]]:
    suffix = suffix.lower()
    if suffix in TEXT_SUFFIXES:
        return _read_text(data), None
    if suffix in EXCEL_SUFFIXES:
        return _read_excel(data)
    if suffix in PARQUET_SUFFIXES:
        return _read_parquet(data), None
    if suffix in JSON_SUFFIXES:
        return _read_json(data), None
    raise LoaderError(
        f"Unsupported file type '{suffix}'. Supported: {', '.join(sorted(SUPPORTED_SUFFIXES))}"
    )


# --------------------------------------------------------------------------- #
# Public loaders
# --------------------------------------------------------------------------- #
def load_from_bytes(data: bytes, name: str, suffix: str) -> DataBundle:
    """Parse a raw byte stream into a :class:`DataBundle`."""
    if not data:
        raise LoaderError("The file is empty (0 bytes).")
    if len(data) > MAX_FILE_BYTES:
        raise LoaderError(
            f"File is {len(data) / 1e6:,.0f} MB — the limit is {MAX_FILE_BYTES // 1_000_000} MB. "
            "Please aggregate or filter the data before uploading."
        )
    df, sheets = _read_any(data, suffix)
    if df.empty:
        raise LoaderError("The file was read successfully but contains no rows.")
    df.columns = sanitize_column_names(df.columns)
    notes = [f"Read '{name}' ({len(df):,} rows × {df.shape[1]} columns)."]
    if sheets and len(sheets) > 1:
        notes.append(
            f"Workbook contains {len(sheets)} sheets; all were read. "
            "Pick the sheet to analyse in the next step."
        )
    logger.info("Loaded dataset '%s': %d rows x %d cols", name, *df.shape)
    return DataBundle(name=name, df=df, source="upload", notes=notes, sheets=sheets)


def load_from_path(path: str | Path) -> DataBundle:
    """Load a local file (used by the bundled samples and the test suite)."""
    path = Path(path)
    if not path.exists():
        raise LoaderError(f"File not found: {path.name}")
    data = path.read_bytes()
    return load_from_bytes(data, path.name, path.suffix)


def load_from_url(url: str) -> DataBundle:
    """Fetch a dataset over http(s) with size and time limits."""
    parsed = requests.utils.urlparse(url.strip())
    if parsed.scheme not in ("http", "https") or not parsed.netloc:
        raise LoaderError("Only http:// and https:// URLs are supported.")
    display_url = f"{parsed.scheme}://{parsed.netloc}{parsed.path}"  # strip credentials
    try:
        response = requests.get(url, timeout=URL_TIMEOUT_SECONDS, stream=True)
        response.raise_for_status()
    except requests.exceptions.RequestException as exc:
        raise LoaderError(f"Could not download '{display_url}': {exc}") from exc

    chunks: List[bytes] = []
    total = 0
    for chunk in response.iter_content(chunk_size=1 << 20):
        if chunk:
            total += len(chunk)
            if total > MAX_FILE_BYTES:
                raise LoaderError(f"Remote file exceeds the {MAX_FILE_BYTES // 1_000_000} MB limit.")
            chunks.append(chunk)
    data = b"".join(chunks)

    suffix = Path(parsed.path).suffix
    if not suffix or suffix.lower() not in SUPPORTED_SUFFIXES:
        raise LoaderError(
            "Could not determine the file type from the URL. Supported extensions: "
            f"{', '.join(sorted(SUPPORTED_SUFFIXES))}"
        )
    bundle = load_from_bytes(data, Path(parsed.path).name, suffix)
    bundle.source = "url"
    bundle.notes.append(f"Downloaded from {display_url}")
    return bundle


def bundled_sample_names() -> Dict[str, str]:
    """Human-friendly name -> filename for the bundled demo datasets."""
    available = {label: fname for label, fname in _BUNDLED_SAMPLES.items() if (DATA_DIR / fname).exists()}
    if not available:
        raise LoaderError("No bundled sample datasets found in the data/ folder.")
    return available


def load_bundled_sample(label: str) -> DataBundle:
    names = bundled_sample_names()
    if label not in names:
        raise LoaderError(f"Unknown sample dataset '{label}'.")
    path = DATA_DIR / names[label]
    bundle = load_from_path(path)
    bundle.source = "sample"
    bundle.notes.append("Public demo dataset bundled with the app (see README for provenance).")
    return bundle


# --------------------------------------------------------------------------- #
# Lightweight, reversible data preparation
# --------------------------------------------------------------------------- #
def parse_date_columns(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """Detect and parse datetime-like columns. Returns (df, parsed column names)."""
    parsed: List[str] = []
    for col in df.columns:
        series = df[col]
        if pd.api.types.is_datetime64_any_dtype(series):
            parsed.append(col)
            continue
        if pd.api.types.is_numeric_dtype(series) or pd.api.types.is_bool_dtype(series):
            continue
        sample = series.dropna().astype(str).head(2000)
        if sample.empty:
            continue
        # Pure 4-digit years (e.g. "2019") are NOT timestamps.
        if sample.str.fullmatch(r"\d{4}").all():
            continue
        try:
            converted = pd.to_datetime(sample, errors="coerce", format="mixed")
        except (TypeError, ValueError):  # format="mixed" unsupported -> plain parse
            converted = pd.to_datetime(sample, errors="coerce")
        ratio = converted.notna().mean()
        if ratio >= 0.9 and converted.nunique() >= 2:
            df[col] = pd.to_datetime(df[col], errors="coerce", format="mixed")
            parsed.append(col)
            logger.info("Parsed '%s' as datetime (%.0f%% matched).", col, 100 * ratio)
    return df, parsed


def parse_numeric_strings(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """Detect numeric columns stored as formatted strings (e.g. '$1,234.50')."""
    parsed: List[str] = []
    for col in df.columns:
        series = df[col]
        if pd.api.types.is_numeric_dtype(series):
            continue
        sample = series.dropna().head(500)
        if sample.empty or len(sample) < 2:
            continue
        # Strip currency symbols, thousand separators and percent signs.
        cleaned = (
            sample.astype(str)
            .str.replace(r"[$,£€¥\s]", "", regex=True)
            .str.replace("%", "", regex=False)
        )
        try:
            numeric = pd.to_numeric(cleaned, errors="coerce")
        except Exception:  # pragma: no cover - defensive
            continue
        if numeric.notna().mean() >= 0.9:
            full = series.astype(str).str.replace(r"[$,£€¥\s]", "", regex=True).str.replace("%", "", regex=False)
            df[col] = pd.to_numeric(full, errors="coerce")
            parsed.append(col)
            logger.info("Parsed '%s' as numeric (%.0f%% matched).", col, 100 * numeric.notna().mean())
    return df, parsed


def prepare_dataframe(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """Apply all reversible cleaning steps. Returns (df, notes)."""
    notes: List[str] = []
    df = df.copy()
    n_before = len(df)
    df = df.dropna(how="all").drop_duplicates()
    if len(df) < n_before:
        notes.append(
            f"Removed {n_before - len(df):,} fully-empty or duplicated row(s)."
        )
    df.columns = sanitize_column_names(df.columns)
    df, date_cols = parse_date_columns(df)
    df, numeric_cols = parse_numeric_strings(df)
    if date_cols:
        notes.append(f"Parsed {len(date_cols)} column(s) as datetime: {', '.join(date_cols)}.")
    if numeric_cols:
        notes.append(f"Parsed {len(numeric_cols)} column(s) as numeric: {', '.join(numeric_cols)}.")
    return df, notes


def data_hash(df: pd.DataFrame) -> str:
    """Stable fingerprint of a DataFrame's *contents* (used for cache keys)."""
    digest = hashlib.md5(usedforsecurity=False)
    digest.update(str(df.shape).encode())
    sample = df.head(200)
    digest.update(sample.to_csv(index=False).encode(errors="ignore"))
    return digest.hexdigest()
