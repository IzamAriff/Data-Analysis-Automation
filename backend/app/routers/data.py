"""Data ingestion router — upload, url, samples."""

from __future__ import annotations

import io
from pathlib import Path
from typing import List

from fastapi import APIRouter, UploadFile, File, HTTPException, Depends
from fastapi.responses import JSONResponse

import sys
REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))
from src import loader
from src.loader import LoaderError

from ..config import get_settings, Settings
from ..models.schemas import (
    DatasetUploadResponse,
    SampleListResponse,
    SampleInfo,
    UrlLoadRequest,
    DatasetPrepareRequest,
    ProfileResponse,
)
from ..services.session_store import get_store, SessionStore
from ..services.data_service import prepare_dataset
from ..utils import df_to_records, clean_nans

router = APIRouter(prefix="/data", tags=["data"])

def _store_dep() -> SessionStore:
    return get_store(ttl_seconds=get_settings().session_ttl_seconds)


@router.get("/health")
def data_health():
    return {"status": "ok", "service": "data"}


@router.get("/samples", response_model=SampleListResponse)
def list_samples(settings: Settings = Depends(get_settings)):
    try:
        samples = loader.bundled_sample_names()
    except LoaderError as e:
        raise HTTPException(status_code=500, detail=str(e))
    infos = []
    for label, fname in samples.items():
        # Try to give row hint from filename
        hint = None
        if "superstore" in fname.lower():
            hint = "9,994 rows"
        elif "video" in fname.lower() or "vgsales" in fname.lower():
            hint = "16,595 rows"
        infos.append(SampleInfo(label=label, filename=fname, rows_hint=hint))
    return SampleListResponse(samples=infos)


@router.post("/upload", response_model=DatasetUploadResponse)
async def upload_file(
    file: UploadFile = File(...),
    store: SessionStore = Depends(_store_dep),
    settings: Settings = Depends(get_settings),
):
    data = await file.read()
    if len(data) > settings.max_file_bytes:
        raise HTTPException(status_code=413, detail=f"File exceeds {settings.max_file_mb} MB limit")
    suffix = Path(file.filename or "upload.csv").suffix or ".csv"
    try:
        bundle = loader.load_from_bytes(data, file.filename or "upload", suffix)
    except LoaderError as e:
        raise HTTPException(status_code=400, detail=str(e))

    sess = store.create(
        df=bundle.df,
        name=bundle.name,
        source=bundle.source,
        notes=bundle.notes,
        sheets=bundle.sheets,
    )
    sheets = list(bundle.sheets.keys()) if bundle.sheets else None
    return DatasetUploadResponse(
        dataset_id=sess.dataset_id,
        name=sess.name,
        rows=len(sess.df_raw),
        cols=sess.df_raw.shape[1],
        notes=sess.load_notes,
        sheets=sheets,
        source=sess.source,
    )


@router.post("/url", response_model=DatasetUploadResponse)
def load_from_url(
    req: UrlLoadRequest,
    store: SessionStore = Depends(_store_dep),
):
    try:
        bundle = loader.load_from_url(str(req.url))
    except LoaderError as e:
        raise HTTPException(status_code=400, detail=str(e))
    sess = store.create(
        df=bundle.df,
        name=bundle.name,
        source=bundle.source,
        notes=bundle.notes,
        sheets=bundle.sheets,
    )
    sheets = list(bundle.sheets.keys()) if bundle.sheets else None
    return DatasetUploadResponse(
        dataset_id=sess.dataset_id,
        name=sess.name,
        rows=len(sess.df_raw),
        cols=sess.df_raw.shape[1],
        notes=sess.load_notes,
        sheets=sheets,
        source=sess.source,
    )


@router.post("/sample/{label:path}", response_model=DatasetUploadResponse)
def load_sample(
    label: str,
    store: SessionStore = Depends(_store_dep),
):
    # label is url-encoded full label string
    try:
        bundle = loader.load_bundled_sample(label)
    except LoaderError as e:
        # try fuzzy match if exact label not found
        try:
            available = loader.bundled_sample_names()
            # if label is filename, find by filename
            for k, v in available.items():
                if label in v or v in label or label == k:
                    bundle = loader.load_bundled_sample(k)
                    break
            else:
                raise e
        except Exception:
            raise HTTPException(status_code=400, detail=str(e))

    sess = store.create(
        df=bundle.df,
        name=bundle.name,
        source=bundle.source,
        notes=bundle.notes,
        sheets=bundle.sheets,
    )
    sheets = list(bundle.sheets.keys()) if bundle.sheets else None
    return DatasetUploadResponse(
        dataset_id=sess.dataset_id,
        name=sess.name,
        rows=len(sess.df_raw),
        cols=sess.df_raw.shape[1],
        notes=sess.load_notes,
        sheets=sheets,
        source=sess.source,
    )


@router.post("/prepare", response_model=ProfileResponse)
def prepare(
    req: DatasetPrepareRequest,
    store: SessionStore = Depends(_store_dep),
):
    sess = store.get(req.dataset_id)
    if not sess:
        raise HTTPException(status_code=404, detail="Dataset not found or expired")
    df_raw = sess.df_raw
    # handle sheet selection for Excel
    if req.sheet and sess.sheets:
        if req.sheet in sess.sheets:
            df_raw = sess.sheets[req.sheet]
        else:
            raise HTTPException(status_code=400, detail=f"Sheet '{req.sheet}' not found")

    try:
        result = prepare_dataset(df_raw, drop_duplicates=req.drop_duplicates)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

    sess.df_prepared = result["df"]
    sess.roles = result["roles"]
    sess.prep_notes = result["notes"]
    store.update(sess)

    return ProfileResponse(
        dataset_id=sess.dataset_id,
        roles=result["roles"],
        summary=clean_nans(result["summary"]),
        column_profile=df_to_records(result["column_profile"]),
        structure_hint=result["structure"],
        prep_notes=result["notes"] + sess.load_notes,
        numeric_describe=df_to_records(result["numeric_describe"]) if not result["numeric_describe"].empty else [],
    )


@router.get("/{dataset_id}")
def get_dataset_info(dataset_id: str, store: SessionStore = Depends(_store_dep)):
    sess = store.get(dataset_id)
    if not sess:
        raise HTTPException(status_code=404, detail="Dataset not found")
    return {
        "dataset_id": sess.dataset_id,
        "name": sess.name,
        "source": sess.source,
        "rows_raw": len(sess.df_raw),
        "cols_raw": sess.df_raw.shape[1],
        "rows_prepared": len(sess.df_prepared) if sess.df_prepared is not None else None,
        "roles": sess.roles,
        "load_notes": sess.load_notes,
        "prep_notes": sess.prep_notes,
        "sheets": list(sess.sheets.keys()) if sess.sheets else None,
    }


@router.delete("/{dataset_id}")
def delete_dataset(dataset_id: str, store: SessionStore = Depends(_store_dep)):
    sess = store.get(dataset_id)
    if not sess:
        raise HTTPException(status_code=404, detail="Dataset not found")
    store.delete(dataset_id)
    return {"status": "deleted", "dataset_id": dataset_id}
