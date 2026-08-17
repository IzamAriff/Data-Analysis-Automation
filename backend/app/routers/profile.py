"""Profile & data dictionary endpoints."""

from __future__ import annotations

from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from fastapi import APIRouter, HTTPException, Depends
from src import profile as profile_mod

from ..config import get_settings, Settings
from ..models.schemas import RoleOverrideRequest, DataDictionaryResponse, ProfileResponse
from ..services.session_store import get_store, SessionStore
from ..utils import df_to_records, clean_nans

router = APIRouter(prefix="/profile", tags=["profile"])

def _store_dep() -> SessionStore:
    return get_store(ttl_seconds=get_settings().session_ttl_seconds)


@router.post("/override", response_model=ProfileResponse)
def override_roles(req: RoleOverrideRequest, store: SessionStore = Depends(_store_dep)):
    sess = store.get(req.dataset_id)
    if not sess or sess.df_prepared is None:
        raise HTTPException(status_code=404, detail="Dataset not prepared yet")
    df = sess.df_prepared
    # validate roles
    allowed = set(profile_mod.ROLE_ORDER)
    for col, role in req.roles.items():
        if col not in df.columns:
            raise HTTPException(status_code=400, detail=f"Column '{col}' not in dataset")
        if role not in allowed:
            raise HTTPException(status_code=400, detail=f"Role '{role}' not allowed; must be one of {allowed}")

    sess.roles = req.roles
    # rebuild derived artefacts
    col_prof = profile_mod.column_profile(df, req.roles)
    summary = profile_mod.profile_summary(df, req.roles)
    structure = profile_mod.dataset_structure_hint(req.roles)
    numeric_desc = profile_mod.numeric_describe(df, [c for c, r in req.roles.items() if r == "numeric"])
    store.update(sess)

    return ProfileResponse(
        dataset_id=sess.dataset_id,
        roles=req.roles,
        summary=clean_nans(summary),
        column_profile=df_to_records(col_prof),
        structure_hint=structure,
        prep_notes=sess.prep_notes + sess.load_notes,
        numeric_describe=df_to_records(numeric_desc) if not numeric_desc.empty else [],
    )


@router.get("/dictionary/{dataset_id}", response_model=DataDictionaryResponse)
def get_dictionary(dataset_id: str, store: SessionStore = Depends(_store_dep)):
    sess = store.get(dataset_id)
    if not sess or sess.df_prepared is None or sess.roles is None:
        raise HTTPException(status_code=404, detail="Dataset not prepared")
    data_dict = profile_mod.build_data_dictionary(sess.df_prepared, sess.roles)
    return DataDictionaryResponse(dataset_id=dataset_id, dictionary=df_to_records(data_dict))


@router.get("/roles/{dataset_id}")
def get_roles(dataset_id: str, store: SessionStore = Depends(_store_dep)):
    sess = store.get(dataset_id)
    if not sess or sess.roles is None:
        raise HTTPException(status_code=404, detail="Dataset roles not found")
    return {"dataset_id": dataset_id, "roles": sess.roles}
