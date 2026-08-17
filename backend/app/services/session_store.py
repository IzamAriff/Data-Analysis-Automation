"""In-memory session store for datasets — thread-safe dict with TTL."""

from __future__ import annotations

import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Dict, Optional

import pandas as pd


@dataclass
class DatasetSession:
    dataset_id: str
    df_raw: pd.DataFrame
    df_prepared: Optional[pd.DataFrame] = None
    roles: Optional[Dict[str, str]] = None
    prep_notes: list = field(default_factory=list)
    load_notes: list = field(default_factory=list)
    name: str = "dataset"
    source: str = "upload"
    created_at: float = field(default_factory=time.time)
    sheets: Optional[Dict[str, pd.DataFrame]] = None

    def touch(self):
        self.created_at = time.time()


class SessionStore:
    def __init__(self, ttl_seconds: int = 3600):
        self._store: Dict[str, DatasetSession] = {}
        self._lock = threading.Lock()
        self._ttl = ttl_seconds

    def create(self, df: pd.DataFrame, name: str, source: str, notes: list, sheets=None) -> DatasetSession:
        dataset_id = uuid.uuid4().hex[:12]
        session = DatasetSession(
            dataset_id=dataset_id,
            df_raw=df,
            name=name,
            source=source,
            load_notes=notes,
            sheets=sheets,
        )
        with self._lock:
            self._store[dataset_id] = session
        return session

    def get(self, dataset_id: str) -> Optional[DatasetSession]:
        with self._lock:
            sess = self._store.get(dataset_id)
            if not sess:
                return None
            # TTL check
            if time.time() - sess.created_at > self._ttl:
                del self._store[dataset_id]
                return None
            sess.touch()
            return sess

    def update(self, session: DatasetSession):
        with self._lock:
            self._store[session.dataset_id] = session

    def delete(self, dataset_id: str):
        with self._lock:
            self._store.pop(dataset_id, None)

    def list_ids(self):
        with self._lock:
            return list(self._store.keys())

    def cleanup(self):
        now = time.time()
        with self._lock:
            expired = [k for k, v in self._store.items() if now - v.created_at > self._ttl]
            for k in expired:
                del self._store[k]


# Singleton
_store_instance: Optional[SessionStore] = None

def get_store(ttl_seconds: int = 3600) -> SessionStore:
    global _store_instance
    if _store_instance is None:
        _store_instance = SessionStore(ttl_seconds=ttl_seconds)
    return _store_instance
