"""Run context: shared state that lets operators avoid recomputing each other's work.

The embedding cache is the core efficiency lever of this project. CLIP image
embeddings are needed by three operators (clip_score_filter,
aesthetic_score_filter, semantic_dedup). Computing them once and sharing via the
cache -- rather than three separate forward passes -- is the single biggest
throughput win measured in bench/.
"""
from __future__ import annotations

import os
from typing import Any, Dict, List, Optional


def auto_device() -> str:
    try:
        import torch
    except ImportError:
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class EmbeddingCache:
    """Keyed float-vector cache with optional npy-backed persistence.

    Keys are sample ids (image embeddings) or ``"txt:" + sample_id`` (text
    embeddings). Persistence lets a re-run skip the expensive CLIP pass entirely.
    """

    def __init__(self, path: Optional[str] = None):
        self.path = path
        self._store: Dict[str, Any] = {}
        self.hits = 0
        self.misses = 0
        if path and os.path.exists(path):
            self.load()

    def get(self, key: str):
        v = self._store.get(key)
        if v is None:
            self.misses += 1
        else:
            self.hits += 1
        return v

    def put(self, key: str, vec) -> None:
        self._store[key] = vec

    def missing(self, keys: List[str]) -> List[str]:
        return [k for k in keys if k not in self._store]

    def __contains__(self, key: str) -> bool:
        return key in self._store

    def __len__(self) -> int:
        return len(self._store)

    def save(self) -> None:
        if not self.path or not self._store:
            return
        import numpy as np

        os.makedirs(os.path.dirname(os.path.abspath(self.path)), exist_ok=True)
        keys = list(self._store)
        mat = np.stack([np.asarray(self._store[k], dtype="float32") for k in keys])
        np.savez(self.path, keys=np.array(keys, dtype=object), mat=mat)

    def load(self) -> None:
        import numpy as np

        data = np.load(self.path, allow_pickle=True)
        for k, v in zip(list(data["keys"]), data["mat"]):
            self._store[str(k)] = v


class Context:
    """Shared per-run state handed to every operator."""

    def __init__(
        self,
        work_dir: str = "outputs/run",
        image_root: str = "",
        device: Optional[str] = None,
        cache_embeddings: bool = True,
    ):
        self.work_dir = work_dir
        self.image_root = image_root
        self.device = device or auto_device()
        os.makedirs(work_dir, exist_ok=True)
        cache_path = (
            os.path.join(work_dir, "embeddings.npz") if cache_embeddings else None
        )
        self.embeddings = EmbeddingCache(cache_path)
        # Free-form slot for operators that need to stash a loaded model or an
        # index and share it (e.g. one CLIP instance across three operators).
        self.shared: Dict[str, Any] = {}

    def resolve_image(self, rel_or_abs: Optional[str]) -> Optional[str]:
        if not rel_or_abs:
            return None
        if os.path.isabs(rel_or_abs) or not self.image_root:
            return rel_or_abs
        return os.path.join(self.image_root, rel_or_abs)

    def close(self) -> None:
        self.embeddings.save()
