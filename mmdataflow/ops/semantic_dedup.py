"""Semantic deduplication over CLIP embeddings (SemDeDup-style).

pHash catches pixel-level copies. This catches the layer above: different
photographs of the same scene, re-encodes that shifted the hash, crops beyond
the Hamming threshold. It reads image embeddings straight out of
``ctx.embeddings`` -- clip_score_filter already computed them, so running this
operator after it costs no extra forward passes.

Backends: faiss when installed, otherwise a chunked numpy matmul. The fallback
keeps the operator testable without faiss and is fine up to ~25k samples; the
chunking is what stops a 25k x 25k similarity matrix from being materialised.

Like phash_dedup, duplicates are keyed on the (image, text) pair -- see that
operator's docstring for why image-only dedup corrupts instruction data.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Set, Tuple

from ..core.context import Context
from ..core.operator import Operator
from ..core.registry import register_op
from ..core.sample import Sample
from .phash_dedup import _tokens, jaccard


class UnionFind:
    def __init__(self, n: int):
        self.parent = list(range(n))

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]  # path halving
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            # Always point at the lower index so the survivor of each cluster is
            # its earliest member -- makes the run order-deterministic.
            lo, hi = (ra, rb) if ra < rb else (rb, ra)
            self.parent[hi] = lo


@register_op("semantic_dedup")
class SemanticDedup(Operator):
    stage = "dedup"
    batch_size = None  # global operator

    def __init__(
        self,
        sim_threshold: float = 0.92,
        model_name: str = "openai/clip-vit-large-patch14",
        text_aware: bool = True,
        text_sim_threshold: float = 0.90,
        chunk_size: int = 1024,
        encode_batch_size: int = 128,
        backend: str = "auto",
    ):
        self.sim_threshold = sim_threshold
        self.model_name = model_name
        self.text_aware = text_aware
        self.text_sim_threshold = text_sim_threshold
        self.chunk_size = chunk_size
        self.encode_batch_size = encode_batch_size
        # auto | faiss | numpy -- pinning the backend makes the two paths
        # separately testable and lets a run prove they agree.
        self.backend = backend

    def _ensure_embeddings(self, samples: List[Sample], ctx: Context) -> None:
        """Encode only what the cache is missing.

        In the standard pipeline order clip_score_filter has already populated
        every embedding, so this is a no-op and the operator is nearly free.
        """
        from .clip_score_filter import encode_images

        encode_images(samples, ctx, self.model_name, self.encode_batch_size)

    def _neighbor_pairs(self, mat) -> List[Tuple[int, int]]:
        """All (i, j) with i < j and cosine similarity >= threshold."""
        if self.backend in ("auto", "faiss"):
            try:
                import faiss

                index = faiss.IndexFlatIP(mat.shape[1])
                index.add(mat)
                # range_search returns every neighbour above the threshold,
                # unlike a fixed top-k which silently truncates large clusters.
                lims, _, idxs = index.range_search(mat, float(self.sim_threshold))
                return [
                    (i, int(j))
                    for i in range(mat.shape[0])
                    for j in idxs[lims[i] : lims[i + 1]]
                    if i < int(j)
                ]
            except ImportError:
                if self.backend == "faiss":
                    raise

        import numpy as np

        pairs: List[Tuple[int, int]] = []
        n = mat.shape[0]
        for start in range(0, n, self.chunk_size):
            block = mat[start : start + self.chunk_size] @ mat.T
            rows, cols = np.where(block >= self.sim_threshold)
            for r, c in zip(rows, cols):
                i, j = start + int(r), int(c)
                if i < j:
                    pairs.append((i, j))
        return pairs

    def process(self, batch: List[Sample], ctx: Context) -> List[Sample]:
        import numpy as np

        alive = [s for s in batch if s.keep]
        if len(alive) < 2:
            return batch
        self._ensure_embeddings(alive, ctx)

        indexed = [s for s in alive if ctx.embeddings.get(s.id) is not None]
        indexed_ids = {s.id for s in indexed}
        for s in alive:
            if s.id not in indexed_ids:
                s.drop(f"{self.name}:no_embedding")
        if len(indexed) < 2:
            return batch

        mat = np.stack(
            [np.asarray(ctx.embeddings.get(s.id), dtype="float32") for s in indexed]
        )
        # Renormalise defensively: inner product only equals cosine on unit
        # vectors, and a cache written by another run may not guarantee that.
        mat /= np.linalg.norm(mat, axis=1, keepdims=True).clip(min=1e-12)

        toks: Dict[int, Set[str]] = (
            {i: _tokens(s.text) for i, s in enumerate(indexed)}
            if self.text_aware
            else {}
        )
        uf = UnionFind(len(indexed))
        for i, j in self._neighbor_pairs(mat):
            if self.text_aware and jaccard(toks[i], toks[j]) < self.text_sim_threshold:
                continue  # same scene, genuinely different caption
            uf.union(i, j)

        for i, s in enumerate(indexed):
            root = uf.find(i)
            if root != i:
                s.meta["duplicate_of"] = indexed[root].id
                s.scores["dedup_cluster"] = float(root)
                s.drop(f"{self.name}:semantic_duplicate")
        return batch
