"""LAION aesthetic predictor: a small MLP head on CLIP ViT-L/14 embeddings.

Reproducing the open-source LAION aesthetic predictor is the cheapest operator
in the pipeline to add, because the expensive half already ran: the MLP takes
the same 768-d image embedding clip_score_filter cached, so scoring the whole
dataset is a handful of matrix multiplies with no image decoding at all.

Weights (sac+logos+ava1-l14-linearMSE.pth, ~4MB):
    https://github.com/christophschuhmann/improved-aesthetic-predictor
    curl -L -o models/aesthetic_l14.pth \\
      https://github.com/christophschuhmann/improved-aesthetic-predictor/raw/main/sac+logos+ava1-l14-linearMSE.pth

Scores run roughly 1-10; LAION-Aesthetics used 5.0+ for its high-quality subset.
"""
from __future__ import annotations

import os
from typing import List, Optional

from ..core.context import Context
from ..core.operator import ScoreFilter
from ..core.registry import register_op
from ..core.sample import Sample

AESTHETIC_SHARED_KEY = "aesthetic_mlp"


def build_mlp(input_dim: int = 768):
    """The predictor's architecture, matched exactly to the released weights."""
    import torch.nn as nn

    return nn.Sequential(
        nn.Linear(input_dim, 1024),
        nn.Dropout(0.2),
        nn.Linear(1024, 128),
        nn.Dropout(0.2),
        nn.Linear(128, 64),
        nn.Dropout(0.1),
        nn.Linear(64, 16),
        nn.Linear(16, 1),
    )


@register_op("aesthetic_score_filter")
class AestheticScoreFilter(ScoreFilter):
    stage = "perception"
    score_key = "aesthetic"
    parallel_safe = False

    def __init__(
        self,
        weights_path: str = "models/aesthetic_l14.pth",
        min_score: float = 4.5,
        model_name: str = "openai/clip-vit-large-patch14",
        batch_size: int = 512,
    ):
        self.weights_path = weights_path
        self.min_score = min_score
        self.model_name = model_name
        self.batch_size = batch_size

    def setup(self, ctx: Context) -> None:
        if AESTHETIC_SHARED_KEY in ctx.shared:
            return
        # Check the file before importing torch: a missing 4MB weight file is a
        # config error and should fail in milliseconds, not after a slow import.
        if not os.path.exists(self.weights_path):
            raise FileNotFoundError(
                f"aesthetic weights not found at {self.weights_path}. Download with:\n"
                f"  curl -L -o {self.weights_path} https://github.com/"
                f"christophschuhmann/improved-aesthetic-predictor/raw/main/"
                f"sac+logos+ava1-l14-linearMSE.pth"
            )
        import torch

        state = torch.load(self.weights_path, map_location="cpu")
        mlp = build_mlp()
        mlp.load_state_dict(state)
        mlp.to(ctx.device).eval()
        ctx.shared[AESTHETIC_SHARED_KEY] = mlp

    def compute_scores(self, batch: List[Sample], ctx: Context) -> List[Optional[float]]:
        import numpy as np
        import torch

        from .clip_score_filter import encode_images

        mlp = ctx.shared[AESTHETIC_SHARED_KEY]

        # Reuse cached embeddings; only encode what is genuinely missing (which
        # is nothing when clip_score_filter ran earlier in the pipeline).
        encode_images(batch, ctx, self.model_name)

        vecs, rows = [], []
        for i, s in enumerate(batch):
            v = ctx.embeddings.get(s.id)
            if v is not None:
                vecs.append(np.asarray(v, dtype="float32"))
                rows.append(i)

        scores: List[Optional[float]] = [None] * len(batch)
        if not vecs:
            return scores
        mat = torch.from_numpy(np.stack(vecs)).to(ctx.device)
        with torch.no_grad():
            out = mlp(mat).squeeze(-1).float().cpu().numpy()
        for i, v in zip(rows, out):
            scores[i] = float(v)
        return scores
