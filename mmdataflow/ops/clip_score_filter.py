"""CLIP image-text alignment score -- the main defence against mismatched pairs.

This operator owns the CLIP forward pass and writes both image and text
embeddings into ``ctx.embeddings``. aesthetic_score_filter and semantic_dedup
then read from that cache instead of running their own passes, which is the
single largest throughput win in the pipeline (three passes -> one).

Cache keys: ``<sample_id>`` for image embeddings, ``txt:<sample_id>`` for text.
"""
from __future__ import annotations

from typing import List, Optional

from ..core.context import Context
from ..core.operator import ScoreFilter
from ..core.registry import register_op
from ..core.sample import Sample
from ._utils import open_image

CLIP_SHARED_KEY = "clip_model"


def load_clip(ctx: Context, model_name: str):
    """Load CLIP once per run and share it via the context."""
    if CLIP_SHARED_KEY in ctx.shared:
        return ctx.shared[CLIP_SHARED_KEY]
    import torch
    from transformers import CLIPModel, CLIPProcessor

    dtype = torch.float16 if ctx.device == "cuda" else torch.float32
    model = CLIPModel.from_pretrained(model_name, torch_dtype=dtype).to(ctx.device)
    model.eval()
    processor = CLIPProcessor.from_pretrained(model_name)
    ctx.shared[CLIP_SHARED_KEY] = (model, processor, dtype)
    return ctx.shared[CLIP_SHARED_KEY]


def encode_images(
    samples: List[Sample],
    ctx: Context,
    model_name: str,
    batch_size: int = 128,
) -> None:
    """Fill ``ctx.embeddings`` with image embeddings for cache misses only.

    The shared entry point for downstream operators (aesthetic_score_filter,
    semantic_dedup) so none of them needs to know how CLIP is loaded. In the
    standard pipeline order clip_score_filter ran first, every embedding is
    already cached, and this returns without loading a model at all.
    """
    missing = [s for s in samples if s.id not in ctx.embeddings]
    if not missing:
        return
    import torch

    load_clip(ctx, model_name)
    model, processor, dtype = ctx.shared[CLIP_SHARED_KEY]
    for i in range(0, len(missing), batch_size):
        chunk = missing[i : i + batch_size]
        imgs, keep = [], []
        for s in chunk:
            img = open_image(s, ctx, convert="RGB")
            if img is None:
                s.meta["image_error"] = True
                continue
            imgs.append(img)
            keep.append(s)
        if not imgs:
            continue
        inputs = processor(images=imgs, return_tensors="pt").to(ctx.device)
        with torch.no_grad():
            feats = model.get_image_features(
                pixel_values=inputs["pixel_values"].to(dtype)
            )
            feats = feats / feats.norm(dim=-1, keepdim=True)
        for s, v in zip(keep, feats.float().cpu().numpy()):
            ctx.embeddings.put(s.id, v)


@register_op("clip_score_filter")
class ClipScoreFilter(ScoreFilter):
    stage = "perception"
    score_key = "clip_score"
    parallel_safe = False  # holds GPU model state

    def __init__(
        self,
        model_name: str = "openai/clip-vit-large-patch14",
        threshold: float = 0.20,
        batch_size: int = 256,
        max_text_len: int = 77,
    ):
        self.model_name = model_name
        self.min_score = threshold
        self.batch_size = batch_size
        self.max_text_len = max_text_len

    def setup(self, ctx: Context) -> None:
        load_clip(ctx, self.model_name)

    def _encode_images(self, samples: List[Sample], ctx: Context) -> None:
        """Fill the cache with L2-normalised image embeddings for cache misses."""
        import torch

        model, processor, dtype = ctx.shared[CLIP_SHARED_KEY]
        todo, images = [], []
        for s in samples:
            if s.id in ctx.embeddings:
                ctx.embeddings.get(s.id)  # count the hit
                continue
            img = open_image(s, ctx, convert="RGB")
            if img is None:
                s.meta["image_error"] = True
                continue
            todo.append(s)
            images.append(img)
        if not todo:
            return
        inputs = processor(images=images, return_tensors="pt").to(ctx.device)
        with torch.no_grad():
            feats = model.get_image_features(pixel_values=inputs["pixel_values"].to(dtype))
            feats = feats / feats.norm(dim=-1, keepdim=True)
        for s, v in zip(todo, feats.float().cpu().numpy()):
            ctx.embeddings.put(s.id, v)

    def _encode_texts(self, samples: List[Sample], ctx: Context) -> None:
        import torch

        model, processor, _ = ctx.shared[CLIP_SHARED_KEY]
        todo = [s for s in samples if f"txt:{s.id}" not in ctx.embeddings]
        if not todo:
            return
        inputs = processor(
            text=[(s.text or "")[:400] for s in todo],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_text_len,
        ).to(ctx.device)
        with torch.no_grad():
            feats = model.get_text_features(
                input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"]
            )
            feats = feats / feats.norm(dim=-1, keepdim=True)
        for s, v in zip(todo, feats.float().cpu().numpy()):
            ctx.embeddings.put(f"txt:{s.id}", v)

    def compute_scores(self, batch: List[Sample], ctx: Context) -> List[Optional[float]]:
        import numpy as np

        self._encode_images(batch, ctx)
        self._encode_texts(batch, ctx)
        scores: List[Optional[float]] = []
        for s in batch:
            iv = ctx.embeddings.get(s.id)
            tv = ctx.embeddings.get(f"txt:{s.id}")
            if iv is None or tv is None:
                scores.append(None)
            else:
                scores.append(float(np.dot(iv, tv)))
        return scores
