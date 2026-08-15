"""Drop images that are too small or too extreme in aspect ratio.

Cheapest operator in the pipeline (reads headers only, never decodes pixels),
so it runs first to shrink the working set before anything expensive.
"""
from __future__ import annotations

from typing import List

from ..core.context import Context
from ..core.operator import Operator
from ..core.registry import register_op
from ..core.sample import Sample
from ._utils import image_size


@register_op("image_resolution_filter")
class ImageResolutionFilter(Operator):
    stage = "rule"
    parallel_safe = True

    def __init__(self, min_side: int = 224, max_side: int = 10000,
                 max_aspect_ratio: float = 3.0):
        self.min_side = min_side
        self.max_side = max_side
        self.max_aspect_ratio = max_aspect_ratio

    def process(self, batch: List[Sample], ctx: Context) -> List[Sample]:
        for s in batch:
            if not s.keep:
                continue
            size = image_size(s, ctx)
            if size is None:
                s.drop(f"{self.name}:unreadable")
                continue
            w, h = size
            short, long_ = min(w, h), max(w, h)
            ratio = long_ / short if short else float("inf")
            s.scores["min_side"] = float(short)
            s.scores["aspect_ratio"] = round(ratio, 3)
            if short < self.min_side:
                s.drop(f"{self.name}:too_small")
            elif long_ > self.max_side:
                s.drop(f"{self.name}:too_large")
            elif ratio > self.max_aspect_ratio:
                s.drop(f"{self.name}:bad_aspect")
        return batch
