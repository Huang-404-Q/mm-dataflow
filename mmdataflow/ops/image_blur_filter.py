"""Drop blurry images via variance of the Laplacian.

Implemented with a numpy convolution rather than OpenCV: one less heavyweight
dependency, and the 3x3 kernel is the same operator cv2.Laplacian applies.
Sharp images have strong high-frequency content -> high variance.
"""
from __future__ import annotations

from typing import List, Optional

from ..core.context import Context
from ..core.operator import ScoreFilter
from ..core.registry import register_op
from ..core.sample import Sample
from ._utils import open_image

_KERNEL = [[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]]


def laplacian_variance(gray) -> float:
    """gray: 2-D float numpy array in [0, 255]."""
    import numpy as np

    k = np.asarray(_KERNEL, dtype="float32")
    # Manual 3x3 valid convolution: shifted-slice accumulation is ~10x faster
    # than scipy-free nested loops and avoids a scipy dependency.
    h, w = gray.shape
    if h < 3 or w < 3:
        return 0.0
    out = np.zeros((h - 2, w - 2), dtype="float32")
    for dy in range(3):
        for dx in range(3):
            coeff = k[dy, dx]
            if coeff:
                out += coeff * gray[dy : dy + h - 2, dx : dx + w - 2]
    return float(out.var())


@register_op("image_blur_filter")
class ImageBlurFilter(ScoreFilter):
    stage = "rule"
    score_key = "blur_var"
    parallel_safe = True

    def __init__(self, min_variance: float = 100.0, resize_to: int = 512):
        self.min_score = min_variance
        # Downscale first so the score is resolution-independent and cheap.
        self.resize_to = resize_to

    def compute_scores(self, batch: List[Sample], ctx: Context) -> List[Optional[float]]:
        import numpy as np

        scores: List[Optional[float]] = []
        for s in batch:
            img = open_image(s, ctx, convert="L")
            if img is None:
                scores.append(None)
                continue
            if self.resize_to and max(img.size) > self.resize_to:
                ratio = self.resize_to / max(img.size)
                img = img.resize(
                    (max(1, int(img.width * ratio)), max(1, int(img.height * ratio)))
                )
            scores.append(laplacian_variance(np.asarray(img, dtype="float32")))
        return scores
