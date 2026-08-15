"""OCR text-density filter (PaddleOCR).

Screenshots, scanned documents and meme images are mostly rendered text. They
pass every other filter -- sharp, well-proportioned, and CLIP happily matches a
caption describing the words -- but as VLM training data they teach layout, not
grounding. Text area ratio is the signal that separates them from photographs
that merely contain a sign or a label.

PaddleOCR is the deliberate choice here over EasyOCR/Tesseract: this project
targets a Paddle-ecosystem team, and PaddleOCR's PP-OCRv4 detection model is
strong on the dense-text case that matters.

    pip install paddlepaddle paddleocr
"""
from __future__ import annotations

from typing import List, Optional

from ..core.context import Context
from ..core.operator import ScoreFilter
from ..core.registry import register_op
from ..core.sample import Sample
from ._utils import open_image

OCR_SHARED_KEY = "paddle_ocr"


def polygon_area(points) -> float:
    """Shoelace formula -- OCR boxes are quadrilaterals, not axis-aligned."""
    n = len(points)
    area = 0.0
    for i in range(n):
        x1, y1 = points[i]
        x2, y2 = points[(i + 1) % n]
        area += x1 * y2 - x2 * y1
    return abs(area) / 2.0


@register_op("ocr_density_filter")
class OcrDensityFilter(ScoreFilter):
    stage = "perception"
    score_key = "text_area_ratio"
    parallel_safe = False

    def __init__(
        self,
        max_text_ratio: float = 0.25,
        lang: str = "en",
        batch_size: int = 32,
        det_only: bool = True,
    ):
        # Only the max side is bounded: an image with no text at all is fine.
        self.max_score = max_text_ratio
        self.lang = lang
        self.batch_size = batch_size
        # Detection alone gives the area ratio; skipping recognition is roughly
        # 3x faster and this operator never looks at the transcribed strings.
        self.det_only = det_only

    def setup(self, ctx: Context) -> None:
        if OCR_SHARED_KEY in ctx.shared:
            return
        try:
            from paddleocr import PaddleOCR
        except ImportError as e:
            raise ImportError(
                "ocr_density_filter needs PaddleOCR: pip install paddlepaddle paddleocr"
            ) from e
        ctx.shared[OCR_SHARED_KEY] = PaddleOCR(
            use_angle_cls=False, lang=self.lang, show_log=False
        )

    def compute_scores(self, batch: List[Sample], ctx: Context) -> List[Optional[float]]:
        import numpy as np

        ocr = ctx.shared[OCR_SHARED_KEY]
        scores: List[Optional[float]] = []
        for s in batch:
            img = open_image(s, ctx, convert="RGB")
            if img is None:
                scores.append(None)
                continue
            arr = np.asarray(img)
            try:
                result = ocr.ocr(arr, det=True, rec=not self.det_only, cls=False)
            except Exception:
                # A single OCR failure should not abort a 25k run; the sample is
                # simply unscorable and gets dropped with an explicit reason.
                scores.append(None)
                continue
            boxes = result[0] if result else None
            if not boxes:
                scores.append(0.0)
                continue
            total = 0.0
            for item in boxes:
                poly = item[0] if isinstance(item, (list, tuple)) else item
                try:
                    total += polygon_area(poly)
                except Exception:
                    continue
            scores.append(min(1.0, total / float(arr.shape[0] * arr.shape[1])))
        return scores
