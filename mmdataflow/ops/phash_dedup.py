"""Perceptual-hash deduplication (pixel-level near-duplicates).

Naive pairwise comparison is O(n^2) -- 312M comparisons at 25k samples. Instead
the 64-bit pHash is split into 4 bands of 16 bits: two hashes within Hamming
distance <= 3 must agree exactly on at least one band (pigeonhole principle), so
banding gives an exact candidate set at a fraction of the cost. Candidates are
then verified with a true Hamming distance.

First occurrence in input order wins, which keeps runs reproducible.

Duplicates are keyed on the (image, text) PAIR, not the image alone. This is not
an optimisation -- it is a correctness requirement for instruction data.
LLaVA-Instruct-150K attaches several different conversations to the same COCO
image, so image-only dedup silently deletes legitimate training samples. It also
mis-attributes noise: a mismatched or corrupted-text sample shares its image with
the clean original, and image-only dedup drops one of the two at random instead
of leaving the sample for clip_score_filter or text_quality_filter to judge on
its merits. Measured on the synthetic smoke set, adding the text guard moved this
operator from 52.9% to 100% precision.

Known limitation: pHash hashes a *grayscale* DCT, so it is blind to colour.
Two images with identical geometry but different hue collide and, if their texts
also match, one is dropped as a false positive (pinned in tests/test_ops.py).
This is the accepted cost of a cheap first-pass filter -- semantic_dedup (CLIP
embeddings, Week 2) catches what matters here, and running pHash first keeps the
expensive pass small.
"""
from __future__ import annotations

import re
from collections import defaultdict
from typing import List, Optional, Set

from ..core.context import Context
from ..core.operator import Operator
from ..core.registry import register_op
from ..core.sample import Sample
from ._utils import open_image

_HASH_BITS = 64
_N_BANDS = 4
_WORD_RE = re.compile(r"\w+", re.UNICODE)


def _bands(h: int) -> List[int]:
    width = _HASH_BITS // _N_BANDS
    mask = (1 << width) - 1
    return [(h >> (i * width)) & mask for i in range(_N_BANDS)]


def _tokens(text: str) -> Set[str]:
    return set(_WORD_RE.findall((text or "").lower()))


def jaccard(a: Set[str], b: Set[str]) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


@register_op("phash_dedup")
class PHashDedup(Operator):
    stage = "dedup"
    batch_size = None  # global operator: needs the whole dataset at once

    def __init__(self, hamming_threshold: int = 3, hash_size: int = 8,
                 text_aware: bool = True, text_sim_threshold: float = 0.90):
        # threshold 0 == exact duplicates only; 3 also catches mild
        # rescale/recompress variants.
        self.hamming_threshold = hamming_threshold
        self.hash_size = hash_size
        # text_aware=False reproduces naive image-only dedup, kept so the
        # difference can be measured rather than asserted.
        self.text_aware = text_aware
        self.text_sim_threshold = text_sim_threshold

    def process(self, batch: List[Sample], ctx: Context) -> List[Sample]:
        import imagehash

        # (band_idx, band_value) -> [(phash, sample_id, text_tokens)]
        band_index = defaultdict(list)
        for s in batch:
            if not s.keep:
                continue
            img = open_image(s, ctx)
            if img is None:
                s.drop(f"{self.name}:unreadable")
                continue
            h = int(str(imagehash.phash(img, hash_size=self.hash_size)), 16)
            toks = _tokens(s.text) if self.text_aware else set()

            dup_of: Optional[str] = None
            seen = set()
            for i, b in enumerate(_bands(h)):
                for other_h, other_id, other_toks in band_index[(i, b)]:
                    if other_id in seen:
                        continue
                    seen.add(other_id)
                    if bin(h ^ other_h).count("1") > self.hamming_threshold:
                        continue
                    if self.text_aware and jaccard(toks, other_toks) < self.text_sim_threshold:
                        continue  # same picture, different caption -- keep both
                    dup_of = other_id
                    break
                if dup_of:
                    break

            if dup_of:
                s.meta["duplicate_of"] = dup_of
                s.drop(f"{self.name}:duplicate")
            else:
                for i, b in enumerate(_bands(h)):
                    band_index[(i, b)].append((h, s.id, toks))
        return batch
