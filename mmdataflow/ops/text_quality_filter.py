"""Rule-based text quality gate: length, n-gram repetition, junk-character ratio.

Targets the "gibberish text" noise class -- truncation, n-gram loops (a classic
generation artefact) and random-character corruption.
"""
from __future__ import annotations

import re
from collections import Counter
from typing import List

from ..core.context import Context
from ..core.operator import Operator
from ..core.registry import register_op
from ..core.sample import Sample

_WORD_RE = re.compile(r"\w+", re.UNICODE)
# Keep letters, digits, CJK, whitespace and common punctuation; the rest counts
# as junk (mojibake, control characters, stray symbol runs).
_JUNK_RE = re.compile(r"[^\w\s一-鿿.,!?;:'\"()\[\]{}\-–—/%$&@#*+=<>~`|\\]")


def repeated_ngram_ratio(text: str, n: int = 5) -> float:
    """Fraction of n-grams that are duplicates. ~0 for natural text, high for loops."""
    words = _WORD_RE.findall(text.lower())
    if len(words) < n + 1:
        return 0.0
    grams = [tuple(words[i : i + n]) for i in range(len(words) - n + 1)]
    counts = Counter(grams)
    duplicated = sum(c - 1 for c in counts.values() if c > 1)
    return duplicated / len(grams)


def junk_char_ratio(text: str) -> float:
    if not text:
        return 1.0
    return len(_JUNK_RE.findall(text)) / len(text)


@register_op("text_quality_filter")
class TextQualityFilter(Operator):
    stage = "rule"
    score_key = "rep_ratio"
    parallel_safe = True

    def __init__(
        self,
        min_chars: int = 10,
        max_chars: int = 4000,
        max_rep_ratio: float = 0.30,
        max_junk_ratio: float = 0.15,
        ngram: int = 5,
    ):
        self.min_chars = min_chars
        self.max_chars = max_chars
        self.max_rep_ratio = max_rep_ratio
        self.max_junk_ratio = max_junk_ratio
        self.ngram = ngram

    def process(self, batch: List[Sample], ctx: Context) -> List[Sample]:
        for s in batch:
            if not s.keep:
                continue
            text = (s.text or "").strip()
            rep = repeated_ngram_ratio(text, self.ngram)
            junk = junk_char_ratio(text)
            s.scores["rep_ratio"] = round(rep, 4)
            s.scores["junk_ratio"] = round(junk, 4)
            s.scores["n_chars"] = float(len(text))
            if len(text) < self.min_chars:
                s.drop(f"{self.name}:too_short")
            elif len(text) > self.max_chars:
                s.drop(f"{self.name}:too_long")
            elif rep > self.max_rep_ratio:
                s.drop(f"{self.name}:repetitive")
            elif junk > self.max_junk_ratio:
                s.drop(f"{self.name}:junk_chars")
        return batch
