"""Language identification filter (fastText lid.176).

Backends:
  fasttext  -- the real thing, needs lid.176.bin (~126MB) downloaded
  heuristic -- Unicode-script majority vote; no model file, used in unit tests
               and as an automatic fallback so the pipeline never hard-fails on
               a missing weight file
  auto      -- fasttext if available, else heuristic (default)
"""
from __future__ import annotations

import os
import re
import warnings
from typing import List, Optional

from ..core.context import Context
from ..core.operator import Operator
from ..core.registry import register_op
from ..core.sample import Sample

_SCRIPTS = [
    ("zh", re.compile(r"[一-鿿]")),
    ("ja", re.compile(r"[぀-ヿ]")),
    ("ko", re.compile(r"[가-힯]")),
    ("ru", re.compile(r"[Ѐ-ӿ]")),
    ("ar", re.compile(r"[؀-ۿ]")),
    ("en", re.compile(r"[A-Za-z]")),
]


def heuristic_lang(text: str):
    """Majority Unicode script -> (lang, confidence). Coarse but dependency-free."""
    counts = {name: len(rx.findall(text)) for name, rx in _SCRIPTS}
    total = sum(counts.values())
    if total == 0:
        return None, 0.0
    lang = max(counts, key=lambda k: counts[k])
    return lang, counts[lang] / total


@register_op("lang_id_filter")
class LangIdFilter(Operator):
    stage = "rule"
    score_key = "lang_conf"
    parallel_safe = True

    def __init__(
        self,
        keep_langs: Optional[List[str]] = None,
        min_confidence: float = 0.5,
        backend: str = "auto",
        model_path: str = "models/lid.176.bin",
    ):
        self.keep_langs = keep_langs or ["en"]
        self.min_confidence = min_confidence
        self.backend = backend
        self.model_path = model_path
        self._model = None
        self._active = backend

    def setup(self, ctx: Context) -> None:
        if self.backend == "heuristic":
            self._active = "heuristic"
            return
        try:
            import fasttext

            if not os.path.exists(self.model_path):
                raise FileNotFoundError(self.model_path)
            # fastText prints a deprecation banner on load; keep run logs clean.
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self._model = fasttext.load_model(self.model_path)
            self._active = "fasttext"
        except Exception as e:
            if self.backend == "fasttext":
                raise
            warnings.warn(
                f"lang_id_filter: fastText unavailable ({e}); "
                f"falling back to heuristic backend"
            )
            self._active = "heuristic"

    def _predict(self, text: str):
        if self._active == "fasttext" and self._model is not None:
            # fastText chokes on newlines inside a single prediction.
            labels, probs = self._model.predict(text.replace("\n", " "), k=1)
            return labels[0].replace("__label__", ""), float(probs[0])
        return heuristic_lang(text)

    def process(self, batch: List[Sample], ctx: Context) -> List[Sample]:
        for s in batch:
            if not s.keep:
                continue
            text = (s.text or "").strip()
            if not text:
                s.drop(f"{self.name}:empty")
                continue
            lang, conf = self._predict(text)
            s.scores["lang_conf"] = round(conf, 4)
            s.meta["lang"] = lang
            if lang not in self.keep_langs:
                s.drop(f"{self.name}:wrong_lang")
            elif conf < self.min_confidence:
                s.drop(f"{self.name}:low_confidence")
        return batch
