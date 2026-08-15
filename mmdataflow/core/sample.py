"""Core data record flowing through the pipeline."""
from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional


@dataclass
class Sample:
    """A single image-text record.

    ``keep``/``drop_reason`` are mutated in place by operators. ``meta`` carries
    provenance, including the ``noise_type`` ground-truth label written by
    scripts/inject_noise.py -- that label is what makes operator precision/recall
    computable in scripts/eval_ops.py.
    """

    id: str
    image_path: Optional[str] = None
    text: str = ""
    # Original training-format payload (e.g. LLaVA "conversations"), preserved
    # untouched so a cleaned dataset can be exported back for fine-tuning.
    conversations: Optional[List[Dict[str, Any]]] = None
    keep: bool = True
    drop_reason: Optional[str] = None
    scores: Dict[str, float] = field(default_factory=dict)
    meta: Dict[str, Any] = field(default_factory=dict)

    def drop(self, reason: str) -> None:
        # First operator to reject a sample owns the reason; later ops do not
        # overwrite it, so drop attribution stays stable regardless of op order.
        if self.keep:
            self.keep = False
            self.drop_reason = reason

    @property
    def noise_type(self) -> Optional[str]:
        """Ground-truth noise label, or None for clean samples."""
        return self.meta.get("noise_type")

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Sample":
        known = {f for f in cls.__dataclass_fields__}
        extra = {k: v for k, v in d.items() if k not in known}
        base = {k: v for k, v in d.items() if k in known}
        s = cls(**base)
        if extra:
            s.meta.update(extra)
        return s


def read_jsonl(path: str, limit: Optional[int] = None) -> List[Sample]:
    samples: List[Sample] = []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            if limit is not None and len(samples) >= limit:
                break
            samples.append(Sample.from_dict(json.loads(line)))
    return samples


def write_jsonl(path: str, samples: List[Sample]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(s.to_dict(), ensure_ascii=False) + "\n")
