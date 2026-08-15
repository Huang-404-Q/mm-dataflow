"""Operator base classes.

Every operator is one file under mmdataflow/ops/ and declares three things:
its stage (rule | perception | dedup | mapper), how it wants to be fed
(``batch_size = None`` means "give me the whole dataset", required by dedup),
and whether it is safe to run in a process pool.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Optional

from .context import Context
from .sample import Sample


class Operator(ABC):
    name: str = "operator"
    stage: str = "rule"
    # None => operator receives every surviving sample in one call (global ops
    # such as dedup, which need cross-sample state).
    batch_size: Optional[int] = 256
    # Rule operators touching only stdlib/PIL are fork-safe; model-backed ones
    # are not (each worker would load its own copy of the weights).
    parallel_safe: bool = False

    def setup(self, ctx: Context) -> None:
        """Lazy-load models here, not in __init__, so config parsing stays cheap."""

    def teardown(self, ctx: Context) -> None:
        pass

    @abstractmethod
    def process(self, batch: List[Sample], ctx: Context) -> List[Sample]:
        """Mutate and return the batch. Dropped samples stay in the list with
        ``keep=False`` so the pipeline can report per-operator drop attribution."""

    def describe(self) -> dict:
        skip = {"name", "stage", "batch_size", "parallel_safe"}
        return {k: v for k, v in vars(self).items() if k not in skip and not k.startswith("_")}


class ScoreFilter(Operator):
    """Operator that assigns a numeric score and drops samples outside a range.

    Subclasses implement :meth:`compute_scores`. Keeping the score on the sample
    (even when it passes) is deliberate: threshold selection in
    scripts/eval_ops.py sweeps the recorded distribution instead of re-running
    the expensive scoring pass.
    """

    score_key: str = "score"
    min_score: Optional[float] = None
    max_score: Optional[float] = None

    @abstractmethod
    def compute_scores(self, batch: List[Sample], ctx: Context) -> List[Optional[float]]:
        """Return one score per input sample. None means "could not score"."""

    def process(self, batch: List[Sample], ctx: Context) -> List[Sample]:
        alive = [s for s in batch if s.keep]
        if not alive:
            return batch
        scores = self.compute_scores(alive, ctx)
        for s, sc in zip(alive, scores):
            if sc is None:
                s.drop(f"{self.name}:unscorable")
                continue
            s.scores[self.score_key] = float(sc)
            if self.min_score is not None and sc < self.min_score:
                s.drop(f"{self.name}:below_min")
            elif self.max_score is not None and sc > self.max_score:
                s.drop(f"{self.name}:above_max")
        return batch
