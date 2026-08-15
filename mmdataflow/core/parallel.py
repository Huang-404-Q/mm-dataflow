"""Process-pool execution for CPU-bound rule operators.

The rule stage is where the wall-clock goes on a CPU box: image_blur_filter
decodes and convolves every surviving image, image_resolution_filter opens every
file. Both are pure functions of one sample, hold no model state, and share
nothing -- exactly the ``parallel_safe = True`` contract.

Why processes and not threads: the work is numpy/PIL driven from Python-level
loops, so the GIL is the binding constraint. Why not parallelise everything:
model-backed operators would load one copy of the weights per worker, and global
operators (dedup) need cross-sample state by definition. Those stay serial,
which is why ``parallel_safe`` is a per-operator declaration, not a global flag.

Two things here were driven by measurement, not by design instinct (see
``mmdataflow/bench/throughput.py`` and the numbers in the README):

1. **One pool per run, not per operator.** macOS and Windows spawn a fresh
   interpreter per worker; each pool costs ~150-300ms to stand up. Creating one
   per operator paid that four times over a rule stage that only takes a second.
   The pool is created lazily on first use and reused, so each worker also keeps
   its constructed operators warm across stages.

2. **A minimum-work threshold.** Below a few thousand samples the pool costs
   more than it saves -- the first benchmark run showed parallel execution
   *slower* than serial at 349 samples, for every operator. Rather than quietly
   shipping a pessimisation, the executor declines the pool below
   ``MIN_SAMPLES_FOR_POOL`` and says so.

Workers are given the operator's *config spec*, not a pickled operator instance:
rebuilding from (name, params) through the registry is exactly what the pipeline
does, so a worker's operator is identical by construction rather than by
whatever __init__ happened to store on self.
"""
from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional, Tuple

from .context import Context
from .operator import Operator
from .sample import Sample

# Below this, pool startup dominates and serial is faster. Measured on the
# synthetic set; re-run the benchmark if the operator mix changes materially.
MIN_SAMPLES_FOR_POOL = 2000

# Per-worker state, keyed by operator spec. Built once per process rather than
# once per batch, and kept across pipeline stages because the pool outlives them.
_WORKER: Dict[str, Tuple[Operator, Context]] = {}
_WORKER_ENV: Dict[str, str] = {}


def default_workers() -> int:
    """Leave one core for the parent process and the OS."""
    return max(1, (os.cpu_count() or 2) - 1)


def _spec_key(spec: dict) -> str:
    return json.dumps([spec["name"], spec.get("params") or {}], sort_keys=True)


def _init_worker(image_root: str, work_dir: str) -> None:
    # A spawned worker starts from a bare interpreter: it imports only what is
    # needed to unpickle this function, so the operator registry is empty until
    # mmdataflow.ops is imported here. The parent's imports do not carry over on
    # macOS/Windows (spawn), only on fork.
    from .. import ops  # noqa: F401

    _WORKER_ENV["image_root"] = image_root
    _WORKER_ENV["work_dir"] = work_dir


def _get_worker_op(spec: dict) -> Tuple[Operator, Context]:
    key = _spec_key(spec)
    if key not in _WORKER:
        from .registry import build_op

        op = build_op(spec["name"], spec.get("params"))
        # cache_embeddings=False: N workers must not race on one npz file.
        # Parallel-safe operators do not use embeddings anyway.
        ctx = Context(
            work_dir=_WORKER_ENV["work_dir"],
            image_root=_WORKER_ENV["image_root"],
            device="cpu",
            cache_embeddings=False,
        )
        op.setup(ctx)
        _WORKER[key] = (op, ctx)
    return _WORKER[key]


def _process_batch(
    task: Tuple[dict, List[Sample]]
) -> List[Tuple[str, bool, Optional[str], dict, dict]]:
    """Run one batch in a worker and return only the mutated fields.

    Sending deltas instead of whole Samples keeps the return payload small:
    image_path and text are unchanged by filters and would otherwise be pickled
    a second time for nothing.
    """
    spec, batch = task
    op, ctx = _get_worker_op(spec)
    op.process(batch, ctx)
    return [(s.id, s.keep, s.drop_reason, s.scores, s.meta) for s in batch]


class WorkerPool:
    """Lazily-created process pool, shared by every parallel_safe operator in a run.

    Held by the pipeline for the duration of the run and closed in ``close()``.
    Lazy because a config with no parallel_safe operators, or a small input,
    should never pay for a pool at all.
    """

    def __init__(self, num_workers: int, image_root: str, work_dir: str):
        self.num_workers = num_workers
        self.image_root = image_root
        self.work_dir = work_dir
        self._pool: Any = None
        self.failed = False

    def get(self):
        if self._pool is None and not self.failed:
            from concurrent.futures import ProcessPoolExecutor

            self._pool = ProcessPoolExecutor(
                max_workers=self.num_workers,
                initializer=_init_worker,
                initargs=(self.image_root, self.work_dir),
            )
        return self._pool

    def close(self) -> None:
        if self._pool is not None:
            self._pool.shutdown(wait=True)
            self._pool = None


def should_parallelise(op: Operator, n_samples: int, num_workers: int) -> bool:
    return (
        num_workers > 1
        and op.parallel_safe
        and op.batch_size is not None
        and n_samples >= MIN_SAMPLES_FOR_POOL
        and n_samples > op.batch_size  # more than one batch to hand out
    )


def run_parallel(
    op: Operator,
    spec: dict,
    alive: List[Sample],
    ctx: Context,
    num_workers: int,
    pool: Optional[WorkerPool] = None,
) -> bool:
    """Execute ``op`` over ``alive`` in a process pool, mutating samples in place.

    Returns False if the pool was declined or unusable, so the caller falls back
    to serial execution rather than failing the run.
    """
    if not should_parallelise(op, len(alive), num_workers):
        return False

    owned = pool is None
    pool = pool or WorkerPool(num_workers, ctx.image_root, ctx.work_dir)
    batches = [
        (spec, alive[i : i + op.batch_size])
        for i in range(0, len(alive), op.batch_size)
    ]
    by_id = {s.id: s for s in alive}
    try:
        executor = pool.get()
        for result in executor.map(_process_batch, batches):
            for sid, keep, reason, scores, meta in result:
                s = by_id[sid]
                s.scores.update(scores)
                s.meta.update(meta)
                if not keep:
                    s.drop(reason or f"{op.name}:dropped")
    except Exception as e:  # noqa: BLE001
        import warnings

        warnings.warn(f"parallel execution failed ({e}); falling back to serial")
        pool.failed = True
        pool.close()
        return False
    finally:
        if owned:
            pool.close()
    return True
