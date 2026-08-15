"""The parallel executor must be an optimisation, never a behaviour change.

Every test here checks the same invariant from a different angle: running an
operator through the process pool produces exactly the decisions serial
execution would. A speedup that changes which samples survive is worthless,
because the A/B/C fine-tuning comparison downstream assumes the cleaned set is a
function of the config alone.
"""
from __future__ import annotations

import copy

import pytest

from mmdataflow.core import parallel as parallel_mod
from mmdataflow.core.parallel import default_workers, run_parallel, should_parallelise
from mmdataflow.ops.clip_score_filter import ClipScoreFilter
from mmdataflow.ops.image_blur_filter import ImageBlurFilter
from mmdataflow.ops.image_resolution_filter import ImageResolutionFilter
from mmdataflow.ops.phash_dedup import PHashDedup


def decisions(samples):
    return [(s.id, s.keep, s.drop_reason) for s in samples]


@pytest.fixture
def tiny_pool_ok(monkeypatch):
    """Drop the minimum-work threshold so equivalence is testable on 12 samples.

    The threshold exists because pool startup dominates at small n -- which is a
    performance property, not a correctness one. Correctness must hold at any n.
    """
    monkeypatch.setattr(parallel_mod, "MIN_SAMPLES_FOR_POOL", 1)


@pytest.fixture
def mixed(make_sample):
    """A batch with survivors, a too-small image and a blurred one."""
    shapes = ["circle", "square", "triangle", "bars"]
    out = []
    for i in range(12):
        out.append(
            make_sample(
                f"p{i}",
                size=(120, 120) if i % 5 == 0 else (400, 400),
                blur=9.0 if i % 4 == 1 else 0.0,
                shape=shapes[i % 4],
                text=f"Sample number {i} showing a {shapes[i % 4]} on white.",
            )
        )
    return out


class TestParallelEquivalence:
    @pytest.mark.parametrize(
        "spec",
        [
            {"name": "image_resolution_filter", "params": {"min_side": 224}},
            {"name": "image_blur_filter", "params": {"min_variance": 3.0}},
            {"name": "text_quality_filter", "params": {"min_chars": 10}},
        ],
    )
    def test_same_decisions_as_serial(self, ctx, mixed, spec, tiny_pool_ok):
        from mmdataflow.core.registry import build_op

        serial_samples = copy.deepcopy(mixed)
        op = build_op(spec["name"], spec["params"])
        op.batch_size = 4  # force several batches from a small fixture
        op.setup(ctx)
        for i in range(0, len(serial_samples), 4):
            op.process(serial_samples[i : i + 4], ctx)

        par_samples = copy.deepcopy(mixed)
        par_op = build_op(spec["name"], spec["params"])
        par_op.batch_size = 4
        assert run_parallel(par_op, spec, par_samples, ctx, num_workers=2)

        assert decisions(par_samples) == decisions(serial_samples)

    def test_scores_survive_the_round_trip(self, ctx, mixed, tiny_pool_ok):
        """Only mutated fields are sent back from workers -- scores must be
        among them, or the threshold sweep downstream sees an empty column."""
        spec = {"name": "image_blur_filter", "params": {"min_variance": 0.0}}
        samples = copy.deepcopy(mixed)
        op = ImageBlurFilter(min_variance=0.0)
        op.batch_size = 4
        assert run_parallel(op, spec, samples, ctx, num_workers=2)
        assert all("blur_var" in s.scores for s in samples)

    def test_one_pool_serves_several_operators(self, ctx, mixed, tiny_pool_ok):
        """The pool outlives a single operator, so a run pays interpreter
        startup once instead of once per stage."""
        pool = parallel_mod.WorkerPool(2, ctx.image_root, ctx.work_dir)
        try:
            specs = [
                {"name": "image_resolution_filter", "params": {"min_side": 224}},
                {"name": "text_quality_filter", "params": {"min_chars": 10}},
            ]
            samples = copy.deepcopy(mixed)
            for spec in specs:
                from mmdataflow.core.registry import build_op

                op = build_op(spec["name"], spec["params"])
                op.batch_size = 4
                alive = [s for s in samples if s.keep]
                assert run_parallel(op, spec, alive, ctx, 2, pool)
            executor = pool.get()
        finally:
            pool.close()
        assert executor is not None


class TestParallelGuards:
    def test_refuses_model_backed_operators(self, ctx, mixed, tiny_pool_ok):
        """clip_score_filter would load one copy of the weights per worker."""
        op = ClipScoreFilter()
        assert not op.parallel_safe
        assert run_parallel(op, {"name": "clip_score_filter"}, mixed, ctx, 4) is False

    def test_refuses_global_operators(self, ctx, mixed, tiny_pool_ok):
        """Dedup needs cross-sample state; splitting it would miss pairs that
        land in different batches."""
        op = PHashDedup()
        assert op.batch_size is None
        assert run_parallel(op, {"name": "phash_dedup"}, mixed, ctx, 4) is False

    def test_single_worker_stays_serial(self, ctx, mixed, tiny_pool_ok):
        op = ImageResolutionFilter()
        assert run_parallel(op, {"name": "image_resolution_filter"}, mixed, ctx, 1) is False

    def test_single_batch_does_not_pay_pool_startup(self, ctx, mixed, tiny_pool_ok):
        op = ImageResolutionFilter()
        op.batch_size = 1000  # one batch for the whole fixture
        assert run_parallel(op, {"name": "image_resolution_filter"}, mixed, ctx, 4) is False

    def test_small_input_declines_the_pool(self, ctx, mixed):
        """Without the threshold override, 12 samples must stay serial: the
        benchmark measured the pool losing to serial well past this size."""
        op = ImageResolutionFilter()
        op.batch_size = 4
        assert should_parallelise(op, len(mixed), 4) is False
        assert should_parallelise(op, parallel_mod.MIN_SAMPLES_FOR_POOL, 4) is True

    def test_default_workers_leaves_a_core_free(self):
        import os

        assert default_workers() == max(1, (os.cpu_count() or 2) - 1)
