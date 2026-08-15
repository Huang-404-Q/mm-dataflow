"""Unit tests for the P1 operators.

These run without torch, faiss or paddle: semantic_dedup is driven by seeding
``ctx.embeddings`` directly, and ocr_density_filter by injecting a stub detector
into ``ctx.shared``. That is only possible because both operators read their
expensive inputs from shared state rather than computing them inline -- the same
property that makes the pipeline cheap at 25k scale.
"""
from __future__ import annotations

import numpy as np
import pytest

from mmdataflow.ops.aesthetic_score_filter import AestheticScoreFilter, build_mlp
from mmdataflow.ops.ocr_density_filter import OCR_SHARED_KEY, OcrDensityFilter, polygon_area
from mmdataflow.ops.semantic_dedup import SemanticDedup, UnionFind


def unit(*v) -> np.ndarray:
    a = np.asarray(v, dtype="float32")
    return a / np.linalg.norm(a)


def seed(ctx, samples, vectors):
    for s, v in zip(samples, vectors):
        ctx.embeddings.put(s.id, v)


class TestUnionFind:
    def test_transitive_closure(self):
        uf = UnionFind(5)
        uf.union(0, 1)
        uf.union(1, 2)
        assert uf.find(2) == uf.find(0)
        assert uf.find(3) != uf.find(0)

    def test_lowest_index_wins(self):
        """The survivor of a cluster must be its earliest member, so a rerun on
        the same input drops exactly the same samples."""
        uf = UnionFind(4)
        uf.union(3, 1)
        uf.union(2, 3)
        assert uf.find(1) == uf.find(2) == uf.find(3) == 1

    def test_union_order_does_not_change_root(self):
        a, b = UnionFind(4), UnionFind(4)
        a.union(0, 3)
        a.union(1, 3)
        b.union(3, 1)
        b.union(3, 0)
        assert [a.find(i) for i in range(4)] == [b.find(i) for i in range(4)]


class TestSemanticDedup:
    @pytest.fixture
    def op(self):
        # numpy backend explicitly: faiss is optional, and the fallback is the
        # path that must stay correct on machines without it.
        return SemanticDedup(sim_threshold=0.95, backend="numpy", text_aware=False)

    def test_collapses_near_identical_embeddings(self, ctx, make_sample, op):
        a = make_sample("a")
        b = make_sample("b")
        far = make_sample("far")
        seed(ctx, [a, b, far], [unit(1, 0, 0), unit(1, 0.05, 0), unit(0, 1, 0)])

        op.process([a, b, far], ctx)
        assert a.keep and far.keep
        assert not b.keep and b.drop_reason.endswith("semantic_duplicate")
        assert b.meta["duplicate_of"] == "a"

    def test_transitive_cluster_keeps_exactly_one(self, ctx, make_sample, op):
        """A chain a~b~c where a and c are below threshold to each other still
        collapses to one survivor -- that is what union-find buys over pairwise
        deletion."""
        xs = [make_sample(f"c{i}") for i in range(3)]
        seed(ctx, xs, [unit(1, 0, 0), unit(1, 0.2, 0), unit(1, 0.4, 0)])
        assert float(np.dot(unit(1, 0, 0), unit(1, 0.4, 0))) < 0.95

        op.process(xs, ctx)
        assert [s.keep for s in xs] == [True, False, False]
        assert all(s.meta["duplicate_of"] == "c0" for s in xs[1:])

    def test_chunking_does_not_change_the_result(self, ctx, make_sample):
        xs = [make_sample(f"k{i}") for i in range(9)]
        vecs = [unit(1, 0.01 * i, 0) for i in range(9)]
        seed(ctx, xs, vecs)
        # chunk_size smaller than n forces multiple blocks through the fallback.
        SemanticDedup(sim_threshold=0.95, backend="numpy", text_aware=False,
                      chunk_size=2).process(xs, ctx)
        survivors = [s.id for s in xs if s.keep]
        assert survivors == ["k0"]

    def test_text_aware_keeps_same_scene_different_caption(self, ctx, make_sample):
        """Same failure mode as phash_dedup: one image, several conversations."""
        a = make_sample("t_a", text="A red circle centred on a white canvas.")
        b = make_sample("t_b", text="How many distinct shapes appear near the edge?")
        seed(ctx, [a, b], [unit(1, 0, 0), unit(1, 0.01, 0)])

        SemanticDedup(sim_threshold=0.95, backend="numpy", text_aware=True).process(
            [a, b], ctx
        )
        assert a.keep and b.keep

    def test_text_aware_still_drops_matching_captions(self, ctx, make_sample):
        cap = "A red circle centred on a white canvas."
        a = make_sample("m_a", text=cap)
        b = make_sample("m_b", text=cap)
        seed(ctx, [a, b], [unit(1, 0, 0), unit(1, 0.01, 0)])

        SemanticDedup(sim_threshold=0.95, backend="numpy", text_aware=True).process(
            [a, b], ctx
        )
        assert a.keep and not b.keep

    def test_already_dropped_samples_are_left_alone(self, ctx, make_sample, op):
        a = make_sample("d_a")
        b = make_sample("d_b")
        b.drop("upstream:whatever")
        seed(ctx, [a, b], [unit(1, 0, 0), unit(1, 0.01, 0)])

        op.process([a, b], ctx)
        # First operator to reject owns the reason -- dedup must not overwrite it.
        assert b.drop_reason == "upstream:whatever"

    def test_unembeddable_sample_is_reported_not_silently_kept(
        self, ctx, make_sample, monkeypatch
    ):
        a = make_sample("e_a")
        b = make_sample("e_b")
        gone = make_sample("e_gone")
        seed(ctx, [a, b], [unit(1, 0, 0), unit(0, 1, 0)])
        # Stub out encoding: this test is about the drop path for a sample that
        # has no embedding, not about how embeddings get produced.
        monkeypatch.setattr(SemanticDedup, "_ensure_embeddings", lambda *a, **k: None)

        SemanticDedup(sim_threshold=0.95, backend="numpy", text_aware=False).process(
            [a, b, gone], ctx
        )
        assert not gone.keep and gone.drop_reason.endswith("no_embedding")


class TestOcrDensityFilter:
    def test_polygon_area_handles_rotated_boxes(self):
        square = [(0, 0), (10, 0), (10, 10), (0, 10)]
        assert polygon_area(square) == pytest.approx(100.0)
        # Same square rotated 45 degrees: axis-aligned width x height would give
        # ~200, the shoelace formula gives the true area.
        rotated = [(5, 0), (10, 5), (5, 10), (0, 5)]
        assert polygon_area(rotated) == pytest.approx(50.0)

    def test_winding_direction_does_not_matter(self):
        cw = [(0, 0), (0, 10), (10, 10), (10, 0)]
        assert polygon_area(cw) == pytest.approx(100.0)

    def test_drops_text_heavy_image_only(self, ctx, make_sample):
        """Stub detector: one big box for the screenshot, one small one for the
        photo that merely contains a sign."""
        screenshot = make_sample("shot", size=(400, 400))
        photo = make_sample("photo", size=(400, 400))
        pending = [
            [[(0, 0), (400, 0), (400, 200), (0, 200)]],   # 50% of pixels
            [[(0, 0), (40, 0), (40, 40), (0, 40)]],       # 1% of pixels
        ]

        class StubOcr:
            def ocr(self, arr, **kw):
                return [[[poly] for poly in pending.pop(0)]]

        ctx.shared[OCR_SHARED_KEY] = StubOcr()
        op = OcrDensityFilter(max_text_ratio=0.25)
        op.setup(ctx)  # returns early, stub is already in place
        op.process([screenshot, photo], ctx)

        assert screenshot.scores["text_area_ratio"] == pytest.approx(0.5)
        assert not screenshot.keep and screenshot.drop_reason.endswith("above_max")
        assert photo.keep

    def test_multiple_boxes_accumulate(self, ctx, make_sample):
        s = make_sample("many", size=(400, 400))
        polys = [
            [(0, 0), (400, 0), (400, 100), (0, 100)],
            [(0, 300), (400, 300), (400, 400), (0, 400)],
        ]

        class StubOcr:
            def ocr(self, arr, **kw):
                return [[[p] for p in polys]]

        ctx.shared[OCR_SHARED_KEY] = StubOcr()
        OcrDensityFilter(max_text_ratio=0.9).process([s], ctx)
        assert s.scores["text_area_ratio"] == pytest.approx(0.5)

    def test_no_text_scores_zero(self, ctx, make_sample):
        s = make_sample("blank")

        class EmptyOcr:
            def ocr(self, arr, **kw):
                return [None]

        ctx.shared[OCR_SHARED_KEY] = EmptyOcr()
        OcrDensityFilter().process([s], ctx)
        assert s.keep and s.scores["text_area_ratio"] == 0.0

    def test_detector_failure_does_not_abort_the_batch(self, ctx, make_sample):
        bad = make_sample("bad")
        good = make_sample("good")

        class FlakyOcr:
            def ocr(self, arr, **kw):
                if not hasattr(self, "_seen"):
                    self._seen = True
                    raise RuntimeError("cudnn hiccup")
                return [None]

        ctx.shared[OCR_SHARED_KEY] = FlakyOcr()
        OcrDensityFilter().process([bad, good], ctx)
        assert not bad.keep and bad.drop_reason.endswith("unscorable")
        assert good.keep


class TestAestheticScoreFilter:
    def test_missing_weights_fail_fast_with_the_download_command(self, ctx, tmp_path):
        """The file check must precede the torch import, so a config typo fails
        in milliseconds instead of after loading a multi-hundred-MB library."""
        op = AestheticScoreFilter(weights_path=str(tmp_path / "nope.pth"))
        with pytest.raises(FileNotFoundError, match="improved-aesthetic-predictor"):
            op.setup(ctx)

    def test_mlp_shape_matches_released_weights(self):
        pytest.importorskip("torch")
        linears = [m for m in build_mlp(768) if m.__class__.__name__ == "Linear"]
        assert [(m.in_features, m.out_features) for m in linears] == [
            (768, 1024), (1024, 128), (128, 64), (64, 16), (16, 1)
        ]

    def test_scores_come_from_the_shared_embedding_cache(self, ctx, make_sample):
        """The point of this operator: it must never re-encode an image that
        clip_score_filter already put in the cache."""
        torch = pytest.importorskip("torch")

        pretty = make_sample("pretty")
        ugly = make_sample("ugly")
        seed(ctx, [pretty, ugly], [unit(1, 0, 0) * 6.0, unit(1, 0, 0) * 2.0])

        # Stand-in head: returns the vector norm, so the seeded magnitudes above
        # are the expected scores.
        ctx.shared["aesthetic_mlp"] = lambda mat: mat.norm(dim=1, keepdim=True)
        op = AestheticScoreFilter(min_score=4.5)
        op.process([pretty, ugly], ctx)

        assert pretty.scores["aesthetic"] == pytest.approx(6.0, abs=1e-4)
        assert ugly.scores["aesthetic"] == pytest.approx(2.0, abs=1e-4)
        assert pretty.keep
        assert not ugly.keep and ugly.drop_reason.endswith("below_min")
        assert ctx.embeddings.misses == 0  # nothing was re-encoded

