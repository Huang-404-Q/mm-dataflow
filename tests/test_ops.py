"""Unit tests for the P0 operators: one positive and one negative case each."""
from __future__ import annotations

from mmdataflow.ops.image_blur_filter import ImageBlurFilter
from mmdataflow.ops.image_resolution_filter import ImageResolutionFilter
from mmdataflow.ops.lang_id_filter import LangIdFilter, heuristic_lang
from mmdataflow.ops.phash_dedup import PHashDedup
from mmdataflow.ops.text_quality_filter import (
    TextQualityFilter,
    junk_char_ratio,
    repeated_ngram_ratio,
)


def run(op, samples, ctx):
    op.setup(ctx)
    op.process(samples, ctx)
    op.teardown(ctx)
    return samples


class TestResolutionFilter:
    def test_keeps_normal_and_drops_small(self, ctx, make_sample):
        ok = make_sample("ok", size=(400, 400))
        small = make_sample("small", size=(100, 100))
        run(ImageResolutionFilter(min_side=224), [ok, small], ctx)
        assert ok.keep
        assert not small.keep and small.drop_reason.endswith("too_small")

    def test_drops_extreme_aspect_ratio(self, ctx, make_sample):
        wide = make_sample("wide", size=(2000, 250))
        run(ImageResolutionFilter(min_side=224, max_aspect_ratio=3.0), [wide], ctx)
        assert not wide.keep and wide.drop_reason.endswith("bad_aspect")

    def test_missing_image_is_reported_not_silently_kept(self, ctx, make_sample):
        gone = make_sample("gone", image=False)
        run(ImageResolutionFilter(), [gone], ctx)
        assert not gone.keep and gone.drop_reason.endswith("unreadable")


class TestBlurFilter:
    def test_sharp_scores_above_blurred(self, ctx, make_sample):
        sharp = make_sample("sharp", blur=0.0)
        blurry = make_sample("blurry", blur=8.0)
        run(ImageBlurFilter(min_variance=0.0), [sharp, blurry], ctx)
        assert sharp.scores["blur_var"] > blurry.scores["blur_var"]

    def test_threshold_drops_blurred_only(self, ctx, make_sample):
        sharp = make_sample("sharp2", blur=0.0)
        blurry = make_sample("blurry2", blur=8.0)
        # Midpoint threshold derived from the observed scores, mirroring how
        # eval_ops.py picks a cutoff from the score distribution.
        run(ImageBlurFilter(min_variance=0.0), [sharp, blurry], ctx)
        mid = (sharp.scores["blur_var"] + blurry.scores["blur_var"]) / 2
        s2 = make_sample("sharp3", blur=0.0)
        b2 = make_sample("blurry3", blur=8.0)
        run(ImageBlurFilter(min_variance=mid), [s2, b2], ctx)
        assert s2.keep and not b2.keep


class TestTextQualityFilter:
    def test_repeated_ngram_ratio(self):
        natural = "the quick brown fox jumps over the lazy dog near a quiet river bank"
        looped = " ".join(["the quick brown fox jumps"] * 12)
        assert repeated_ngram_ratio(natural) < 0.1
        assert repeated_ngram_ratio(looped) > 0.8

    def test_junk_char_ratio(self):
        assert junk_char_ratio("A normal English sentence, with punctuation.") < 0.05
        assert junk_char_ratio("§§¤¶‡†□▯╳" * 10) > 0.5

    def test_drops_short_repetitive_and_junk(self, ctx, make_sample):
        good = make_sample("g", text="A red circle sits at the center of a white canvas.")
        short = make_sample("s", text="hi")
        loop = make_sample("l", text=" ".join(["red circle on white bg"] * 15))
        junk = make_sample("j", text="a" * 20 + "§¤¶‡†□▯╳" * 40)
        run(TextQualityFilter(), [good, short, loop, junk], ctx)
        assert good.keep
        assert short.drop_reason.endswith("too_short")
        assert loop.drop_reason.endswith("repetitive")
        assert junk.drop_reason.endswith("junk_chars")


class TestLangIdFilter:
    def test_heuristic_detects_scripts(self):
        assert heuristic_lang("A perfectly ordinary English sentence.")[0] == "en"
        assert heuristic_lang("这张图片展示了一个非常有趣的场景。")[0] == "zh"
        assert heuristic_lang("Эта фотография показывает интересную сцену.")[0] == "ru"

    def test_drops_non_english(self, ctx, make_sample):
        en = make_sample("en", text="A red circle on a white background.")
        zh = make_sample("zh", text="这张图片展示了一个非常有趣的场景，值得仔细观察。")
        run(LangIdFilter(keep_langs=["en"], backend="heuristic"), [en, zh], ctx)
        assert en.keep
        assert not zh.keep and zh.drop_reason.endswith("wrong_lang")


class TestPHashDedup:
    def test_keeps_first_drops_copy(self, ctx, make_sample, img_dir):
        import shutil, os

        a = make_sample("a", shape="circle")
        b = make_sample("b", shape="bars", color=(40, 80, 200))
        # Byte-identical copy of a's image under a new sample id.
        dup = make_sample("a_copy", image=False)
        shutil.copy(os.path.join(img_dir, "a.jpg"), os.path.join(img_dir, "a_copy.jpg"))
        dup.image_path = "a_copy.jpg"

        run(PHashDedup(hamming_threshold=3), [a, b, dup], ctx)
        assert a.keep and b.keep
        assert not dup.keep and dup.drop_reason.endswith("duplicate")
        assert dup.meta["duplicate_of"] == "a"

    def test_catches_rescaled_near_duplicate(self, ctx, make_sample, img_dir):
        import os

        from PIL import Image

        a = make_sample("orig", shape="triangle")
        near = make_sample("near", image=False)
        # Rescale + recompress: different bytes, same content -- exactly the
        # duplicate_near noise class from inject_noise.py.
        img = Image.open(os.path.join(img_dir, "orig.jpg"))
        img.resize((int(img.width * 0.85), int(img.height * 0.85))).save(
            os.path.join(img_dir, "near.jpg"), quality=80
        )
        near.image_path = "near.jpg"

        run(PHashDedup(hamming_threshold=3), [a, near], ctx)
        assert a.keep
        assert not near.keep and near.meta["duplicate_of"] == "orig"

    def test_distinct_images_survive(self, ctx, make_sample):
        xs = [
            make_sample(f"d{i}", shape=sh, color=c)
            for i, (sh, c) in enumerate(
                [("circle", (200, 40, 40)), ("square", (40, 160, 70)),
                 ("triangle", (140, 60, 180)), ("bars", (230, 200, 50))]
            )
        ]
        run(PHashDedup(hamming_threshold=3), xs, ctx)
        assert all(s.keep for s in xs)

    def test_same_image_different_caption_is_not_a_duplicate(self, ctx, make_sample,
                                                             img_dir):
        """The multi-caption case: LLaVA attaches several conversations to one
        COCO image, so image-only dedup would delete real training samples."""
        import os
        import shutil

        a = make_sample("cap_a", shape="square",
                        text="A blue square sits in the upper left of the frame.")
        b = make_sample("cap_b", image=False,
                        text="Two small shapes are scattered near the bottom edge.")
        shutil.copy(os.path.join(img_dir, "cap_a.jpg"),
                    os.path.join(img_dir, "cap_b.jpg"))
        b.image_path = "cap_b.jpg"

        run(PHashDedup(hamming_threshold=3, text_aware=True), [a, b], ctx)
        assert a.keep and b.keep

    def test_text_aware_can_be_disabled(self, ctx, make_sample, img_dir):
        """Image-only mode is retained so the difference stays measurable."""
        import os
        import shutil

        a = make_sample("io_a", shape="triangle", text="A green triangle.")
        b = make_sample("io_b", image=False, text="Completely unrelated caption text.")
        shutil.copy(os.path.join(img_dir, "io_a.jpg"), os.path.join(img_dir, "io_b.jpg"))
        b.image_path = "io_b.jpg"

        run(PHashDedup(hamming_threshold=3, text_aware=False), [a, b], ctx)
        assert a.keep and not b.keep

    def test_is_colour_blind_by_design(self, ctx, make_sample):
        """Documents a real pHash limitation, covered later by semantic_dedup.

        pHash hashes a grayscale DCT, so two images with identical geometry but
        different hue collide. Recording it as a test means the behaviour is a
        known trade-off rather than a surprise during the full run.
        """
        red = make_sample("red_sq", shape="square", color=(200, 40, 40))
        blue = make_sample("blue_sq", shape="square", color=(40, 80, 200))
        run(PHashDedup(hamming_threshold=3), [red, blue], ctx)
        assert red.keep
        assert not blue.keep  # false positive: different image, same structure
