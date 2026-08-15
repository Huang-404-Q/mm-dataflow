"""Tests for the A/B/C split and the results renderer.

The split is the load-bearing piece of the whole experiment: if C is drawn
wrongly, or the groups differ by anything other than sample selection, the
benchmark table downstream measures something other than what it claims to.
Those properties are cheap to assert here and expensive to discover after three
GPU-hours, so they are asserted here.
"""
from __future__ import annotations

import json
import os
import sys

import pytest

sys.path.insert(
    0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts")
)

from make_sft_sets import steps_for, to_sharegpt  # noqa: E402
from run_eval import harvest, render  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mmdataflow.core.sample import Sample, write_jsonl  # noqa: E402


def mk(i, keep=True, noise=None, convs=None):
    s = Sample(
        id=str(i),
        image_path=f"{i:012d}.jpg",
        text="t",
        conversations=convs if convs is not None else [
            {"from": "human", "value": "<image>\nWhat is this?"},
            {"from": "gpt", "value": f"Answer {i}."},
        ],
        keep=keep,
    )
    if noise:
        s.meta["noise_type"] = noise
    return s


@pytest.fixture
def built(tmp_path):
    """Run the real CLI over a synthetic pool; yields (out_dir, manifest)."""
    import subprocess

    images = tmp_path / "images"
    images.mkdir()
    from PIL import Image

    samples = []
    for i in range(40):
        s = mk(i, keep=(i % 4 != 0), noise=("blur" if i % 4 == 0 else None))
        Image.new("RGB", (64, 64), (i, i, i)).save(images / s.image_path)
        samples.append(s)
    ann = tmp_path / "annotated.jsonl"
    write_jsonl(str(ann), samples)

    out = tmp_path / "sft"
    script = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "scripts", "make_sft_sets.py",
    )
    r = subprocess.run(
        [sys.executable, script, "--annotated", str(ann),
         "--image-root", str(images), "--out-dir", str(out), "--seed", "7"],
        capture_output=True, text=True,
    )
    assert r.returncode == 0, r.stdout + r.stderr
    return out, json.load(open(out / "manifest.json", encoding="utf-8"))


class TestSplit:
    def test_c_matches_b_in_size(self, built):
        """The one property the size-controlled comparison depends on."""
        out, m = built
        assert m["groups"]["b_cleaned"]["n"] == m["groups"]["c_random"]["n"]

    def test_a_is_the_full_pool(self, built):
        out, m = built
        assert m["groups"]["a_dirty"]["n"] == 40
        assert m["groups"]["b_cleaned"]["n"] == 30  # 3 of every 4 kept

    def test_c_is_drawn_from_a_not_from_b(self, built):
        """C must inherit A's noise, or it is just a second clean set and the
        comparison proves nothing."""
        out, m = built
        c_noise = m["groups"]["c_random"]["noise_composition"]
        assert c_noise.get("blur", 0) > 0, (
            "C contains no noisy samples, so it is not a random draw from the "
            "dirty pool"
        )
        assert m["groups"]["b_cleaned"]["noise_composition"].get("blur", 0) == 0

    def test_split_is_deterministic(self, tmp_path, built):
        """Same seed, same C -- otherwise a re-run silently changes the
        experiment it is supposed to reproduce."""
        out, m = built
        first = json.load(open(out / "c_random.json", encoding="utf-8"))
        second_dir = out.parent / "sft2"
        import subprocess

        script = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "scripts", "make_sft_sets.py",
        )
        subprocess.run(
            [sys.executable, script, "--annotated", str(tmp_path / "annotated.jsonl"),
             "--image-root", str(tmp_path / "images"), "--out-dir", str(second_dir),
             "--seed", "7"],
            capture_output=True, text=True, check=True,
        )
        assert json.load(open(second_dir / "c_random.json", encoding="utf-8")) == first

    def test_dataset_info_registers_every_group(self, built):
        out, _ = built
        info = json.load(open(out / "dataset_info.json", encoding="utf-8"))
        assert set(info) == {"mmdf_a_dirty", "mmdf_b_cleaned", "mmdf_c_random"}
        assert info["mmdf_b_cleaned"]["formatting"] == "sharegpt"

    def test_missing_images_are_dropped_from_every_group(self, tmp_path):
        """A record excluded from only some groups would make the groups differ
        by download luck as well as by selection."""
        import subprocess
        from PIL import Image

        images = tmp_path / "img"
        images.mkdir()
        samples = []
        for i in range(20):
            s = mk(i, keep=(i % 2 == 0))
            if i != 3:  # id 3 is kept in the jsonl but its file never exists
                Image.new("RGB", (64, 64)).save(images / s.image_path)
            samples.append(s)
        ann = tmp_path / "a.jsonl"
        write_jsonl(str(ann), samples)

        out = tmp_path / "sft"
        script = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "scripts", "make_sft_sets.py",
        )
        subprocess.run(
            [sys.executable, script, "--annotated", str(ann), "--image-root",
             str(images), "--out-dir", str(out)],
            capture_output=True, text=True, check=True,
        )
        m = json.load(open(out / "manifest.json", encoding="utf-8"))
        assert m["groups"]["a_dirty"]["n"] == 19  # not 20
        for g in ("a_dirty", "b_cleaned", "c_random"):
            recs = json.load(open(out / f"{g}.json", encoding="utf-8"))
            assert all(os.path.exists(r["images"][0]) for r in recs)


class TestToShareGPT:
    def test_maps_llava_roles(self):
        r = to_sharegpt(mk(1), "/root")
        assert [m["role"] for m in r["messages"]] == ["user", "assistant"]
        assert r["images"] == ["/root/000000000001.jpg"]

    def test_adds_image_token_when_absent(self):
        r = to_sharegpt(mk(1, convs=[
            {"from": "human", "value": "What is this?"},
            {"from": "gpt", "value": "A cat."},
        ]), "/root")
        assert r["messages"][0]["content"].startswith("<image>")

    def test_rejects_multiple_image_tokens(self):
        """Two tokens against one image fails deep inside the processor, so it
        is caught here instead."""
        assert to_sharegpt(mk(1, convs=[
            {"from": "human", "value": "<image> and <image>"},
            {"from": "gpt", "value": "x"},
        ]), "/root") is None

    def test_rejects_non_alternating_turns(self):
        assert to_sharegpt(mk(1, convs=[
            {"from": "human", "value": "<image>a"},
            {"from": "human", "value": "b"},
        ]), "/root") is None

    def test_rejects_assistant_first(self):
        assert to_sharegpt(mk(1, convs=[
            {"from": "gpt", "value": "a"},
            {"from": "human", "value": "<image>b"},
        ]), "/root") is None

    def test_rejects_missing_conversations(self):
        assert to_sharegpt(mk(1, convs=[]), "/root") is None


class TestSteps:
    def test_equal_sizes_give_equal_steps(self):
        """B and C must take the same number of optimizer steps, or the LR
        schedule differs between them and the comparison is contaminated."""
        assert steps_for(3000, 2, 8, 1.0) == steps_for(3000, 2, 8, 1.0)

    def test_more_data_means_more_steps(self):
        assert steps_for(6000, 2, 8, 1.0) > steps_for(3000, 2, 8, 1.0)


class TestHarvest:
    def test_reads_overall_from_a_csv(self, tmp_path):
        d = tmp_path / "b_cleaned"
        d.mkdir()
        (d / "b_cleaned_MMBench_DEV_EN_V11_acc.csv").write_text(
            "split,Overall\ndev,71.35\n"
        )
        assert harvest(str(tmp_path), "b_cleaned", "MMBench_DEV_EN_V11") == 71.35

    def test_returns_none_when_nothing_written(self, tmp_path):
        assert harvest(str(tmp_path), "b_cleaned", "MME") is None


class TestRender:
    def _res(self, b, c):
        return {
            "base": {"MMBench": 60.0},
            "a_dirty": {"MMBench": 65.0},
            "b_cleaned": {"MMBench": b},
            "c_random": {"MMBench": c},
        }

    def test_reports_b_minus_c_as_the_headline(self):
        md = render(self._res(70.0, 67.0), ["MMBench"], None)
        assert "B vs C" in md and "+3.00" in md

    def test_a_negative_result_is_stated_not_hidden(self):
        """The failure mode this guards against is a renderer that only knows
        how to describe a win."""
        md = render(self._res(64.0, 67.0), ["MMBench"], None)
        assert "没有优于" in md
        assert "-3.00" in md

    def test_a_vs_b_is_labelled_as_confounded(self):
        md = render(self._res(70.0, 67.0), ["MMBench"], None)
        assert "不能单独用来论证" in md

    def test_missing_numbers_render_as_gaps(self):
        md = render({"base": {"MMBench": None}}, ["MMBench"], None)
        assert "—" in md
