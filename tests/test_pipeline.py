"""End-to-end pipeline tests: config -> run -> report -> operator evaluation."""
from __future__ import annotations

import json
import os
import sys

import pytest
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mmdataflow.core import Pipeline, Sample, read_jsonl, write_jsonl  # noqa: E402
from mmdataflow import ops  # noqa: E402,F401  (registers operators)


def _dataset(tmp_path, make_sample):
    """Six samples: three clean, three noisy with ground-truth labels."""
    samples = [
        make_sample("clean1", text="A red circle on a white background.",
                    shape="circle"),
        make_sample("clean2", text="A blue square on a grey background.",
                    shape="square", color=(40, 80, 200)),
        make_sample("clean3", text="A green triangle on a black background.",
                    shape="triangle", color=(40, 160, 70)),
        make_sample("tiny", size=(80, 80), shape="bars", is_noise=True,
                    noise_type="lowquality_image"),
        make_sample("loop", text=" ".join(["red circle white bg"] * 15),
                    shape="bars", color=(140, 60, 180), is_noise=True,
                    noise_type="gibberish"),
        make_sample("zh", text="这张图片展示了一个非常有趣的场景，值得仔细观察分析。",
                    shape="circle", color=(230, 130, 40), is_noise=True,
                    noise_type="wrong_lang"),
    ]
    path = str(tmp_path / "input.jsonl")
    write_jsonl(path, samples)
    return path


def _config(tmp_path, input_path, img_dir):
    return {
        "name": "test",
        "input": input_path,
        "image_root": img_dir,
        "output_dir": str(tmp_path / "out"),
        "device": "cpu",
        "cache_embeddings": False,
        "ops": [
            {"name": "image_resolution_filter", "params": {"min_side": 224}},
            {"name": "text_quality_filter", "params": {"max_rep_ratio": 0.30}},
            {"name": "lang_id_filter",
             "params": {"keep_langs": ["en"], "backend": "heuristic"}},
            {"name": "phash_dedup", "params": {"hamming_threshold": 3}},
        ],
    }


class TestPipeline:
    def test_run_drops_exactly_the_noisy_samples(self, tmp_path, img_dir, make_sample):
        cfg = _config(tmp_path, _dataset(tmp_path, make_sample), img_dir)
        report = Pipeline(cfg).run()

        assert report.n_input == 6
        assert report.n_output == 3

        out_dir = cfg["output_dir"]
        kept = {s.id for s in read_jsonl(os.path.join(out_dir, "cleaned.jsonl"))}
        assert kept == {"clean1", "clean2", "clean3"}

        annotated = read_jsonl(os.path.join(out_dir, "annotated.jsonl"))
        assert len(annotated) == 6  # dropped samples are retained for evaluation
        dropped = {s.id: s.drop_reason for s in annotated if not s.keep}
        assert dropped["tiny"].startswith("image_resolution_filter")
        assert dropped["loop"].startswith("text_quality_filter")
        assert dropped["zh"].startswith("lang_id_filter")

    def test_report_files_written(self, tmp_path, img_dir, make_sample):
        cfg = _config(tmp_path, _dataset(tmp_path, make_sample), img_dir)
        Pipeline(cfg).run()
        out_dir = cfg["output_dir"]

        with open(os.path.join(out_dir, "report.json"), encoding="utf-8") as f:
            data = json.load(f)
        assert data["n_input"] == 6 and data["n_output"] == 3
        assert len(data["operators"]) == 4
        for op in data["operators"]:
            # Every operator owes the report three numbers.
            assert "drop_rate" in op and "elapsed_s" in op and "throughput_sps" in op

        md = open(os.path.join(out_dir, "report.md"), encoding="utf-8").read()
        assert "Pipeline report" in md and "text_quality_filter" in md
        # Ground-truth labels present -> the noise breakdown section appears.
        assert "Drops by ground-truth noise type" in md

    def test_limit_and_checkpoints(self, tmp_path, img_dir, make_sample):
        cfg = _config(tmp_path, _dataset(tmp_path, make_sample), img_dir)
        report = Pipeline(cfg).run(limit=3)
        assert report.n_input == 3
        ckpts = os.listdir(os.path.join(cfg["output_dir"], "checkpoints"))
        assert len(ckpts) == 4  # one per operator

    def test_resume_skips_completed_stages(self, tmp_path, img_dir, make_sample):
        cfg = _config(tmp_path, _dataset(tmp_path, make_sample), img_dir)
        Pipeline(cfg).run()
        # Resuming from a complete run replays nothing but reproduces the output.
        report = Pipeline(cfg).run(resume=True)
        assert report.n_output == 3
        assert report.stats == []

    def test_unknown_operator_fails_fast(self, tmp_path, img_dir, make_sample):
        cfg = _config(tmp_path, _dataset(tmp_path, make_sample), img_dir)
        cfg["ops"].append({"name": "no_such_op", "params": {}})
        with pytest.raises(KeyError):
            Pipeline(cfg)


class TestSampleIO:
    def test_roundtrip_preserves_scores_and_meta(self, tmp_path):
        s = Sample(id="x", text="hello", scores={"clip_score": 0.31},
                   meta={"is_noise": True, "noise_type": "mismatch"})
        s.drop("some_op:reason")
        path = str(tmp_path / "rt.jsonl")
        write_jsonl(path, [s])
        got = read_jsonl(path)[0]
        assert got.id == "x" and not got.keep
        assert got.drop_reason == "some_op:reason"
        assert got.scores["clip_score"] == 0.31
        assert got.noise_type == "mismatch"

    def test_first_drop_reason_wins(self):
        s = Sample(id="x")
        s.drop("first:a")
        s.drop("second:b")
        assert s.drop_reason == "first:a"
