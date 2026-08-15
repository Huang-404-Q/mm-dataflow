"""YAML-driven pipeline runner with per-stage checkpointing."""
from __future__ import annotations

import glob
import os
import time
from typing import List, Optional

import yaml

from ..report.report import RunReport
from .context import Context
from .operator import Operator
from .parallel import WorkerPool, default_workers, run_parallel
from .registry import build_op
from .sample import Sample, read_jsonl, write_jsonl


class Pipeline:
    def __init__(self, config: dict, config_name: str = "pipeline"):
        self.cfg = config
        self.config_name = config.get("name", config_name)
        # The raw specs are kept alongside the built operators: process workers
        # rebuild their operator from (name, params) rather than unpickling an
        # instance, so a worker's operator is identical by construction.
        self.op_specs: List[dict] = list(config.get("ops", []))
        self.ops: List[Operator] = [
            build_op(o["name"], o.get("params")) for o in self.op_specs
        ]

    @classmethod
    def from_yaml(cls, path: str) -> "Pipeline":
        with open(path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        return cls(cfg, os.path.splitext(os.path.basename(path))[0])

    # -- checkpointing ---------------------------------------------------
    def _ckpt_path(self, work_dir: str, idx: int, op_name: str) -> str:
        return os.path.join(work_dir, "checkpoints", f"{idx:02d}_{op_name}.jsonl")

    def _latest_checkpoint(self, work_dir: str):
        files = sorted(glob.glob(os.path.join(work_dir, "checkpoints", "*.jsonl")))
        if not files:
            return None, -1
        last = files[-1]
        return last, int(os.path.basename(last).split("_")[0])

    # -- execution -------------------------------------------------------
    def run(
        self,
        limit: Optional[int] = None,
        resume: bool = False,
        work_dir: Optional[str] = None,
        num_workers: Optional[int] = None,
    ) -> RunReport:
        cfg = self.cfg
        work_dir = work_dir or cfg.get("output_dir", "outputs/run")
        os.makedirs(os.path.join(work_dir, "checkpoints"), exist_ok=True)

        if num_workers is None:
            num_workers = cfg.get("num_workers", 1)
        if num_workers in ("auto", -1):
            num_workers = default_workers()
        num_workers = int(num_workers)

        device = cfg.get("device") or None
        if device == "auto":
            device = None
        ctx = Context(
            work_dir=work_dir,
            image_root=cfg.get("image_root", ""),
            device=device,
            cache_embeddings=cfg.get("cache_embeddings", True),
        )

        start_idx = 0
        samples: List[Sample]
        if resume:
            ckpt, idx = self._latest_checkpoint(work_dir)
            if ckpt:
                samples = read_jsonl(ckpt)
                start_idx = idx + 1
                print(f"[resume] {ckpt} -> skipping first {start_idx} operator(s)")
            else:
                samples = read_jsonl(cfg["input"], limit=limit)
        else:
            samples = read_jsonl(cfg["input"], limit=limit)

        report = RunReport(self.config_name, ctx.device)
        report.n_input = len(samples)
        parallelised: List[str] = []
        # One pool for the whole run: spawning a fresh one per operator paid the
        # ~200ms interpreter startup once per stage, which measured slower than
        # serial on the rule stage. Lazy, so a serial run never creates it.
        pool = WorkerPool(num_workers, ctx.image_root, work_dir)
        print(
            f"[pipeline] {report.n_input} samples, device={ctx.device}, "
            f"workers={num_workers}"
        )

        t_run = time.time()
        for idx, op in enumerate(self.ops):
            if idx < start_idx:
                continue
            alive_before = sum(1 for s in samples if s.keep)
            if alive_before == 0:
                print(f"[{op.name}] no samples left, stopping early")
                break

            op.setup(ctx)
            t0 = time.time()
            if op.batch_size is None:
                # Global operator (dedup): needs to see everything at once.
                op.process([s for s in samples if s.keep], ctx)
            else:
                alive = [s for s in samples if s.keep]
                # A failed pool falls back to serial; re-running a filter over
                # partially-processed samples is safe because Sample.drop keeps
                # the first reason and scores are overwritten, not accumulated.
                if not run_parallel(
                    op, self.op_specs[idx], alive, ctx, num_workers, pool
                ):
                    for i in range(0, len(alive), op.batch_size):
                        op.process(alive[i : i + op.batch_size], ctx)
                else:
                    parallelised.append(op.name)
            elapsed = time.time() - t0
            op.teardown(ctx)

            stat = report.record(op, alive_before, samples, elapsed)
            mark = " [x%d]" % num_workers if op.name in parallelised else ""
            print(
                f"[{idx:02d} {op.name}] {stat.n_in} -> {stat.n_out} "
                f"(drop {stat.drop_rate:.1%}) in {elapsed:.1f}s "
                f"({stat.throughput:.0f}/s){mark}"
            )
            write_jsonl(self._ckpt_path(work_dir, idx, op.name), samples)

        report.total_elapsed = time.time() - t_run
        report.n_output = sum(1 for s in samples if s.keep)
        report.cache_hits = ctx.embeddings.hits
        report.cache_misses = ctx.embeddings.misses
        pool.close()
        ctx.close()

        # annotated.jsonl keeps every sample with its scores and drop reason --
        # this is the input scripts/eval_ops.py needs to compute precision/recall.
        write_jsonl(os.path.join(work_dir, "annotated.jsonl"), samples)
        write_jsonl(
            os.path.join(work_dir, "cleaned.jsonl"), [s for s in samples if s.keep]
        )
        report.save(work_dir)
        print(
            f"[pipeline] {report.n_input} -> {report.n_output} in "
            f"{report.total_elapsed:.1f}s -> {work_dir}/report.md"
        )
        return report
