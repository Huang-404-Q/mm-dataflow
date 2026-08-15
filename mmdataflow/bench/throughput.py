"""Throughput benchmark for the CPU rule stage.

Measures each ``parallel_safe`` operator serially and then at N workers over the
same input, so the speedup column is a measurement rather than a claim. Run it
before quoting any number in the write-up:

    python -m mmdataflow.bench.throughput --config configs/pipeline_dev.yaml \\
        --workers 1,2,4,8 --repeat 3

Notes on reading the output:
  * Speedup is capped by how much of the operator is actually Python-level work.
    image_resolution_filter only reads image headers, so it is I/O bound and will
    scale worse than image_blur_filter, which decodes and convolves.
  * Pool startup (~50-200ms per run on spawn platforms such as macOS) is charged
    to the parallel column on purpose -- it is a real cost the pipeline pays.
"""
from __future__ import annotations

import argparse
import copy
import statistics
import time
from typing import Dict, List

import yaml

from .. import ops  # noqa: F401  (registers operators)
from ..core import parallel as parallel_mod
from ..core.context import Context
from ..core.parallel import WorkerPool, default_workers, run_parallel
from ..core.registry import build_op
from ..core.sample import Sample, read_jsonl


def _fresh(samples: List[Sample]) -> List[Sample]:
    """Deep-copy so each timed repetition starts from identical state."""
    return copy.deepcopy(samples)


def time_op(spec: dict, samples: List[Sample], ctx: Context, workers: int,
            pool=None) -> float:
    op = build_op(spec["name"], spec.get("params"))
    batch = _fresh(samples)
    op.setup(ctx)
    bs = op.batch_size or len(batch)
    t0 = time.perf_counter()
    if workers <= 1 or not run_parallel(op, spec, batch, ctx, workers, pool):
        for i in range(0, len(batch), bs):
            op.process(batch[i : i + bs], ctx)
    elapsed = time.perf_counter() - t0
    op.teardown(ctx)
    return elapsed


def benchmark(config_path: str, worker_counts: List[int], repeat: int,
              limit: int = None, warm_pool: bool = True) -> str:
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # The benchmark measures the executor, so it must not be gated by the
    # production minimum-work threshold -- seeing the pool lose at small n is
    # exactly what the threshold was derived from.
    parallel_mod.MIN_SAMPLES_FOR_POOL = 1

    samples = read_jsonl(cfg["input"], limit=limit)
    ctx = Context(
        work_dir=cfg.get("output_dir", "outputs/bench") + "/bench",
        image_root=cfg.get("image_root", ""),
        device="cpu",
        cache_embeddings=False,
    )

    specs = [s for s in cfg.get("ops", [])
             if build_op(s["name"], s.get("params")).parallel_safe]
    if not specs:
        return "No parallel_safe operators in this config.\n"

    rows: List[Dict] = []
    pools: Dict[int, WorkerPool] = {}
    try:
        for spec in specs:
            timings: Dict[int, float] = {}
            for w in worker_counts:
                if warm_pool and w > 1 and w not in pools:
                    pools[w] = WorkerPool(w, ctx.image_root, ctx.work_dir)
                    # Warm up outside the timer: interpreter startup is a
                    # once-per-run cost in production, not a per-operator one,
                    # so charging it to every cell would misreport steady state.
                    time_op(specs[0], samples[:8], ctx, w, pools[w])
                pool = pools.get(w)
                # Median, not mean: one GC pause should not decide the headline.
                timings[w] = statistics.median(
                    time_op(spec, samples, ctx, w, pool) for _ in range(repeat)
                )
            rows.append({"name": spec["name"], "timings": timings})
    finally:
        for p in pools.values():
            p.close()

    n = len(samples)
    base = worker_counts[0]
    head = " | ".join(f"{w}w (samples/s)" for w in worker_counts)
    out = [
        f"# Throughput: {config_path}",
        "",
        f"{n} samples, {repeat} repetitions per cell (median reported), "
        f"host has {default_workers() + 1} cores.",
        "",
        f"| operator | {head} | speedup ({worker_counts[-1]}w vs {base}w) |",
        "|---|" + "---|" * (len(worker_counts) + 1),
    ]
    tot = {w: 0.0 for w in worker_counts}
    for r in rows:
        cells = " | ".join(f"{n / r['timings'][w]:,.0f}" for w in worker_counts)
        speed = r["timings"][base] / r["timings"][worker_counts[-1]]
        out.append(f"| `{r['name']}` | {cells} | {speed:.2f}x |")
        for w in worker_counts:
            tot[w] += r["timings"][w]
    cells = " | ".join(f"{n / tot[w]:,.0f}" for w in worker_counts)
    out.append(
        f"| **rule stage total** | {cells} | "
        f"**{tot[base] / tot[worker_counts[-1]]:.2f}x** |"
    )
    out.append("")
    return "\n".join(out)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="mmdataflow.bench.throughput")
    p.add_argument("--config", required=True)
    p.add_argument("--workers", default="1,2,4",
                   help="comma-separated worker counts; first is the baseline")
    p.add_argument("--repeat", type=int, default=3)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--cold-pool", action="store_true",
                   help="charge interpreter startup to every cell (worst case)")
    p.add_argument("--out", default=None, help="write the markdown table here")
    args = p.parse_args(argv)

    counts = [int(w) if w != "auto" else default_workers()
              for w in args.workers.split(",")]
    md = benchmark(args.config, counts, args.repeat, args.limit,
                   warm_pool=not args.cold_pool)
    print(md)
    if args.out:
        import os

        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
