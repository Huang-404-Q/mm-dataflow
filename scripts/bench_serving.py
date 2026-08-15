#!/usr/bin/env python3
"""Measure what a vLLM deployment of the fine-tuned VLM actually delivers.

Start the server first:

    vllm serve outputs/sft/b_cleaned_merged --served-model-name mmdf-b \\
        --max-model-len 4096 --limit-mm-per-prompt image=1

Then sweep concurrency:

    python scripts/bench_serving.py --model mmdf-b --concurrency 1,4,8,16 \\
        --num-prompts 128 --out docs/serving.md

Three things this measures that a single average latency would hide:

  TTFT   time to first token -- what a user perceives as responsiveness, and
         the number that degrades first as concurrency rises
  TPOT   time per output token after the first -- the steady-state stream rate
  tput   output tokens/s across all streams -- what the GPU is actually worth

Reported as p50/p90/p99, not means: a mean TTFT hides the queueing tail that
appears exactly when the server is under the load you care about.

Requests are issued from threads and the payload is real -- actual images from
the training set, base64-inlined the way an OpenAI-compatible client sends them.
Benchmarking with a text-only prompt would measure a different model path
entirely, since the vision encoder and the multimodal cache are most of what
makes VLM serving interesting.
"""
from __future__ import annotations

import argparse
import base64
import json
import mimetypes
import os
import random
import statistics
import sys
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional, Tuple

DEFAULT_PROMPT = "Describe this image in detail."


def load_requests(dataset: str, n: int, seed: int) -> List[Tuple[str, str]]:
    """Return (image_path, prompt) pairs drawn from the SFT set."""
    with open(dataset, "r", encoding="utf-8") as f:
        recs = json.load(f)
    rng = random.Random(seed)
    picked = rng.sample(recs, min(n, len(recs)))
    out = []
    for r in picked:
        img = r["images"][0]
        if not os.path.exists(img):
            continue
        user = next((m["content"] for m in r["messages"] if m["role"] == "user"), "")
        prompt = user.replace("<image>", "").strip() or DEFAULT_PROMPT
        out.append((img, prompt))
    if not out:
        raise SystemExit(f"no usable records in {dataset} (images missing?)")
    while len(out) < n:            # pad by repetition so every concurrency level
        out.extend(out[: n - len(out)])   # sees the same request count
    return out[:n]


def data_url(path: str) -> str:
    mime = mimetypes.guess_type(path)[0] or "image/jpeg"
    with open(path, "rb") as f:
        return f"data:{mime};base64,{base64.b64encode(f.read()).decode()}"


def one_request(base_url: str, model: str, img: str, prompt: str,
                max_tokens: int, timeout: float, api_key: str) -> Dict[str, Any]:
    """Issue one streaming completion; return per-request timings.

    Streaming is not optional here: TTFT is only observable if we read the
    response incrementally. A non-streaming call would collapse TTFT and total
    latency into one number and lose the queueing signal entirely.
    """
    body = json.dumps({
        "model": model,
        "messages": [{
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": data_url(img)}},
                {"type": "text", "text": prompt},
            ],
        }],
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stream": True,
        # Ask vLLM for real token counts rather than inferring them from chunk
        # boundaries, which over-counts when a token spans two chunks.
        "stream_options": {"include_usage": True},
    }).encode()

    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    req = urllib.request.Request(f"{base_url.rstrip('/')}/chat/completions",
                                 data=body, headers=headers)

    start = time.perf_counter()
    ttft: Optional[float] = None
    chunks = 0
    usage_out: Optional[int] = None
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            for raw in r:
                line = raw.decode("utf-8", "replace").strip()
                if not line.startswith("data:"):
                    continue
                payload = line[5:].strip()
                if payload == "[DONE]":
                    break
                try:
                    obj = json.loads(payload)
                except json.JSONDecodeError:
                    continue
                if obj.get("usage"):
                    usage_out = obj["usage"].get("completion_tokens")
                for ch in obj.get("choices", []):
                    piece = (ch.get("delta") or {}).get("content")
                    if piece:
                        if ttft is None:
                            ttft = time.perf_counter() - start
                        chunks += 1
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "error": f"{type(e).__name__}: {e}"}

    total = time.perf_counter() - start
    n_out = usage_out if usage_out else chunks
    if ttft is None or n_out == 0:
        return {"ok": False, "error": "empty response"}
    # TPOT excludes the first token by construction, so it isolates decode speed
    # from prefill + queueing.
    tpot = (total - ttft) / max(1, n_out - 1)
    return {"ok": True, "ttft": ttft, "total": total, "tpot": tpot, "out_tokens": n_out}


def pct(xs: List[float], q: float) -> float:
    if not xs:
        return float("nan")
    s = sorted(xs)
    k = min(len(s) - 1, int(round(q * (len(s) - 1))))
    return s[k]


def run_level(base_url: str, model: str, reqs: List[Tuple[str, str]], conc: int,
              max_tokens: int, timeout: float, api_key: str) -> Dict[str, Any]:
    results: List[Dict[str, Any]] = []
    lock = threading.Lock()

    def work(item):
        r = one_request(base_url, model, item[0], item[1], max_tokens, timeout, api_key)
        with lock:
            results.append(r)

    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=conc) as pool:
        list(pool.map(work, reqs))
    wall = time.perf_counter() - t0

    ok = [r for r in results if r["ok"]]
    failed = len(results) - len(ok)
    if not ok:
        return {"concurrency": conc, "failed": failed, "ok": 0, "wall": wall}
    ttfts = [r["ttft"] for r in ok]
    tpots = [r["tpot"] for r in ok]
    tot_out = sum(r["out_tokens"] for r in ok)
    return {
        "concurrency": conc,
        "ok": len(ok),
        "failed": failed,
        "wall": wall,
        "req_per_s": len(ok) / wall,
        "out_tok_per_s": tot_out / wall,
        "ttft_p50": pct(ttfts, 0.50), "ttft_p90": pct(ttfts, 0.90),
        "ttft_p99": pct(ttfts, 0.99),
        "tpot_p50": statistics.median(tpots),
        "mean_out_tokens": tot_out / len(ok),
    }


def to_markdown(model: str, rows: List[Dict[str, Any]], meta: Dict[str, Any]) -> str:
    lines = [
        "# 服务吞吐基准",
        "",
        f"模型 `{model}`，{meta['num_prompts']} 条真实图文请求"
        f"（采样自 `{os.path.basename(meta['dataset'])}`，seed={meta['seed']}），"
        f"`max_tokens={meta['max_tokens']}`。",
        "",
        "| 并发 | 吞吐 req/s | 输出 tok/s | TTFT p50 | TTFT p90 | TTFT p99 | TPOT p50 | 失败 |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        if not r.get("ok"):
            lines.append(f"| {r['concurrency']} | — | — | — | — | — | — | "
                         f"{r['failed']} (全部失败) |")
            continue
        lines.append(
            f"| {r['concurrency']} | {r['req_per_s']:.2f} | {r['out_tok_per_s']:.0f} | "
            f"{r['ttft_p50']*1000:.0f}ms | {r['ttft_p90']*1000:.0f}ms | "
            f"{r['ttft_p99']*1000:.0f}ms | {r['tpot_p50']*1000:.1f}ms | {r['failed']} |"
        )
    good = [r for r in rows if r.get("ok")]
    if len(good) >= 2:
        lo, hi = good[0], good[-1]
        lines += [
            "",
            f"并发从 {lo['concurrency']} 提到 {hi['concurrency']}，输出吞吐 "
            f"{lo['out_tok_per_s']:.0f} → {hi['out_tok_per_s']:.0f} tok/s "
            f"（{hi['out_tok_per_s']/max(lo['out_tok_per_s'], 1e-9):.1f}x），"
            f"代价是 TTFT p99 从 {lo['ttft_p99']*1000:.0f}ms 涨到 "
            f"{hi['ttft_p99']*1000:.0f}ms。",
            "",
            "> 吞吐和首字延迟是一对权衡，不存在「最优并发」——取决于服务的是"
            "离线批处理还是交互式请求。上表的意义是把这个权衡量化出来。",
        ]
    return "\n".join(lines) + "\n"


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--base-url", default="http://localhost:8000/v1")
    p.add_argument("--model", required=True, help="--served-model-name given to vllm")
    p.add_argument("--dataset", default="data/sft/b_cleaned.json")
    p.add_argument("--num-prompts", type=int, default=128)
    p.add_argument("--concurrency", default="1,4,8,16",
                   help="comma-separated levels, run in ascending order")
    p.add_argument("--max-tokens", type=int, default=128)
    p.add_argument("--timeout", type=float, default=300.0)
    p.add_argument("--api-key", default=os.environ.get("OPENAI_API_KEY", ""))
    p.add_argument("--warmup", type=int, default=4,
                   help="requests issued and discarded first; the very first "
                        "request pays CUDA graph capture and weight paging, "
                        "which would otherwise land in the concurrency-1 p99")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="docs/serving.md")
    args = p.parse_args()

    levels = [int(x) for x in args.concurrency.split(",") if x.strip()]
    reqs = load_requests(args.dataset, args.num_prompts, args.seed)
    print(f"[bench] {len(reqs)} requests, concurrency {levels}, model {args.model}")

    if args.warmup:
        print(f"[bench] warmup x{args.warmup} (discarded)")
        for img, prompt in reqs[: args.warmup]:
            r = one_request(args.base_url, args.model, img, prompt,
                            args.max_tokens, args.timeout, args.api_key)
            if not r["ok"]:
                print(f"[bench] warmup failed: {r['error']}")
                print(f"[bench] is the server up at {args.base_url}?")
                return 1

    rows = []
    for c in levels:
        print(f"[bench] concurrency {c} ...", end=" ", flush=True)
        r = run_level(args.base_url, args.model, reqs, c, args.max_tokens,
                      args.timeout, args.api_key)
        rows.append(r)
        if r.get("ok"):
            print(f"{r['req_per_s']:.2f} req/s, {r['out_tok_per_s']:.0f} tok/s, "
                  f"TTFT p50 {r['ttft_p50']*1000:.0f}ms"
                  + (f", {r['failed']} failed" if r["failed"] else ""))
        else:
            print(f"all {r['failed']} requests failed")

    meta = {"num_prompts": len(reqs), "dataset": args.dataset,
            "seed": args.seed, "max_tokens": args.max_tokens}
    md = to_markdown(args.model, rows, meta)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        f.write(md)
    with open(os.path.splitext(args.out)[0] + ".json", "w", encoding="utf-8") as f:
        json.dump({"model": args.model, "meta": meta, "rows": rows}, f, indent=2)
    print(f"[bench] -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
