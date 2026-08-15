#!/usr/bin/env python3
"""Download only the COCO images a prepared dataset actually references.

LLaVA-Instruct-150K draws on COCO: 81,479 unique images, 13GB as a single
train2014 zip (26GB peak, since the zip and its contents coexist during
extraction). A 25k-record sample references roughly 23k unique images -- about
3.5GB at the measured 149KB average. Fetching per-image is the difference
between an overnight download and a coffee break, and it resumes for free: a
re-run only fetches what is still missing.

    python scripts/prepare_data.py --annotations data/raw/llava_instruct_150k.json \\
        --image-root data/images --n 25000 --output data/clean_25k.jsonl \\
        --no-require-image
    python scripts/fetch_images.py --input data/clean_25k.jsonl --image-root data/images

Threads, not processes: this is network-bound, so the GIL is released during
socket waits and threads are the right tool. (Contrast mmdataflow/core/parallel.py,
where the rule operators are CPU-bound and processes are required.)

Behind a slow link, mirror the base URL:

    --base-url https://<your-mirror>/train2017
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Optional, Set, Tuple

# llava_instruct_150k.json stores bare COCO ids ("000000033471.jpg"). That is the
# train2017 naming; train2014 hosts the same bytes under a COCO_train2014_ prefix.
# Since train2014 is a subset of train2017, serving from train2017 covers every
# referenced image and needs no filename rewriting. Verified by HEAD-probing both.
COCO_TRAIN2017 = "http://images.cocodataset.org/train2017"


def referenced_images(path: str) -> List[str]:
    """Unique image paths from a pipeline JSONL, in first-seen order.

    Order matters for resumability: a re-run after an interrupt walks the same
    sequence, so progress is monotonic rather than randomly reshuffled.
    """
    seen: Set[str] = set()
    out: List[str] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rel = json.loads(line).get("image_path")
            if rel and rel not in seen:
                seen.add(rel)
                out.append(rel)
    return out


def is_valid_image(path: str) -> bool:
    """Reject truncated files that a previous interrupt left behind.

    An empty or half-written JPEG still 'exists', so a plain os.path.exists
    check would skip it forever and the pipeline would only find out much later
    when the operator drops it as unreadable.
    """
    try:
        if os.path.getsize(path) < 1024:
            return False
        from PIL import Image

        with Image.open(path) as img:
            img.verify()
        return True
    except Exception:
        return False


def download_one(
    rel: str,
    image_root: str,
    base_url: str,
    retries: int,
    timeout: float,
    verify: bool,
) -> Tuple[str, str, Optional[str]]:
    """Return (rel, status, error). Status is one of skip | ok | fail."""
    dest = os.path.join(image_root, rel)
    if os.path.exists(dest) and (not verify or is_valid_image(dest)):
        return rel, "skip", None

    os.makedirs(os.path.dirname(dest) or ".", exist_ok=True)
    url = f"{base_url.rstrip('/')}/{os.path.basename(rel)}"
    # Write to .part and rename: os.replace is atomic on the same filesystem, so
    # a Ctrl-C can never leave a truncated file that later runs would skip.
    tmp = f"{dest}.part"
    last_err = None
    for attempt in range(retries + 1):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "mm-dataflow/0.1"})
            with urllib.request.urlopen(req, timeout=timeout) as r, open(tmp, "wb") as f:
                while True:
                    chunk = r.read(65536)
                    if not chunk:
                        break
                    f.write(chunk)
            if verify and not is_valid_image(tmp):
                raise OSError("downloaded file is not a readable image")
            os.replace(tmp, dest)
            return rel, "ok", None
        except Exception as e:  # noqa: BLE001
            last_err = f"{type(e).__name__}: {e}"
            if isinstance(e, urllib.error.HTTPError) and e.code == 404:
                break  # a missing image will not appear on retry
            if attempt < retries:
                # Exponential backoff with jitter: a burst of N threads hitting
                # the same rate limit should not retry in lockstep.
                time.sleep((2 ** attempt) * 0.5 + random.random() * 0.3)
    if os.path.exists(tmp):
        os.remove(tmp)
    return rel, "fail", last_err


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--input", required=True,
                   help="JSONL produced by prepare_data.py")
    p.add_argument("--image-root", required=True)
    p.add_argument("--base-url", default=COCO_TRAIN2017,
                   help="images are requested as <base-url>/<basename>; point "
                        "this at a mirror if the CDN is slow")
    p.add_argument("--workers", type=int, default=16,
                   help="concurrent downloads; raise for a fast link, lower if "
                        "the host starts rate-limiting")
    p.add_argument("--retries", type=int, default=3)
    p.add_argument("--timeout", type=float, default=30.0)
    p.add_argument("--limit", type=int, default=None,
                   help="only fetch the first N unique images (dry runs)")
    p.add_argument("--no-verify", dest="verify", action="store_false", default=True,
                   help="skip the decode check (faster, but truncated files "
                        "will be treated as complete)")
    p.add_argument("--failed-list", default=None,
                   help="write still-missing paths here for a targeted retry")
    args = p.parse_args()

    rels = referenced_images(args.input)
    if args.limit:
        rels = rels[: args.limit]
    print(f"[fetch] {len(rels)} unique images referenced by {args.input}")

    counts = {"ok": 0, "skip": 0, "fail": 0}
    failures: List[Tuple[str, str]] = []
    lock = threading.Lock()
    t0 = time.time()

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [
            pool.submit(download_one, rel, args.image_root, args.base_url,
                        args.retries, args.timeout, args.verify)
            for rel in rels
        ]
        for i, fut in enumerate(as_completed(futures), 1):
            rel, status, err = fut.result()
            with lock:
                counts[status] += 1
                if status == "fail":
                    failures.append((rel, err or "unknown"))
            if i % 200 == 0 or i == len(rels):
                done = time.time() - t0
                rate = i / done if done else 0
                eta = (len(rels) - i) / rate if rate else 0
                print(f"  {i}/{len(rels)}  ok={counts['ok']} skip={counts['skip']} "
                      f"fail={counts['fail']}  {rate:.0f} img/s  eta {eta / 60:.1f}min",
                      flush=True)

    print(f"[fetch] done in {(time.time() - t0) / 60:.1f}min: "
          f"{counts['ok']} downloaded, {counts['skip']} already present, "
          f"{counts['fail']} failed")

    if failures:
        # Print a few so a systematic problem (wrong base URL, dead mirror) is
        # obvious immediately rather than after re-reading a log file.
        print("[fetch] first failures:")
        for rel, err in failures[:5]:
            print(f"    {rel}: {err}")
        if args.failed_list:
            with open(args.failed_list, "w", encoding="utf-8") as f:
                for rel, err in failures:
                    f.write(f"{rel}\t{err}\n")
            print(f"[fetch] full list -> {args.failed_list}")
        print("[fetch] re-run the same command to retry only what is missing.")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
