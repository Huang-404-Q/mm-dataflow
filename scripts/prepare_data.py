#!/usr/bin/env python3
"""Convert LLaVA-Instruct-150K into the pipeline's JSONL format and sample it.

Expects the raw annotation JSON (llava_instruct_150k.json) plus a COCO
train2014 image directory. Samples whose image is missing on disk are skipped,
so a partial image download still yields a usable dataset.

    export HF_ENDPOINT=https://hf-mirror.com
    huggingface-cli download liuhaotian/LLaVA-Instruct-150K \
        llava_instruct_150k.json --repo-type dataset --local-dir data/raw

    python scripts/prepare_data.py \
        --annotations data/raw/llava_instruct_150k.json \
        --image-root data/images --n 15000 --output data/clean_15k.jsonl
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from typing import List

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mmdataflow.core.sample import Sample, write_jsonl  # noqa: E402


def extract_text(conversations: List[dict]) -> str:
    """Concatenate the assistant turns.

    Filtering targets the *response* content: the human turn is a templated
    question that carries little quality signal, while a corrupted or mismatched
    answer is exactly what the operators need to see.
    """
    parts = [
        c.get("value", "")
        for c in conversations
        if c.get("from") in ("gpt", "assistant")
    ]
    return "\n".join(p.strip() for p in parts if p.strip())


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--annotations", required=True)
    p.add_argument("--image-root", required=True,
                   help="directory holding COCO train2014 images")
    p.add_argument("--image-prefix", default="",
                   help="prefix prepended to each image filename, e.g. train2014/")
    p.add_argument("--n", type=int, default=15000)
    p.add_argument("--output", required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--require-image", action="store_true", default=True,
                   help="skip records whose image file is absent (default on)")
    args = p.parse_args()

    with open(args.annotations, "r", encoding="utf-8") as f:
        raw = json.load(f)
    print(f"[load] {len(raw)} raw records")

    rng = random.Random(args.seed)
    rng.shuffle(raw)

    samples, missing, empty = [], 0, 0
    for rec in raw:
        if len(samples) >= args.n:
            break
        img_rel = os.path.join(args.image_prefix, rec.get("image", ""))
        if not rec.get("image"):
            continue
        if args.require_image and not os.path.exists(
            os.path.join(args.image_root, img_rel)
        ):
            missing += 1
            continue
        text = extract_text(rec.get("conversations", []))
        if not text:
            empty += 1
            continue
        samples.append(
            Sample(
                id=str(rec.get("id")),
                image_path=img_rel,
                text=text,
                conversations=rec.get("conversations"),
                meta={"source": "llava_instruct_150k", "is_noise": False},
            )
        )

    write_jsonl(args.output, samples)
    print(f"[write] {args.output}: {len(samples)} samples "
          f"(skipped {missing} missing images, {empty} empty texts)")
    if len(samples) < args.n:
        print(f"[warn] requested {args.n} but only produced {len(samples)}; "
              f"more images may still be downloading")


if __name__ == "__main__":
    main()
