#!/usr/bin/env python3
"""Convert LLaVA-Instruct-150K into the pipeline's JSONL format and sample it.

Expects the raw annotation JSON (llava_instruct_150k.json). LLaVA-Instruct-150K
references COCO **train2014** images (not train2017).

Two flows, depending on whether the images are already local:

  A. Images already on disk -- sample only records whose image exists:

        python scripts/prepare_data.py --annotations data/raw/llava_instruct_150k.json \\
            --image-root data/images --n 25000 --output data/clean_25k.jsonl

  B. No images yet -- decide the sample set first, then fetch only what it
     references (~20k unique images / ~3.5GB, versus 13GB for all of train2014):

        python scripts/prepare_data.py --annotations data/raw/llava_instruct_150k.json \\
            --image-root data/images --n 25000 --output data/clean_25k.jsonl \\
            --no-require-image
        python scripts/fetch_images.py --input data/clean_25k.jsonl \\
            --image-root data/images

Get the annotations with:

    export HF_ENDPOINT=https://hf-mirror.com
    huggingface-cli download liuhaotian/LLaVA-Instruct-150K \\
        llava_instruct_150k.json --repo-type dataset --local-dir data/raw
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
    p.add_argument("--require-image", dest="require_image", action="store_true",
                   default=True,
                   help="skip records whose image file is absent (default)")
    p.add_argument("--no-require-image", dest="require_image", action="store_false",
                   help="keep records regardless -- use this to decide the sample "
                        "set first, then feed it to scripts/fetch_images.py")
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
