#!/usr/bin/env python3
"""Inject labelled noise into a clean dataset.

This is the keystone of the project's experimental design. Because every
corrupted sample carries a ``meta.noise_type`` ground-truth label, operator
precision/recall becomes exactly computable (scripts/eval_ops.py) instead of a
qualitative claim -- and the A/B/C fine-tuning comparison gets a known-quality
dirty baseline.

Noise classes and their intended detector:

    mismatch          shuffled image-text pairing      -> clip_score_filter
    duplicate_exact   verbatim copy of another sample  -> phash_dedup
    duplicate_near    rescaled/recompressed copy       -> phash_dedup / semantic_dedup
    gibberish         truncation / n-gram loop / junk  -> text_quality_filter
    lowquality_image  downscaled or blurred image      -> resolution / blur filters
    wrong_lang        text replaced with another language -> lang_id_filter

Usage:
    python scripts/inject_noise.py \
        --input data/clean_15k.jsonl --image-root data/images \
        --output data/mixed_25k.jsonl --noise-image-dir data/images_noise
"""
from __future__ import annotations

import argparse
import copy
import os
import random
import string
import sys
from collections import Counter
from typing import Dict, List

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mmdataflow.core.sample import Sample, read_jsonl, write_jsonl  # noqa: E402

# Default counts follow PROJECT_PLAN.md section 6.1 (15k clean + 10k noise).
DEFAULT_COUNTS = {
    "mismatch": 4000,
    "duplicate_exact": 2000,
    "duplicate_near": 1000,
    "gibberish": 1500,
    "lowquality_image": 1000,
    "wrong_lang": 500,
}

FOREIGN_TEXTS = [
    "这张图片展示了一个非常有趣的场景，值得仔细观察和分析。",
    "Das Bild zeigt eine interessante Szene, die genauer betrachtet werden sollte.",
    "Эта фотография показывает интересную сцену, которую стоит рассмотреть.",
    "この写真は非常に興味深い場面を示しており、詳しく見る価値があります。",
    "Cette image montre une scène intéressante qui mérite d'être analysée.",
    "الصورة تظهر مشهدا مثيرا للاهتمام يستحق التحليل بعناية.",
    "이 사진은 자세히 살펴볼 가치가 있는 흥미로운 장면을 보여줍니다.",
    "Esta imagen muestra una escena interesante que vale la pena analizar.",
]


def _mark(s: Sample, noise_type: str, **extra) -> Sample:
    s.meta["is_noise"] = True
    s.meta["noise_type"] = noise_type
    s.meta.update(extra)
    return s


def corrupt_text(text: str, rng: random.Random) -> str:
    """One of three real-world text failure modes, chosen at random."""
    mode = rng.choice(["truncate", "ngram_loop", "junk"])
    if mode == "truncate":
        cut = max(3, int(len(text) * rng.uniform(0.02, 0.08)))
        return text[:cut]
    if mode == "ngram_loop":
        words = text.split()
        if len(words) < 6:
            return (text + " ") * 12
        gram = " ".join(words[: rng.randint(4, 6)])
        return " ".join([gram] * rng.randint(10, 25))
    junk_alphabet = string.punctuation + "�□▯╳¤§¶‡†"
    junk = "".join(rng.choice(junk_alphabet) for _ in range(rng.randint(60, 200)))
    keep = text[: max(0, int(len(text) * 0.2))]
    return keep + " " + junk


def degrade_image(src: str, dst: str, rng: random.Random) -> str:
    """Downscale below the resolution gate, or blur heavily. Returns mode used."""
    from PIL import Image, ImageFilter

    img = Image.open(src).convert("RGB")
    mode = rng.choice(["downscale", "blur"])
    if mode == "downscale":
        target = rng.randint(64, 200)  # under the 224 min_side gate
        ratio = target / max(1, min(img.size))
        img = img.resize(
            (max(1, int(img.width * ratio)), max(1, int(img.height * ratio)))
        )
    else:
        img = img.filter(ImageFilter.GaussianBlur(radius=rng.uniform(4.0, 9.0)))
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    img.save(dst, quality=rng.randint(30, 60))
    return mode


def near_duplicate_image(src: str, dst: str, rng: random.Random) -> str:
    """Rescale + light crop + recompress: same content, different bytes."""
    from PIL import Image

    img = Image.open(src).convert("RGB")
    scale = rng.uniform(0.75, 0.95)
    img = img.resize((max(1, int(img.width * scale)), max(1, int(img.height * scale))))
    if rng.random() < 0.5:
        dx, dy = int(img.width * 0.03), int(img.height * 0.03)
        img = img.crop((dx, dy, img.width - dx, img.height - dy))
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    img.save(dst, quality=rng.randint(70, 92))
    return "rescale_crop"


def build(args) -> None:
    rng = random.Random(args.seed)
    clean = read_jsonl(args.input)
    if not clean:
        raise SystemExit(f"no samples in {args.input}")
    for s in clean:
        s.meta.setdefault("is_noise", False)
    print(f"[load] {len(clean)} clean samples")

    counts: Dict[str, int] = dict(DEFAULT_COUNTS)
    if args.scale != 1.0:
        counts = {k: max(1, int(v * args.scale)) for k, v in counts.items()}
    total_noise = sum(counts.values())
    if total_noise > len(clean):
        raise SystemExit(
            f"need {total_noise} source samples for noise but only {len(clean)} clean "
            f"ones exist; use --scale to shrink"
        )

    # Draw disjoint source pools so one clean sample is never the basis of two
    # different noise samples (that would blur the ground truth).
    pool = list(range(len(clean)))
    rng.shuffle(pool)
    cursor = 0
    noisy: List[Sample] = []

    def take(n: int) -> List[Sample]:
        nonlocal cursor
        picked = [clean[i] for i in pool[cursor : cursor + n]]
        cursor += n
        return picked

    # 1. mismatch -- rotate image assignments so no sample keeps its own image
    src = take(counts["mismatch"])
    for i, s in enumerate(src):
        partner = src[(i + 1) % len(src)]
        new = copy.deepcopy(s)
        new.id = f"{s.id}__mismatch"
        new.image_path = partner.image_path
        noisy.append(_mark(new, "mismatch", original_image=s.image_path))

    # 2a. exact duplicates
    for s in take(counts["duplicate_exact"]):
        new = copy.deepcopy(s)
        new.id = f"{s.id}__dupexact"
        noisy.append(_mark(new, "duplicate_exact", duplicate_source=s.id))

    # 2b. near duplicates (new image file on disk)
    for s in take(counts["duplicate_near"]):
        new = copy.deepcopy(s)
        new.id = f"{s.id}__dupnear"
        rel = f"near/{new.id}.jpg"
        try:
            near_duplicate_image(
                os.path.join(args.image_root, s.image_path),
                os.path.join(args.noise_image_dir, rel),
                rng,
            )
        except Exception as e:
            print(f"[warn] near-dup skipped for {s.id}: {e}")
            continue
        new.image_path = os.path.join(args.noise_image_rel_prefix, rel)
        noisy.append(_mark(new, "duplicate_near", duplicate_source=s.id))

    # 3. gibberish text
    for s in take(counts["gibberish"]):
        new = copy.deepcopy(s)
        new.id = f"{s.id}__gibberish"
        new.text = corrupt_text(s.text, rng)
        noisy.append(_mark(new, "gibberish"))

    # 4. low-quality images
    for s in take(counts["lowquality_image"]):
        new = copy.deepcopy(s)
        new.id = f"{s.id}__lowq"
        rel = f"lowq/{new.id}.jpg"
        try:
            mode = degrade_image(
                os.path.join(args.image_root, s.image_path),
                os.path.join(args.noise_image_dir, rel),
                rng,
            )
        except Exception as e:
            print(f"[warn] degrade skipped for {s.id}: {e}")
            continue
        new.image_path = os.path.join(args.noise_image_rel_prefix, rel)
        noisy.append(_mark(new, "lowquality_image", degrade_mode=mode))

    # 5. wrong language
    for s in take(counts["wrong_lang"]):
        new = copy.deepcopy(s)
        new.id = f"{s.id}__lang"
        new.text = rng.choice(FOREIGN_TEXTS)
        noisy.append(_mark(new, "wrong_lang"))

    mixed = clean + noisy
    rng.shuffle(mixed)
    write_jsonl(args.output, mixed)

    stats = Counter(s.meta.get("noise_type", "clean") for s in mixed)
    print(f"[write] {args.output}: {len(mixed)} samples "
          f"({len(noisy)} noisy = {len(noisy) / len(mixed):.1%})")
    for k, v in sorted(stats.items(), key=lambda kv: -kv[1]):
        print(f"  {k:18s} {v:6d}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", required=True, help="clean dataset jsonl")
    p.add_argument("--output", required=True)
    p.add_argument("--image-root", default="data/images",
                   help="root the clean image_path values resolve against")
    p.add_argument("--noise-image-dir", default="data/images_noise",
                   help="where generated (degraded/near-dup) images are written")
    p.add_argument("--noise-image-rel-prefix", default="../images_noise",
                   help="prefix stored in image_path so noise images resolve "
                        "from image_root at pipeline time")
    p.add_argument("--scale", type=float, default=1.0,
                   help="scale all noise counts (e.g. 0.02 for a smoke test)")
    p.add_argument("--seed", type=int, default=42)
    build(p.parse_args())


if __name__ == "__main__":
    main()
