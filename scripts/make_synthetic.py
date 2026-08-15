#!/usr/bin/env python3
"""Generate a synthetic image-text dataset so the pipeline is runnable today.

COCO train2014 is 13GB. This script fabricates structurally identical data --
procedurally drawn shapes plus captions that truthfully describe them -- so the
framework, noise injection, and operator evaluation can all be exercised
end-to-end (including CLIP, which genuinely scores these pairs) while the real
download runs.

    python scripts/make_synthetic.py --n 200 --out-dir data/synthetic
"""
from __future__ import annotations

import argparse
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mmdataflow.core.sample import Sample, write_jsonl  # noqa: E402

COLORS = {
    "red": (200, 40, 40), "blue": (40, 80, 200), "green": (40, 160, 70),
    "yellow": (230, 200, 50), "purple": (140, 60, 180), "orange": (230, 130, 40),
}
SHAPES = ["circle", "square", "triangle"]
BACKGROUNDS = {"white": (245, 245, 245), "grey": (130, 130, 130), "black": (25, 25, 25)}


def draw(path: str, color: str, shape: str, bg: str, size: int,
         rng: random.Random) -> None:
    """Draw one dominant shape plus 1-3 random distractors.

    Structural diversity matters more than it looks: pHash is colour-blind, so a
    generator that only varies hue produces hundreds of structurally identical
    images and the dedup operator (correctly) collapses them all. Randomising
    position, scale and distractors makes each image structurally unique, which
    is what real photographs are.
    """
    from PIL import Image, ImageDraw

    img = Image.new("RGB", (size, size), BACKGROUNDS[bg])
    d = ImageDraw.Draw(img)
    rgb = COLORS[color]

    # Dominant shape: random position and scale, still clearly the main subject.
    side = rng.randint(int(size * 0.40), int(size * 0.65))
    x0 = rng.randint(0, size - side)
    y0 = rng.randint(0, size - side)
    _shape(d, shape, x0, y0, side, rgb)

    # Distractors: smaller, dimmer, never large enough to change the caption.
    for _ in range(rng.randint(1, 3)):
        s2 = rng.randint(int(size * 0.08), int(size * 0.20))
        x, y = rng.randint(0, size - s2), rng.randint(0, size - s2)
        c2 = COLORS[rng.choice(list(COLORS))]
        _shape(d, rng.choice(SHAPES), x, y, s2, c2)

    # Fine speckle so sharp images carry high-frequency detail (blur filter).
    for _ in range(rng.randint(20, 40)):
        x, y = rng.randint(0, size - 1), rng.randint(0, size - 1)
        r = rng.randint(1, 3)
        d.ellipse([x, y, x + r, y + r],
                  fill=tuple(rng.randint(0, 255) for _ in range(3)))

    os.makedirs(os.path.dirname(path), exist_ok=True)
    img.save(path, quality=92)


def _shape(d, shape: str, x: int, y: int, side: int, rgb) -> None:
    box = [x, y, x + side, y + side]
    if shape == "circle":
        d.ellipse(box, fill=rgb)
    elif shape == "square":
        d.rectangle(box, fill=rgb)
    else:
        d.polygon([(x + side // 2, y), (x + side, y + side), (x, y + side)], fill=rgb)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--n", type=int, default=200)
    p.add_argument("--out-dir", default="data/synthetic")
    p.add_argument("--size", type=int, default=384)
    p.add_argument("--seed", type=int, default=7)
    args = p.parse_args()

    rng = random.Random(args.seed)
    img_dir = os.path.join(args.out_dir, "images")
    samples = []
    for i in range(args.n):
        color = rng.choice(list(COLORS))
        shape = rng.choice(SHAPES)
        bg = rng.choice(list(BACKGROUNDS))
        rel = f"img_{i:05d}.jpg"
        draw(os.path.join(img_dir, rel), color, shape, bg, args.size, rng)
        samples.append(
            Sample(
                id=f"syn_{i:05d}",
                image_path=rel,
                text=f"A {color} {shape} centered on a {bg} background. "
                     f"The {shape} is the main subject of this simple image.",
                meta={"source": "synthetic", "is_noise": False,
                      "shape": shape, "color": color},
            )
        )

    out = os.path.join(args.out_dir, "clean.jsonl")
    write_jsonl(out, samples)
    print(f"[write] {out}: {len(samples)} samples, images in {img_dir}/")


if __name__ == "__main__":
    main()
