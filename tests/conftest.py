"""Shared fixtures: tiny procedurally drawn images written to a temp dir."""
from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mmdataflow.core import Context, Sample  # noqa: E402


def _draw(path: str, size=(400, 400), color=(200, 40, 40), blur=0.0, noise=True,
          shape="circle"):
    from PIL import Image, ImageDraw, ImageFilter

    img = Image.new("RGB", size, (245, 245, 245))
    d = ImageDraw.Draw(img)
    w, h = size
    m = min(w, h) // 5
    box = [m, m, w - m, h - m]
    # Vary geometry, not just colour: pHash works on a grayscale DCT and is
    # deliberately insensitive to hue, so images that differ only in colour do
    # collide. Fixtures must differ structurally to be genuinely distinct.
    if shape == "circle":
        d.ellipse(box, fill=color)
    elif shape == "square":
        d.rectangle(box, fill=color)
    elif shape == "triangle":
        d.polygon([(w // 2, m), (w - m, h - m), (m, h - m)], fill=color)
    elif shape == "bars":
        for i in range(m, w - m, max(2, (w - 2 * m) // 6)):
            d.rectangle([i, m, i + max(2, (w - 2 * m) // 12), h - m], fill=color)
    if noise:
        # High-frequency detail so the sharp image clears the blur threshold.
        for i in range(0, w, 7):
            d.line([(i, 0), (i, h)], fill=(0, 0, 0), width=1)
    if blur:
        img = img.filter(ImageFilter.GaussianBlur(radius=blur))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    img.save(path, quality=95)
    return path


@pytest.fixture
def img_dir(tmp_path):
    return str(tmp_path / "images")


@pytest.fixture
def ctx(tmp_path, img_dir):
    return Context(
        work_dir=str(tmp_path / "work"),
        image_root=img_dir,
        device="cpu",
        cache_embeddings=False,
    )


@pytest.fixture
def make_sample(img_dir):
    """Create a Sample backed by a real image file on disk."""
    def _make(sid: str, text: str = "A red circle on a white background.",
              size=(400, 400), color=(200, 40, 40), blur=0.0, noise=True,
              image: bool = True, shape: str = "circle", **meta):
        rel = f"{sid}.jpg"
        if image:
            _draw(os.path.join(img_dir, rel), size, color, blur, noise, shape)
        return Sample(id=sid, image_path=rel if image else None, text=text, meta=meta)

    return _make
