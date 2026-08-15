"""Shared helpers for operators that touch images."""
from __future__ import annotations

from typing import Optional

from ..core.context import Context
from ..core.sample import Sample


def open_image(sample: Sample, ctx: Context, convert: Optional[str] = None):
    """Open a sample's image, or return None if missing/corrupt.

    Callers decide what a None means -- most operators drop the sample with an
    explicit ``:unreadable`` reason so broken paths show up in the report rather
    than silently passing through.
    """
    from PIL import Image

    path = ctx.resolve_image(sample.image_path)
    if not path:
        return None
    try:
        img = Image.open(path)
        img.load()
        return img.convert(convert) if convert else img
    except Exception:
        return None


def image_size(sample: Sample, ctx: Context):
    """Read (width, height) without decoding pixel data -- much cheaper."""
    from PIL import Image

    path = ctx.resolve_image(sample.image_path)
    if not path:
        return None
    try:
        with Image.open(path) as img:
            return img.size
    except Exception:
        return None
