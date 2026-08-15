"""Tests for the per-image COCO fetcher.

Served by a real local HTTP server rather than a mocked urlopen: the failure
modes worth testing here (truncated files, 404s, atomic rename) all live in the
interaction with the network stack, and a mock would assert my assumptions about
urllib instead of urllib's behaviour.
"""
from __future__ import annotations

import http.server
import json
import os
import socketserver
import sys
import threading

import pytest

sys.path.insert(
    0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts")
)

from fetch_images import download_one, is_valid_image, referenced_images  # noqa: E402


@pytest.fixture
def served(tmp_path):
    """Serve a directory over HTTP; yields (base_url, directory)."""
    root = tmp_path / "remote"
    root.mkdir()

    class Handler(http.server.SimpleHTTPRequestHandler):
        def __init__(self, *a, **kw):
            super().__init__(*a, directory=str(root), **kw)

        def log_message(self, *a):
            pass  # keep pytest output readable

    with socketserver.TCPServer(("127.0.0.1", 0), Handler) as httpd:
        t = threading.Thread(target=httpd.serve_forever, daemon=True)
        t.start()
        yield f"http://127.0.0.1:{httpd.server_address[1]}", root
        httpd.shutdown()


@pytest.fixture
def remote_image(served):
    base, root = served
    from PIL import Image

    Image.new("RGB", (300, 300), (120, 30, 200)).save(root / "COCO_x.jpg", quality=90)
    return base


class TestReferencedImages:
    def test_dedupes_and_preserves_order(self, tmp_path):
        p = tmp_path / "s.jsonl"
        rows = [
            {"id": "1", "image_path": "b.jpg"},
            {"id": "2", "image_path": "a.jpg"},
            {"id": "3", "image_path": "b.jpg"},  # LLaVA reuses one image
            {"id": "4", "image_path": None},
        ]
        p.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
        # Order must be stable so an interrupted run resumes monotonically.
        assert referenced_images(str(p)) == ["b.jpg", "a.jpg"]

    def test_tolerates_blank_lines(self, tmp_path):
        p = tmp_path / "s.jsonl"
        p.write_text(json.dumps({"id": "1", "image_path": "a.jpg"}) + "\n\n")
        assert referenced_images(str(p)) == ["a.jpg"]


class TestIsValidImage:
    def test_rejects_truncated_file(self, tmp_path):
        f = tmp_path / "half.jpg"
        f.write_bytes(b"\xff\xd8\xff\xe0" + b"\x00" * 4000)  # JPEG header, no body
        assert is_valid_image(str(f)) is False

    def test_rejects_empty_file(self, tmp_path):
        f = tmp_path / "empty.jpg"
        f.write_bytes(b"")
        assert is_valid_image(str(f)) is False

    def test_accepts_real_image(self, tmp_path):
        from PIL import Image

        f = tmp_path / "real.jpg"
        Image.new("RGB", (300, 300), (10, 200, 90)).save(f, quality=90)
        assert is_valid_image(str(f)) is True


class TestDownloadOne:
    def test_downloads_and_verifies(self, tmp_path, remote_image):
        local = tmp_path / "local"
        rel, status, err = download_one(
            "COCO_x.jpg", str(local), remote_image, retries=1, timeout=10, verify=True
        )
        assert (rel, status, err) == ("COCO_x.jpg", "ok", None)
        assert is_valid_image(str(local / "COCO_x.jpg"))

    def test_skips_an_image_already_present(self, tmp_path, remote_image):
        from PIL import Image

        local = tmp_path / "local"
        local.mkdir()
        Image.new("RGB", (300, 300), (1, 2, 3)).save(local / "COCO_x.jpg", quality=90)
        _, status, _ = download_one(
            "COCO_x.jpg", str(local), remote_image, retries=0, timeout=10, verify=True
        )
        assert status == "skip"

    def test_replaces_a_truncated_leftover(self, tmp_path, remote_image):
        """The bug a plain os.path.exists check would cause: an interrupted run
        leaves a half file, and every later run skips it forever."""
        local = tmp_path / "local"
        local.mkdir()
        (local / "COCO_x.jpg").write_bytes(b"\xff\xd8\xff\xe0" + b"\x00" * 4000)

        _, status, _ = download_one(
            "COCO_x.jpg", str(local), remote_image, retries=1, timeout=10, verify=True
        )
        assert status == "ok"
        assert is_valid_image(str(local / "COCO_x.jpg"))

    def test_missing_image_fails_without_leaving_a_partial(self, tmp_path, remote_image):
        local = tmp_path / "local"
        rel, status, err = download_one(
            "nope.jpg", str(local), remote_image, retries=1, timeout=10, verify=True
        )
        assert status == "fail" and "404" in err
        # No .part file survives a failure, or the next run would see junk.
        assert not os.path.exists(str(local / "nope.jpg.part"))
        assert not os.path.exists(str(local / "nope.jpg"))

    def test_404_is_not_retried(self, tmp_path, served, monkeypatch):
        """A missing image will not appear on retry; burning the retry budget on
        it just slows a 20k-image run down."""
        base, _ = served
        calls = []
        real = __import__("urllib.request", fromlist=["urlopen"]).urlopen

        def counting(req, *a, **kw):
            calls.append(req.full_url)
            return real(req, *a, **kw)

        monkeypatch.setattr("urllib.request.urlopen", counting)
        download_one("gone.jpg", str(tmp_path / "l"), base, retries=3, timeout=10,
                     verify=True)
        assert len(calls) == 1

    def test_nested_relative_paths_are_created(self, tmp_path, remote_image):
        """prepare_data.py can emit a train2014/ prefix."""
        local = tmp_path / "local"
        _, status, _ = download_one(
            "train2014/COCO_x.jpg", str(local), remote_image, retries=1, timeout=10,
            verify=True,
        )
        assert status == "ok"
        assert os.path.exists(str(local / "train2014" / "COCO_x.jpg"))
