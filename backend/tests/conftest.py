"""Shared test fixtures.

Every test gets an isolated app: its own temporary storage root + SQLite DB, the
mock provider with a short step delay, and `.env` files ignored (no real config or
secrets can leak into tests).
"""

from __future__ import annotations

import io
import re
import subprocess
import time
from collections.abc import Callable, Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app.core.config import Settings
from app.main import create_app
from app.providers.mock_video import ffmpeg_binary

TERMINAL = {"completed", "failed", "cancelled"}


def make_settings(tmp_path: Path, **overrides) -> Settings:
    base = dict(
        app_env="test",
        storage_local_root=tmp_path / "storage",
        mock_step_delay_seconds=0.002,
        inference_timeout_seconds=60,
        cors_origins="http://localhost:5173",
        log_level="WARNING",
    )
    base.update(overrides)
    return Settings(_env_file=None, **base)


@pytest.fixture
def make_client(tmp_path) -> Iterator[Callable[..., TestClient]]:
    """Factory: make_client(**settings_overrides) -> started TestClient."""
    clients: list[TestClient] = []

    def factory(**overrides) -> TestClient:
        client = TestClient(create_app(make_settings(tmp_path, **overrides)))
        client.__enter__()          # runs the lifespan (starts workers)
        clients.append(client)
        return client

    yield factory
    for c in reversed(clients):
        c.__exit__(None, None, None)


@pytest.fixture
def client(make_client) -> TestClient:
    return make_client()


# ---------------------------------------------------------------- helpers
def submit(client: TestClient, image: bytes | None = None, image_name: str = "in.png",
           image_type: str = "image/png", **fields):
    data = {"mode": "t2v", "prompt": "a red fox running through snow", "num_frames": "17", "steps": "8"}
    data.update({k: str(v) for k, v in fields.items()})
    files = {"image": (image_name, image, image_type)} if image is not None else None
    return client.post("/api/jobs", data=data, files=files)


def wait_for(client: TestClient, job_id: str, statuses=TERMINAL, timeout: float = 30.0) -> dict:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        job = client.get(f"/api/jobs/{job_id}").json()
        if job["status"] in statuses:
            return job
        time.sleep(0.01)
    raise AssertionError(f"job {job_id} did not reach {statuses} in {timeout}s (last: {job['status']})")


def make_image(fmt: str = "PNG", size=(640, 480), color=(200, 40, 40), exif: bool = False) -> bytes:
    img = Image.new("RGB", size, color)
    # A visible pattern so the I2V first-frame check is meaningful.
    for x in range(0, size[0], 40):
        for y in range(size[1]):
            img.putpixel((x, y), (20, 20, 220))
    buf = io.BytesIO()
    kwargs = {}
    if exif:
        ex = Image.Exif()
        ex[0x010F] = "SecretCameraMaker"   # Make
        kwargs["exif"] = ex.tobytes()
    img.save(buf, format=fmt, **kwargs)
    return buf.getvalue()


def probe_video(data: bytes, tmp_path: Path) -> dict:
    """Decode an MP4 with ffmpeg; return width, height, fps and decoded frame count."""
    f = tmp_path / "probe.mp4"
    f.write_bytes(data)
    r = subprocess.run([ffmpeg_binary(), "-hide_banner", "-i", str(f), "-map", "0:v:0", "-f", "null", "-"],
                       capture_output=True, text=True, check=True)
    err = r.stderr
    w, h = map(int, re.search(r"Video: h264.*?, (\d{2,5})x(\d{2,5})", err).groups())
    fps = float(re.search(r"([\d.]+) fps", err).group(1))
    frames = int(re.findall(r"frame=\s*(\d+)", err)[-1])
    return {"width": w, "height": h, "fps": fps, "frames": frames}


def first_frame_rgb(data: bytes, tmp_path: Path, width: int, height: int):
    import numpy as np

    f = tmp_path / "first.mp4"
    f.write_bytes(data)
    raw = subprocess.run([ffmpeg_binary(), "-hide_banner", "-loglevel", "error", "-i", str(f), "-frames:v", "1",
                          "-f", "rawvideo", "-pix_fmt", "rgb24", "-"], capture_output=True, check=True).stdout
    return np.frombuffer(raw, dtype=np.uint8).reshape(height, width, 3)
