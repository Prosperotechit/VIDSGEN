"""Synthesises a small but real H.264 MP4 for the mock provider (no GPU, no model).

- T2V: animated colour gradient with a moving circle. The colours come from the seed
  and the prompt, so the same request always produces the same video.
- I2V: the input image is cover-cropped to the target size and slowly zoomed, so the
  first frame equals the input image, like real image-to-video output.

Frames are generated with numpy and piped as raw RGB into ffmpeg (the system ffmpeg
if present, otherwise the binary bundled with `imageio-ffmpeg`).
"""

from __future__ import annotations

import hashlib
import io
import shutil
import subprocess
import tempfile
from functools import lru_cache
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


class VideoEncodeError(RuntimeError):
    pass


@lru_cache
def ffmpeg_binary() -> str:
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception as e:  # pragma: no cover - depends on host
        raise VideoEncodeError("ffmpeg is not available (install ffmpeg or imageio-ffmpeg)") from e


def _palette(seed: int, prompt: str) -> np.ndarray:
    h = hashlib.sha256(f"{seed}:{prompt}".encode()).digest()
    return np.array([[h[0], h[1], h[2]], [h[3], h[4], h[5]]], dtype=np.float32)


def _t2v_frames(width: int, height: int, n: int, seed: int, prompt: str):
    c0, c1 = _palette(seed, prompt)
    yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
    radius = min(width, height) * 0.12
    for i in range(n):
        phase = i / max(n - 1, 1)
        mix = ((xx / width + yy / height) * 0.5 + phase) % 1.0
        frame = c0 * (1 - mix[..., None]) + c1 * mix[..., None]
        cx = width * (0.15 + 0.7 * phase)
        cy = height * (0.5 + 0.25 * np.sin(phase * 2 * np.pi))
        inside = (xx - cx) ** 2 + (yy - cy) ** 2 < radius**2
        frame[inside] = 255 - frame[inside]
        yield frame.astype(np.uint8)


def _cover(img: Image.Image, width: int, height: int, zoom: float) -> Image.Image:
    scale = max(width / img.width, height / img.height) * zoom
    w, h = round(img.width * scale), round(img.height * scale)
    img = img.resize((w, h), Image.BICUBIC)
    left, top = (w - width) // 2, (h - height) // 2
    return img.crop((left, top, left + width, top + height))


def _i2v_frames(width: int, height: int, n: int, image_png: bytes):
    with Image.open(io.BytesIO(image_png)) as im:
        src = im.convert("RGB")
    for i in range(n):
        zoom = 1.0 + 0.15 * (i / max(n - 1, 1))   # first frame: zoom 1.0 == input image
        yield np.asarray(_cover(src, width, height, zoom), dtype=np.uint8)


def _label(frame: np.ndarray) -> np.ndarray:
    img = Image.fromarray(frame)
    ImageDraw.Draw(img).text((8, 8), "VIDSGEN MOCK", fill=(255, 255, 255))
    return np.asarray(img)


def encode_mock_video(
    out_path: Path, *, width: int, height: int, num_frames: int, fps: int,
    seed: int, prompt: str, image_png: bytes | None = None,
) -> None:
    frames = (_i2v_frames(width, height, num_frames, image_png) if image_png
              else _t2v_frames(width, height, num_frames, seed, prompt))
    cmd = [
        ffmpeg_binary(), "-hide_banner", "-loglevel", "error", "-y",
        "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{width}x{height}", "-r", str(fps), "-i", "-",
        "-an", "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p",
        "-movflags", "+faststart", str(out_path),
    ]
    # stderr goes to a temp file rather than a pipe, so ffmpeg can never block on a full pipe.
    # The `with` blocks guarantee every handle is closed and the process is reaped.
    with tempfile.TemporaryFile() as errlog, \
            subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=errlog) as proc:
        try:
            for i, f in enumerate(frames):
                proc.stdin.write((_label(f) if i > 0 else f).tobytes())  # keep frame 0 untouched for I2V
            proc.stdin.close()
            code = proc.wait()
        except BrokenPipeError:
            code = proc.wait()
        except BaseException:
            proc.kill()
            raise
        if code != 0:
            errlog.seek(0)
            raise VideoEncodeError(f"ffmpeg failed ({code}): {errlog.read().decode(errors='replace').strip()[:500]}")
