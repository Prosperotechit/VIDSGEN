import io

import numpy as np
import pytest
from PIL import Image

from conftest import first_frame_rgb, make_image, probe_video, submit, wait_for


@pytest.mark.parametrize("fmt,mime", [("PNG", "image/png"), ("JPEG", "image/jpeg"), ("WEBP", "image/webp")])
def test_i2v_accepts_supported_formats(client, fmt, mime):
    r = submit(client, image=make_image(fmt), image_name=f"in.{fmt.lower()}", image_type=mime, mode="i2v")
    assert r.status_code == 202, r.text
    job = r.json()
    assert job["mode"] == "i2v"
    assert job["input_image"]["width"] == 640 and job["input_image"]["height"] == 480
    assert job["input_image"]["original_format"] == fmt.lower()
    assert wait_for(client, job["id"])["status"] == "completed"


def test_i2v_video_starts_from_input_image(client, tmp_path):
    img = make_image("PNG", size=(832, 480))
    job_id = submit(client, image=img, mode="i2v", resolution="832x480", num_frames=17).json()["id"]
    assert wait_for(client, job_id)["status"] == "completed"
    video = client.get(f"/api/jobs/{job_id}/download").content
    info = probe_video(video, tmp_path)
    assert (info["width"], info["height"], info["frames"]) == (832, 480, 17)
    frame0 = first_frame_rgb(video, tmp_path, 832, 480).astype(int)
    src = np.asarray(Image.open(io.BytesIO(img)).convert("RGB")).astype(int)
    assert np.abs(frame0 - src).mean() < 12   # H.264 is lossy, but the frame matches the input


def test_i2v_input_is_normalised_and_exif_stripped(client):
    job = submit(client, image=make_image("JPEG", exif=True), image_name="p.jpg", image_type="image/jpeg",
                 mode="i2v").json()
    manager = client.app.state.manager
    key = manager.get(job["id"]).input_image_key
    assert key == f"uploads/{job['id']}/input.png"
    stored = manager.storage.local_path(key).read_bytes()
    with Image.open(io.BytesIO(stored)) as im:
        assert im.format == "PNG"
        assert not im.getexif()
    assert b"SecretCameraMaker" not in stored


def test_i2v_requires_image(client):
    r = submit(client, mode="i2v")
    assert r.status_code == 422
    assert r.json()["detail"][0]["loc"] == ["body", "image"]


def test_t2v_rejects_image(client):
    r = submit(client, image=make_image(), mode="t2v")
    assert r.status_code == 422
    assert "text-to-video" in r.json()["detail"][0]["msg"]


def test_i2v_rejects_non_image_bytes(client):
    r = submit(client, image=b"this is not an image at all", image_type="image/png", mode="i2v")
    assert r.status_code == 422
    assert "not a valid image" in r.json()["detail"][0]["msg"]


def test_i2v_rejects_truncated_image(client):
    data = make_image("PNG")
    r = submit(client, image=data[: len(data) // 2], mode="i2v")
    assert r.status_code == 422


def test_i2v_rejects_unsupported_format_even_if_mislabelled(client):
    r = submit(client, image=make_image("GIF"), image_name="x.png", image_type="image/png", mode="i2v")
    assert r.status_code == 415
    assert r.json()["detail"]["code"] == "invalid_image"


def test_i2v_rejects_too_small_image(client):
    r = submit(client, image=make_image(size=(100, 300)), mode="i2v")
    assert r.status_code == 422
    assert "too small" in r.json()["detail"][0]["msg"]


def test_i2v_rejects_oversized_dimensions(client):
    r = submit(client, image=make_image(size=(5000, 300)), mode="i2v")
    assert r.status_code == 422


def test_i2v_rejects_file_over_size_limit(make_client):
    client = make_client(max_upload_mb=0.05)  # ~51 KB
    noisy = Image.fromarray(np.random.default_rng(0).integers(0, 255, (600, 600, 3), dtype=np.uint8))
    buf = io.BytesIO()
    noisy.save(buf, format="PNG")
    assert len(buf.getvalue()) > 0.05 * 1024 * 1024
    r = submit(client, image=buf.getvalue(), mode="i2v")
    assert r.status_code == 413


def test_rejected_upload_creates_no_job(client):
    submit(client, image=b"junk", mode="i2v")
    assert client.get("/api/jobs").json()["total"] == 0
