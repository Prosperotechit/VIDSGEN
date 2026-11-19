"""The model downloader must be safe by default. These tests only ever run it in DRY-RUN mode
(no --execute), in a subprocess with network access deliberately broken."""

import os
import subprocess
import sys

from app.core.config import REPO_ROOT

SCRIPT = REPO_ROOT / "scripts" / "download_models.py"


def run(*args, cwd=None):
    env = {**os.environ, "HTTPS_PROXY": "http://127.0.0.1:9", "HTTP_PROXY": "http://127.0.0.1:9",
           "HF_HUB_OFFLINE": "1"}  # any network attempt would fail loudly
    return subprocess.run([sys.executable, str(SCRIPT), *args], capture_output=True, text=True, env=env, cwd=cwd, timeout=60)


def test_dry_run_is_default_and_lists_files_and_destinations(tmp_path):
    r = run("--comfyui-dir", str(tmp_path / "ComfyUI"))
    assert r.returncode == 0, r.stderr
    out = r.stdout
    assert "DRY RUN: nothing was downloaded" in out
    for name, folder in [("wan2.2_ti2v_5B_fp16.safetensors", "diffusion_models"),
                         ("wan2.2_vae.safetensors", "vae"),
                         ("umt5_xxl_fp8_e4m3fn_scaled.safetensors", "text_encoders")]:
        assert name in out
        assert str(tmp_path / "ComfyUI" / "models" / folder / name) in out
    assert "To download: 3 file(s), 18.1 GB" in out
    assert not (tmp_path / "ComfyUI").exists()   # dry run creates nothing


def test_dry_run_detects_present_files(tmp_path):
    vae = tmp_path / "ComfyUI" / "models" / "vae" / "wan2.2_vae.safetensors"
    vae.parent.mkdir(parents=True)
    vae.write_bytes(b"x")           # wrong size: must not count as present
    out = run("--comfyui-dir", str(tmp_path / "ComfyUI")).stdout
    assert "[wrong size] vae" in out and "To download: 3 file(s)" in out


def test_script_does_not_import_downloader_unless_executing():
    src = SCRIPT.read_text()
    head = src.split("def download(", 1)[0]
    assert "huggingface_hub" not in head.split('"""', 2)[-1]   # only imported inside download()
