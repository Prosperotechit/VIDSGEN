"""Tests for scripts/backend_e2e_test.py (Phase 6B): the loopback backend-to-ComfyUI end-to-end tooling.

No GPU, no weights, no network: the real FastAPI backend runs as a subprocess against a fake ComfyUI server, so the
whole tool (process management, HTTP driving, checks, report) is exercised. This proves the TOOLING. It does not
prove real GPU integration: that is only established by running it on Kaggle.
"""

from __future__ import annotations

import importlib.util
import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

import pytest
from PIL import Image

from app.core.config import REPO_ROOT
from app.jobs.images import validate_image
from app.providers.mock_video import encode_mock_video
from comfyui_support import FakeComfyUI
import test_wan22_gpu_test  # noqa: F401  (registers scripts/wan22_gpu_test.py as module "wan22_gpu_test")

SCRIPT = REPO_ROOT / "scripts" / "backend_e2e_test.py"
_spec = importlib.util.spec_from_file_location("backend_e2e_test", SCRIPT)
e2e = importlib.util.module_from_spec(_spec)
sys.modules["backend_e2e_test"] = e2e
_spec.loader.exec_module(e2e)
gt = e2e.gt

BACKEND_DIR = REPO_ROOT / "backend"
WORKFLOWS = REPO_ROOT / "workflows" / "comfyui"
PYTHON = sys.executable                        # has the backend's dependencies


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture(scope="module")
def good_video(tmp_path_factory) -> bytes:
    path = tmp_path_factory.mktemp("vid") / "v.mp4"
    encode_mock_video(path, width=832, height=480, num_frames=17, fps=24, seed=3, prompt="e2e")
    return path.read_bytes()


@pytest.fixture
def fake(good_video):
    server = FakeComfyUI(good_video).start()
    server.step_delay = 0.05                   # 20 steps take about a second: polling can see intermediate progress
    yield server
    server.stop()


def run_e2e(tmp_path, url, *extra, capsys=None):
    base = tmp_path / "base"
    base.mkdir(exist_ok=True)
    port = free_port()
    argv = ["e2e", "--base", str(base), "--backend-dir", str(BACKEND_DIR), "--workflows-dir", str(WORKFLOWS), "--python", PYTHON,
            "--comfyui-url", url, "--backend-port", str(port), "--poll", "0.1", "--timeout", "60",
            "--out", str(tmp_path / "out" / "e2e.mp4"), "--report", str(tmp_path / "out" / "report.json"), *extra]
    rc = e2e.main(argv)
    report = json.loads((tmp_path / "out" / "report.json").read_text()) if (tmp_path / "out" / "report.json").exists() else None
    return rc, report, base, port


def assert_stopped(base: Path, port: int):
    assert not (base / "logs" / "backend.pid").exists()
    assert gt.listening_addresses(port) == []


def posted_prompts(fake) -> list[dict]:
    return [r["body"] for r in fake.requests if r["path"] == "/prompt" and r["method"] == "POST"]


# ====================================================================== configuration (pure functions)
@pytest.mark.parametrize("url,ok", [
    ("http://127.0.0.1:8188", True), ("http://localhost:8188", True), ("http://[::1]:8188", True),
    ("https://abc.trycloudflare.com", False), ("http://0.0.0.0:8188", False), ("http://192.168.1.5:8188", False),
    ("http://example.com:8188", False), ("http://127.0.0.1", False), ("https://127.0.0.1:8188", False), ("", False),
])
def test_only_loopback_comfyui_urls_are_accepted(url, ok):
    if ok:
        host, port = e2e.loopback_endpoint(url)
        assert port == 8188 and host in ("127.0.0.1", "localhost", "::1")
    else:
        with pytest.raises(ValueError, match="loopback only"):
            e2e.loopback_endpoint(url)


def test_app_settings_are_exactly_the_verified_loopback_configuration(tmp_path):
    s = e2e.app_settings(tmp_path, WORKFLOWS, "http://127.0.0.1:8188", 8000)
    assert s["GENERATION_PROFILE"] == "t4_verified" and s["INFERENCE_MAX_CONCURRENT_JOBS"] == "1"
    assert s["INFERENCE_PROVIDER"] == "comfyui" and s["COMFYUI_BASE_URL"] == "http://127.0.0.1:8188"
    assert s["API_HOST"] == "127.0.0.1" and s["CORS_ORIGINS"] == "" and s["COMFYUI_CONNECTION_GRACE_SECONDS"] == "30"
    assert not any(k in s for k in ("COMFYUI_API_TOKEN", "HF_TOKEN", "S3_ACCESS_KEY_ID"))
    assert s["STORAGE_LOCAL_ROOT"] == str(tmp_path / "storage")
    with pytest.raises(ValueError, match="loopback only"):
        e2e.app_settings(tmp_path, WORKFLOWS, "https://x.trycloudflare.com", 8000)


def test_the_settings_are_accepted_by_the_real_backend_configuration(tmp_path):
    from app.core.config import Settings
    s = e2e.app_settings(tmp_path, WORKFLOWS, "http://127.0.0.1:8188", 8000)
    cfg = Settings(_env_file=None, **{k.lower(): v for k, v in s.items()})
    assert cfg.generation_profile == "t4_verified" and cfg.inference_max_concurrent_jobs == 1
    assert cfg.api_host == "127.0.0.1" and cfg.cors_origin_list == []


def test_the_backend_environment_is_an_allowlist_and_carries_no_credential(tmp_path):
    secrets = {"HF_TOKEN": "hf_" + "a" * 30, "GITHUB_TOKEN": "ghp_" + "b" * 30, "AWS_ACCESS_KEY_ID": "AKIA" + "C" * 16,
               "KAGGLE_KEY": "kk-" + "d" * 20, "COMFYUI_API_TOKEN": "tok-" + "e" * 20, "SOME_RANDOM_VAR": "x",
               "S3_SECRET_ACCESS_KEY": "s" * 20, "INFERENCE_PROVIDER": "mock", "GENERATION_PROFILE": "unrestricted"}
    environ = {"PATH": "/usr/bin", "HOME": "/root", "LANG": "C.UTF-8", **secrets}
    s = e2e.app_settings(tmp_path, WORKFLOWS, "http://127.0.0.1:8188", 8000)
    env = e2e.backend_env(tmp_path, s, environ)
    assert env["PATH"] == "/usr/bin" and env["HOME"] == "/root"
    for name, value in secrets.items():
        assert value not in env.values(), name
    assert env["INFERENCE_PROVIDER"] == "comfyui" and env["GENERATION_PROFILE"] == "t4_verified"   # ours, not the ambient ones
    assert "SOME_RANDOM_VAR" not in env and "KAGGLE_KEY" not in env


def test_backend_requirements_come_from_pyproject_and_never_include_pytorch():
    reqs = e2e.backend_requirements(BACKEND_DIR)
    names = {r.split("[")[0].split(">")[0].split("=")[0].split("<")[0].strip().lower() for r in reqs}
    assert {"fastapi", "uvicorn", "pydantic-settings", "python-multipart", "pillow", "httpx", "websockets", "pyyaml"} <= names
    assert not names & {"torch", "torchvision", "torchaudio"} and all("#" not in r for r in reqs)


def test_the_probe_png_is_valid_and_large_enough_for_the_backends_image_rules():
    data = e2e.png_bytes()
    assert Image.open(__import__("io").BytesIO(data)).size == (256, 256)
    validate_image(data, max_bytes=10 * 1024 * 1024, min_side=256, max_side=4096)      # the backend accepts it as an image


# ====================================================================== the verdict
def test_verdict_passes_only_when_every_evaluated_claim_is_true():
    v = e2e.Verdict()
    assert v.verdict == "FAIL"                      # nothing was checked: never a pass
    v.claim("a", True); v.skip("b")
    assert v.verdict == "PASS" and v.claims == {"a": True, "b": None}
    v.claim("c", False, "because")
    assert v.verdict == "FAIL" and v.reasons == ["c: because"]


def test_poll_job_gives_up_after_the_timeout():
    class Api:
        def get(self, path):
            class R:
                def raise_for_status(self): pass
                def json(self): return {"status": "processing", "progress": 12.0}
            return R()
    t0 = time.monotonic()
    r = e2e.poll_job(Api(), "job_x", timeout=0.3, interval=0.05)
    assert r["timed_out"] is True and time.monotonic() - t0 < 5 and r["timeline"] == [{"t": r["timeline"][0]["t"], "status": "processing", "progress": 12.0}]


# ====================================================================== install-backend (pip is spied on: no network)
@pytest.fixture
def install_env(tmp_path, monkeypatch):
    base = tmp_path / "base"
    (base / "venv" / "bin").mkdir(parents=True)
    (base / "venv" / "bin" / "python").write_text("")
    (base / "constraints.txt").write_text("numpy==2.0.0\ntorch==2.6.0\n")
    calls = []

    def fake_run(cmd, **kw):
        calls.append(list(cmd))
        return subprocess.CompletedProcess(cmd, 0, "", "")
    host = {"python": "3.13.1", "packages": {n: [f"{n}-1", f"/host/{n}/__init__.py"] for n in gt.PINNED_PACKAGES}}
    monkeypatch.setattr(e2e.subprocess, "run", fake_run)
    monkeypatch.setattr(gt, "host_inventory", lambda *a, **k: {"torch": ["2.6.0"], "numpy": ["2.0.0"], "pip": ["24"]})
    monkeypatch.setattr(gt, "pip_info", lambda *a, **k: {"version": "24", "location": "/host/pip", "source": "host"})
    monkeypatch.setattr(gt, "import_probe", lambda py, env: json.loads(json.dumps(host)))
    monkeypatch.setattr(e2e, "backend_importable", lambda *a, **k: "")
    return base, calls, host


def test_install_backend_runs_pip_in_the_venv_under_the_constraints_file(install_env):
    base, calls, _ = install_env
    r = e2e.install_backend(base, BACKEND_DIR, "/usr/bin/python3")
    pip = next(c for c in calls if "install" in c)
    assert pip[0] == str(base / "venv" / "bin" / "python") and pip[1:4] == ["-m", "pip", "install"]
    assert pip[pip.index("-c") + 1] == str(base / "constraints.txt") and pip[pip.index("-r") + 1] == str(base / "requirements-backend.txt")
    assert "--no-cache-dir" in pip
    written = (base / "requirements-backend.txt").read_text()
    assert "fastapi" in written and "torch" not in written
    assert r["host_environment_unchanged"] is True and r["backend_importable"] is True and r["pip"]["source"] == "host"
    assert set(r["protected_packages"]) == set(gt.PINNED_PACKAGES)


def test_install_backend_needs_the_comfyui_install_first(tmp_path):
    with pytest.raises(RuntimeError, match="run install-comfyui first"):
        e2e.install_backend(tmp_path / "empty", BACKEND_DIR, "python")


def test_install_backend_needs_phase_6a_in_the_checkout(install_env, tmp_path):
    base, _, _ = install_env
    old = tmp_path / "old-backend"
    (old / "app").mkdir(parents=True)
    (old / "pyproject.toml").write_text("[project]\nname='x'\ndependencies=[]\n")
    (old / "app" / "main.py").write_text("")
    with pytest.raises(RuntimeError, match="Phase 6A"):
        e2e.install_backend(base, old, "python")


def test_install_backend_fails_if_a_protected_package_was_replaced(install_env, monkeypatch):
    base, _, host = install_env
    venv_py = str(base / "venv" / "bin" / "python")

    def probe(py, env):
        r = json.loads(json.dumps(host))
        if py == venv_py:
            r["packages"]["numpy"] = ["9.9.9", str(base / "venv" / "lib" / "numpy" / "__init__.py")]
        return r
    monkeypatch.setattr(gt, "import_probe", probe)
    with pytest.raises(RuntimeError, match="numpy changed"):
        e2e.install_backend(base, BACKEND_DIR, "python")


def test_install_backend_fails_if_the_host_environment_changed(install_env, monkeypatch):
    base, _, _ = install_env
    seen = iter([{"torch": ["2.6.0"], "pip": ["24"]}, {"torch": ["2.6.0"], "pip": ["24"], "evil": ["1.0"]}])
    monkeypatch.setattr(gt, "host_inventory", lambda *a, **k: next(seen))
    with pytest.raises(RuntimeError, match="evil was ADDED"):
        e2e.install_backend(base, BACKEND_DIR, "python")


def test_install_backend_reports_a_pip_failure_redacted(install_env, monkeypatch):
    base, _, _ = install_env
    monkeypatch.setattr(e2e.subprocess, "run", lambda cmd, **kw: subprocess.CompletedProcess(
        cmd, 1, "", "ERROR: no matching distribution Authorization: Bearer abc.def-123"))
    with pytest.raises(RuntimeError, match="no matching distribution") as e:
        e2e.install_backend(base, BACKEND_DIR, "python")
    assert "abc.def-123" not in str(e.value)


def test_install_backend_fails_if_the_backend_cannot_be_imported_afterwards(install_env, monkeypatch):
    base, _, _ = install_env
    monkeypatch.setattr(e2e, "backend_importable", lambda *a, **k: "ModuleNotFoundError: No module named 'fastapi'")
    with pytest.raises(RuntimeError, match=r"(?s)cannot be imported from the venv.*fastapi"):
        e2e.install_backend(base, BACKEND_DIR, "python")


def test_backend_importable_is_true_for_the_real_backend(tmp_path):
    assert e2e.backend_importable(PYTHON, BACKEND_DIR, tmp_path) == ""
    assert "No module named" in e2e.backend_importable(PYTHON, tmp_path, tmp_path)        # not a backend directory


# ====================================================================== the end-to-end tool against a fake ComfyUI
def test_full_run_passes_and_proves_every_claim(tmp_path, fake, capsys):
    rc, report, base, port = run_e2e(tmp_path, fake.url)
    assert rc == 0, report["reasons"]
    assert report["verdict"] == "PASS" and report["reasons"] == []
    claims = report["claims"]
    assert claims.pop("cancel_works") is None                                   # not requested
    assert claims and all(v is True for v in claims.values()), claims
    assert {"backend_loopback_only", "comfyui_loopback_only", "envelope_enforced", "job_completed", "progress_monotonic",
            "progress_intermediate_values", "download_ok", "range_request_ok", "preview_inline_ok", "mp4_decodes_and_matches",
            "inference_verified_by_generation", "backend_stopped"} <= set(claims)
    # exactly ONE generation reached ComfyUI: the four out-of-envelope probes were stopped by the backend
    prompts = posted_prompts(fake)
    assert len(prompts) == 1
    graph = prompts[0]["prompt"]
    assert (graph["55"]["inputs"]["width"], graph["55"]["inputs"]["height"], graph["55"]["inputs"]["length"]) == (832, 480, 17)
    assert (graph["3"]["inputs"]["steps"], graph["3"]["inputs"]["cfg"], graph["57"]["inputs"]["fps"]) == (20, 5.0, 24)
    assert report["envelope_probes"]["resolution"]["status"] == 422 and report["envelope_probes"]["mode"]["fields"] == ["mode"]
    # the saved file is a real, verified MP4
    out = tmp_path / "out" / "e2e.mp4"
    assert gt.probe_mp4(out, frames=17, size=(832, 480), fps=24.0)["problems"] == []
    assert report["output"]["decoded_frames"] == 17 and report["output"]["codec"] == "h264"
    assert report["job"]["status"] == "completed" and report["job"]["provider"] == "comfyui"
    assert report["settings_applied"]["GENERATION_PROFILE"] == "t4_verified"
    # no credentials: none were sent to ComfyUI, none appear in the report
    assert all(r["auth"] is None for r in fake.requests)
    text = json.dumps(report)
    assert "Bearer " not in text and "hf_" not in text
    assert_stopped(base, port)
    assert json.loads(capsys.readouterr().out)["verdict"] == "PASS"


def test_progress_timeline_shows_real_intermediate_values(tmp_path, fake):
    _, report, _, _ = run_e2e(tmp_path, fake.url)
    t = report["timeline"]
    assert t[0]["status"] in ("queued", "processing") and t[-1] == {"t": t[-1]["t"], "status": "completed", "progress": 100.0}
    assert any(0 < x["progress"] < 100 for x in t) and [x["progress"] for x in t] == sorted(x["progress"] for x in t)


def test_cancel_stage_cancels_a_second_job_and_interrupts_comfyui(tmp_path, fake):
    fake.next_scenarios = ["success", "hang"]
    rc, report, base, port = run_e2e(tmp_path, fake.url, "--with-cancel")
    assert rc == 0, report["reasons"]
    assert report["claims"]["cancel_works"] is True and report["cancel_test"]["ok"] is True
    assert len(fake.interrupts) >= 1
    assert_stopped(base, port)


def test_gpu_out_of_memory_is_a_clear_failure_with_no_video(tmp_path, fake):
    fake.default_scenario = "oom"
    rc, report, base, port = run_e2e(tmp_path, fake.url)
    assert rc == 1 and report["verdict"] == "FAIL"
    assert any("gpu_out_of_memory" in r for r in report["reasons"]), report["reasons"]
    assert report["claims"]["job_completed"] is False and "download_ok" not in report["claims"]
    assert not (tmp_path / "out" / "e2e.mp4").exists() and report["backend_log_tail"]
    assert_stopped(base, port)


def test_a_video_of_the_wrong_size_is_not_a_pass(tmp_path, fake):
    small = tmp_path / "small.mp4"
    encode_mock_video(small, width=64, height=64, num_frames=17, fps=24, seed=1, prompt="x")
    fake.video_bytes = small.read_bytes()
    rc, report, base, port = run_e2e(tmp_path, fake.url)
    assert rc == 1 and report["claims"]["job_completed"] is True and report["claims"]["mp4_decodes_and_matches"] is False
    assert "size is 64x64, expected 832x480" in " ".join(report["reasons"])
    assert_stopped(base, port)


def test_an_unreachable_comfyui_fails_clearly_and_never_hangs(tmp_path):
    rc, report, base, port = run_e2e(tmp_path, f"http://127.0.0.1:{free_port()}")
    assert rc == 1 and report["verdict"] == "FAIL"
    assert report["claims"]["provider_ready"] is False and report["claims"]["job_completed"] is False
    assert any("comfyui_unreachable" in r for r in report["reasons"]), report["reasons"]
    assert_stopped(base, port)


def test_a_comfyui_that_is_not_loopback_is_refused_before_anything_starts(tmp_path, capsys):
    base = tmp_path / "base"
    rc = e2e.main(["e2e", "--base", str(base), "--backend-dir", str(BACKEND_DIR), "--workflows-dir", str(WORKFLOWS), "--python", PYTHON,
                   "--comfyui-url", "https://abc.trycloudflare.com", "--out", str(tmp_path / "o.mp4"), "--report", str(tmp_path / "r.json")])
    assert rc == 2 and "loopback only" in capsys.readouterr().err
    assert not (base / "logs").exists() and not (tmp_path / "r.json").exists()          # no process was started, no report written


def test_a_missing_backend_installation_points_to_install_backend(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(e2e, "backend_importable", lambda *a, **k: "ModuleNotFoundError: No module named 'fastapi'")
    rc = e2e.main(["e2e", "--base", str(tmp_path / "b"), "--backend-dir", str(BACKEND_DIR), "--workflows-dir", str(WORKFLOWS),
                   "--python", PYTHON, "--out", str(tmp_path / "o.mp4"), "--report", str(tmp_path / "r.json")])
    assert rc == 2 and "run install-backend first" in capsys.readouterr().err


def test_the_backend_process_is_started_with_a_credential_free_loopback_environment(tmp_path, fake, monkeypatch):
    """Captures what the tool really hands to the backend process (the environment and the command)."""
    monkeypatch.setenv("HF_TOKEN", "hf_" + "z" * 30)
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "AKIA" + "Q" * 16)
    seen = {}
    real = e2e.subprocess.Popen

    def spy(cmd, **kw):
        seen["cmd"], seen["env"] = cmd, kw["env"]
        return real(cmd, **kw)
    monkeypatch.setattr(e2e.subprocess, "Popen", spy)
    rc, report, base, port = run_e2e(tmp_path, fake.url)
    assert rc == 0
    assert seen["cmd"][:4] == [PYTHON, "-m", "uvicorn", "app.main:app"] and seen["cmd"][seen["cmd"].index("--host") + 1] == "127.0.0.1"
    assert "0.0.0.0" not in seen["cmd"]
    assert "HF_TOKEN" not in seen["env"] and "AWS_ACCESS_KEY_ID" not in seen["env"]
    assert not any("hf_zzz" in v or "AKIAQQQ" in v for v in seen["env"].values())


# ====================================================================== static guarantees about the tool itself
def test_the_tool_contains_no_tunnel_and_no_public_binding():
    src = SCRIPT.read_text()
    for forbidden in ("cloudflared", "trycloudflare", "ngrok", "localtunnel", "0.0.0.0", "--listen", "--share"):
        assert forbidden not in src.replace("(2) ", ""), forbidden
    assert 'loopback' in src and '"--host", "127.0.0.1"' in src


def test_the_tool_only_uses_helpers_that_exist_in_the_phase_5b_script():
    """6B reuses the verified 5B helpers; every `gt.<name>` it calls must exist (guards against a rename or removal)."""
    import re
    names = set(re.findall(r"\bgt\.([A-Za-z_]\w*)", SCRIPT.read_text()))
    assert names and not [n for n in names if not hasattr(gt, n)], sorted(n for n in names if not hasattr(gt, n))
