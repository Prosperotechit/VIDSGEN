"""Tests for scripts/wan22_gpu_test.py (the Phase 5B GPU-host tooling). No GPU, no weights, no network."""

from __future__ import annotations

import importlib.util
import json
import os
import socket
import subprocess
import sys
import textwrap
from argparse import Namespace
from pathlib import Path

import pytest

from app.core.config import REPO_ROOT
from app.providers.mock_video import encode_mock_video, ffmpeg_binary
from comfyui_support import FakeComfyUI

SCRIPT = REPO_ROOT / "scripts" / "wan22_gpu_test.py"
_spec = importlib.util.spec_from_file_location("wan22_gpu_test", SCRIPT)
g = importlib.util.module_from_spec(_spec)
sys.modules["wan22_gpu_test"] = g
_spec.loader.exec_module(g)

# ComfyUI v0.46.0 @ 995456d requirements.txt (verbatim), the file the installer must filter.
REAL_REQUIREMENTS = """comfyui-frontend-package==1.59.6
comfyui-workflow-templates==0.11.111
comfyui-embedded-docs==0.5.17
torch
torchsde
torchvision
numpy>=1.25.0
einops
transformers>=4.50.3
tokenizers>=0.13.3
sentencepiece
safetensors>=0.4.2
aiohttp>=3.11.8
yarl>=1.18.0
pyyaml
Pillow>=10.1.0
scipy
tqdm
psutil
alembic
SQLAlchemy>=2.0.0
filelock
av>=17.0.0
comfy-aimdo==0.6.0
comfy-kitchen==0.2.41
requests
simpleeval>=1.0.0
blake3

#non essential dependencies:
kornia>=0.7.1
spandrel
pydantic~=2.0
pydantic-settings~=2.0
PyOpenGL>=3.1.8
comfy-angle
"""


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def make_video(path: Path, frames=17, size=(832, 480), fps=24, kind="mock") -> Path:
    if kind == "mock":
        encode_mock_video(path, width=size[0], height=size[1], num_frames=frames, fps=fps, seed=7, prompt="smoke")
    else:
        colour = {"black": "black", "flat": "0x808080"}[kind]
        subprocess.run([ffmpeg_binary(), "-hide_banner", "-loglevel", "error", "-y", "-f", "lavfi", "-i",
                        f"color=c={colour}:s={size[0]}x{size[1]}:r={fps}", "-frames:v", str(frames), "-c:v", "libx264",
                        "-pix_fmt", "yuv420p", str(path)], check=True)
    return path


# ====================================================================== PyTorch protection
def test_filter_requirements_removes_only_the_pytorch_packages():
    kept, removed = g.filter_requirements(REAL_REQUIREMENTS)
    assert removed == ["torch", "torchvision"]
    names = [k.split("==")[0].split(">=")[0].split("~=")[0] for k in kept if k and not k.startswith("#")]
    for must_keep in ("torchsde", "numpy", "comfy-kitchen", "comfy-aimdo", "av", "comfyui-frontend-package", "pydantic"):
        assert must_keep in names
    assert not any(n in ("torch", "torchvision", "torchaudio") for n in names)


@pytest.mark.parametrize("line", ["torch", "torch==2.6.0", "torch>=2.4", "Torch", "torchvision~=0.20", "torchaudio ; python_version>'3.9'", "torch[extra]"])
def test_filter_catches_every_spelling_of_torch(line):
    assert g.filter_requirements(line)[1] == [line.strip()]


def test_filter_does_not_drop_lookalikes():
    kept, removed = g.filter_requirements("torchsde\ntorchmetrics\ntorch-fidelity\n")
    assert removed == [] and len(kept) == 3


HOST = {"python": "3.11.13", "packages": {
    "torch": ["2.6.0+cu124", "/opt/conda/lib/python3.11/site-packages/torch/__init__.py"],
    "torchvision": ["0.21.0+cu124", "/opt/conda/lib/python3.11/site-packages/torchvision/__init__.py"],
    "torchaudio": ["2.6.0+cu124", "/opt/conda/lib/python3.11/site-packages/torchaudio/__init__.py"],
    "numpy": ["1.26.4", "/opt/conda/lib/python3.11/site-packages/numpy/__init__.py"]}}
PROTECTED = ("torch", "torchvision", "torchaudio", "numpy")


def seen_with(name=None, value="same"):
    seen = json.loads(json.dumps(HOST))
    if name is not None:
        seen["packages"][name] = value
    return seen


def test_the_protected_packages_are_torch_torchvision_torchaudio_and_numpy():
    assert g.PINNED_PACKAGES == PROTECTED


def test_protected_packages_accepts_the_identical_installation(tmp_path):
    assert g.protected_packages_replaced(HOST, seen_with(), tmp_path / "venv") == []


@pytest.mark.parametrize("name", PROTECTED)
def test_a_changed_version_of_any_protected_package_is_detected(tmp_path, name):
    changed = [HOST["packages"][name][0] + ".post1", HOST["packages"][name][1]]
    problems = g.protected_packages_replaced(HOST, seen_with(name, changed), tmp_path / "venv")
    assert any(f"{name} changed" in p for p in problems), problems


@pytest.mark.parametrize("name", PROTECTED)
def test_a_different_copy_of_any_protected_package_is_detected(tmp_path, name):
    other = [HOST["packages"][name][0], f"/usr/lib/python3/dist-packages/{name}/__init__.py"]     # same version, other file
    problems = g.protected_packages_replaced(HOST, seen_with(name, other), tmp_path / "venv")
    assert any(f"a different {name}" in p for p in problems), problems


@pytest.mark.parametrize("name", PROTECTED)
def test_a_second_copy_inside_the_venv_is_detected_even_at_the_same_version(tmp_path, name):
    venv = tmp_path / "venv"
    inside = [HOST["packages"][name][0], str(venv / "lib/python3.11/site-packages" / name / "__init__.py")]
    problems = g.protected_packages_replaced(HOST, seen_with(name, inside), venv)
    assert any(f"a second {name} was installed inside the venv" in p for p in problems), problems


@pytest.mark.parametrize("name", PROTECTED)
def test_a_protected_package_the_venv_cannot_import_is_detected(tmp_path, name):
    problems = g.protected_packages_replaced(HOST, seen_with(name, None), tmp_path / "venv")
    assert any(f"{name} {HOST['packages'][name][0]} cannot be imported inside the venv" in p for p in problems), problems


def test_a_package_the_host_does_not_have_is_not_protected(tmp_path):
    host = json.loads(json.dumps(HOST)); host["packages"]["torchaudio"] = None
    seen = json.loads(json.dumps(HOST)); seen["packages"]["torchaudio"] = ["2.6.0", str(tmp_path / "venv/lib/torchaudio/__init__.py")]
    assert g.protected_packages_replaced(host, seen, tmp_path / "venv") == []


def test_import_probe_reports_each_protected_package_or_none():
    r = g.import_probe(sys.executable, dict(os.environ))
    assert list(r["packages"]) == list(PROTECTED) and r["python"] == sys.version.split()[0]
    version, file = r["packages"]["numpy"]                              # installed for the backend
    assert version and file.endswith("numpy/__init__.py")
    assert r["packages"]["torch"] is None                               # not installed in the backend venv: reported, not an error


def test_constraints_pin_the_installed_versions(tmp_path):
    p = tmp_path / "c.txt"
    g.write_constraints(p, {"torch": "2.6.0+cu124", "numpy": "1.26.4", "torchvision": "0.21.0+cu124"})
    assert p.read_text().splitlines() == ["numpy==1.26.4", "torch==2.6.0+cu124", "torchvision==0.21.0+cu124"]


# ====================================================================== secrets
def test_clean_env_drops_credentials_and_keeps_the_rest():
    env = g.clean_env({"HF_TOKEN": "x", "KAGGLE_USER_SECRETS_TOKEN": "y", "COMFYUI_API_TOKEN": "z", "AWS_SECRET_ACCESS_KEY": "q",
                       "AWS_ACCESS_KEY_ID": "a", "GIT_CONFIG_KEY_0": "g", "GIT_CONFIG_VALUE_0": "v", "GIT_ASKPASS": "ask",
                       "SERVICE_PRIVATE_THING": "p", "KAGGLE_KEY": "k", "MY_PASSWORD": "p", "PATH": "/usr/bin", "HOME": "/root",
                       "HTTPS_PROXY": "http://proxy:3128", "CUDA_VISIBLE_DEVICES": "0"}, TMPDIR="/kaggle/tmp/x")
    assert env == {"PATH": "/usr/bin", "HOME": "/root", "HTTPS_PROXY": "http://proxy:3128", "CUDA_VISIBLE_DEVICES": "0",
                   "TMPDIR": "/kaggle/tmp/x"}


def test_server_env_is_an_allowlist(tmp_path):
    source = {"PATH": "/usr/bin", "HOME": "/root", "LANG": "C.UTF-8", "LD_LIBRARY_PATH": "/usr/local/nvidia/lib64",
              "NVIDIA_VISIBLE_DEVICES": "all", "CUDA_HOME": "/usr/local/cuda", "PYTHONPATH": "/x",
              # credential-like, with names no denylist could be sure to cover:
              "HF_TOKEN": "t", "AWS_ACCESS_KEY_ID": "a", "GIT_CONFIG_VALUE_0": "extraheader=Authorization: Basic abc",
              "MYSTERY_VENDOR_CREDS": "c", "DATABASE_URL": "postgres://u:p@h/db", "HTTPS_PROXY": "http://u:p@proxy:3128",
              "KAGGLE_USER_SECRETS_TOKEN": "s", "COMFYUI_API_TOKEN": "z"}
    env = g.server_env(tmp_path, "0", source)
    assert set(env) == {"PATH", "HOME", "LANG", "LD_LIBRARY_PATH", "NVIDIA_VISIBLE_DEVICES", "CUDA_HOME", "PYTHONPATH",
                        "PIP_CACHE_DIR", "HF_HOME", "TORCH_HOME", "XDG_CACHE_HOME", "TMPDIR", "PYTHONDONTWRITEBYTECODE",
                        "CUDA_VISIBLE_DEVICES"}
    assert env["CUDA_VISIBLE_DEVICES"] == "0" and all(str(tmp_path) in env[k] for k in ("TMPDIR", "HF_HOME", "XDG_CACHE_HOME"))
    assert "CUDA_VISIBLE_DEVICES" not in g.server_env(tmp_path, "none", source)


def test_redact_hides_tokens():
    fake_hf = "hf_" + "a1b2c3d4e5" * 3                 # built at runtime: no token-shaped literal in the source tree
    text = f"Authorization: Bearer abc.def-123 and {fake_hf} token=SECRETVALUE ok"
    out = g.redact(text)
    for leaked in ("abc.def-123", fake_hf, "SECRETVALUE"):
        assert leaked not in out
    assert out.endswith("ok")


def test_cache_env_keeps_everything_under_the_base_dir(tmp_path):
    env = g.cache_env(tmp_path)
    assert all(str(tmp_path) in v for k, v in env.items() if k != "PYTHONDONTWRITEBYTECODE")
    assert {"PIP_CACHE_DIR", "HF_HOME", "TORCH_HOME", "XDG_CACHE_HOME", "TMPDIR"} <= set(env)


# ====================================================================== loopback detection
def test_listening_addresses_parses_proc_net(tmp_path):
    header = "  sl  local_address rem_address   st tx_queue rx_queue tr tm->when retrnsmt   uid  timeout inode\n"
    (tmp_path / "tcp").write_text(header
        + "   0: 0100007F:2018 00000000:0000 0A 00000000:00000000 00:00000000 00000000     0        0 1\n"   # 127.0.0.1:8216
        + "   1: 00000000:1F90 00000000:0000 0A 00000000:00000000 00:00000000 00000000     0        0 2\n"   # 0.0.0.0:8080
        + "   2: 0100007F:2018 0100007F:9999 01 00000000:00000000 00:00000000 00000000     0        0 3\n")  # ESTABLISHED: ignored
    (tmp_path / "tcp6").write_text(header
        + "   0: 00000000000000000000000001000000:2018 00000000000000000000000000000000:0000 0A 0:0 0:0 0 0 0 4\n"
        + "   1: 00000000000000000000000000000000:1F90 00000000000000000000000000000000:0000 0A 0:0 0:0 0 0 0 5\n")
    assert g.listening_addresses(0x2018, tmp_path) == ["127.0.0.1", "::1"]
    assert g.listening_addresses(0x1F90, tmp_path) == ["0.0.0.0", "::"]
    assert all(g.is_loopback(a) for a in ["127.0.0.1", "::1"]) and not any(g.is_loopback(a) for a in ["0.0.0.0", "::"])


@pytest.mark.parametrize("host,loopback", [("127.0.0.1", True), ("0.0.0.0", False)])
def test_listening_addresses_on_a_real_socket(host, loopback):
    if not Path("/proc/net/tcp").exists():
        pytest.skip("needs /proc/net")
    with socket.socket() as s:
        s.bind((host, 0)); s.listen(1)
        addrs = g.listening_addresses(s.getsockname()[1])
    assert addrs and all(g.is_loopback(a) for a in addrs) is loopback


# ====================================================================== MP4 verification
def test_probe_accepts_a_good_video(tmp_path):
    r = g.probe_mp4(make_video(tmp_path / "ok.mp4"), frames=17, size=(832, 480), fps=24.0)
    assert r["problems"] == [] and r["codec"] == "h264" and "mp4" in r["container"] and r["decoded_frames"] == 17
    assert (r["width"], r["height"]) == (832, 480) and r["luma_std_max"] > 1 and r["pix_fmt"] == "yuv420p"


@pytest.mark.parametrize("video_kwargs,expect_kwargs,fragment", [
    (dict(frames=9), {}, "decoded 9 frames, expected 17"),
    (dict(), dict(size=(640, 640)), "size is 832x480, expected 640x640"),
    (dict(kind="black"), {}, "every frame is black"),
    (dict(kind="flat"), {}, "flat single colour"),
])
def test_probe_catches_bad_videos(tmp_path, video_kwargs, expect_kwargs, fragment):
    video = make_video(tmp_path / "v.mp4", **video_kwargs)
    r = g.probe_mp4(video, **{"frames": 17, "size": (832, 480), **expect_kwargs})
    assert any(fragment in p for p in r["problems"]), r["problems"]


def test_probe_catches_broken_files(tmp_path):
    good = make_video(tmp_path / "good.mp4")
    data = good.read_bytes()
    (tmp_path / "truncated.mp4").write_bytes(data[: len(data) // 2])
    (tmp_path / "garbage.mp4").write_bytes(b"x" * 5000)
    (tmp_path / "tiny.mp4").write_bytes(b"abc")
    assert g.probe_mp4(tmp_path / "missing.mp4")["problems"] == ["file is missing or implausibly small"]
    assert g.probe_mp4(tmp_path / "tiny.mp4")["problems"] == ["file is missing or implausibly small"]
    assert any("not an MP4" in p for p in g.probe_mp4(tmp_path / "garbage.mp4")["problems"])
    t = g.probe_mp4(tmp_path / "truncated.mp4", frames=17)
    assert t["problems"], "a truncated MP4 must not pass"


# ====================================================================== verdict logic
GOOD_PROBE = {"exists": True, "opened": True, "problems": []}
CPU_STATS = {"gpu_sampling": "unavailable"}
GPU_STATS = {"gpu_sampling": "nvidia-smi"}


@pytest.mark.parametrize("res,probe,stats,gpu,expect,fragment", [
    ({"queued": True, "final_state": "completed"}, GOOD_PROBE, GPU_STATS, True, "PASS", None),
    ({"queued": True, "final_state": "completed"}, GOOD_PROBE, CPU_STATS, False, "PASS", None),
    # "only the server started" / "the workflow queued": never a pass
    ({"queued": False, "error": {"code": "provider_not_ready"}}, None, GPU_STATS, True, "FAIL", "never queued"),
    ({"queued": True, "final_state": None}, None, GPU_STATS, True, "FAIL", "ended as None"),
    ({"queued": True, "final_state": "running"}, None, GPU_STATS, True, "FAIL", "ended as 'running'"),
    ({"queued": True, "final_state": "queued"}, None, GPU_STATS, True, "FAIL", "ended as 'queued'"),
    ({"queued": True, "final_state": "failed", "error": {"code": "gpu_out_of_memory"}}, None, GPU_STATS, True, "FAIL", "gpu_out_of_memory"),
    ({"queued": True, "final_state": "cancelled"}, None, GPU_STATS, True, "FAIL", "cancelled"),
    # completed but nothing usable came out
    ({"queued": True, "final_state": "completed"}, None, GPU_STATS, True, "FAIL", "no output file"),
    ({"queued": True, "final_state": "completed"}, {**GOOD_PROBE, "problems": ["every frame is black"]}, GPU_STATS, True, "FAIL", "black"),
    # cannot measure VRAM on a GPU run
    ({"queued": True, "final_state": "completed"}, GOOD_PROBE, CPU_STATS, True, "FAIL", "GPU memory could not be measured"),
])
def test_verdict_never_claims_success_without_a_real_video(res, probe, stats, gpu, expect, fragment):
    verdict, reasons = g.evaluate(res, probe, stats, gpu)
    assert verdict == expect
    assert (not reasons) if expect == "PASS" else any(fragment in r for r in reasons), reasons


def test_step_timing():
    ups = [{"t": 1.0, "state": "queued", "message": "queued"}, {"t": 30.0, "state": "running", "message": "step 1/20"},
           {"t": 50.0, "state": "running", "message": "step 11/20"}, {"t": 70.0, "state": "running", "message": "step 20/20"}]
    t = g.step_timing(ups)
    assert t["seconds_per_step"] == round(40 / 19, 2) and t["first_step_at_s"] == 30.0 and t["steps_total"] == 20


# ====================================================================== smoke test vs the fake ComfyUI
@pytest.fixture
def smoke_env(tmp_path):
    started = []

    def make(video: Path | None = None, scenario: str = "success", **overrides):
        v = video or make_video(tmp_path / "served.mp4")
        fake = FakeComfyUI(v.read_bytes(), token=None).start()
        fake.default_scenario = scenario
        started.append(fake)
        args = Namespace(base=None, url=fake.url, backend_dir=str(REPO_ROOT / "backend"), workflows_dir=str(REPO_ROOT / "workflows/comfyui"),
                         resolution="832x480", frames=17, fps=24, steps=8, cfg=5.0, seed=42, prompt=g.DEFAULT_PROMPT, gpu=0,
                         timeout=60, require_gpu=False, out=str(tmp_path / "out" / "smoke.mp4"), report=str(tmp_path / "report.json"))
        for k, val in overrides.items():
            setattr(args, k, val)
        return fake, args

    yield make
    for f in started:
        f.stop()


def run_smoke(args) -> tuple[int, dict]:
    code = g.smoke(args)
    return code, json.loads(Path(args.report).read_text())


def test_smoke_passes_only_with_a_real_decodable_video(smoke_env, capsys):
    fake, args = smoke_env()
    code, rep = run_smoke(args)
    assert code == 0 and rep["verdict"] == "PASS" and rep["reasons"] == []
    assert all(rep["claims_checked"].values())
    assert rep["output"]["decoded_frames"] == 17 and rep["output"]["codec"] == "h264" and (rep["output"]["width"], rep["output"]["height"]) == (832, 480)
    assert Path(args.out).read_bytes()[4:8] == b"ftyp"
    assert rep["timing"]["generation_wall_s"] is not None and rep["timing"]["steps_total"] == 8
    assert rep["resources"]["gpu_sampling"] == "unavailable" and rep["resources"]["ram_peak_used_gib"] is not None
    # honest health: before = unverified; after the real generation = verified
    before = {c["name"]: c["status"] for c in rep["health_before"]["checks"]}
    after = {c["name"]: c["status"] for c in rep["health_after"]["checks"]}
    assert before["inference"] == "unverified" and after["inference"] == "ok"
    # the request used the real workflow and the requested values
    graph = [r["body"] for r in fake.requests if r["path"] == "/prompt"][0]["prompt"]
    assert graph["55"]["inputs"]["length"] == 17 and (graph["55"]["inputs"]["width"], graph["55"]["inputs"]["height"]) == (832, 480)
    assert "56" not in graph                                            # text-to-video: no image
    captured = capsys.readouterr()
    assert '"verdict": "PASS"' in captured.out
    assert "completed" in captured.err and "100.0%" in captured.err          # live heartbeat lines for long runs


@pytest.mark.parametrize("scenario,fragment", [
    ("error", "comfyui_execution_error"),
    ("oom", "gpu_out_of_memory"),
    ("no_output", "no_video_output"),
])
def test_smoke_fails_when_generation_fails(smoke_env, scenario, fragment):
    _, args = smoke_env(scenario=scenario)
    code, rep = run_smoke(args)
    assert code == 1 and rep["verdict"] == "FAIL" and any(fragment in r for r in rep["reasons"])
    assert rep["claims_checked"]["workflow_queued"] and not rep["claims_checked"]["generation_completed"]
    assert not rep["claims_checked"]["mp4_exists"] and rep["output"] is None


@pytest.mark.parametrize("make_kwargs,fragment", [
    (dict(frames=9), "decoded 9 frames, expected 17"),
    (dict(size=(640, 640)), "size is 640x640, expected 832x480"),
    (dict(kind="black"), "every frame is black"),
    (dict(kind="flat"), "flat single colour"),
])
def test_smoke_fails_when_the_video_is_wrong_even_though_comfyui_said_success(tmp_path, smoke_env, make_kwargs, fragment):
    bad = make_video(tmp_path / "bad.mp4", **make_kwargs)
    _, args = smoke_env(video=bad)
    code, rep = run_smoke(args)
    assert rep["claims_checked"]["generation_completed"] is True       # ComfyUI reported success ...
    assert code == 1 and rep["verdict"] == "FAIL"                       # ... but we do not
    assert any(fragment in r for r in rep["reasons"]), rep["reasons"]


def test_smoke_fails_fast_when_the_server_is_not_ready(smoke_env):
    fake, args = smoke_env()
    fake.models["vae"] = []                                             # a model file is missing
    code, rep = run_smoke(args)
    assert code == 1 and rep["claims_checked"]["workflow_queued"] is False
    assert any("never queued" in r and "provider_not_ready" in r for r in rep["reasons"])
    assert [r for r in fake.requests if r["path"] == "/prompt"] == []   # nothing was submitted


def test_smoke_reports_a_missing_dependency_cleanly(smoke_env, monkeypatch):
    _, args = smoke_env()

    async def broken(*a, **k):
        raise ImportError("No module named 'pydantic_settings'")
    monkeypatch.setattr(g, "run_provider", broken)
    code, rep = run_smoke(args)
    assert code == 1 and rep["verdict"] == "FAIL" and rep["claims_checked"]["workflow_queued"] is False
    assert any("missing_dependency" in r and "venv" in r for r in rep["reasons"])


def test_smoke_requires_gpu_measurements_when_asked(smoke_env):
    _, args = smoke_env(require_gpu=True)
    code, rep = run_smoke(args)
    assert code == 1 and any("GPU memory could not be measured" in r for r in rep["reasons"])
    assert rep["claims_checked"]["generation_completed"] is True        # the video itself was fine


def test_smoke_report_never_contains_secrets(smoke_env, monkeypatch):
    monkeypatch.setenv("COMFYUI_API_TOKEN", "super-secret-token-value")
    fake, args = smoke_env()
    fake.token = "super-secret-token-value"
    code, rep = run_smoke(args)
    assert code == 0
    assert "super-secret-token-value" not in Path(args.report).read_text()


# ====================================================================== server lifecycle (fake ComfyUI process)
FAKE_MAIN = textwrap.dedent('''
    import http.server, json, os, sys
    # Control comes from a file, not the environment: the server environment is an allowlist.
    cfg = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "fake.json")))
    a = sys.argv[1:]
    listen = a[a.index("--listen") + 1] if "--listen" in a else "0.0.0.0"
    port = int(a[a.index("--port") + 1])
    if cfg.get("ignore_listen"):
        listen = "0.0.0.0"
    if cfg.get("crash"):
        print("boom: Authorization: Bearer leaked-token-value", file=sys.stderr); sys.exit(3)
    open(cfg["dump"], "w").write(json.dumps({"argv": a, "env": dict(os.environ), "cwd": os.getcwd()}))
    class H(http.server.BaseHTTPRequestHandler):
        def log_message(self, *x): ...
        def do_GET(self):
            self.send_response(200); self.send_header("Content-Length", "2"); self.end_headers(); self.wfile.write(b"{}")
    http.server.HTTPServer((listen, port), H).serve_forever()
''')


def configure_fake(base: Path, **cfg) -> None:
    (base / "ComfyUI" / "fake.json").write_text(json.dumps({"dump": str(base / "dump.json"), **cfg}))


@pytest.fixture
def fake_install(tmp_path):
    base = tmp_path / "gpu"
    (base / "ComfyUI").mkdir(parents=True)
    (base / "ComfyUI" / "main.py").write_text(FAKE_MAIN)
    (base / "venv" / "bin").mkdir(parents=True)
    os.symlink(sys.executable, base / "venv" / "bin" / "python")
    configure_fake(base)
    yield base
    g.stop(base, quiet=True)


def test_serve_binds_loopback_only_on_one_gpu_with_a_clean_environment(fake_install, monkeypatch):
    dump = fake_install / "dump.json"
    for secret in ("HF_TOKEN", "KAGGLE_USER_SECRETS_TOKEN", "COMFYUI_API_TOKEN", "AWS_ACCESS_KEY_ID", "MYSTERY_VENDOR_CREDS",
                   "GIT_CONFIG_VALUE_0", "DATABASE_URL"):
        monkeypatch.setenv(secret, "must-not-reach-comfyui")
    port = free_port()
    info = g.serve(fake_install, port, "0", ["--lowvram"], wait=30)
    assert info["loopback_only"] is True and info["listening_on"] == ["127.0.0.1"] and info["cuda_visible_devices"] == "0"
    seen = json.loads(dump.read_text())
    argv = seen["argv"]
    assert argv[argv.index("--listen") + 1] == "127.0.0.1" and argv[argv.index("--port") + 1] == str(port)
    assert argv[argv.index("--cuda-device") + 1] == "0" and "--lowvram" in argv and "--disable-auto-launch" in argv
    assert argv[argv.index("--base-directory") + 1] == str(fake_install / "comfy-data")
    assert seen["env"]["CUDA_VISIBLE_DEVICES"] == "0" and seen["cwd"] == str(fake_install / "ComfyUI")
    assert "must-not-reach-comfyui" not in json.dumps(seen["env"]), "a credential reached the ComfyUI process"
    assert not [k for k in seen["env"] if not g.SERVER_ENV_ALLOW.match(k)], "a variable outside the allowlist reached ComfyUI"
    for key in ("TMPDIR", "HF_HOME", "XDG_CACHE_HOME", "TORCH_HOME", "PIP_CACHE_DIR"):
        assert str(fake_install) in seen["env"][key]                    # every cache lives under the base dir
    for sub in ("models/diffusion_models", "models/text_encoders", "models/vae", "custom_nodes", "input", "output"):
        assert (fake_install / "comfy-data" / sub).is_dir()
    assert g.http_get(f"http://127.0.0.1:{port}/system_stats")[0] == 200
    assert g.stop(fake_install) is True
    with pytest.raises(OSError):
        g.http_get(f"http://127.0.0.1:{port}/system_stats", timeout=1)


def test_serve_refuses_and_stops_a_server_that_is_not_loopback_only(fake_install, monkeypatch):
    configure_fake(fake_install, ignore_listen=True)                          # a server that binds 0.0.0.0 regardless
    port = free_port()
    with pytest.raises(RuntimeError, match="not loopback only"):
        g.serve(fake_install, port, "0", [], wait=30)
    with pytest.raises(OSError):
        g.http_get(f"http://127.0.0.1:{port}/system_stats", timeout=1)  # it was stopped


def test_serve_reports_a_crash_with_a_redacted_log(fake_install, monkeypatch):
    configure_fake(fake_install, crash=True)
    with pytest.raises(RuntimeError) as e:
        g.serve(fake_install, free_port(), "0", [], wait=30)
    assert "exited with code 3" in str(e.value) and "boom" in str(e.value)
    assert "leaked-token-value" not in str(e.value) and "Bearer [redacted]" in str(e.value)


def test_serve_requires_an_installation(tmp_path):
    with pytest.raises(RuntimeError, match="run install-comfyui first"):
        g.serve(tmp_path, free_port(), "0", [], wait=5)


def test_comfy_args_cpu_mode_has_no_cuda_device(tmp_path):
    args = g.comfy_args(tmp_path, 8188, "none", ["--cpu"])
    assert "--cuda-device" not in args and args[:2] == ["--listen", "127.0.0.1"] and args[-1] == "--cpu"


# ====================================================================== pinned checkout
def _git(cwd, *a):
    return subprocess.run(["git", *a], cwd=cwd, check=True, capture_output=True, text=True,
                          env={**os.environ, "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@t", "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@t"}).stdout.strip()


@pytest.fixture
def upstream(tmp_path):
    repo = tmp_path / "upstream"
    repo.mkdir()
    _git(repo, "init", "-q", "-b", "master")
    _git(repo, "config", "uploadpack.allowAnySHA1InWant", "true")
    (repo / "comfyui_version.py").write_text('__version__ = "0.46.0"\n')
    (repo / "requirements.txt").write_text("torch\neinops\n")
    _git(repo, "add", "-A"); _git(repo, "commit", "-q", "-m", "pinned")
    pinned = _git(repo, "rev-parse", "HEAD")
    (repo / "comfyui_version.py").write_text('__version__ = "0.47.0"\n')
    _git(repo, "commit", "-q", "-am", "newer")
    return repo, pinned, _git(repo, "rev-parse", "HEAD")


def test_checkout_fetches_exactly_the_pinned_commit_not_the_tip(tmp_path, upstream):
    repo, pinned, tip = upstream
    dest = tmp_path / "ComfyUI"
    assert g.checkout_commit(f"file://{repo}", pinned, dest) == pinned
    assert g.comfyui_version(dest) == "0.46.0"                          # not the newer 0.47.0 at the tip
    assert _git(dest, "rev-parse", "HEAD") == pinned and _git(dest, "status", "--porcelain") == ""
    assert g.checkout_commit(f"file://{repo}", pinned, dest) == pinned  # idempotent


def test_checkout_fails_for_an_unknown_commit(tmp_path, upstream):
    repo, _, _ = upstream
    with pytest.raises(RuntimeError, match="git fetch"):
        g.checkout_commit(f"file://{repo}", "1" * 40, tmp_path / "x")


def test_install_rejects_a_version_mismatch_and_a_dirty_checkout(tmp_path, upstream):
    repo, pinned, _ = upstream
    with pytest.raises(RuntimeError, match="reports version 0.46.0, expected 0.99.0"):
        g.install_comfyui(tmp_path / "a", f"file://{repo}", pinned, "0.99.0")
    base = tmp_path / "b"
    g.checkout_commit(f"file://{repo}", pinned, base / "ComfyUI")
    (base / "ComfyUI" / "requirements.txt").write_text("torch\nsomething-else\n")   # local modification
    with pytest.raises(RuntimeError, match="local modifications"):
        g.install_comfyui(base, f"file://{repo}", pinned, "0.46.0")


# ====================================================================== live template validation
def test_check_templates_passes_against_a_matching_server(smoke_env):
    fake, args = smoke_env()
    r = g.check_templates(fake.url, REPO_ROOT / "backend", REPO_ROOT / "workflows/comfyui")
    assert r["ok"] is True and r["node_classes_missing"] == []
    assert r["templates"] == {"t2v": {"problems": [], "ok": True}, "i2v": {"problems": [], "ok": True}}
    assert len(r["node_classes_required"]) == 11


def test_check_templates_reports_a_missing_node_class(smoke_env):
    fake, args = smoke_env()
    fake.missing_nodes = {"Wan22ImageToVideoLatent"}
    r = g.check_templates(fake.url, REPO_ROOT / "backend", REPO_ROOT / "workflows/comfyui")
    assert r["ok"] is False and r["node_classes_missing"] == ["Wan22ImageToVideoLatent"]
    assert all(not t["ok"] for t in r["templates"].values())


def test_check_templates_cli_exit_code(smoke_env):
    fake, _ = smoke_env()
    cmd = [sys.executable, str(SCRIPT), "check-templates", "--url", fake.url, "--backend-dir", str(REPO_ROOT / "backend"),
           "--workflows-dir", str(REPO_ROOT / "workflows/comfyui")]
    ok = subprocess.run(cmd, capture_output=True, text=True, cwd=REPO_ROOT / "backend")
    assert ok.returncode == 0 and json.loads(ok.stdout)["ok"] is True
    fake.missing_nodes = {"SaveVideo"}
    bad = subprocess.run(cmd, capture_output=True, text=True, cwd=REPO_ROOT / "backend")
    assert bad.returncode == 1 and json.loads(bad.stdout)["node_classes_missing"] == ["SaveVideo"]


def test_gpu_probe_on_a_machine_without_a_gpu():
    r = g.gpu_probe(0)
    assert r["nvidia_smi"] is None and r["devices"] == []


def test_cli_verify_mp4_exit_codes(tmp_path):
    ok, bad = make_video(tmp_path / "ok.mp4"), make_video(tmp_path / "black.mp4", kind="black")
    run = lambda f: subprocess.run([sys.executable, str(SCRIPT), "verify-mp4", str(f), "--frames", "17", "--size", "832x480"], capture_output=True, text=True)
    assert run(ok).returncode == 0
    r = run(bad)
    assert r.returncode == 1 and "every frame is black" in r.stdout


# ====================================================================== preflight: authoritative Hugging Face metadata
import threading
from collections import namedtuple
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import yaml

MANIFEST = yaml.safe_load((REPO_ROOT / "workflows/comfyui/models.yaml").read_text())
XET = {f["role"]: ("e" * 63 + str(i)) for i, f in enumerate(MANIFEST["files"])}      # fake Xet hashes (a different identifier)


class FakeHfApi:
    """Serves the metadata endpoints the preflight reads, using the REAL manifest values."""

    def __init__(self):
        self.oid_override: dict[str, str] = {}
        self.etag_override: dict[str, str] = {}
        self.size_override: dict[str, int] = {}
        self.unknown_revisions: set[str] = set()
        self.pypi_status = 200
        api = self
        files = {(f["repo"], f["path"]): f for f in MANIFEST["files"]}

        class H(BaseHTTPRequestHandler):
            def log_message(self, *a): ...

            def _send(self, code, body=b"", headers=None):
                self.send_response(code)
                for k, v in (headers or {}).items():
                    self.send_header(k, v)
                self.send_header("Content-Length", str(len(body)) if self.command != "HEAD" else "0")
                self.end_headers()
                if self.command != "HEAD":
                    self.wfile.write(body)

            def do_GET(self):
                if self.path.startswith("/simple/pip"):
                    return self._send(api.pypi_status)
                parts = self.path.split("/")      # '', api, models, org, name, revision, rev
                if len(parts) == 7 and parts[5] == "revision" and parts[6] not in api.unknown_revisions:
                    return self._send(200, json.dumps({"id": f"{parts[3]}/{parts[4]}", "sha": parts[6]}).encode())
                self._send(404)

            def do_POST(self):
                parts = self.path.split("/")      # '', api, models, org, name, paths-info, rev
                form = dict(x.split("=", 1) for x in self.rfile.read(int(self.headers["Content-Length"])).decode().split("&"))
                path = urllib_unquote(form["paths"])
                f = files.get((f"{parts[3]}/{parts[4]}", path))
                if not f:
                    return self._send(404)
                lfs = {"oid": api.oid_override.get(f["role"], f["sha256"]), "size": api.size_override.get(f["role"], f["size_bytes"]), "pointerSize": 135}
                self._send(200, json.dumps([{"path": path, "type": "file", "lfs": lfs, "xetHash": XET[f["role"]], "oid": "b" * 40}]).encode())

            def do_HEAD(self):
                parts = self.path.split("/", 5)   # '', org, name, resolve, rev, path
                f = files.get((f"{parts[1]}/{parts[2]}", urllib_unquote(parts[5]))) if len(parts) > 5 else None
                if not f:
                    return self._send(404)
                self._send(302, headers={"Location": "http://cdn.invalid/x", "x-linked-size": str(api.size_override.get(f["role"], f["size_bytes"])),
                                         "x-linked-etag": f'"{api.etag_override.get(f["role"], f["sha256"])}"', "x-xet-hash": XET[f["role"]]})

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), H)
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}"
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def close(self):
        self.server.shutdown(); self.server.server_close()


def urllib_unquote(s: str) -> str:
    import urllib.parse
    return urllib.parse.unquote(s)


@pytest.fixture
def hf():
    api = FakeHfApi()
    yield api
    api.close()


ROW = {r["role"]: r for r in MANIFEST["files"]}


def test_hf_metadata_confirms_the_real_manifest_values(hf):
    for role, row in ROW.items():
        status, detail = g.hf_metadata_check(hf.url, row)
        assert status == "PASS" and "confirmed by 2 sources" in detail and row["sha256"][:16] in detail, (role, detail)


def test_hf_metadata_requires_both_sources_to_agree(hf):
    hf.oid_override["vae"] = "0" * 64
    st, detail = g.hf_metadata_check(hf.url, ROW["vae"])
    assert st == "FAIL" and "paths-info SHA-256" in detail
    hf.oid_override.clear(); hf.etag_override["vae"] = "1" * 64          # only the HEAD header disagrees
    st, detail = g.hf_metadata_check(hf.url, ROW["vae"])
    assert st == "FAIL" and "x-linked-etag" in detail


def test_a_manifest_that_used_the_xet_hash_as_sha256_is_caught(hf):
    wrong = {**ROW["vae"], "sha256": XET["vae"]}
    st, detail = g.hf_metadata_check(hf.url, wrong)
    assert st == "FAIL" and "disagrees with the manifest" in detail


def test_hf_metadata_detects_size_and_revision_problems(hf):
    hf.size_override["text_encoder"] = 123
    assert "size 123" in g.hf_metadata_check(hf.url, ROW["text_encoder"])[1]
    hf.unknown_revisions.add(ROW["vae"]["revision"])
    st, detail = g.hf_metadata_check(hf.url, ROW["vae"])
    assert st == "FAIL" and "pinned revision" in detail and "not found" in detail


def test_hf_metadata_unreachable_is_a_failure():
    st, detail = g.hf_metadata_check("http://127.0.0.1:9", ROW["vae"])
    assert st == "FAIL" and "could not read Hugging Face metadata" in detail


# ====================================================================== preflight: disk and GPU gates
def test_disk_requirement_counts_only_what_is_still_missing(tmp_path):
    comfy = tmp_path / "comfy-data"
    total = sum(f["size_bytes"] for f in MANIFEST["files"])
    assert g.disk_requirement(MANIFEST, comfy, 8) == (total, total + 8 * g.GIB)
    vae = comfy / ROW["vae"]["comfyui_dir"] / Path(ROW["vae"]["path"]).name
    vae.parent.mkdir(parents=True)
    vae.write_bytes(b"x" * 10)                                          # wrong size: still needed in full
    assert g.disk_requirement(MANIFEST, comfy, 0)[0] == total
    part = vae.with_name(vae.name + ".part")
    part.write_bytes(b"x" * 1000)                                       # a partial download counts as progress
    assert g.disk_requirement(MANIFEST, comfy, 0)[0] == total - 1000


T4 = {"devices": [{"index": 0, "name": "Tesla T4", "capability": "7.5", "total_bytes": 15 * g.GIB, "free_bytes": 14.5 * g.GIB},
                  {"index": 1, "name": "Tesla T4", "capability": "7.5", "total_bytes": 15 * g.GIB, "free_bytes": 14.5 * g.GIB}],
      "torch": {"version": "2.6.0+cu124", "cuda_build": "12.4", "cuda_available": True, "arch_list": ["sm_50", "sm_75", "sm_80", "sm_90"]},
      "driver_cuda": "12.6", "fp16_matmul_ok": True, "bf16_native": False}


def gstat(probe, **kw):
    return {name: st for name, st, _ in g.gpu_checks(probe, kw.pop("gpu", 0), kw.pop("min_vram_gib", 14.0), kw.pop("allow_no_gpu", False))}


def test_gpu_checks_pass_on_a_kaggle_t4_x2_and_use_one_gpu():
    s = gstat(T4)
    assert s == {"gpu_detected": "PASS", "gpu_selection": "INFO", "gpu_vram": "PASS", "torch_supports_gpu": "PASS",
                 "fp16_kernel": "PASS", "cuda_driver": "PASS", "bf16": "INFO"}
    detail = {n: d for n, _, d in g.gpu_checks(T4, 0, 14.0, False)}
    assert "VRAM is NOT pooled" in detail["gpu_selection"] and "GPU 0 only" in detail["gpu_selection"]
    assert "Tesla T4" in detail["gpu_detected"] and "GPU1" in detail["gpu_detected"]


@pytest.mark.parametrize("mutate,check,expected", [
    (lambda p: p.update(devices=[]), "gpu_detected", "FAIL"),
    (lambda p: p["devices"][0].update(total_bytes=8 * g.GIB), "gpu_vram", "FAIL"),
    (lambda p: p["devices"][0].update(total_bytes=13 * g.GIB), "gpu_vram", "WARN"),
    (lambda p: p["torch"].update(arch_list=["sm_80", "sm_90"]), "torch_supports_gpu", "FAIL"),
    (lambda p: p.update(fp16_matmul_ok=False), "fp16_kernel", "FAIL"),
    (lambda p: p.update(driver_cuda="11.8"), "cuda_driver", "WARN"),
])
def test_gpu_checks_gates(mutate, check, expected):
    probe = json.loads(json.dumps(T4))
    mutate(probe)
    assert gstat(probe)[check] == expected


def test_gpu_checks_selecting_a_missing_gpu_fails_and_no_gpu_can_be_downgraded():
    assert gstat(T4, gpu=5)["gpu_selected"] == "FAIL"
    assert gstat({"devices": [], "torch_error": "No module named torch"}, allow_no_gpu=True)["gpu_detected"] == "WARN"


# ====================================================================== preflight: composed
def small_manifest(tmp_path: Path) -> Path:
    m = json.loads(json.dumps(MANIFEST))
    for f in m["files"]:
        f["size_bytes"] = 1000
    p = tmp_path / "small.yaml"
    p.write_text(yaml.safe_dump(m))
    return p


def run_preflight(tmp_path, hf=None, base=None, **kw):
    opts = dict(gpu=0, min_ram_gib=1, overhead_gib=0.001, offline=hf is None, probe=json.loads(json.dumps(T4)))
    if hf is not None:
        opts.update(endpoint=hf.url, pypi_url=hf.url + "/simple/pip/")
    opts.update(kw)
    return g.preflight(base or tmp_path / "tmp" / "vidsgen-gpu", small_manifest(tmp_path), **opts)


def statuses(r):
    return {c["name"]: c["status"] for c in r["checks"]}


def test_preflight_ok_on_a_healthy_host_with_verified_metadata(tmp_path, hf, monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "never-printed-token-value")
    # the small manifest has other sizes, so serve those from the fake API as well
    for f in MANIFEST["files"]:
        hf.size_override[f["role"]] = 1000
    r = run_preflight(tmp_path, hf)
    s = statuses(r)
    assert r["ok"] is True and r["counts"]["FAIL"] == 0
    assert s["python"] == s["install_location"] == s["disk_space"] == s["system_ram"] == s["pypi"] == "PASS"
    assert [s[f"hf_metadata {role}"] for role in ("diffusion_model", "vae", "text_encoder")] == ["PASS"] * 3
    assert "never-printed-token-value" not in json.dumps(r)


def test_preflight_stops_when_hash_metadata_disagrees(tmp_path, hf):
    for f in MANIFEST["files"]:
        hf.size_override[f["role"]] = 1000
    hf.oid_override["diffusion_model"] = "f" * 64
    r = run_preflight(tmp_path, hf)
    assert r["ok"] is False and statuses(r)["hf_metadata diffusion_model"] == "FAIL"


def test_preflight_stops_without_internet(tmp_path, hf):
    hf.pypi_status = 503
    for f in MANIFEST["files"]:
        hf.size_override[f["role"]] = 1000
    r = run_preflight(tmp_path, hf)
    assert r["ok"] is False and statuses(r)["pypi"] == "FAIL"


def test_preflight_refuses_kaggle_working_as_the_install_location(tmp_path):
    r = run_preflight(tmp_path, base=Path("/kaggle/working/vidsgen-gpu"))
    assert r["ok"] is False and statuses(r)["install_location"] == "FAIL"
    assert "disk_space" not in statuses(r)


def test_preflight_stops_when_the_disk_is_too_small(tmp_path, monkeypatch):
    usage = namedtuple("usage", "total used free")
    monkeypatch.setattr(g.shutil, "disk_usage", lambda p: usage(30 * g.GIB, 29 * g.GIB, 1 * g.GIB))
    r = run_preflight(tmp_path, overhead_gib=8)
    assert r["ok"] is False and statuses(r)["disk_space"] == "FAIL"
    assert "need 8.0 GiB" in next(c["detail"] for c in r["checks"] if c["name"] == "disk_space")


def test_preflight_stops_when_there_is_too_little_ram(tmp_path):
    assert statuses(run_preflight(tmp_path, min_ram_gib=10_000))["system_ram"] == "FAIL"


def test_preflight_cli_exit_codes_on_a_machine_without_a_gpu(tmp_path):
    base = ["--base", str(tmp_path / "tmp" / "gpu"), "--manifest", str(small_manifest(tmp_path)), "--offline", "--min-ram-gib", "1", "--overhead-gib", "0.001"]
    stop = subprocess.run([sys.executable, str(SCRIPT), "preflight", *base], capture_output=True, text=True)
    assert stop.returncode == 1 and "[FAIL] gpu_detected" in stop.stdout and "STOP: fix the FAIL items first" in stop.stdout
    ok = subprocess.run([sys.executable, str(SCRIPT), "preflight", *base, "--allow-no-gpu", "--report", str(tmp_path / "pf.json")], capture_output=True, text=True)
    assert ok.returncode == 0 and "OK to continue" in ok.stdout and json.loads((tmp_path / "pf.json").read_text())["ok"] is True


# ====================================================================== out-of-memory fallback (exactly one retry)
@pytest.fixture
def fallback_env(tmp_path, monkeypatch):
    calls: list = []
    outcomes: list = []

    def fake_smoke_once(args):
        code, error = outcomes.pop(0)
        Path(args.report).write_text(json.dumps({"verdict": "PASS" if code == 0 else "FAIL", "error": error}))
        calls.append(("smoke", None))
        return code

    monkeypatch.setattr(g, "smoke_once", fake_smoke_once)
    monkeypatch.setattr(g, "stop", lambda base, quiet=False: calls.append(("stop", None)))
    monkeypatch.setattr(g, "serve", lambda base, port, gpu, extra, wait: calls.append(("serve", list(extra))) or {"args": ["--listen", "127.0.0.1", *extra]})
    args = Namespace(base=str(tmp_path), report=str(tmp_path / "r.json"), gpu=0, port=8188, fallback_args=["--disable-dynamic-vram", "--disable-smart-memory"])
    return args, calls, outcomes


OOM = {"code": "gpu_out_of_memory", "message": "The GPU ran out of memory."}


def test_oom_triggers_one_restart_with_fallback_args_and_a_second_attempt(fallback_env):
    args, calls, outcomes = fallback_env
    outcomes += [(1, OOM), (0, None)]
    assert g.smoke(args) == 0
    assert calls == [("smoke", None), ("stop", None), ("serve", ["--disable-dynamic-vram", "--disable-smart-memory"]), ("smoke", None)]
    final = json.loads(Path(args.report).read_text())
    assert final["verdict"] == "PASS" and final["retry"]["first_attempt_error"] == OOM
    assert final["retry"]["fallback_args"] == ["--disable-dynamic-vram", "--disable-smart-memory"]
    assert json.loads((Path(args.report).with_name("r_attempt1.json")).read_text())["error"] == OOM     # the first attempt is kept


def test_a_second_oom_is_reported_and_never_retried_again(fallback_env):
    args, calls, outcomes = fallback_env
    outcomes += [(1, OOM), (1, OOM)]
    assert g.smoke(args) == 1
    assert [c[0] for c in calls].count("smoke") == 2 and [c[0] for c in calls].count("serve") == 1


@pytest.mark.parametrize("outcome", [(0, None), (1, {"code": "comfyui_execution_error", "message": "x"}), (1, {"code": "provider_not_ready", "message": "x"})])
def test_no_retry_unless_the_failure_was_out_of_memory(fallback_env, outcome):
    args, calls, outcomes = fallback_env
    outcomes.append(outcome)
    assert g.smoke(args) == outcome[0]
    assert calls == [("smoke", None)]


def test_no_retry_without_fallback_args(fallback_env):
    args, calls, outcomes = fallback_env
    args.fallback_args = []
    outcomes.append((1, OOM))
    assert g.smoke(args) == 1 and calls == [("smoke", None)]


def test_fallback_args_string_is_parsed_through_the_real_cli(tmp_path, monkeypatch):
    seen = {}
    monkeypatch.setattr(g, "smoke", lambda a: seen.update(fb=a.fallback_args, port=a.port) or 0)
    base = ["smoke-t2v", "--backend-dir", "b", "--workflows-dir", "w", "--out", "o", "--report", "r"]
    assert g.main([*base, "--fallback-args=--disable-dynamic-vram --disable-smart-memory", "--port", "9000"]) == 0
    assert seen == {"fb": "--disable-dynamic-vram --disable-smart-memory", "port": 9000}
    assert g.main([*base, "--fallback-args=--lowvram"]) == 0 and seen["fb"] == "--lowvram"      # a single flag works too


def test_fallback_args_string_is_split_into_arguments(fallback_env):
    args, calls, outcomes = fallback_env
    args.fallback_args = "--disable-dynamic-vram --disable-smart-memory"
    outcomes += [(1, OOM), (0, None)]
    assert g.smoke(args) == 0
    assert ("serve", ["--disable-dynamic-vram", "--disable-smart-memory"]) in calls



# ====================================================================== the install venv: --without-pip (Kaggle Python 3.13 has no ensurepip)
import zipfile

BASE_PY = getattr(sys, "_base_executable", sys.executable)          # the interpreter the backend venv was made from ("the host")


def _ok(*cmd) -> bool:
    return subprocess.run(list(cmd), capture_output=True).returncode == 0


HOST_HAS_PIP = _ok(BASE_PY, "-m", "pip", "--version")
HOST_HAS_ENSUREPIP = _ok(BASE_PY, "-c", "import ensurepip")
needs_host_pip = pytest.mark.skipif(not HOST_HAS_PIP, reason="the host Python has no pip")
needs_ensurepip = pytest.mark.skipif(not HOST_HAS_ENSUREPIP, reason="the host Python has no ensurepip")
HOST_HAS_STACK = HOST_HAS_PIP and _ok(BASE_PY, "-c", "import torch, numpy")
needs_host_stack = pytest.mark.skipif(not HOST_HAS_STACK, reason="the host Python needs pip, torch and numpy")


def venv_env(tmp_path):
    return g.clean_env(None, **g.cache_env(tmp_path))


def make_wheel(directory: Path, name="vidsgen_tinypkg", version="1.0") -> Path:
    """A tiny pure-Python wheel, so a real `pip install` can run with no network."""
    dist = f"{name}-{version}.dist-info"
    files = {f"{name}/__init__.py": "VALUE = 42\n",
             f"{dist}/METADATA": f"Metadata-Version: 2.1\nName: {name}\nVersion: {version}\n",
             f"{dist}/WHEEL": "Wheel-Version: 1.0\nGenerator: test\nRoot-Is-Purelib: true\nTag: py3-none-any\n"}
    files[f"{dist}/RECORD"] = "".join(f"{p},,\n" for p in files) + f"{dist}/RECORD,,\n"
    wheel = directory / f"{name}-{version}-py3-none-any.whl"
    with zipfile.ZipFile(wheel, "w") as z:
        for path, content in files.items():
            z.writestr(path, content)
    return wheel


def test_venv_flags_are_system_site_packages_and_without_pip():
    assert g.VENV_FLAGS == ("--system-site-packages", "--without-pip")


def test_create_venv_runs_the_exact_command_with_without_pip(tmp_path, monkeypatch):
    venv, calls = tmp_path / "venv", []

    def fake_run(cmd, **kw):
        calls.append(cmd)
        (venv / "bin").mkdir(parents=True)
        (venv / "bin" / "python").write_text("")
        return subprocess.CompletedProcess(cmd, 0, "", "")
    monkeypatch.setattr(subprocess, "run", fake_run)
    info = g.create_venv("/opt/conda/bin/python", venv, {"PATH": "/usr/bin"})
    assert calls == [["/opt/conda/bin/python", "-m", "venv", "--system-site-packages", "--without-pip", str(venv)]]
    assert info["created"] is True and "--without-pip" in info["command"]


def test_create_venv_failure_is_reported_clearly_and_redacted(tmp_path, monkeypatch):
    monkeypatch.setattr(subprocess, "run", lambda cmd, **kw: subprocess.CompletedProcess(
        cmd, 1, "", "ensurepip is not available Authorization: Bearer abc.def-123"))
    with pytest.raises(RuntimeError) as e:
        g.create_venv("python", tmp_path / "venv", {})
    assert "could not create the virtual environment" in str(e.value) and "--without-pip" in str(e.value)
    assert "ensurepip is not available" in str(e.value) and "abc.def-123" not in str(e.value)


@needs_host_pip
def test_a_real_without_pip_venv_has_no_pip_of_its_own_but_uses_the_hosts(tmp_path):
    venv, env = tmp_path / "venv", venv_env(tmp_path)
    assert g.create_venv(BASE_PY, venv, env) == {"created": True, "command": [BASE_PY, "-m", "venv", *g.VENV_FLAGS, str(venv)]}
    assert g.venv_includes_system_packages(venv)
    assert not (venv / "bin" / "pip").exists() and not list(venv.glob("lib/python*/site-packages/pip*"))
    info = g.pip_info(str(g.venv_python(venv)), venv, env)
    assert info["source"] == "host" and str(venv) not in info["location"] and info["version"]


@needs_host_pip
def test_pip_installs_go_into_the_venv_and_never_into_the_host(tmp_path):
    """The core claim of --without-pip, proven for real (no network): a package installed by the venv's interpreter, using
    the HOST's pip, lands in the venv; the host cannot import it and the host's installed packages are unchanged."""
    venv, env = tmp_path / "venv", venv_env(tmp_path)
    g.create_venv(BASE_PY, venv, env)
    wheel = make_wheel(tmp_path)
    before = g.host_inventory(BASE_PY, env, tmp_path)
    install = subprocess.run([str(g.venv_python(venv)), "-m", "pip", "install", "--no-index", "--no-deps", "--no-cache-dir",
                              "--disable-pip-version-check", "-q", str(wheel)], capture_output=True, text=True, env=env)
    assert install.returncode == 0, install.stderr
    assert list(venv.glob("lib/python*/site-packages/vidsgen_tinypkg")), "the package must be inside the venv"
    assert subprocess.run([str(g.venv_python(venv)), "-c", "import vidsgen_tinypkg"], capture_output=True, env=env).returncode == 0
    assert subprocess.run([BASE_PY, "-c", "import vidsgen_tinypkg"], capture_output=True, env=env).returncode != 0   # not on the host
    after = g.host_inventory(BASE_PY, env, tmp_path)
    assert g.inventory_changes(before, after) == [] and "vidsgen-tinypkg" not in after


# --- existing virtual environments stay compatible ---
@needs_ensurepip
def test_an_existing_venv_with_its_own_pip_is_reused_untouched(tmp_path, monkeypatch):
    venv, env = tmp_path / "venv", venv_env(tmp_path)
    subprocess.run([BASE_PY, "-m", "venv", "--system-site-packages", str(venv)], check=True, env=env)       # the old style: with pip
    assert list(venv.glob("lib/python*/site-packages/pip"))
    marker = (venv / "pyvenv.cfg").read_text()
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: pytest.fail("an existing venv must not be re-created"))
    assert g.create_venv(BASE_PY, venv, env) == {"created": False}
    monkeypatch.undo()
    assert (venv / "pyvenv.cfg").read_text() == marker
    assert g.pip_info(str(g.venv_python(venv)), venv, env)["source"] == "venv"


@needs_host_pip
def test_rerunning_with_an_existing_without_pip_venv_is_idempotent(tmp_path):
    venv, env = tmp_path / "venv", venv_env(tmp_path)
    assert g.create_venv(BASE_PY, venv, env)["created"] is True
    py = g.venv_python(venv)
    stamp = (py.lstat().st_mtime_ns, (venv / "pyvenv.cfg").stat().st_mtime_ns)
    assert g.create_venv(BASE_PY, venv, env) == {"created": False}
    assert (py.lstat().st_mtime_ns, (venv / "pyvenv.cfg").stat().st_mtime_ns) == stamp


def test_an_existing_venv_that_cannot_see_the_host_is_rejected_and_left_alone(tmp_path):
    venv, env = tmp_path / "venv", venv_env(tmp_path)
    subprocess.run([BASE_PY, "-m", "venv", "--without-pip", str(venv)], check=True, env=env)               # no system site-packages
    cfg = (venv / "pyvenv.cfg").read_text()
    with pytest.raises(RuntimeError, match="does not include the host's site-packages.*Nothing was changed"):
        g.create_venv(BASE_PY, venv, env)
    assert g.venv_python(venv).exists() and (venv / "pyvenv.cfg").read_text() == cfg


@pytest.mark.parametrize("cfg,expected", [
    ("home = /usr/bin\ninclude-system-site-packages = true\nversion = 3.13.1\n", True),
    ("home = /usr/bin\ninclude-system-site-packages = True\n", True),
    ("home = /usr/bin\ninclude-system-site-packages = false\n", False),
    ("home = /usr/bin\n", False),
    ("", False),
])
def test_venv_includes_system_packages_parses_pyvenv_cfg(tmp_path, cfg, expected):
    (tmp_path / "pyvenv.cfg").write_text(cfg)
    assert g.venv_includes_system_packages(tmp_path) is expected
    assert g.venv_includes_system_packages(tmp_path / "missing") is False


def test_pip_info_explains_when_no_pip_is_available_at_all(tmp_path):
    """No own pip (--without-pip) AND no host pip visible (no system-site-packages) = the situation to explain clearly."""
    venv, env = tmp_path / "venv", venv_env(tmp_path)
    subprocess.run([BASE_PY, "-m", "venv", "--without-pip", str(venv)], check=True, env=env)
    with pytest.raises(RuntimeError) as e:
        g.pip_info(str(g.venv_python(venv)), venv, env)
    assert "HOST Python's pip" in str(e.value) and "ensurepip" in str(e.value) and "run the install again" in str(e.value)


# --- the host's environment must stay untouched ---
def test_inventory_changes_reports_added_removed_and_changed_packages():
    before = {"torch": ["2.6.0"], "numpy": ["1.26.4"], "wrapt": ["1.16.0"], "pip": ["24.0"]}
    after = {"torch": ["2.6.0"], "numpy": ["2.0.0"], "pip": ["24.0"], "evil": ["1.0"]}
    assert g.inventory_changes(before, after) == ["evil was ADDED (1.0)", "numpy changed: 1.26.4 -> 2.0.0", "wrapt was REMOVED (was 1.16.0)"]
    assert g.inventory_changes(before, dict(before)) == []


def test_host_inventory_reads_the_hosts_installed_packages(tmp_path):
    inv = g.host_inventory(sys.executable, venv_env(tmp_path), tmp_path)
    assert "pytest" in inv and "numpy" in inv and all(k == k.lower() and "_" not in k for k in inv)
    assert isinstance(inv["pytest"], list) and inv["pytest"][0][0].isdigit()


# --- the whole installer, end to end, with no network: every requirement is already satisfied by the host ---
@pytest.fixture(scope="module")
def offline_upstream(tmp_path_factory):
    repo = tmp_path_factory.mktemp("upstream")
    _git(repo, "init", "-q", "-b", "master")
    (repo / "comfyui_version.py").write_text('__version__ = "0.46.0"\n')
    (repo / "requirements.txt").write_text("torch\ntorchvision\ntorchaudio\nnumpy>=1.20\npip\n")      # all satisfied by the host
    _git(repo, "add", "-A"); _git(repo, "commit", "-q", "-m", "pinned")
    return f"file://{repo}", _git(repo, "rev-parse", "HEAD")


@pytest.fixture(scope="module")
def installed(tmp_path_factory, offline_upstream):
    url, sha = offline_upstream
    base = tmp_path_factory.mktemp("gpu") / "tmp" / "vidsgen-gpu"
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(g, "EXTRA_REQUIREMENTS", ["pip"])
        yield base, g.install_comfyui(base, url, sha, "0.46.0", python=BASE_PY), (url, sha)


@needs_host_stack
def test_install_builds_a_without_pip_venv_and_leaves_the_host_untouched(installed):
    base, r, _ = installed
    venv = base / "venv"
    assert r["venv_created"] is True and r["venv_flags"] == ["--system-site-packages", "--without-pip"]
    assert r["pip"]["source"] == "host" and str(venv) not in r["pip"]["location"]
    assert g.venv_includes_system_packages(venv) and not list(venv.glob("lib/python*/site-packages/pip*"))
    assert r["torch_replaced"] is False and r["host_environment_unchanged"] is True and r["host_packages_checked"] > 10
    assert {"torch", "numpy"} <= set(r["protected_packages"]) and all(p["identical_to_host"] for p in r["protected_packages"].values())
    assert set(r["pinned_by_constraints"]) <= set(PROTECTED) and "torch" in r["pinned_by_constraints"] and "numpy" in r["pinned_by_constraints"]
    assert r["requirements_removed"] == ["torch", "torchvision", "torchaudio"]
    constraints = (base / "constraints.txt").read_text().splitlines()
    assert any(l.startswith("torch==") for l in constraints) and any(l.startswith("numpy==") for l in constraints)
    assert "torch" not in (base / "requirements-comfyui.txt").read_text().split()                           # never asked to be installed
    # the venv really uses the host's copies
    probe = lambda py: json.loads(subprocess.run([py, "-c", g.IMPORT_PROBE], capture_output=True, text=True).stdout)["packages"]
    assert probe(str(g.venv_python(venv))) == probe(BASE_PY)


@needs_host_stack
def test_rerunning_the_installer_reuses_the_existing_venv(installed):
    base, first, (url, sha) = installed
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(g, "EXTRA_REQUIREMENTS", ["pip"])
        second = g.install_comfyui(base, url, sha, "0.46.0", python=BASE_PY)
    assert second["venv_created"] is False and second["torch_file"] == first["torch_file"] and second["host_environment_unchanged"] is True


@needs_host_stack
@needs_ensurepip
def test_the_installer_accepts_a_preexisting_venv_that_has_its_own_pip(tmp_path, offline_upstream, monkeypatch):
    url, sha = offline_upstream
    base = tmp_path / "tmp" / "vidsgen-gpu"
    base.mkdir(parents=True)
    subprocess.run([BASE_PY, "-m", "venv", "--system-site-packages", str(base / "venv")], check=True, env=venv_env(tmp_path))
    monkeypatch.setattr(g, "EXTRA_REQUIREMENTS", ["pip"])
    r = g.install_comfyui(base, url, sha, "0.46.0", python=BASE_PY)
    assert r["venv_created"] is False and r["pip"]["source"] == "venv" and r["torch_replaced"] is False


@needs_host_stack
def test_the_installer_refuses_an_existing_venv_that_cannot_see_the_host(tmp_path, offline_upstream, monkeypatch):
    url, sha = offline_upstream
    base = tmp_path / "tmp" / "vidsgen-gpu"
    base.mkdir(parents=True)
    subprocess.run([BASE_PY, "-m", "venv", "--without-pip", str(base / "venv")], check=True, env=venv_env(tmp_path))
    monkeypatch.setattr(g, "EXTRA_REQUIREMENTS", ["pip"])
    with pytest.raises(RuntimeError, match="does not include the host's site-packages"):
        g.install_comfyui(base, url, sha, "0.46.0", python=BASE_PY)


@needs_host_stack
@pytest.mark.parametrize("name", PROTECTED)
def test_the_installer_fails_if_any_protected_package_would_be_replaced(installed, monkeypatch, name):
    base, _, (url, sha) = installed
    venv_py = str(g.venv_python(base / "venv"))
    host = {"python": "3.13.1", "packages": {n: [f"{n}-1.0", f"/usr/lib/{n}/__init__.py"] for n in PROTECTED}}

    def fake_probe(py, env):
        r = json.loads(json.dumps(host))
        if py == venv_py:
            r["packages"][name] = ["9.9.9", str(base / "venv" / "lib" / name / "__init__.py")]      # a copy installed into the venv
        return r
    monkeypatch.setattr(g, "import_probe", fake_probe)
    monkeypatch.setattr(g, "EXTRA_REQUIREMENTS", ["pip"])
    with pytest.raises(RuntimeError, match=rf"PyTorch/NumPy stack was replaced.*{name} changed: {name}-1.0 -> 9.9.9"):
        g.install_comfyui(base, url, sha, "0.46.0", python=BASE_PY)


@needs_host_stack
def test_the_installer_fails_if_the_hosts_environment_was_modified(installed, monkeypatch):
    base, _, (url, sha) = installed
    real, n = g.host_inventory, {"calls": 0}

    def fake_inventory(py, env, cwd):
        inv = real(py, env, cwd)
        n["calls"] += 1
        return {**inv, "numpy": ["0.0.1"], "wrapt": ["0.0.1"], "evil-pkg": ["1.0"]} if n["calls"] == 2 else inv     # the AFTER snapshot
    monkeypatch.setattr(g, "host_inventory", fake_inventory)
    monkeypatch.setattr(g, "EXTRA_REQUIREMENTS", ["pip"])
    with pytest.raises(RuntimeError, match="the host's Python environment was modified by the install") as e:
        g.install_comfyui(base, url, sha, "0.46.0", python=BASE_PY)
    assert "evil-pkg was ADDED" in str(e.value) and "numpy changed" in str(e.value)


# ====================================================================== regression: a stale NumPy record on the host (Kaggle: numpy 2.1.3 + numpy-1.26.4 metadata)
# The installer used to pin "the first of the sorted versions" from the host's package inventory. With a stale numpy-1.26.4 record next to
# the real numpy 2.1.3, that is "1.26.4": pip was told numpy==1.26.4 and installed a second NumPy into the venv, every time.
def make_stale_record(directory: Path, name="numpy", version="1.26.4") -> Path:
    dist = directory / f"{name}-{version}.dist-info"
    dist.mkdir(parents=True)
    (dist / "METADATA").write_text(f"Metadata-Version: 2.1\nName: {name}\nVersion: {version}\n")
    return directory


def host_numpy_version() -> str | None:
    p = subprocess.run([BASE_PY, "-c", "import numpy; print(numpy.__version__)"], capture_output=True, text=True)
    return p.stdout.strip() if p.returncode == 0 else None


needs_host_numpy = pytest.mark.skipif(host_numpy_version() is None, reason="the host Python has no numpy")


@needs_host_numpy
def test_reproduction_the_inventorys_first_sorted_version_is_the_stale_record(tmp_path):
    """The old rule: it picks 1.26.4 although the host imports the real NumPy."""
    stale = make_stale_record(tmp_path / "stale")
    env = {**venv_env(tmp_path), "PYTHONPATH": str(stale)}
    inventory = g.host_inventory(BASE_PY, env, tmp_path)
    assert "1.26.4" in inventory["numpy"] and host_numpy_version() in inventory["numpy"]
    assert inventory["numpy"][0] == "1.26.4"                      # <- what `inventory["numpy"][0]` used to feed into constraints.txt


@needs_host_numpy
def test_host_pins_follow_the_imported_numpy_not_the_stale_record(tmp_path):
    stale = make_stale_record(tmp_path / "stale")
    env = {**venv_env(tmp_path), "PYTHONPATH": str(stale)}
    pins, conflicts = g.host_pins(BASE_PY, env, tmp_path)
    assert pins["numpy"] == host_numpy_version() != "1.26.4"
    assert {v for v, _ in conflicts["numpy"]} == {"1.26.4", host_numpy_version()}               # visible, with locations
    assert any(str(stale) in path for _, path in conflicts["numpy"])
    # and with nothing stale there is nothing to report
    pins2, conflicts2 = g.host_pins(BASE_PY, venv_env(tmp_path), tmp_path)
    assert pins2 == pins and "numpy" not in conflicts2


def probe_entry(imported, *records):
    return {"imported": imported, "metadata": [[v, f"/site/{i}"] for i, v in enumerate(records)]}


KAGGLE_LIKE = {
    "torch": probe_entry(["2.11.0+cu130", "/p/torch/__init__.py"], "2.11.0+cu130"),
    "torchvision": probe_entry(["0.26.0+cu130", "/p/torchvision/__init__.py"], "0.26.0+cu130"),
    "torchaudio": probe_entry(["2.11.0+cu130", "/p/torchaudio/__init__.py"], "2.11.0+cu130"),
    "numpy": probe_entry(["2.1.3", "/p/numpy/__init__.py"], "2.1.3"),
}


def with_entry(name, entry):
    return {**json.loads(json.dumps(KAGGLE_LIKE)), name: entry}


def test_select_pins_for_the_verified_kaggle_stack():
    pins, conflicts = g.select_pins(KAGGLE_LIKE)
    assert pins == {"torch": "2.11.0+cu130", "torchvision": "0.26.0+cu130", "torchaudio": "2.11.0+cu130", "numpy": "2.1.3"}
    assert conflicts == {}


@pytest.mark.parametrize("records", [("1.26.4", "2.1.3"), ("2.1.3", "1.26.4"), ("1.26.4", "2.1.3", "1.24.0"), ("2.1.3", "1.26.4", "2.1.3")])
def test_a_stale_record_never_wins_whatever_the_order(records):
    pins, conflicts = g.select_pins(with_entry("numpy", probe_entry(["2.1.3", "/p/numpy/__init__.py"], *records)))
    assert pins["numpy"] == "2.1.3" and "numpy" in conflicts and len(conflicts["numpy"]) == len(records)


def test_a_local_build_tag_the_metadata_omits_still_matches_the_imported_release():
    """pip's metadata may say 2.15.0 while the module says 2.15.0+cu132: the pin is the metadata form pip understands."""
    pins, _ = g.select_pins(with_entry("torch", probe_entry(["2.15.0+cu132", "/p/torch/__init__.py"], "2.15.0")))
    assert pins["torch"] == "2.15.0"


def test_a_pin_that_cannot_be_matched_to_the_imported_release_is_an_error_not_a_guess():
    with pytest.raises(RuntimeError, match=r"cannot establish which numpy the host uses.*imports 2\.1\.3.*1\.26\.4"):
        g.select_pins(with_entry("numpy", probe_entry(["2.1.3", "/p/numpy/__init__.py"], "1.26.4")))
    with pytest.raises(RuntimeError, match="cannot establish which numpy"):
        g.select_pins(with_entry("numpy", probe_entry(["2.1.3", "/p/numpy/__init__.py"])))                  # imported, no record at all


def test_a_package_that_fails_to_import_on_the_host_stops_the_installer():
    with pytest.raises(RuntimeError, match=r"numpy is installed on the host but importing it fails \(ImportError\)"):
        g.select_pins(with_entry("numpy", {"imported": ["error", "ImportError"], "metadata": [["2.1.3", "/x"]]}))


def test_absent_packages_are_skipped_and_installed_but_unimportable_ones_stay_protected():
    probe = with_entry("torchaudio", {"imported": None, "metadata": []})
    assert "torchaudio" not in g.select_pins(probe)[0]
    probe = with_entry("torchaudio", {"imported": None, "metadata": [["2.11.0", "/x"]]})
    assert g.select_pins(probe)[0]["torchaudio"] == "2.11.0"


@needs_host_stack
def test_install_pins_the_imported_numpy_even_when_the_inventory_lists_a_stale_record_first(installed, offline_upstream, monkeypatch):
    """End to end (offline): the inventory is made to report numpy ['1.26.4', <real>] exactly as on Kaggle. The constraints file and the
    resulting venv must still use the real NumPy, and the verification must pass (it is not weakened)."""
    base, first, (url, sha) = installed
    real = g.host_pins(BASE_PY, venv_env(base.parent), base)[0]["numpy"]
    original = g.host_inventory

    def with_stale_record(python, env, cwd):
        inv = original(python, env, cwd)
        return {**inv, "numpy": sorted({"1.26.4", *inv["numpy"]})}

    monkeypatch.setattr(g, "host_inventory", with_stale_record)
    monkeypatch.setattr(g, "EXTRA_REQUIREMENTS", ["pip"])
    r = g.install_comfyui(base, url, sha, "0.46.0", python=BASE_PY)
    constraints = (base / "constraints.txt").read_text().splitlines()
    assert f"numpy=={real}" in constraints and "numpy==1.26.4" not in constraints
    assert r["pinned_by_constraints"]["numpy"] == real and r["host_environment_unchanged"] is True
    assert r["protected_packages"]["numpy"]["identical_to_host"] is True and r["torch_replaced"] is False
    assert not list((base / "venv").glob("lib/python*/site-packages/numpy"))          # no second NumPy inside the venv


@needs_host_stack
def test_rerunning_the_installer_rewrites_the_constraints_from_the_hosts_real_packages(installed, offline_upstream, monkeypatch):
    """The Kaggle symptom: fixing constraints.txt by hand did not stick, because every run rewrote it. Now every run rewrites it correctly."""
    base, _, (url, sha) = installed
    (base / "constraints.txt").write_text("numpy==1.26.4\ntorch==0.0.1\n")           # a wrong file left behind by a failed earlier run
    monkeypatch.setattr(g, "EXTRA_REQUIREMENTS", ["pip"])
    r = g.install_comfyui(base, url, sha, "0.46.0", python=BASE_PY)
    text = (base / "constraints.txt").read_text()
    assert "1.26.4" not in text and "0.0.1" not in text and r["host_environment_unchanged"] is True


@needs_host_stack
def test_a_failed_verification_still_names_what_was_pinned(installed, monkeypatch):
    base, _, (url, sha) = installed
    venv_py = str(g.venv_python(base / "venv"))
    real_probe = g.import_probe

    def probe(py, env):
        r = real_probe(py, env)
        if py == venv_py:
            r["packages"]["numpy"] = ["1.26.4", str(base / "venv" / "lib" / "numpy" / "__init__.py")]     # the Kaggle symptom
        return r
    monkeypatch.setattr(g, "import_probe", probe)
    monkeypatch.setattr(g, "EXTRA_REQUIREMENTS", ["pip"])
    with pytest.raises(RuntimeError, match=r"(?s)numpy changed: .* -> 1\.26\.4.*a second numpy was installed inside the venv.*pinned from the host: .*numpy=="):
        g.install_comfyui(base, url, offline_upstream_sha(base), "0.46.0", python=BASE_PY)


def offline_upstream_sha(base):
    return subprocess.run(["git", "-C", str(base / "ComfyUI"), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
