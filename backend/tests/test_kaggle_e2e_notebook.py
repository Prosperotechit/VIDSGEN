"""Tests for notebooks/kaggle_backend_e2e_phase6b.ipynb (Phase 6B).

Static checks (sync with the generator and the repository, safety invariants) plus a harness that EXECUTES the notebook's
real cells with a stubbed command runner, to prove the gating and the honesty of its final message. It does not run Kaggle,
a GPU, ComfyUI or the backend: only an actual Kaggle run can establish that the integration works.
"""

from __future__ import annotations

import importlib.util
import json
import re
import subprocess
import sys
import types
import urllib.request
from pathlib import Path

import pytest

from app.core.config import REPO_ROOT
from app.providers.mock_video import encode_mock_video

NOTEBOOK = REPO_ROOT / "notebooks" / "kaggle_backend_e2e_phase6b.ipynb"
NOTEBOOK_5B = REPO_ROOT / "notebooks" / "kaggle_comfyui_wan22.ipynb"


def _load(name: str, rel: str):
    spec = importlib.util.spec_from_file_location(name, REPO_ROOT / rel)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


builder5 = _load("build_kaggle_notebook", "notebooks/tools/build_kaggle_notebook.py")
builder = _load("build_kaggle_e2e_notebook", "notebooks/tools/build_kaggle_e2e_notebook.py")

nb = json.loads(NOTEBOOK.read_text())
CELLS = ["".join(c["source"]) for c in nb["cells"]]
CODE = [("".join(c["source"]), i) for i, c in enumerate(nb["cells"]) if c["cell_type"] == "code"]
EMBED_RE = re.compile(r"^# Embedded verbatim from (\S+),.*\nPath\(TOOLS, '([^']+)'\)\.write_text\(r'''(.*)'''\)\nprint\(", re.S)
EMBEDDED = {m.group(1): m.group(3) for src, _ in CODE if (m := EMBED_RE.match(src))}
OWN = "\n".join(src for src, _ in CODE if not EMBED_RE.match(src))


def cell_with(text: str) -> int:
    hits = [i for i, c in enumerate(CELLS) if text in c and nb["cells"][i]["cell_type"] == "code" and not EMBED_RE.match(c)]
    assert len(hits) == 1, (text, hits)
    return hits[0]


# ====================================================================== static
def test_notebook_is_valid_and_has_no_outputs():
    assert nb["nbformat"] == 4 and nb["metadata"]["kernelspec"]["name"] == "python3"
    for c in nb["cells"]:
        if c["cell_type"] == "code":
            assert c["outputs"] == [] and c["execution_count"] is None
            compile("".join(c["source"]), "cell", "exec")
    assert len({c["id"] for c in nb["cells"]}) == len(nb["cells"])


def test_notebook_is_in_sync_with_its_generator():
    assert json.dumps(builder.build(), indent=1, ensure_ascii=False) + "\n" == NOTEBOOK.read_text(), \
        "run: python notebooks/tools/build_kaggle_e2e_notebook.py"


def test_the_phase_5b_notebook_is_unaffected():
    """Phase 6B adds a notebook; it must not change the verified 5B one or its generator output."""
    assert json.dumps(builder5.build(), indent=1, ensure_ascii=False) + "\n" == NOTEBOOK_5B.read_text()
    assert builder.NOTEBOOK != builder5.NOTEBOOK and builder.NOTEBOOK.name == "kaggle_backend_e2e_phase6b.ipynb"


def test_embedded_files_are_identical_to_the_repository_copies():
    assert set(EMBEDDED) == {"scripts/download_models.py", "scripts/wan22_gpu_test.py", "workflows/comfyui/models.yaml",
                             "scripts/backend_e2e_test.py"}
    for rel, content in EMBEDDED.items():
        assert content == (REPO_ROOT / rel).read_text(), f"{rel} differs from the embedded copy"


def test_the_pinned_vidsgen_commit_contains_the_backend_with_the_generation_limits():
    commit = builder.BACKEND_COMMIT
    assert f'"{commit}"' in CELLS[1] and "4b6fc67" not in OWN
    for path in ("backend/app/jobs/envelope.py", "backend/app/main.py", "workflows/comfyui/wan22_ti2v_5b_t2v.json"):
        r = subprocess.run(["git", "cat-file", "-e", f"{commit}:{path}"], cwd=REPO_ROOT, capture_output=True)
        assert r.returncode == 0, f"{path} is not in the pinned commit {commit}"
    assert 'COMFYUI_COMMIT, COMFYUI_VERSION = "995456d0188ffb51032c28f41527cfa82d87d8a9", "0.46.0"' in CELLS[1]


def test_the_verified_configuration_is_fixed_and_everything_is_on_loopback():
    e2e_cell = CELLS[cell_with('"e2e"')]
    assert 'URL = f"http://127.0.0.1:{PORT}"' in CELLS[1]
    assert "--comfyui-url" in e2e_cell and "URL" in e2e_cell and "--require-gpu" in e2e_cell
    assert "BACKEND_PORT = 8000" in CELLS[2]
    # the request values are not user settings: the test script fixes them (832x480, 17, 20, 24, 5.0, t4_verified)
    for flag in ("--resolution", "--frames", "--steps", "--fps", "--cfg"):
        assert flag not in e2e_cell
    assert '"resolution": "832x480", "num_frames": 17, "steps": 20, "fps": 24, "guidance_scale": 5.0' in EMBEDDED["scripts/backend_e2e_test.py"]
    assert 'GENERATION_PROFILE": "t4_verified"' in EMBEDDED["scripts/backend_e2e_test.py"]


def test_no_tunnel_no_public_binding_no_secrets():
    for forbidden in ("cloudflared", "trycloudflare", "ngrok", "localtunnel", "0.0.0.0", "--listen 0", "--share"):
        assert forbidden not in OWN, forbidden
    assert not re.search(r"hf_[A-Za-z0-9]{20,}|ghp_[A-Za-z0-9]{20,}|Bearer\s+[A-Za-z0-9._-]{10,}", "\n".join(CELLS))
    assert "get_secret(\"HF_TOKEN\")" in OWN and "GITHUB" not in OWN          # the public repo needs no GitHub credential
    assert "/kaggle/working" in OWN and 'BASE = "/kaggle/tmp/vidsgen-gpu"' in OWN


def test_every_new_stage_stops_on_failure_and_runs_in_the_right_order():
    for name in ("backend_install", "backend_e2e"):
        assert f'stage_failed("{name}"' in OWN
    order = [cell_with(t) for t in ('"preflight"', "checkout_commit", '"install-comfyui"', "download_models.py", '"serve"',
                                    "check-templates", '"install-backend"', '"e2e"', "vidsgen_phase6b_report.json")]
    assert order == sorted(order)


# ====================================================================== executing the notebook's cells
@pytest.fixture
def harness(tmp_path, monkeypatch):
    gmod = _load("wan22_gpu_test", "scripts/wan22_gpu_test.py")
    monkeypatch.setitem(sys.modules, "wan22_gpu_test", gmod)
    base, out = tmp_path / "kaggle-tmp" / "vidsgen-gpu", tmp_path / "kaggle-working"
    out.mkdir(parents=True)
    h = types.SimpleNamespace(base=base, out=out, calls=[], rc={}, e2e_claims=None)

    def fake_checkout(repo, commit, dest):
        for f in ["workflows/comfyui/wan22_ti2v_5b_t2v.json", "workflows/comfyui/wan22_ti2v_5b_i2v.json", "workflows/comfyui/wan22_ti2v_5b.mapping.json",
                  "backend/app/providers/comfyui.py", "backend/app/workflows/templates.py", "backend/tests/comfyui_support.py"]:
            (Path(dest) / f).parent.mkdir(parents=True, exist_ok=True)
            (Path(dest) / f).write_text("x")
        h.checked_out = commit
        return commit
    monkeypatch.setattr(gmod, "checkout_commit", fake_checkout)

    disp = types.ModuleType("IPython.display")
    disp.Video, disp.display = (lambda *a, **k: None), (lambda *a, **k: None)
    monkeypatch.setitem(sys.modules, "IPython", types.ModuleType("IPython")); monkeypatch.setitem(sys.modules, "IPython.display", disp)

    sizes = {"diffusion_models": ("wan2.2_ti2v_5B_fp16.safetensors", 9999658848), "text_encoders": ("umt5_xxl_fp8_e4m3fn_scaled.safetensors", 6735906897),
             "vae": ("wan2.2_vae.safetensors", 1409400960)}

    class Resp:
        def __init__(self, body): self._b = body
        def read(self): return json.dumps(self._b).encode()
        def __enter__(self): return self
        def __exit__(self, *a): return False

    def urlopen(url, timeout=30):
        path = url.split("8188", 1)[1]
        if path == "/system_stats":
            return Resp({"system": {"comfyui_version": "0.46.0", "pytorch_version": "2.6.0+cu124"},
                         "devices": [{"name": "cuda:0 Tesla T4", "vram_total": 15 * 2**30, "vram_free": 14 * 2**30}]})
        name, size = sizes[path.rsplit("/", 1)[1]]
        return Resp([{"name": name, "size": size}])
    monkeypatch.setattr(urllib.request, "urlopen", urlopen)

    def fake_run(cmd, env=None, check=True):
        cmd = [str(c) for c in cmd]
        h.calls.append(cmd)
        sub = next((s for s in ("preflight", "install-comfyui", "install-backend", "serve", "check-templates", "e2e", "stop-backend", "stop") if s in cmd), None)
        if "download_models.py" in " ".join(cmd):
            sub = "verify-only" if "--verify-only" in cmd else "download"
        code, text = h.rc.get(sub, 0), ""
        if sub == "preflight":
            Path(cmd[cmd.index("--report") + 1]).write_text(json.dumps({"ok": code == 0, "counts": {"PASS": 9, "WARN": 0, "FAIL": 0, "INFO": 2}, "checks": []}))
        elif sub == "install-comfyui":
            text = json.dumps({"commit": cmd[cmd.index("--commit") + 1], "comfyui_version": "0.46.0", "torch_replaced": False}, indent=1)
        elif sub == "install-backend":
            text = "noise\n" + json.dumps({"host_environment_unchanged": True, "backend_importable": True,
                                           "protected_packages": {"torch": {}, "numpy": {}}}, indent=1)
        elif sub == "serve":
            log = h.base / "logs" / "comfyui.log"; log.parent.mkdir(parents=True, exist_ok=True); log.write_text("Total VRAM 15360 MB\nAuthorization: Bearer leakedtoken\n")
            text = json.dumps({"pid": 1, "port": 8188, "listening_on": ["127.0.0.1"], "loopback_only": True, "log": str(log), "cuda_visible_devices": "0"}, indent=1)
        elif sub == "check-templates":
            text = json.dumps({"ok": True, "node_classes_missing": [], "templates": {}}, indent=1)
        elif sub == "e2e":
            ok = code == 0
            claims = h.e2e_claims or {"backend_loopback_only": ok, "job_completed": ok, "mp4_decodes_and_matches": ok, "cancel_works": None}
            Path(cmd[cmd.index("--report") + 1]).write_text(json.dumps({
                "verdict": "PASS" if ok else "FAIL", "reasons": [] if ok else ["job_completed: job ended as failed: gpu_out_of_memory"],
                "claims": claims, "timing": {"job_wall_s": 100.0, "total_wall_s": 130.0},
                "resources": {"vram_peak_used_mib": 14885, "vram_total_mib": 15360, "ram_peak_used_gib": 20.6, "ram_total_gib": 31.4},
                "envelope_probes": {}, "output": {"bytes": 70000, "container": "mov,mp4", "codec": "h264", "width": 832, "height": 480,
                                                  "decoded_frames": 17, "fps": 24.0} if ok else None}))
            if ok:
                encode_mock_video(Path(cmd[cmd.index("--out") + 1]), width=64, height=64, num_frames=17, fps=24, seed=1, prompt="x")
        if check and code != 0:
            raise RuntimeError(f"STOP: exit code {code}.")
        return code, text

    def execute(overrides: dict[str, str] | None = None):
        ns = {"__name__": "__main__"}
        for src, idx in CODE:
            if idx == 1:
                src = src.replace('BASE = "/kaggle/tmp/vidsgen-gpu"', f"BASE = {str(base)!r}").replace('OUT = Path("/kaggle/working")', f"OUT = Path({str(out)!r})")
                assert str(base) in src and str(out) in src
            if idx == 2:
                for old, new in (overrides or {}).items():
                    assert old in src, old
                    src = src.replace(old, new)
            exec(compile(src, f"<cell {idx}>", "exec"), ns)
            if idx == 1:
                ns["run"], ns["hf_env"] = fake_run, lambda: {}
        return ns

    h.execute = execute
    return h


def names(h):
    out = []
    for cmd in h.calls:
        j = " ".join(cmd)
        out.append("verify-only" if "--verify-only" in cmd else "download" if "download_models.py" in j else
                   next(s for s in ("preflight", "install-comfyui", "install-backend", "serve", "check-templates", "e2e", "stop-backend", "stop") if s in cmd))
    return out


def test_happy_path_runs_every_stage_in_order_with_one_gpu_and_loopback(harness, capsys):
    ns = harness.execute()
    assert names(harness) == ["preflight", "install-comfyui", "download", "verify-only", "serve", "check-templates", "install-backend",
                              "e2e", "stop", "stop-backend"]
    assert harness.checked_out == builder.BACKEND_COMMIT
    e2e = next(c for c in harness.calls if "e2e" in c and "backend_e2e_test.py" in " ".join(c))
    assert e2e[0] == str(harness.base / "venv" / "bin" / "python")                  # the venv python: it has the backend's dependencies
    assert e2e[e2e.index("--comfyui-url") + 1] == "http://127.0.0.1:8188" and e2e[e2e.index("--gpu") + 1] == "0" and "--with-cancel" not in e2e
    inst = next(c for c in harness.calls if "install-backend" in c)
    assert inst[0] == sys.executable                                                   # the HOST python, to compare against the venv
    report = json.loads((harness.out / "vidsgen_phase6b_report.json").read_text())
    assert {k: v["ok"] for k, v in report["stages"].items()} == {
        "preflight": True, "vidsgen_checkout": True, "comfyui_install": True, "models": True, "server": True, "verify_server": True,
        "backend_install": True, "backend_e2e": True}
    assert report["settings"]["profile"] == "t4_verified" and report["settings"]["max_concurrent_jobs"] == 1
    assert report["settings"]["request"] == {"resolution": "832x480", "frames": 17, "steps": 20, "fps": 24, "cfg": 5.0}
    assert "Bearer" not in json.dumps(report) and "leakedtoken" not in json.dumps(report)
    assert (harness.out / "vidsgen_phase6b_e2e.mp4").read_bytes()[4:8] == b"ftyp"
    assert sorted(p.name for p in harness.out.iterdir()) == ["vidsgen_phase6b_e2e.mp4", "vidsgen_phase6b_e2e_report.json",
                                                             "vidsgen_phase6b_report.json", "vidsgen_preflight_5b.json"]
    shown = capsys.readouterr().out
    assert "BACKEND END-TO-END VERDICT: PASS" in shown and "EVERY CHECK PASSED (localhost only)" in shown
    assert "Not validated by this notebook: image-to-video" in shown and "leakedtoken" not in shown


def test_a_failed_backend_install_stops_before_the_end_to_end_test(harness):
    harness.rc["install-backend"] = 2
    with pytest.raises(RuntimeError, match="STOP at stage 'backend_install'"):
        harness.execute()
    assert names(harness)[-1] == "install-backend" and "e2e" not in names(harness)


def test_a_failed_end_to_end_test_is_reported_as_a_failure_not_success(harness, capsys):
    harness.rc["e2e"] = 1
    with pytest.raises(RuntimeError, match="STOP at stage 'backend_e2e'.*gpu_out_of_memory"):
        harness.execute()
    shown = capsys.readouterr().out
    assert "BACKEND END-TO-END VERDICT: FAIL" in shown and "EVERY CHECK PASSED" not in shown
    assert not (harness.out / "vidsgen_phase6b_report.json").exists()               # no success report is written


def test_skipping_the_end_to_end_test_never_claims_success(harness, capsys):
    harness.execute({"RUN_E2E = True": "RUN_E2E = False"})
    shown = capsys.readouterr().out
    assert "e2e" not in names(harness) and "did NOT pass" in shown and "EVERY CHECK PASSED" not in shown
    assert json.loads((harness.out / "vidsgen_phase6b_report.json").read_text())["stages"]["backend_e2e"]["ok"] is None


def test_the_cancel_test_is_opt_in(harness):
    harness.execute({"RUN_CANCEL_TEST = False": "RUN_CANCEL_TEST = True"})
    e2e = next(c for c in harness.calls if "backend_e2e_test.py" in " ".join(c) and "e2e" in c)
    assert "--with-cancel" in e2e


def test_comfyui_can_be_left_running_when_asked(harness):
    harness.execute({"STOP_SERVER_AT_END = True": "STOP_SERVER_AT_END = False"})
    assert names(harness)[-1] == "e2e" and "stop" not in names(harness)


def test_failed_earlier_stages_still_stop_everything_after_them(harness):
    harness.rc["serve"] = 1
    with pytest.raises(RuntimeError, match="STOP at stage 'server'"):
        harness.execute()
    assert "install-backend" not in names(harness) and "e2e" not in names(harness)
