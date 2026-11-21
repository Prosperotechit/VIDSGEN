"""Tests for notebooks/kaggle_comfyui_wan22.ipynb (Phase 5B).

Static checks (sync with the repo, safety invariants) plus a harness that EXECUTES the notebook's real cells with a
stubbed command runner, to prove the gating: a failed stage stops everything after it, and no download is ever started
unless the preflight and install stages passed.
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
import yaml

from app.core.config import REPO_ROOT
from app.providers.mock_video import encode_mock_video

NOTEBOOK = REPO_ROOT / "notebooks" / "kaggle_comfyui_wan22.ipynb"
_bspec = importlib.util.spec_from_file_location("build_kaggle_notebook", REPO_ROOT / "notebooks/tools/build_kaggle_notebook.py")
builder = importlib.util.module_from_spec(_bspec)
_bspec.loader.exec_module(builder)

nb = json.loads(NOTEBOOK.read_text())
CELLS = ["".join(c["source"]) for c in nb["cells"]]
CODE = [("".join(c["source"]), i) for i, c in enumerate(nb["cells"]) if c["cell_type"] == "code"]
EMBED_RE = re.compile(r"^# Embedded verbatim from (\S+)\..*\nPath\(TOOLS, '([^']+)'\)\.write_text\(r'''(.*)'''\)\nprint\(", re.S)
EMBEDDED = {m.group(1): m.group(3) for src, _ in CODE if (m := EMBED_RE.match(src))}
OWN_CODE = [src for src, _ in CODE if not EMBED_RE.match(src)]          # the notebook's own cells (not the embedded files)
GATE_NAMES = ["settings", "tools-dir", "e1", "e2", "e3", "preflight", "checkout", "install", "download", "serve", "verify", "smoke", "report"]


# ====================================================================== static
def test_notebook_is_valid_and_has_no_outputs():
    assert nb["nbformat"] == 4 and nb["metadata"]["kernelspec"]["name"] == "python3"
    for c in nb["cells"]:
        if c["cell_type"] == "code":
            assert c["outputs"] == [] and c["execution_count"] is None
            compile("".join(c["source"]), "cell", "exec")
    assert len({c["id"] for c in nb["cells"]}) == len(nb["cells"])


def test_notebook_is_in_sync_with_the_generator():
    assert json.dumps(builder.build(), indent=1, ensure_ascii=False) + "\n" == NOTEBOOK.read_text(), \
        "run: python notebooks/tools/build_kaggle_notebook.py"


def test_embedded_files_are_identical_to_the_repository_copies():
    assert set(EMBEDDED) == {"scripts/download_models.py", "scripts/wan22_gpu_test.py", "workflows/comfyui/models.yaml"}
    for rel, content in EMBEDDED.items():
        assert content == (REPO_ROOT / rel).read_text(), f"{rel} differs from the embedded copy"


def test_pinned_sources():
    settings = CELLS[1]
    assert '"https://github.com/Prosperotechit/VIDSGEN.git", "4b6fc67243b53cb2dcfb7d2b114dca2517380176"' in settings
    assert 'COMFYUI_COMMIT, COMFYUI_VERSION = "995456d0188ffb51032c28f41527cfa82d87d8a9", "0.46.0"' in settings
    assert all(re.fullmatch(r"[0-9a-f]{40}", sha) for sha in re.findall(r'"([0-9a-f]{40})"', settings)) and len(re.findall(r'"[0-9a-f]{40}"', settings)) == 2
    p = subprocess.run(["git", "cat-file", "-t", "4b6fc67243b53cb2dcfb7d2b114dca2517380176"], capture_output=True, text=True, cwd=REPO_ROOT)
    assert p.stdout.strip() == "commit", "the pinned VIDSGEN commit must exist in this repository"


def test_everything_large_lives_in_kaggle_tmp_and_one_gpu_is_used():
    settings = CELLS[1]
    assert 'BASE = "/kaggle/tmp/vidsgen-gpu"' in settings and 'GPU = "0"' in settings
    assert 'TOOLS, DATA = f"{BASE}/tools", f"{BASE}/comfy-data"' in settings and 'VENV_PY = f"{BASE}/venv/bin/python"' in settings
    for src in OWN_CODE:
        for line in src.splitlines():
            code = line.split("#", 1)[0]                           # comments may say "never /kaggle/working"
            if "/kaggle/working" in code:
                assert "OUT" in code, f"only the OUT reports may use /kaggle/working: {line}"
    assert "--comfyui-dir\", DATA" in CELLS[9]                       # weights go to DATA (under BASE), never OUT


def test_only_the_downloader_downloads_and_only_in_one_guarded_place():
    joined = "\n".join(OWN_CODE)
    assert joined.count("--execute") == 1 and "if not DOWNLOAD_MODELS:" in CELLS[9]
    for banned in ("hf_hub_download", "snapshot_download", "wget", "curl ", "huggingface-cli", "from_pretrained"):
        assert banned not in joined
    manifest = yaml.safe_load(EMBEDDED["workflows/comfyui/models.yaml"])
    files = manifest["files"]
    assert sorted(f["role"] for f in files) == ["diffusion_model", "text_encoder", "vae"]       # exactly the three approved files
    assert all(re.fullmatch(r"[0-9a-f]{64}", f["sha256"]) and re.fullmatch(r"[0-9a-f]{40}", f["revision"]) for f in files)
    assert sum(f["size_bytes"] for f in files) == manifest["total_bytes"] == 18_144_966_705
    unapproved = re.compile(r"14B|high_noise|low_noise|fun_|lora|wan_2\.1_vae|animate|s2v|inpaint|camera", re.I)
    assert not [f["path"] for f in files if unapproved.search(f["path"])]                        # (the 'incompatible' notes may mention them)


def test_no_tunnel_no_public_binding_no_secrets():
    joined = "\n".join(OWN_CODE)
    for banned in ("cloudflared", "trycloudflare", "ngrok", "localtunnel", "0.0.0.0", "--listen"):
        assert banned not in joined, banned
    assert "cloudflared" not in EMBEDDED["scripts/wan22_gpu_test.py"] and "ngrok" not in EMBEDDED["scripts/wan22_gpu_test.py"]
    assert not re.search(r"hf_[A-Za-z0-9]{20,}|Bearer [A-Za-z0-9._-]{12,}|AKIA[0-9A-Z]{16}|BEGIN [A-Z ]*PRIVATE KEY", "\n".join(CELLS))
    assert "UserSecretsClient" in CELLS[1] and "value not shown" in CELLS[1]
    assert not re.search(r"print\([^)]*(token|TOKEN)\b[^)]*\)", CELLS[1].replace("HF_TOKEN secret", ""))


def test_every_stage_stops_on_failure():
    for name, idx in (("preflight", 6), ("comfyui_install", 8), ("models", 9), ("server", 10), ("verify_server", 11), ("smoke_t2v", 12)):
        assert f'stage_failed("{name}"' in CELLS[idx], name
    assert "check=False" in CELLS[6] and "check=False" in CELLS[8]       # exit codes are inspected, then stage_failed raises


# ====================================================================== executing the notebook's cells
@pytest.fixture
def harness(tmp_path, monkeypatch):
    g_spec = importlib.util.spec_from_file_location("wan22_gpu_test", REPO_ROOT / "scripts/wan22_gpu_test.py")
    gmod = importlib.util.module_from_spec(g_spec)
    g_spec.loader.exec_module(gmod)
    monkeypatch.setitem(sys.modules, "wan22_gpu_test", gmod)

    base, out = tmp_path / "kaggle-tmp" / "vidsgen-gpu", tmp_path / "kaggle-working"
    out.mkdir(parents=True)
    h = types.SimpleNamespace(base=base, out=out, calls=[], rc={}, ran_cells=0, ns=None)

    def fake_checkout(repo, commit, dest):
        for f in ["workflows/comfyui/wan22_ti2v_5b_t2v.json", "workflows/comfyui/wan22_ti2v_5b_i2v.json", "workflows/comfyui/wan22_ti2v_5b.mapping.json",
                  "backend/app/providers/comfyui.py", "backend/app/workflows/templates.py", "backend/tests/comfyui_support.py"]:
            (Path(dest) / f).parent.mkdir(parents=True, exist_ok=True)
            (Path(dest) / f).write_text("x")
        return commit
    monkeypatch.setattr(gmod, "checkout_commit", fake_checkout)

    ipy = types.ModuleType("IPython"); disp = types.ModuleType("IPython.display")
    disp.Video, disp.display = (lambda *a, **k: ("video", a)), (lambda *a, **k: None)
    monkeypatch.setitem(sys.modules, "IPython", ipy); monkeypatch.setitem(sys.modules, "IPython.display", disp)

    def fake_urlopen(url, timeout=30):
        path = url.split("8188", 1)[1]
        sizes = {"diffusion_models": ("wan2.2_ti2v_5B_fp16.safetensors", 9999658848), "text_encoders": ("umt5_xxl_fp8_e4m3fn_scaled.safetensors", 6735906897),
                 "vae": ("wan2.2_vae.safetensors", 1409400960)}
        if path == "/system_stats":
            body = {"system": {"comfyui_version": "0.46.0", "pytorch_version": "2.6.0+cu124"},
                    "devices": [{"name": "cuda:0 Tesla T4", "vram_total": 15 * 2**30, "vram_free": 14 * 2**30}]}
        else:
            name, size = sizes[path.rsplit("/", 1)[1]]
            body = [{"name": name, "size": h.model_size.get(name, size)}]
        return types.SimpleNamespace(read=lambda: json.dumps(body).encode(), __enter__=lambda s: s, __exit__=lambda *a: None)
    h.model_size = {}
    class _Resp:
        def __init__(self, body): self._b = body
        def read(self): return json.dumps(self._b).encode()
        def __enter__(self): return self
        def __exit__(self, *a): return False
    monkeypatch.setattr(urllib.request, "urlopen", lambda url, timeout=30: _Resp(json.loads(fake_urlopen(url).read())))

    def fake_run(cmd, env=None, check=True):
        cmd = [str(c) for c in cmd]
        h.calls.append(cmd)
        joined = " ".join(cmd)
        sub = next((s for s in ("preflight", "install-comfyui", "serve", "check-templates", "smoke-t2v", "stop") if s in cmd), None)
        if "download_models.py" in joined:
            sub = "verify-only" if "--verify-only" in cmd else "download"
        code, text = h.rc.get(sub, 0), ""
        if sub == "preflight":
            Path(cmd[cmd.index("--report") + 1]).write_text(json.dumps({"ok": code == 0, "counts": {"PASS": 9, "WARN": 0, "FAIL": 0 if code == 0 else 1, "INFO": 2}, "checks": []}))
        elif sub == "install-comfyui":
            text = "pip noise\n" + json.dumps({"commit": cmd[cmd.index("--commit") + 1], "comfyui_version": "0.46.0", "torch_replaced": False}, indent=1)
        elif sub == "serve":
            log = h.base / "logs" / "comfyui.log"; log.parent.mkdir(parents=True, exist_ok=True); log.write_text("Total VRAM 15360 MB\nAuthorization: Bearer leakedtoken\n")
            text = json.dumps({"pid": 1, "port": 8188, "listening_on": ["127.0.0.1"], "loopback_only": True, "log": str(log), "cuda_visible_devices": "0"}, indent=1)
        elif sub == "check-templates":
            text = json.dumps({"ok": code == 0, "node_classes_missing": [], "templates": {"t2v": {"ok": code == 0, "problems": []}, "i2v": {"ok": code == 0, "problems": []}}}, indent=1)
        elif sub == "smoke-t2v":
            report, video = cmd[cmd.index("--report") + 1], cmd[cmd.index("--out") + 1]
            ok = code == 0
            Path(report).write_text(json.dumps({
                "verdict": "PASS" if ok else "FAIL", "reasons": [] if ok else ["generation ended as 'failed': gpu_out_of_memory"],
                "claims_checked": {"generation_completed": ok}, "timing": {"generation_wall_s": 90.0, "total_wall_s": 95.0, "seconds_per_step": 4.1},
                "resources": {"vram_peak_used_mib": 11000, "vram_total_mib": 15360, "vram_peak_above_baseline_mib": 10500, "ram_peak_used_gib": 14.2,
                              "ram_total_gib": 31.4, "comfyui_rss_peak_gib": 12.0},
                "output": {"bytes": 70000, "container": "mov,mp4", "codec": "h264", "width": 832, "height": 480, "decoded_frames": 17, "fps": 24.0, "duration_s": 0.708} if ok else None,
                "error": None, "retry": None}))
            if ok:
                encode_mock_video(Path(video), width=64, height=64, num_frames=17, fps=24, seed=1, prompt="x")
        if check and code != 0:
            raise RuntimeError(f"STOP: exit code {code}.")
        return code, text

    def execute(overrides: dict[str, str] | None = None, stop_after: int | None = None):
        ns = {"__name__": "__main__"}
        h.ns = ns
        for n, (src, idx) in enumerate(CODE):
            if idx == 1:
                src = src.replace('BASE = "/kaggle/tmp/vidsgen-gpu"', f"BASE = {str(base)!r}").replace('OUT = Path("/kaggle/working")', f"OUT = Path({str(out)!r})")
                assert str(base) in src and str(out) in src
                for old, new in (overrides or {}).items():
                    assert old in src, old
                    src = src.replace(old, new)
            exec(compile(src, f"<cell {idx}>", "exec"), ns)
            if idx == 1:
                ns["run"], ns["hf_env"] = fake_run, lambda: {}
            h.ran_cells = n + 1
        return ns

    h.execute = execute
    return h


def commands(h):
    names = []
    for cmd in h.calls:
        j = " ".join(cmd)
        names.append("verify-only" if "--verify-only" in cmd else "download" if "download_models.py" in j else
                     next(s for s in ("preflight", "install-comfyui", "serve", "check-templates", "smoke-t2v", "stop") if s in cmd))
    return names


def test_happy_path_runs_every_stage_in_order_and_writes_a_clean_report(harness):
    ns = harness.execute()
    assert commands(harness) == ["preflight", "install-comfyui", "download", "verify-only", "serve", "check-templates", "smoke-t2v"]
    report = json.loads((harness.out / "vidsgen_phase5b_report.json").read_text())
    assert {k: v["ok"] for k, v in report["stages"].items()} == {
        "preflight": True, "vidsgen_checkout": True, "comfyui_install": True, "models": True, "server": True, "verify_server": True, "smoke_t2v": True}
    assert report["stages"]["smoke_t2v"]["verdict"] == "PASS" and report["stages"]["smoke_t2v"]["resources"]["vram_peak_used_mib"] == 11000
    assert report["pinned"]["comfyui_commit"] == "995456d0188ffb51032c28f41527cfa82d87d8a9"
    assert "Bearer" not in json.dumps(report) and "leakedtoken" not in json.dumps(report)
    assert (harness.out / "vidsgen_smoke_t2v.mp4").read_bytes()[4:8] == b"ftyp"
    # weights, tools and the install all live under BASE; only reports and the test MP4 under OUT
    assert sorted(p.name for p in harness.out.iterdir()) == ["vidsgen_phase5b_report.json", "vidsgen_phase5b_smoke_report.json", "vidsgen_preflight_5b.json", "vidsgen_smoke_t2v.mp4"]
    dl = next(c for c in harness.calls if "--execute" in c)
    assert dl[dl.index("--comfyui-dir") + 1] == str(harness.base / "comfy-data")
    assert ns["STAGES"]["server"]["loopback_only"] is True


def test_server_log_shown_to_the_user_is_redacted(harness, capsys):
    harness.execute()
    shown = capsys.readouterr().out
    assert "Total VRAM 15360 MB" in shown and "leakedtoken" not in shown


def test_commands_use_one_gpu_loopback_and_the_venv_python(harness):
    harness.execute()
    serve = next(c for c in harness.calls if "serve" in c)
    assert serve[serve.index("--gpu") + 1] == "0" and "--listen" not in serve and "0.0.0.0" not in " ".join(serve)
    venv_py = str(harness.base / "venv" / "bin" / "python")
    for sub in ("check-templates", "smoke-t2v"):
        assert next(c for c in harness.calls if sub in c)[0] == venv_py
    smoke = next(c for c in harness.calls if "smoke-t2v" in c)
    assert "--require-gpu" in smoke and "--fallback-args=--disable-dynamic-vram --disable-smart-memory" in smoke
    assert smoke[smoke.index("--frames") + 1] == "17" and smoke[smoke.index("--resolution") + 1] == "832x480"


def test_failed_preflight_stops_before_install_or_download(harness):
    harness.rc["preflight"] = 1
    with pytest.raises(RuntimeError, match="STOP at stage 'preflight'"):
        harness.execute()
    assert commands(harness) == ["preflight"]


def test_failed_install_stops_before_download(harness):
    harness.rc["install-comfyui"] = 2
    with pytest.raises(RuntimeError, match="STOP at stage 'comfyui_install'"):
        harness.execute()
    assert commands(harness) == ["preflight", "install-comfyui"]


@pytest.mark.parametrize("code,fragment", [(4, "VERIFICATION FAILED"), (3, "re-run this cell"), (2, "refused before downloading")])
def test_failed_download_stops_before_the_server_starts(harness, code, fragment):
    harness.rc["download"] = code
    with pytest.raises(RuntimeError, match=fragment):
        harness.execute()
    assert commands(harness) == ["preflight", "install-comfyui", "download"]


def test_failed_reverification_stops_before_the_server_starts(harness):
    harness.rc["verify-only"] = 1
    with pytest.raises(RuntimeError, match="re-verification"):
        harness.execute()
    assert "serve" not in commands(harness)


def test_download_disabled_never_executes_a_download_and_never_starts_the_server(harness):
    with pytest.raises(RuntimeError, match="the model files are not verified; not starting ComfyUI"):
        harness.execute({"DOWNLOAD_MODELS = True": "DOWNLOAD_MODELS = False"})
    assert commands(harness) == ["preflight", "install-comfyui"]
    assert not any("--execute" in c for c in harness.calls)


def test_server_failure_stops_before_verification(harness):
    harness.rc["serve"] = 2
    with pytest.raises(RuntimeError, match="STOP at stage 'server'"):
        harness.execute()
    assert "check-templates" not in commands(harness)


def test_model_sizes_not_matching_stops_before_the_smoke_test(harness):
    harness.model_size["wan2.2_vae.safetensors"] = 0                    # ComfyUI sees a placeholder
    with pytest.raises(RuntimeError, match="exact sizes"):
        harness.execute()
    assert "smoke-t2v" not in commands(harness) and "check-templates" not in commands(harness)


def test_template_mismatch_stops_before_the_smoke_test(harness):
    harness.rc["check-templates"] = 1
    with pytest.raises(RuntimeError, match="do not match this server"):
        harness.execute()
    assert "smoke-t2v" not in commands(harness)


def test_failed_smoke_test_is_reported_as_a_failure_not_success(harness):
    harness.rc["smoke-t2v"] = 1
    with pytest.raises(RuntimeError, match="STOP at stage 'smoke_t2v'.*gpu_out_of_memory"):
        harness.execute()
    assert not (harness.out / "vidsgen_phase5b_report.json").exists()   # the "video generated" summary cell never ran


def test_skipping_the_smoke_test_never_claims_a_video_was_generated(harness, capsys):
    harness.execute({"RUN_SMOKE_TEST = True": "RUN_SMOKE_TEST = False"})
    out = capsys.readouterr().out
    assert "No video was generated and verified in this run" in out and "A REAL VIDEO WAS GENERATED" not in out
    assert "smoke-t2v" not in commands(harness)


def test_final_cell_can_stop_the_server(harness):
    harness.execute({"STOP_SERVER_AT_END = False": "STOP_SERVER_AT_END = True"})
    assert commands(harness)[-1] == "stop"


def test_last_json_helper_extracts_the_tool_output(harness):
    ns = {"__name__": "__main__"}
    src = CELLS[1].replace('BASE = "/kaggle/tmp/vidsgen-gpu"', f"BASE = {str(harness.base)!r}").replace('OUT = Path("/kaggle/working")', f"OUT = Path({str(harness.out)!r})")
    exec(compile(src, "<settings>", "exec"), ns)
    assert ns["last_json"]('{\n "a": 1\n}') == {"a": 1}
    assert ns["last_json"]('pip noise\nWARNING x\n{\n "a": {\n  "b": 2\n }\n}\n') == {"a": {"b": 2}}
