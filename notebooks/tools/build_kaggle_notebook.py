#!/usr/bin/env python3
"""Generates notebooks/kaggle_comfyui_wan22.ipynb (Phase 5B).

The notebook is generated so that the three files it embeds (scripts/download_models.py,
scripts/wan22_gpu_test.py, workflows/comfyui/models.yaml) are byte-for-byte the repository copies.
backend/tests/test_kaggle_notebook.py fails if the committed notebook differs from this output.

    python notebooks/tools/build_kaggle_notebook.py            # rewrite the notebook
    python notebooks/tools/build_kaggle_notebook.py --check    # exit 1 if it is out of date

Why the files are embedded: Phase 5B is not committed to GitHub yet, and the notebook must not depend on
unpublished files. Once it is committed, the embedding cells can be replaced by a pinned `git fetch`.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
NOTEBOOK = REPO / "notebooks" / "kaggle_comfyui_wan22.ipynb"
EMBEDDED = [  # (file name inside TOOLS, repository path)
    ("download_models.py", "scripts/download_models.py"),
    ("wan22_gpu_test.py", "scripts/wan22_gpu_test.py"),
    ("models.yaml", "workflows/comfyui/models.yaml"),
]

MD_INTRO = r"""# VIDSGEN Phase 5B: Wan 2.2 TI2V-5B on one Kaggle GPU

Runs the **existing** VIDSGEN ComfyUI workflows on Kaggle and proves, step by step, that a real video can be generated.
Every stage has a gate: if one fails, the notebook **stops** and tells you why. Nothing is claimed from "the server started"
or "the workflow queued": success means an MP4 that decodes with the right frames, size and non-blank content.

| Stage | What happens |
|---|---|
| 1 Preflight | Python, install location, free disk, GPU + VRAM + CUDA/PyTorch, RAM, internet, and the authoritative Hugging Face **SHA-256 / size / revision** metadata checked against the manifest |
| 2 Install | The pinned VIDSGEN commit (workflows + provider) and the pinned **ComfyUI commit `995456d`** in a venv that **reuses Kaggle's PyTorch** (it is never replaced) |
| 3 Download | Exactly **3 files, 16.9 GiB**, from pinned repository revisions. Each is checked for exact size and SHA-256; any mismatch deletes the file and stops |
| 4 Server | ComfyUI on `127.0.0.1` only, on **GPU 0 only**, with a credential-free environment |
| 5 Verify | Model discovery, and both workflow templates checked against the live server's own node schemas |
| 6 Smoke test | 17-frame text-to-video at 832x480 through the real VIDSGEN `ComfyUIProvider`; peak VRAM/RAM, timing and the MP4 are verified |

**Before you run it**
1. *Settings -> Accelerator:* **GPU T4 x2**. (The two GPUs do **not** pool VRAM. Only GPU 0 is used.)
2. *Settings -> Internet:* **On**.
3. Optional: *Add-ons -> Secrets* -> `HF_TOKEN` (read-only). Not needed for these public repositories; it is never printed or saved.
4. **Run All.** Expect roughly 10-25 minutes, mostly the model download and the first generation.

**It does NOT:** open a tunnel, expose ComfyUI to the internet, use `/kaggle/working` for weights, or touch your GitHub repository.
Everything large lives in `/kaggle/tmp`. Only a small report and the test MP4 are written to `/kaggle/working`."""

C_SETTINGS = r'''# --- 1. Settings (the only cell you should need to edit) ---------------------------------------------------
import json, os, re, subprocess, sys, time
from pathlib import Path

BASE = "/kaggle/tmp/vidsgen-gpu"   # ComfyUI, venv, model weights, caches and temp files. Never /kaggle/working.
GPU = "0"                          # ComfyUI uses this ONE GPU (T4 x2 VRAM is not pooled)
PORT = 8188                        # ComfyUI listens on 127.0.0.1 only

# Pinned sources: nothing here follows a moving branch.
VIDSGEN_REPO, VIDSGEN_COMMIT = "https://github.com/Prosperotechit/VIDSGEN.git", "4b6fc67243b53cb2dcfb7d2b114dca2517380176"
COMFYUI_REPO = "https://github.com/comfyanonymous/ComfyUI.git"
COMFYUI_COMMIT, COMFYUI_VERSION = "995456d0188ffb51032c28f41527cfa82d87d8a9", "0.46.0"

DOWNLOAD_MODELS = True    # ~16.9 GiB (3 files), verified by exact size + SHA-256. False = stop after preflight + install.
RUN_SMOKE_TEST = True
SMOKE = dict(resolution="832x480", frames=17, steps=20, fps=24, cfg=5.0, seed=424242)
FALLBACK_ARGS = ["--disable-dynamic-vram", "--disable-smart-memory"]   # ONE retry, only after an out-of-memory failure
STOP_SERVER_AT_END = False                                              # leave ComfyUI running for the next phase

OUT = Path("/kaggle/working")      # only the small JSON reports and the test MP4 go here
TOOLS, DATA = f"{BASE}/tools", f"{BASE}/comfy-data"
VENV_PY = f"{BASE}/venv/bin/python"
URL = f"http://127.0.0.1:{PORT}"
STAGES = {}                        # collected for the final report


def run(cmd, env=None, check=True):
    """Run a command and stream its output live. Returns (exit code, output text)."""
    print("$", " ".join(str(c) for c in cmd), flush=True)
    proc = subprocess.Popen([str(c) for c in cmd], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env, bufsize=1)
    lines = []
    for line in proc.stdout:
        print(line, end="", flush=True)
        lines.append(line)
    proc.wait()
    if check and proc.returncode != 0:
        raise RuntimeError(f"STOP: exit code {proc.returncode}. Read the output above, fix the cause, then re-run from this cell.")
    return proc.returncode, "".join(lines)


def last_json(text):
    """The last top-level JSON object printed by one of the tools."""
    start = text.rfind("\n{\n")
    start = 0 if text.startswith("{\n") and start < 0 else start + 1
    return json.loads(text[start:])


def hf_env():
    """Environment for the preflight/downloader. HF_TOKEN comes only from a Kaggle Secret and is never printed."""
    env = dict(os.environ)
    try:
        from kaggle_secrets import UserSecretsClient
        token = UserSecretsClient().get_secret("HF_TOKEN")
        if token:
            env["HF_TOKEN"] = token
            print("HF_TOKEN secret: attached (value not shown)")
    except Exception:
        print("HF_TOKEN secret: not attached (fine for these public repositories)")
    return env


def stage_failed(name, why):
    STAGES[name] = {"ok": False, "why": why}
    raise RuntimeError(f"STOP at stage '{name}': {why}")'''

C_EMBED_DIR = r'''# --- 2. Tools used by this notebook (embedded verbatim from the VIDSGEN repository) -------------------------
# They are written to /kaggle/tmp; the next three cells contain the exact source, so you can read what will run.
os.makedirs(TOOLS, exist_ok=True)
print("tools directory:", TOOLS)'''

C_PREFLIGHT = r'''# --- 3. Stage 1: preflight (nothing is installed or downloaded yet) ----------------------------------------
# Gates: Python >= 3.11, a non-persistent install location with enough free disk, GPU 0 + VRAM + CUDA/PyTorch
# compatibility, RAM, internet, and the authoritative Hugging Face metadata (pinned revision, size, SHA-256).
code, out = run([sys.executable, f"{TOOLS}/wan22_gpu_test.py", "preflight", "--base", BASE, "--manifest", f"{TOOLS}/models.yaml",
                 "--gpu", GPU, "--repo", VIDSGEN_REPO, "--repo", COMFYUI_REPO, "--report", f"{OUT}/vidsgen_preflight_5b.json"],
                env=hf_env(), check=False)
STAGES["preflight"] = {"ok": code == 0, **json.loads(Path(f"{OUT}/vidsgen_preflight_5b.json").read_text())}
if code != 0:
    stage_failed("preflight", "one or more checks FAILED (see the [FAIL] lines above). Nothing was installed or downloaded.")'''

C_CHECKOUT = r'''# --- 4. Stage 2a: the pinned VIDSGEN commit (existing workflows, ComfyUIProvider, validators) ----------------
sys.path.insert(0, TOOLS)
import wan22_gpu_test as gt

head = gt.checkout_commit(VIDSGEN_REPO, VIDSGEN_COMMIT, Path(BASE, "vidsgen"))
needed = ["workflows/comfyui/wan22_ti2v_5b_t2v.json", "workflows/comfyui/wan22_ti2v_5b_i2v.json", "workflows/comfyui/wan22_ti2v_5b.mapping.json",
          "backend/app/providers/comfyui.py", "backend/app/workflows/templates.py", "backend/tests/comfyui_support.py"]
missing = [f for f in needed if not Path(BASE, "vidsgen", f).is_file()]
if missing:
    stage_failed("vidsgen_checkout", f"files missing from the pinned commit: {missing}")
STAGES["vidsgen_checkout"] = {"ok": True, "commit": head}
print("VIDSGEN checked out at", head)'''

C_INSTALL = r'''# --- 5. Stage 2b: ComfyUI at the pinned commit, in a venv that REUSES Kaggle's PyTorch --------------------
# pip runs with a constraints file pinning torch/torchvision/torchaudio/numpy to what is already installed, and the
# result is verified: the venv must import the host's identical PyTorch file. Takes a few minutes.
code, out = run([sys.executable, f"{TOOLS}/wan22_gpu_test.py", "install-comfyui", "--base", BASE,
                 "--commit", COMFYUI_COMMIT, "--expect-version", COMFYUI_VERSION], env=hf_env(), check=False)
if code != 0:
    stage_failed("comfyui_install", "installation failed (see the error above); the host's PyTorch was not modified.")
STAGES["comfyui_install"] = {"ok": True, **last_json(out)}'''

C_DOWNLOAD = r'''# --- 6. Stage 3: download + verify the three model files (exactly 16.9 GiB) -----------------------------------
# Only the files in the manifest, from their pinned revisions. Size AND SHA-256 are checked; a mismatch deletes the
# file and stops. An interrupted download resumes (HTTP Range) and is trusted only after the whole-file SHA-256 matches.
if not DOWNLOAD_MODELS:
    STAGES["models"] = {"ok": None, "skipped": "DOWNLOAD_MODELS is False"}
    print("DOWNLOAD_MODELS is False: skipping the download. Later stages need the model files.")
else:
    code, _ = run([sys.executable, f"{TOOLS}/download_models.py", "--manifest", f"{TOOLS}/models.yaml", "--comfyui-dir", DATA, "--execute"],
                  env=hf_env(), check=False)
    if code != 0:
        why = {2: "refused before downloading (manifest or disk space)", 3: "network problem; the partial download is kept: just re-run this cell",
               4: "VERIFICATION FAILED: the file was deleted"}.get(code, f"exit code {code}")
        stage_failed("models", why)
    print("\n--- independent re-verification of the files on disk (size + SHA-256, no network) ---")
    code, out = run([sys.executable, f"{TOOLS}/download_models.py", "--manifest", f"{TOOLS}/models.yaml", "--comfyui-dir", DATA, "--verify-only"],
                    check=False)
    if code != 0:
        stage_failed("models", "re-verification of the downloaded files failed")
    STAGES["models"] = {"ok": True, "verified": "exact size + SHA-256 (download-time and again from disk)", "files": 3}'''

C_SERVE = r'''# --- 7. Stage 4: start ComfyUI (127.0.0.1 only, GPU 0 only, credential-free environment) ---------------------
if STAGES["models"].get("ok") is not True:
    stage_failed("server", "the model files are not verified; not starting ComfyUI")
code, out = run([sys.executable, f"{TOOLS}/wan22_gpu_test.py", "serve", "--base", BASE, "--port", str(PORT), "--gpu", GPU, "--wait", "300"],
                check=False)
if code != 0:
    stage_failed("server", "ComfyUI did not start cleanly (see the error and log tail above)")
server = last_json(out)
STAGES["server"] = {"ok": True, **server}
print("\nloopback only:", server["loopback_only"], "| listening on:", server["listening_on"], "| CUDA_VISIBLE_DEVICES:", server["cuda_visible_devices"])
print("\n--- ComfyUI start-up log (first 30 lines; secrets redacted) ---")
print(gt.redact("\n".join(Path(server["log"]).read_text(errors="replace").splitlines()[:30])))'''

C_VERIFY = r'''# --- 8. Stage 5: model discovery + both workflow templates against the LIVE server ---------------------------
import urllib.request


def api(path):
    with urllib.request.urlopen(URL + path, timeout=30) as r:
        return json.loads(r.read())


stats = api("/system_stats")
dev = stats["devices"][0]
print(f"ComfyUI {stats['system']['comfyui_version']} | device: {dev['name']} | VRAM {dev['vram_total'] / 2**30:.1f} GiB total, "
      f"{dev['vram_free'] / 2**30:.1f} GiB free | PyTorch {stats['system']['pytorch_version']}")
manifest_files = {"diffusion_models": ("wan2.2_ti2v_5B_fp16.safetensors", 9999658848), "text_encoders": ("umt5_xxl_fp8_e4m3fn_scaled.safetensors", 6735906897),
                  "vae": ("wan2.2_vae.safetensors", 1409400960)}
discovery = {}
for folder, (name, size) in manifest_files.items():
    listed = {m["name"]: m["size"] for m in api(f"/experiment/models/{folder}")}
    discovery[folder] = {"discovered": name in listed, "size_matches": listed.get(name) == size}
    print(f"  models/{folder}: {name} -> discovered={discovery[folder]['discovered']}, exact size={discovery[folder]['size_matches']}")
if not all(d["discovered"] and d["size_matches"] for d in discovery.values()):
    stage_failed("verify_server", "ComfyUI does not see all three model files with their exact sizes")

code, out = run([VENV_PY, f"{TOOLS}/wan22_gpu_test.py", "check-templates", "--url", URL, "--backend-dir", f"{BASE}/vidsgen/backend",
                 "--workflows-dir", f"{BASE}/vidsgen/workflows/comfyui"], check=False)
templates = last_json(out)
STAGES["verify_server"] = {"ok": code == 0, "discovery": discovery, "templates": templates,
                           "device": dev["name"], "vram_total_bytes": dev["vram_total"]}
if code != 0:
    stage_failed("verify_server", f"workflow templates do not match this server: missing nodes {templates['node_classes_missing']}, "
                                  f"problems {templates['templates']}")'''

C_SMOKE = r'''# --- 9. Stage 6: the first real inference: a small text-to-video generation ---------------------------------
# Runs through the real VIDSGEN ComfyUIProvider and the existing Phase 4 workflow. PASS requires a generated MP4 that decodes
# with the requested frame count and size and is not blank. If (and only if) it fails with an out-of-memory error, ComfyUI is
# restarted ONCE with FALLBACK_ARGS and the test is repeated. Watch the heartbeat lines below: the first run also loads the models.
if not RUN_SMOKE_TEST:
    STAGES["smoke_t2v"] = {"ok": None, "skipped": "RUN_SMOKE_TEST is False"}
    print("RUN_SMOKE_TEST is False: skipping.")
else:
    report_path, video_path = f"{OUT}/vidsgen_phase5b_smoke_report.json", f"{OUT}/vidsgen_smoke_t2v.mp4"
    code, _ = run([VENV_PY, f"{TOOLS}/wan22_gpu_test.py", "smoke-t2v", "--base", BASE, "--url", URL, "--port", str(PORT), "--gpu", GPU,
                   "--backend-dir", f"{BASE}/vidsgen/backend", "--workflows-dir", f"{BASE}/vidsgen/workflows/comfyui",
                   "--resolution", SMOKE["resolution"], "--frames", str(SMOKE["frames"]), "--steps", str(SMOKE["steps"]),
                   "--fps", str(SMOKE["fps"]), "--cfg", str(SMOKE["cfg"]), "--seed", str(SMOKE["seed"]), "--require-gpu",
                   f"--fallback-args={' '.join(FALLBACK_ARGS)}", "--out", video_path, "--report", report_path], check=False)
    smoke = json.loads(Path(report_path).read_text())
    STAGES["smoke_t2v"] = {"ok": code == 0, "report": report_path, "video": video_path if code == 0 else None, **{
        k: smoke.get(k) for k in ("verdict", "reasons", "claims_checked", "timing", "resources", "output", "error", "retry")}}
    res, tim = smoke["resources"], smoke["timing"]
    print("\n" + "=" * 78)
    print(f"SMOKE TEST VERDICT: {smoke['verdict']}")
    print("=" * 78)
    print("claims checked :", smoke["claims_checked"])
    print(f"peak GPU memory: {res.get('vram_peak_used_mib')} MiB of {res.get('vram_total_mib')} MiB "
          f"(+{res.get('vram_peak_above_baseline_mib')} MiB above the idle baseline)")
    print(f"peak RAM       : {res.get('ram_peak_used_gib')} GiB used of {res.get('ram_total_gib')} GiB | ComfyUI process peak RSS {res.get('comfyui_rss_peak_gib')} GiB")
    print(f"time           : generation {tim.get('generation_wall_s')} s, total {tim.get('total_wall_s')} s, {tim.get('seconds_per_step')} s/step")
    if code == 0:
        o = smoke["output"]
        print(f"output         : {video_path} ({o['bytes']:,} bytes; {o['container']}, {o['codec']}, {o['width']}x{o['height']}, "
              f"{o['decoded_frames']} frames, {o['fps']} fps, {o['duration_s']} s)")
        from IPython.display import Video, display
        display(Video(video_path, embed=True, width=480))
    else:
        print("reasons        :", smoke["reasons"])
        stage_failed("smoke_t2v", "; ".join(smoke["reasons"]))'''

C_REPORT = r'''# --- 10. Final report (contains no secrets) ----------------------------------------------------------------
claims = {name: (s.get("ok")) for name, s in STAGES.items()}
report = {"generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "stages": STAGES,
          "pinned": {"vidsgen_commit": VIDSGEN_COMMIT, "comfyui_commit": COMFYUI_COMMIT, "comfyui_version": COMFYUI_VERSION},
          "settings": {"gpu": GPU, "smoke": SMOKE, "fallback_args": FALLBACK_ARGS, "base": BASE}}
text = json.dumps(report, indent=1, default=str)
assert "Bearer " not in text and not re.search(r"hf_[A-Za-z0-9]{20,}", text), "token-like text found; report not written"
Path(f"{OUT}/vidsgen_phase5b_report.json").write_text(text)

print("Stage results (True = passed, None = skipped by your settings):")
for name, ok in claims.items():
    print(f"  {name:<18} {ok}")
generated = STAGES.get("smoke_t2v", {}).get("ok") is True
print("\n" + ("A REAL VIDEO WAS GENERATED AND VERIFIED (decodes, correct frames and size, not blank)." if generated else
              "No video was generated and verified in this run: do NOT treat the GPU path as working."))
print(f"\nSend back: {OUT}/vidsgen_phase5b_report.json  (and {OUT}/vidsgen_phase5b_smoke_report.json if it exists).")
print("ComfyUI is " + ("stopped." if STOP_SERVER_AT_END else f"still running on 127.0.0.1:{PORT} (loopback only). No tunnel was started."))
if STOP_SERVER_AT_END:
    run([sys.executable, f"{TOOLS}/wan22_gpu_test.py", "stop", "--base", BASE], check=False)'''

MD_END = r"""## What to send back
* `/kaggle/working/vidsgen_phase5b_report.json` (and `vidsgen_phase5b_smoke_report.json`): no secrets, safe to share.
* If a stage stopped the run, the `[FAIL]` / `STOP` lines above it.

## Stop ComfyUI when you are done
Run `!python /kaggle/tmp/vidsgen-gpu/tools/wan22_gpu_test.py stop --base /kaggle/tmp/vidsgen-gpu`, or just end the session.

## Not in this notebook (separate approval)
The token-protected proxy and the Cloudflare quick tunnel, connecting the VIDSGEN backend to this GPU, and the image-to-video test."""


def embed_cell(name: str, rel: str) -> str:
    content = (REPO / rel).read_text()
    if "'''" in content:
        raise ValueError(f"{rel} contains ''' and cannot be embedded in a raw triple-quoted string")
    if content.endswith("\\"):
        raise ValueError(f"{rel} ends with a backslash")
    return (f"# Embedded verbatim from {rel}. (Phase 5B is not on GitHub yet; a test keeps this copy identical.)\n"
            f"Path(TOOLS, {name!r}).write_text(r'''{content}''')\n"
            f"print('wrote', Path(TOOLS, {name!r}), Path(TOOLS, {name!r}).stat().st_size, 'bytes')")


def source_lines(text: str) -> list[str]:
    lines = text.split("\n")
    return [l + "\n" for l in lines[:-1]] + [lines[-1]]


def build() -> dict:
    cells: list[tuple[str, str]] = [("markdown", MD_INTRO), ("code", C_SETTINGS), ("code", C_EMBED_DIR)]
    cells += [("code", embed_cell(name, rel)) for name, rel in EMBEDDED]
    cells += [("code", C_PREFLIGHT), ("code", C_CHECKOUT), ("code", C_INSTALL), ("code", C_DOWNLOAD), ("code", C_SERVE),
              ("code", C_VERIFY), ("code", C_SMOKE), ("code", C_REPORT), ("markdown", MD_END)]
    nb_cells = []
    for i, (kind, text) in enumerate(cells):
        cell = {"cell_type": kind, "id": f"vidsgen-5b-{i:02d}", "metadata": {}, "source": source_lines(text)}
        if kind == "code":
            cell.update(execution_count=None, outputs=[])
        nb_cells.append(cell)
    return {"nbformat": 4, "nbformat_minor": 5,
            "metadata": {"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"},
                         "language_info": {"name": "python"},
                         "kaggle": {"accelerator": "nvidiaTeslaT4", "isInternetEnabled": True, "isGpuEnabled": True}},
            "cells": nb_cells}


def main() -> int:
    text = json.dumps(build(), indent=1, ensure_ascii=False) + "\n"
    if "--check" in sys.argv:
        ok = NOTEBOOK.exists() and NOTEBOOK.read_text() == text
        print("notebook is up to date" if ok else "notebook is OUT OF DATE: run notebooks/tools/build_kaggle_notebook.py")
        return 0 if ok else 1
    NOTEBOOK.write_text(text)
    print(f"wrote {NOTEBOOK.relative_to(REPO)} ({len(text):,} bytes, {len(build()['cells'])} cells)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
