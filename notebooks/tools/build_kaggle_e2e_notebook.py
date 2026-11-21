#!/usr/bin/env python3
"""Generates notebooks/kaggle_backend_e2e_phase6b.ipynb (Phase 6B): the VIDSGEN FastAPI backend and ComfyUI together on
one Kaggle GPU, talking over localhost only.

It reuses the verified Phase 5B stage cells (preflight, pinned ComfyUI install, verified model download, loopback server,
template check) unchanged, imported from build_kaggle_notebook.py, and adds the backend install and the end-to-end test.
The Phase 5B notebook itself is not touched by this generator.

    python notebooks/tools/build_kaggle_e2e_notebook.py            # rewrite the notebook
    python notebooks/tools/build_kaggle_e2e_notebook.py --check    # exit 1 if it is out of date
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_kaggle_notebook as b5  # noqa: E402

REPO = b5.REPO
NOTEBOOK = REPO / "notebooks" / "kaggle_backend_e2e_phase6b.ipynb"
# The VIDSGEN commit the backend and workflows are taken from: Phase 6A (it contains the generation limits).
BACKEND_COMMIT = "1f9d5f40fc62a4bb933333d04ab37c7a5de57342"
EMBEDDED = [*b5.EMBEDDED, ("backend_e2e_test.py", "scripts/backend_e2e_test.py")]

MD_INTRO = r"""# VIDSGEN Phase 6B: backend + ComfyUI on one Kaggle GPU (localhost only)

Runs the **real VIDSGEN FastAPI backend** and **ComfyUI** together on this Kaggle machine and drives the backend through its public API
with one real text-to-video job. Both services listen on `127.0.0.1` only. **No tunnel, no proxy, nothing is exposed.**

Success is never inferred from "the server started" or "the job was queued". The test passes only if **every** check passes:
health, the generation limits rejecting out-of-range requests, a real job with live progress, a completed job, download and
HTTP Range support, an MP4 that fully decodes with the right frames/size/codec and is not blank, and `inference` turning `ok`
in the health report after a real generation.

| Stage | What happens |
|---|---|
| 1-5 | Exactly the verified Phase 5B stages: preflight, pinned ComfyUI install (Kaggle's PyTorch is never replaced), verified model download (16.9 GiB), loopback ComfyUI on GPU 0, live template check |
| 6 Backend install | The backend's own dependencies go into the same venv under the same protection: PyTorch, torchvision, torchaudio and NumPy stay the host's copies and the host's Python environment is verified unchanged |
| 7 End-to-end | The backend starts with `GENERATION_PROFILE=t4_verified` and **one job at a time**; out-of-limit requests are rejected; one job (**832x480, 17 frames, 20 steps, 24 fps, CFG 5.0**) is submitted, tracked, downloaded and verified |

**Before you run it**
1. *Settings -> Accelerator:* **GPU T4 x2** (only GPU 0 is used; VRAM is not pooled). *Internet:* **On**.
2. Optional: *Add-ons -> Secrets* -> `HF_TOKEN` (not needed for these public repositories; never printed or saved).
3. **Run All.** This notebook is self-contained: a new Kaggle session starts empty, so it installs ComfyUI and downloads the models again
   (roughly 15-30 minutes in total). The VIDSGEN repository is public, so no GitHub credential is needed.

**It does NOT:** open a tunnel, expose either service, use `/kaggle/working` for weights, or change your GitHub repository."""

C_E2E_SETTINGS = r'''# --- 1b. Phase 6B settings -----------------------------------------------------------------------------------------
BACKEND_PORT = 8000          # the FastAPI backend listens on 127.0.0.1 only
RUN_E2E = True
RUN_CANCEL_TEST = False      # True: also cancel a SECOND job while it is being processed (extra GPU time, real interrupt)
E2E_TIMEOUT_S = 2400
STOP_SERVER_AT_END = True    # stop ComfyUI at the end of the notebook (the backend stops itself)
# The test always uses the verified values: 832x480, 17 frames, 20 steps, 24 fps, CFG 5.0, GENERATION_PROFILE=t4_verified, 1 job at a time.'''

C_BACKEND_INSTALL = r'''# --- 8. Stage 6: the backend's dependencies, into the SAME venv, with PyTorch/NumPy protected --------------------
# Runs under the HOST python so the host's PyTorch/NumPy can be compared with the venv's. Afterwards the venv must import the
# host's identical torch, torchvision, torchaudio and numpy, the host's installed packages must be unchanged, and `import app.main` must work.
code, out = run([sys.executable, f"{TOOLS}/backend_e2e_test.py", "install-backend", "--base", BASE, "--backend-dir", f"{BASE}/vidsgen/backend"],
                env=hf_env(), check=False)
if code != 0:
    stage_failed("backend_install", "installing the backend failed (see the error above); the host's PyTorch and NumPy were not modified.")
STAGES["backend_install"] = {"ok": True, **last_json(out)}
print("\nbackend installed | host environment unchanged:", STAGES["backend_install"]["host_environment_unchanged"],
      "| protected:", sorted(STAGES["backend_install"]["protected_packages"]))'''

C_E2E = r'''# --- 9. Stage 7: the end-to-end test (backend <-> ComfyUI over 127.0.0.1) -------------------------------------
# The backend runs as a separate process (provider=comfyui, GENERATION_PROFILE=t4_verified, one job at a time, loopback only) and is driven
# ONLY through its HTTP API. The first generation loads the models: watch the heartbeat lines. Nothing here is a mock.
if not RUN_E2E:
    STAGES["backend_e2e"] = {"ok": None, "skipped": "RUN_E2E is False"}
    print("RUN_E2E is False: skipping.")
else:
    report_path, video_path = f"{OUT}/vidsgen_phase6b_e2e_report.json", f"{OUT}/vidsgen_phase6b_e2e.mp4"
    cmd = [VENV_PY, f"{TOOLS}/backend_e2e_test.py", "e2e", "--base", BASE, "--backend-dir", f"{BASE}/vidsgen/backend",
           "--workflows-dir", f"{BASE}/vidsgen/workflows/comfyui", "--comfyui-url", URL, "--backend-port", str(BACKEND_PORT),
           "--gpu", GPU, "--timeout", str(E2E_TIMEOUT_S), "--require-gpu", "--out", video_path, "--report", report_path]
    if RUN_CANCEL_TEST:
        cmd.append("--with-cancel")
    code, _ = run(cmd, check=False)
    e2e = json.loads(Path(report_path).read_text())
    STAGES["backend_e2e"] = {"ok": code == 0, "report": report_path, "video": video_path if code == 0 else None,
                             **{k: e2e.get(k) for k in ("verdict", "reasons", "claims", "timing", "resources", "envelope_probes", "output")}}
    print("\n" + "=" * 78)
    print(f"BACKEND END-TO-END VERDICT: {e2e['verdict']}")
    print("=" * 78)
    for name, ok in e2e["claims"].items():
        print(f"  {'PASS' if ok else ('skip' if ok is None else 'FAIL')}  {name}")
    res, tim = e2e.get("resources", {}), e2e.get("timing", {})
    print(f"\npeak GPU memory: {res.get('vram_peak_used_mib')} MiB of {res.get('vram_total_mib')} MiB | peak RAM {res.get('ram_peak_used_gib')} GiB "
          f"of {res.get('ram_total_gib')} GiB | job {tim.get('job_wall_s')} s, total {tim.get('total_wall_s')} s")
    if code == 0:
        o = e2e["output"]
        print(f"output: {video_path} ({o['bytes']:,} bytes; {o['container']}, {o['codec']}, {o['width']}x{o['height']}, "
              f"{o['decoded_frames']} frames, {o['fps']} fps)")
        from IPython.display import Video, display
        display(Video(video_path, embed=True, width=480))
    else:
        print("reasons:", e2e["reasons"])
        stage_failed("backend_e2e", "; ".join(e2e["reasons"]))'''

C_REPORT = r'''# --- 10. Final report (contains no secrets) ----------------------------------------------------------------
report = {"generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "stages": STAGES,
          "pinned": {"vidsgen_commit": VIDSGEN_COMMIT, "comfyui_commit": COMFYUI_COMMIT, "comfyui_version": COMFYUI_VERSION},
          "settings": {"gpu": GPU, "backend_port": BACKEND_PORT, "profile": "t4_verified", "max_concurrent_jobs": 1,
                       "request": dict(resolution="832x480", frames=17, steps=20, fps=24, cfg=5.0), "cancel_test": RUN_CANCEL_TEST, "base": BASE}}
text = json.dumps(report, indent=1, default=str)
assert "Bearer " not in text and not re.search(r"hf_[A-Za-z0-9]{20,}", text), "token-like text found; report not written"
Path(f"{OUT}/vidsgen_phase6b_report.json").write_text(text)

print("Stage results (True = passed, None = skipped by your settings):")
for name, s in STAGES.items():
    print(f"  {name:<18} {s.get('ok')}")
passed = STAGES.get("backend_e2e", {}).get("ok") is True
print("\n" + ("THE BACKEND GENERATED A REAL VIDEO THROUGH COMFYUI ON THIS GPU, AND EVERY CHECK PASSED (localhost only)." if passed else
              "The backend-to-GPU end-to-end test did NOT pass in this run: do not treat the integration as working."))
print("\nNot validated by this notebook: image-to-video, a frontend in front of this backend, the proxy/tunnel, any production deployment.")
print(f"\nSend back: {OUT}/vidsgen_phase6b_report.json and {OUT}/vidsgen_phase6b_e2e_report.json (no secrets).")
if STOP_SERVER_AT_END:
    run([sys.executable, f"{TOOLS}/wan22_gpu_test.py", "stop", "--base", BASE], check=False)
    run([sys.executable, f"{TOOLS}/backend_e2e_test.py", "stop-backend", "--base", BASE], check=False)
else:
    print(f"ComfyUI is still running on 127.0.0.1:{PORT} (loopback only). No tunnel was started.")'''

MD_END = r"""## What to send back
* `/kaggle/working/vidsgen_phase6b_report.json` and `/kaggle/working/vidsgen_phase6b_e2e_report.json`: no secrets, safe to share.
* If a stage stopped the run, the `[FAIL]` / `STOP` lines above it.

## Stop everything
The last cell stops ComfyUI and the backend (`STOP_SERVER_AT_END`). Ending the session also stops them.

## Not in this notebook (separate approval)
The token-protected proxy, any tunnel, image-to-video on the GPU, and the frontend."""


def embed_cell(name: str, rel: str) -> str:
    content = (REPO / rel).read_text()
    if "'''" in content:
        raise ValueError(f"{rel} contains ''' and cannot be embedded in a raw triple-quoted string")
    if content.endswith("\\"):
        raise ValueError(f"{rel} ends with a backslash")
    return (f"# Embedded verbatim from {rel}, so this notebook is self-contained. A test keeps this copy identical.\n"
            f"Path(TOOLS, {name!r}).write_text(r'''{content}''')\n"
            f"print('wrote', Path(TOOLS, {name!r}), Path(TOOLS, {name!r}).stat().st_size, 'bytes')")


def settings_cell() -> str:
    """The Phase 5B settings cell with the VIDSGEN commit moved to the Phase 6A commit (which has the generation limits)."""
    old = '"4b6fc67243b53cb2dcfb7d2b114dca2517380176"'
    cell = b5.C_SETTINGS
    assert cell.count(old) == 1, "the Phase 5B settings cell no longer pins the expected commit"
    return cell.replace(old, f'"{BACKEND_COMMIT}"').replace("# --- 1. Settings", "# --- 1. Settings (Phase 6B)", 1)


def build() -> dict:
    cells: list[tuple[str, str]] = [("markdown", MD_INTRO), ("code", settings_cell()), ("code", C_E2E_SETTINGS), ("code", b5.C_EMBED_DIR)]
    cells += [("code", embed_cell(name, rel)) for name, rel in EMBEDDED]
    cells += [("code", b5.C_PREFLIGHT), ("code", b5.C_CHECKOUT), ("code", b5.C_INSTALL), ("code", b5.C_DOWNLOAD), ("code", b5.C_SERVE),
              ("code", b5.C_VERIFY), ("code", C_BACKEND_INSTALL), ("code", C_E2E), ("code", C_REPORT), ("markdown", MD_END)]
    nb_cells = []
    for i, (kind, text) in enumerate(cells):
        cell = {"cell_type": kind, "id": f"vidsgen-6b-{i:02d}", "metadata": {}, "source": b5.source_lines(text)}
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
        print("notebook is up to date" if ok else "notebook is OUT OF DATE: run notebooks/tools/build_kaggle_e2e_notebook.py")
        return 0 if ok else 1
    NOTEBOOK.write_text(text)
    print(f"wrote {NOTEBOOK.relative_to(REPO)} ({len(text):,} bytes, {len(build()['cells'])} cells)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
