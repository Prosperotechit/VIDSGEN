#!/usr/bin/env python3
"""VIDSGEN Phase 6B: a controlled backend-to-ComfyUI end-to-end test that runs ENTIRELY on one machine's loopback
(designed for a Kaggle GPU notebook). Nothing is exposed: no tunnel, no proxy, no public address.

    install-backend  Add the VIDSGEN backend's own dependencies to the existing ComfyUI venv, under the same protection
                     as the ComfyUI install: PyTorch, torchvision, torchaudio and NumPy are pinned to the host's copies and
                     the host's Python environment is checked to be unchanged.
    e2e              Start the real FastAPI backend on 127.0.0.1 (provider=comfyui, GENERATION_PROFILE=t4_verified, one job
                     at a time), then drive it ONLY through its public HTTP API: health, rejected out-of-envelope requests,
                     one real text-to-video job (832x480, 17 frames, 20 steps, 24 fps, CFG 5.0), progress polling, download,
                     Range request, and a full decode of the MP4. PASS requires every check; nothing is claimed from "the
                     server started" or "the job was queued".
    stop-backend     Stop the backend started by `e2e` (it also stops it itself when it finishes).

It reuses the helpers of scripts/wan22_gpu_test.py (which must sit next to this file) and does not change them.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import struct
import subprocess
import sys
import time
import tomllib
import zlib
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import wan22_gpu_test as gt  # noqa: E402  (the Phase 5B helpers)

# The one configuration verified on a Kaggle T4 in Phase 5B. The API defaults are NOT used: every value is sent explicitly.
PROVEN = {"mode": "t2v", "resolution": "832x480", "num_frames": 17, "steps": 20, "fps": 24, "guidance_scale": 5.0}
DEFAULT_PROMPT = gt.DEFAULT_PROMPT
LOOPBACK_HOSTS = ("127.0.0.1", "localhost", "::1", "[::1]")
ORDER = {"queued": 0, "processing": 1, "completed": 2, "failed": 2, "cancelled": 2}
_CHILDREN: dict[int, subprocess.Popen] = {}


# ===================================================================================== configuration
def loopback_endpoint(url: str) -> tuple[str, int]:
    """(host, port) of a plain-http loopback URL. Anything else is refused: this test must never leave the machine."""
    from urllib.parse import urlsplit

    p = urlsplit(url)
    if p.scheme != "http" or (p.hostname or "") not in {h.strip("[]") for h in LOOPBACK_HOSTS} or not p.port:
        raise ValueError(f"the ComfyUI URL must be http://127.0.0.1:<port> (loopback only), got {gt.redact(url)!r}")
    return p.hostname, p.port


def backend_requirements(backend_dir: Path) -> list[str]:
    """The backend's runtime dependencies, read from its pyproject.toml so they cannot drift."""
    deps = tomllib.loads((backend_dir / "pyproject.toml").read_text())["project"]["dependencies"]
    return [d.split("#")[0].strip() for d in deps]


def app_settings(base: Path, workflows_dir: Path, comfy_url: str, port: int) -> dict[str, str]:
    """The backend's settings for this test. Every value is non-secret and is recorded in the report."""
    loopback_endpoint(comfy_url)
    return {
        "APP_ENV": "production", "API_HOST": "127.0.0.1", "API_PORT": str(port), "LOG_LEVEL": "INFO",
        "CORS_ORIGINS": "",                                   # no browser origin is allowed: the API is driven by this script
        "INFERENCE_PROVIDER": "comfyui", "COMFYUI_BASE_URL": comfy_url, "COMFYUI_WORKFLOWS_DIR": str(workflows_dir),
        "COMFYUI_TIMEOUT": "60", "COMFYUI_CONNECTION_GRACE_SECONDS": "30", "INFERENCE_TIMEOUT_SECONDS": "1200",
        "GENERATION_PROFILE": "t4_verified", "INFERENCE_MAX_CONCURRENT_JOBS": "1",
        "STORAGE_LOCAL_ROOT": str(base / "storage"),
        # No COMFYUI_API_TOKEN: ComfyUI is on loopback, there is no proxy in this phase.
    }


def backend_env(base: Path, settings: dict[str, str], environ: dict[str, str] | None = None) -> dict[str, str]:
    """Environment for the backend process: ALLOW-LISTED variables only (no token or credential can reach it), caches
    inside the base directory, plus the settings above."""
    env = gt.server_env(base, None, environ)
    env.update(settings)
    return env


def png_bytes(width: int = 256, height: int = 256, rgb: tuple[int, int, int] = (30, 120, 200)) -> bytes:
    """A valid solid-colour PNG without any imaging library (used to probe the image-to-video rejection)."""
    def chunk(kind: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)

    raw = b"".join(b"\x00" + bytes(rgb) * width for _ in range(height))
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b""))


# ===================================================================================== install-backend
def install_backend(base: Path, backend_dir: Path, host_python: str = sys.executable) -> dict:
    """pip-install the backend's dependencies into <base>/venv, protecting the host's PyTorch/NumPy (see install_comfyui)."""
    venv, constraints = base / "venv", base / "constraints.txt"
    if not gt.venv_python(venv).exists() or not constraints.exists():
        raise RuntimeError(f"the ComfyUI venv and constraints file are missing in {base}: run install-comfyui first")
    for needed in ("pyproject.toml", "app/main.py", "app/jobs/envelope.py"):
        if not (backend_dir / needed).is_file():
            raise RuntimeError(f"{backend_dir / needed} is missing: the VIDSGEN checkout must include Phase 6A (the generation limits)")
    env = gt.clean_env(None, **gt.cache_env(base))
    inventory_before = gt.host_inventory(host_python, env, base)
    requirements = base / "requirements-backend.txt"
    requirements.write_text("\n".join(backend_requirements(backend_dir)) + "\n")
    pip = gt.pip_info(str(gt.venv_python(venv)), venv, env)
    cmd = [str(gt.venv_python(venv)), "-m", "pip", "install", "--no-cache-dir", "--disable-pip-version-check", "-q",
           "-c", str(constraints), "-r", str(requirements)]
    t0 = time.monotonic()
    p = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=3600)
    if p.returncode != 0:
        raise RuntimeError("pip install failed (PyTorch and NumPy were protected by the constraints file):\n"
                           + gt.redact(p.stdout + p.stderr)[-1500:])
    host_seen, seen = gt.import_probe(host_python, env), gt.import_probe(str(gt.venv_python(venv)), env)
    problems = gt.protected_packages_replaced(host_seen, seen, venv)
    if problems:
        raise RuntimeError("the host's PyTorch/NumPy stack was replaced: " + "; ".join(problems))
    changes = gt.inventory_changes(inventory_before, gt.host_inventory(host_python, env, base))
    if changes:
        raise RuntimeError("the host's Python environment was modified by the install (it must stay untouched): "
                           + "; ".join(changes[:12]))
    importable = backend_importable(str(gt.venv_python(venv)), backend_dir, base)
    if importable:
        raise RuntimeError("the backend cannot be imported from the venv after the install:\n" + importable)
    chk = subprocess.run([str(gt.venv_python(venv)), "-m", "pip", "check"], capture_output=True, text=True, env=env, timeout=300)
    return {"venv": str(venv), "pip": pip, "requirements": backend_requirements(backend_dir), "pip_seconds": round(time.monotonic() - t0),
            "protected_packages": {n: {"version": v[0], "identical_to_host": True} for n, v in seen["packages"].items()
                                   if host_seen["packages"].get(n)},
            "host_environment_unchanged": True, "backend_importable": True,
            "pip_check": (chk.stdout + chk.stderr).strip()[-600:] or "No broken requirements found."}


def backend_importable(python: str, backend_dir: Path, base: Path) -> str:
    """'' if `app.main` imports with this interpreter, else the (redacted) reason."""
    env = gt.clean_env(None, **gt.cache_env(base))
    env.update(INFERENCE_PROVIDER="mock", STORAGE_LOCAL_ROOT=str(base / "storage-import-check"))
    p = subprocess.run([python, "-c", "import app.main"], cwd=backend_dir, capture_output=True, text=True, env=env, timeout=120)
    return "" if p.returncode == 0 else gt.redact((p.stderr or p.stdout)[-800:])


# ===================================================================================== backend process
def start_backend(base: Path, backend_dir: Path, python: str, settings: dict[str, str], wait: float = 90) -> dict:
    port = int(settings["API_PORT"])
    (base / "logs").mkdir(parents=True, exist_ok=True)
    stop_backend(base, quiet=True)
    pidfile, logfile = base / "logs" / "backend.pid", base / "logs" / "backend.log"
    cmd = [python, "-m", "uvicorn", "app.main:app", "--host", "127.0.0.1", "--port", str(port), "--log-level", "info"]
    with open(logfile, "wb") as log:
        proc = subprocess.Popen(cmd, cwd=backend_dir, env=backend_env(base, settings), stdout=log, stderr=subprocess.STDOUT,
                                start_new_session=True)
    pidfile.write_text(str(proc.pid))
    _CHILDREN[proc.pid] = proc
    deadline = time.monotonic() + wait
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            stop_backend(base, quiet=True)
            raise RuntimeError(f"the backend exited with code {proc.returncode}:\n" + gt.redact(gt.tail(logfile, 30)))
        try:
            if gt.http_get(f"http://127.0.0.1:{port}/api/health")[0] == 200:
                addrs = gt.listening_addresses(port)
                return {"pid": proc.pid, "port": port, "listening_on": addrs,
                        "loopback_only": bool(addrs) and all(gt.is_loopback(a) for a in addrs), "log": str(logfile)}
        except OSError:
            pass
        time.sleep(0.5)
    stop_backend(base, quiet=True)
    raise RuntimeError(f"the backend did not become ready within {wait:.0f}s:\n" + gt.redact(gt.tail(logfile, 30)))


def stop_backend(base: Path, quiet: bool = False) -> bool:
    pidfile = base / "logs" / "backend.pid"
    if not pidfile.exists():
        return False
    try:
        pid = int(pidfile.read_text())
    except ValueError:
        pidfile.unlink(missing_ok=True)
        return False
    proc = _CHILDREN.pop(pid, None)
    try:
        group = os.getpgid(pid)
        os.killpg(group, signal.SIGTERM)
        for _ in range(60):
            time.sleep(0.25)
            if (proc.poll() is not None) if proc is not None else not gt._alive(pid):
                break
        else:
            os.killpg(group, signal.SIGKILL)
    except OSError:
        pass
    if proc is not None:
        try:
            proc.wait(timeout=10)
        except subprocess.SubprocessError:
            pass
    pidfile.unlink(missing_ok=True)
    if not quiet:
        print("backend stopped")
    return True


# ===================================================================================== the test
class Verdict:
    """Collects named claims. PASS only if every claim that was evaluated is True."""

    def __init__(self):
        self.claims: dict[str, bool | None] = {}
        self.reasons: list[str] = []

    def claim(self, name: str, ok: bool, why: str = "") -> bool:
        self.claims[name] = bool(ok)
        if not ok:
            self.reasons.append(f"{name}: {why}" if why else name)
        return bool(ok)

    def skip(self, name: str) -> None:
        self.claims[name] = None

    @property
    def verdict(self) -> str:
        return "PASS" if self.claims and all(v is not False for v in self.claims.values()) and not self.reasons else "FAIL"


def _check_summary(health: dict) -> dict[str, str]:
    return {c["name"]: c["status"] for c in health.get("provider", {}).get("checks", [])}


def poll_job(api, job_id: str, timeout: float, interval: float, sampler=None) -> dict:
    """Poll GET /api/jobs/{id} until it is terminal. Returns the timeline and the last job body."""
    t0 = time.monotonic()
    timeline: list[dict] = []
    job: dict = {}
    last_beat = 0.0
    while time.monotonic() - t0 < timeout:
        r = api.get(f"/api/jobs/{job_id}")
        r.raise_for_status()
        job = r.json()
        now = round(time.monotonic() - t0, 2)
        if not timeline or (timeline[-1]["status"], timeline[-1]["progress"]) != (job["status"], job["progress"]):
            timeline.append({"t": now, "status": job["status"], "progress": job["progress"]})
        if now - last_beat >= 10 or job["status"] in ("completed", "failed", "cancelled"):
            last_beat = now
            vram = f" | GPU memory {sampler.vram_used_mib[-1]} MiB" if sampler is not None and sampler.vram_used_mib else ""
            print(f"  [{now:6.0f}s] {job['status']:<10} {job['progress']:5.1f}%{vram}", file=sys.stderr, flush=True)
        if job["status"] in ("completed", "failed", "cancelled"):
            return {"timeline": timeline, "job": job, "timed_out": False, "seconds": now}
        time.sleep(interval)
    return {"timeline": timeline, "job": job, "timed_out": True, "seconds": round(time.monotonic() - t0, 2)}


def run_e2e(args) -> int:
    try:
        import httpx
    except ImportError as e:
        print(f"FAILED: {e}. Run this command with the ComfyUI venv's python (<base>/venv/bin/python).", file=sys.stderr)
        return 2
    base, backend_dir = Path(args.base), Path(args.backend_dir)
    python = args.python or str(gt.venv_python(base / "venv"))
    comfy_host, comfy_port = loopback_endpoint(args.comfyui_url)
    settings = app_settings(base, Path(args.workflows_dir), args.comfyui_url, args.backend_port)
    reason = backend_importable(python, backend_dir, base)
    if reason:
        raise RuntimeError("the backend cannot be imported with " + python + " (run install-backend first):\n" + reason)

    v = Verdict()
    report: dict = {"settings_applied": settings, "request": {**PROVEN, "seed": args.seed, "prompt": args.prompt}}
    api = None
    sampler = None
    started = False
    t_start = time.monotonic()
    comfy_pidfile = base / "logs" / "comfyui.pid"
    comfy_pid = int(comfy_pidfile.read_text()) if comfy_pidfile.exists() and comfy_pidfile.read_text().strip().isdigit() else None
    try:
        # 1. Services and their bindings --------------------------------------------------------------------------
        backend = start_backend(base, backend_dir, python, settings)
        report["backend"] = backend
        started = True
        v.claim("backend_started", True)
        v.claim("backend_loopback_only", backend["loopback_only"], f"listening on {backend['listening_on']}")
        comfy_addrs = gt.listening_addresses(comfy_port)
        report["comfyui_listening_on"] = comfy_addrs
        v.claim("comfyui_loopback_only", bool(comfy_addrs) and all(gt.is_loopback(a) for a in comfy_addrs), f"listening on {comfy_addrs}")
        api = httpx.Client(base_url=f"http://127.0.0.1:{args.backend_port}", timeout=60)

        # 2. Health -----------------------------------------------------------------------------------------------
        health = api.get("/api/health").json()
        report["health_before"] = health
        v.claim("provider_is_comfyui", health["provider"]["name"] == "comfyui", f"provider is {health['provider']['name']}")
        v.claim("provider_ready", health["provider"]["ok"], health["provider"]["detail"])

        # 3. The envelope is active in this deployment (cheap: nothing reaches the GPU) ----------------------------
        jobs_before = api.get("/api/jobs").json()["total"]
        probes = {
            "resolution": {**_form(), "resolution": "1280x704"},
            "num_frames": {**_form(), "num_frames": "49"},
            "steps": {**_form(), "steps": "30"},
            "mode": {**_form(), "mode": "i2v"},
        }
        rejected: dict[str, dict] = {}
        for field, data in probes.items():
            files = {"image": ("probe.png", png_bytes(), "image/png")} if field == "mode" else None
            r = api.post("/api/jobs", data=data, files=files)
            locs = [e["loc"][-1] for e in r.json().get("detail", [])] if r.status_code == 422 and isinstance(r.json().get("detail"), list) else []
            rejected[field] = {"status": r.status_code, "fields": locs}
        report["envelope_probes"] = rejected
        no_new_jobs = api.get("/api/jobs").json()["total"] == jobs_before
        v.claim("envelope_enforced", all(p["status"] == 422 and f in p["fields"] for f, p in rejected.items()) and no_new_jobs,
                f"probes: {rejected}; no new jobs: {no_new_jobs}")

        # 4. The real job, with the proven values ---------------------------------------------------------------------
        sampler = gt.Sampler(gpu=args.gpu, pid=comfy_pid)
        sampler.start()
        sampler.sample_once()
        sampler.baseline_vram_mib = sampler.vram_used_mib[-1] if sampler.vram_used_mib else None
        body = {**_form(), "prompt": args.prompt, "seed": str(args.seed)}
        r = api.post("/api/jobs", data=body)
        report["submit"] = {"status_code": r.status_code}
        if not v.claim("job_accepted", r.status_code == 202, f"POST /api/jobs returned {r.status_code}: {gt.redact(r.text[:300])}"):
            raise _Stop
        job_id = r.json()["id"]
        report["job_id"] = job_id
        polled = poll_job(api, job_id, args.timeout, args.poll, sampler)
        job, timeline = polled["job"], polled["timeline"]
        report["timeline"] = timeline[:40] + ([{"...": f"{len(timeline) - 60} more"}] + timeline[-20:] if len(timeline) > 60 else timeline[40:])
        report["job_seconds"] = polled["seconds"]
        statuses = [t["status"] for t in timeline]
        v.claim("status_flow_valid", all(ORDER[a] <= ORDER[b] for a, b in zip(statuses, statuses[1:])) and "processing" in statuses,
                f"statuses seen: {statuses}")
        progress = [t["progress"] for t in timeline]
        v.claim("progress_monotonic", all(a <= b for a, b in zip(progress, progress[1:])), f"progress seen: {progress}")
        v.claim("progress_intermediate_values", any(0 < t["progress"] < 100 for t in timeline if t["status"] == "processing"),
                "polling never saw a progress value between 0 and 100 while processing")
        if polled["timed_out"]:
            api.post(f"/api/jobs/{job_id}/cancel")
            v.claim("job_completed", False, f"no result within {args.timeout:.0f}s (cancelled)")
            raise _Stop
        report["job"] = job
        if not v.claim("job_completed", job["status"] == "completed",
                       f"job ended as {job['status']}: {json.dumps(job.get('error'))}"):
            raise _Stop

        # 5. The job record --------------------------------------------------------------------------------------------
        p = job["params"]
        v.claim("job_record_matches_request", job["provider"] == "comfyui" and job["error"] is None and
                (p["resolution"], p["num_frames"], p["steps"], p["fps"], p["guidance_scale"], p["seed"]) ==
                ("832x480", 17, 20, 24, 5.0, args.seed), f"provider={job['provider']} params={p}")
        v.claim("job_in_history", any(j["id"] == job_id for j in api.get("/api/jobs").json()["items"]), "not listed in GET /api/jobs")

        # 6. Video retrieval -------------------------------------------------------------------------------------------------
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        download_url = job["result"]["download_url"]
        with api.stream("GET", download_url) as resp:
            ok_headers = resp.status_code == 200 and resp.headers.get("content-type", "").startswith("video/mp4") and \
                resp.headers.get("content-disposition", "").startswith("attachment")
            with open(out, "wb") as f:
                for chunk in resp.iter_bytes(1024 * 1024):
                    f.write(chunk)
        v.claim("download_ok", ok_headers and out.stat().st_size == job["result"]["size_bytes"] > 0,
                f"status/headers/size mismatch (file {out.stat().st_size if out.exists() else 0} bytes, job says {job['result']['size_bytes']})")
        data = out.read_bytes()
        rr = api.get(download_url, headers={"Range": "bytes=0-1023"})
        v.claim("range_request_ok", rr.status_code == 206 and rr.content == data[:1024] and
                rr.headers.get("content-range", "").startswith("bytes 0-1023/"), f"HTTP {rr.status_code}")
        pv = api.get(job["result"]["preview_url"], headers={"Range": "bytes=0-11"})
        v.claim("preview_inline_ok", pv.status_code in (200, 206) and pv.headers.get("content-disposition", "").startswith("inline") and
                pv.content[4:8] == b"ftyp", f"HTTP {pv.status_code}")

        # 7. The MP4 itself (fully decoded) -------------------------------------------------------------------------------
        probe = gt.probe_mp4(out, frames=PROVEN["num_frames"], size=(832, 480), fps=float(PROVEN["fps"]))
        report["output"] = probe
        v.claim("mp4_decodes_and_matches", not probe["problems"], "; ".join(probe["problems"]))

        # 8. Health after a real generation --------------------------------------------------------------------------------
        health2 = api.get("/api/health").json()
        report["health_after"] = health2
        v.claim("inference_verified_by_generation", _check_summary(health2).get("inference") == "ok",
                f"checks: {_check_summary(health2)}")

        # 9. Optional: cancel a second job (real interrupt) ---------------------------------------------------------------------
        if args.with_cancel:
            cancel = cancel_test(api, body, args.poll)
            report["cancel_test"] = cancel
            v.claim("cancel_works", cancel["ok"], cancel["why"])
        else:
            v.skip("cancel_works")
    except _Stop:
        pass
    except Exception as e:  # noqa: BLE001  (reported, never a traceback with secrets)
        v.claim("test_ran_to_completion", False, f"{type(e).__name__}: {gt.redact(str(e))[:300]}")
    finally:
        stats = sampler.finish() if sampler is not None else {}
        if api is not None:
            api.close()
        stopped = stop_backend(base, quiet=True)
        if started:
            v.claim("backend_stopped", stopped and not gt.listening_addresses(args.backend_port), "the backend is still listening")
        if args.require_gpu and stats and stats.get("gpu_sampling") == "unavailable":
            v.claim("gpu_measured", False, "GPU memory could not be measured (nvidia-smi unavailable)")

    verdict = v.verdict
    report.update({
        "verdict": verdict, "reasons": v.reasons, "claims": v.claims,
        "resources": stats,
        "timing": {"total_wall_s": round(time.monotonic() - t_start, 1), "job_wall_s": report.get("job_seconds")},
        "backend_log_tail": gt.redact(gt.tail(base / "logs" / "backend.log", 60)) if verdict == "FAIL" else None,
    })
    text = json.dumps(report, indent=1, default=str)
    assert "Bearer " not in text.replace("Bearer [redacted]", ""), "a token-like string reached the report"
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text(text)
    gt.emit({k: report[k] for k in ("verdict", "reasons", "claims", "timing", "resources") if k in report} |
            ({"output": report["output"]} if "output" in report else {}))
    return 0 if verdict == "PASS" else 1


class _Stop(Exception):
    """Ends the test early; the reasons are already recorded."""


def _form() -> dict[str, str]:
    """The explicit, proven request fields (form-encoded)."""
    return {"mode": PROVEN["mode"], "prompt": DEFAULT_PROMPT, "resolution": PROVEN["resolution"],
            "num_frames": str(PROVEN["num_frames"]), "steps": str(PROVEN["steps"]), "fps": str(PROVEN["fps"]),
            "guidance_scale": str(PROVEN["guidance_scale"])}


def cancel_test(api, body: dict, interval: float, wait_processing: float = 180) -> dict:
    """Submit another job, cancel it while it is being processed, and check it stays cancelled."""
    r = api.post("/api/jobs", data=body)
    if r.status_code != 202:
        return {"ok": False, "why": f"submit returned {r.status_code}"}
    job_id = r.json()["id"]
    t0 = time.monotonic()
    while time.monotonic() - t0 < wait_processing:
        status = api.get(f"/api/jobs/{job_id}").json()["status"]
        if status == "processing":
            break
        if status in ("completed", "failed", "cancelled"):
            return {"ok": False, "why": f"job ended as {status} before it could be cancelled", "job_id": job_id}
        time.sleep(interval)
    c = api.post(f"/api/jobs/{job_id}/cancel")
    time.sleep(max(interval, 2.0))
    final = api.get(f"/api/jobs/{job_id}").json()
    ok = c.status_code == 200 and c.json()["status"] == "cancelled" and final["status"] == "cancelled" and final["result"] is None
    return {"ok": ok, "why": "" if ok else f"cancel returned {c.status_code}; final status {final['status']}", "job_id": job_id}


# ===================================================================================== CLI
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("install-backend")
    p.add_argument("--base", required=True); p.add_argument("--backend-dir", required=True)
    p = sub.add_parser("e2e")
    p.add_argument("--base", required=True); p.add_argument("--backend-dir", required=True); p.add_argument("--workflows-dir", required=True)
    p.add_argument("--python", help="interpreter that runs the backend (default: <base>/venv/bin/python)")
    p.add_argument("--comfyui-url", default="http://127.0.0.1:8188"); p.add_argument("--backend-port", type=int, default=8000)
    p.add_argument("--gpu", type=int, default=0); p.add_argument("--timeout", type=float, default=2400)
    p.add_argument("--poll", type=float, default=1.0); p.add_argument("--seed", type=int, default=424242)
    p.add_argument("--prompt", default=DEFAULT_PROMPT); p.add_argument("--require-gpu", action="store_true")
    p.add_argument("--with-cancel", action="store_true", help="also cancel a second job while it is being processed")
    p.add_argument("--out", required=True); p.add_argument("--report", required=True)
    p = sub.add_parser("stop-backend"); p.add_argument("--base", required=True)
    a = ap.parse_args(argv)
    try:
        if a.cmd == "install-backend":
            gt.emit(install_backend(Path(a.base), Path(a.backend_dir)))
        elif a.cmd == "e2e":
            return run_e2e(a)
        elif a.cmd == "stop-backend":
            stop_backend(Path(a.base))
    except (RuntimeError, ValueError, subprocess.SubprocessError, OSError) as e:
        print(f"FAILED: {gt.redact(str(e))}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
