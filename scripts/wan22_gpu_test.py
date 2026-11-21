#!/usr/bin/env python3
"""GPU-host tooling for the Wan 2.2 TI2V-5B test on Kaggle (Phase 5B). Used by
notebooks/kaggle_comfyui_wan22.ipynb; every piece is also unit-tested without a GPU.

Subcommands
  preflight       The gates that must pass before anything is installed or downloaded: Python version, a
                  non-persistent install location with enough free disk, GPU model / VRAM / CUDA-PyTorch
                  compatibility, system RAM, internet, and the authoritative Hugging Face metadata
                  (pinned revision, size, SHA-256) cross-checked against the manifest. No downloads.
  gpu-probe       GPU model, VRAM, CUDA/driver/PyTorch compatibility. Runs in its own process so the
                  notebook kernel never holds a CUDA context (that would steal VRAM from ComfyUI).
  install-comfyui Fetch ONE pinned ComfyUI commit, create a venv (--system-site-packages --without-pip: it
                  reuses the host's PyTorch and the host's pip), install ComfyUI's requirements under a
                  constraints file that pins torch/torchvision/torchaudio/numpy to what is already installed,
                  then verify they are the host's identical copies and that the host's own Python
                  environment is unchanged.
  serve / stop    Start ComfyUI in the background, bound to 127.0.0.1 ONLY, on one GPU, with a allow-listed
                  environment (no tokens or other credentials); verify it listens on loopback only. Stop it again.
  check-templates Validate BOTH Phase 4 workflow templates (text-to-video and image-to-video) against the
                  live server's own /object_info: every node class present, every input name, type, range and
                  combo value (including the model filenames) accepted.
  smoke-t2v       Run a small text-to-video generation through the existing VIDSGEN ComfyUIProvider and
                  the existing Phase 4 workflow, measure VRAM/RAM/time, and verify the MP4. Exit code 0
                  only if a real video was generated AND decodes with the requested frames and size.
  verify-mp4      Decode an MP4 and check frames, size, codec and that it is not blank.

Nothing here opens a public tunnel, downloads model weights, or prints secrets.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import shlex
import shutil
import signal
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

GIB = 1024**3
# Used for pip/git style helper processes: drop anything that looks like a credential (best effort).
SECRET_ENV = re.compile(r"TOKEN|SECRET|PASSWORD|PASSWD|PASS$|ASKPASS|API_?KEY|ACCESS_?KEY|PRIVATE|CREDENTIAL|_KEY(_\d+)?$|"
                        r"KAGGLE_KEY|KAGGLE_USERNAME|^GIT_CONFIG_(KEY|VALUE)", re.I)
# The ComfyUI SERVER gets an ALLOWLIST instead: a denylist cannot be trusted to catch every credential name.
SERVER_ENV_ALLOW = re.compile(r"^(PATH|HOME|USER|LOGNAME|LANG|LANGUAGE|LC_[A-Z_]+|TERM|TZ|SHELL|LD_LIBRARY_PATH|LIBRARY_PATH|"
                              r"NVIDIA_[A-Z_]+|CUDA_[A-Z_]+|OMP_[A-Z_]+|MKL_[A-Z_]+|PYTORCH_[A-Z_]+|TORCH[A-Z_]*|NCCL_[A-Z_]+|"
                              r"CONDA_[A-Z_]+|VIRTUAL_ENV|PYTHON[A-Z_]*|TMPDIR|XDG_[A-Z_]+|HF_HOME|PIP_CACHE_DIR)$")
REDACT = [(re.compile(r"Bearer\s+\S+", re.I), "Bearer [redacted]"), (re.compile(r"hf_[A-Za-z0-9]{20,}"), "[redacted]"),
          (re.compile(r"(?i)(token|secret|password)=\S+"), r"\1=[redacted]")]
TORCH_PACKAGES = ("torch", "torchvision", "torchaudio")
PINNED_PACKAGES = (*TORCH_PACKAGES, "numpy")
# Needed by the existing VIDSGEN ComfyUIProvider (httpx, websockets) and the Phase 4 schema validator (starlette, uvicorn).
EXTRA_REQUIREMENTS = ["httpx>=0.27,<1.0", "websockets>=13", "uvicorn>=0.30", "starlette"]


def redact(text: str) -> str:
    for rx, repl in REDACT:
        text = rx.sub(repl, text)
    return text


def clean_env(base: dict[str, str] | None = None, **extra: str) -> dict[str, str]:
    """A copy of the environment without anything that looks like a credential."""
    env = {k: v for k, v in (base if base is not None else os.environ).items() if not SECRET_ENV.search(k)}
    env.update(extra)
    return env


def server_env(base_dir: Path, gpu: str | None, environ: dict[str, str] | None = None) -> dict[str, str]:
    """Environment for the ComfyUI server: only allow-listed variables, caches inside the base dir, one GPU."""
    src = environ if environ is not None else os.environ
    env = {k: v for k, v in src.items() if SERVER_ENV_ALLOW.match(k) and not SECRET_ENV.search(k)}
    env.update(cache_env(base_dir))
    if gpu not in (None, "", "none"):
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    return env


def cache_env(base_dir: Path) -> dict[str, str]:
    """Send every cache/temp location to the (non-persistent) base directory."""
    cache = base_dir / "cache"
    for sub in ("pip", "hf", "torch", "xdg", "tmp"):
        (cache / sub).mkdir(parents=True, exist_ok=True)
    return {"PIP_CACHE_DIR": str(cache / "pip"), "HF_HOME": str(cache / "hf"), "TORCH_HOME": str(cache / "torch"),
            "XDG_CACHE_HOME": str(cache / "xdg"), "TMPDIR": str(cache / "tmp"), "PYTHONDONTWRITEBYTECODE": "1"}


def emit(obj) -> None:
    print(json.dumps(obj, indent=1, default=str))


# ===================================================================================== gpu-probe
def gpu_probe(gpu: int) -> dict:
    out: dict = {"nvidia_smi": None, "devices": [], "torch": None}
    try:
        p = subprocess.run(["nvidia-smi", "--query-gpu=index,name,memory.total,memory.used,driver_version,compute_cap",
                            "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=30)
        if p.returncode == 0:
            out["nvidia_smi"] = [dict(zip(("index", "name", "memory_total_mib", "memory_used_mib", "driver", "compute_cap"),
                                          [x.strip() for x in line.split(",")])) for line in p.stdout.strip().splitlines()]
        banner = subprocess.run(["nvidia-smi"], capture_output=True, text=True, timeout=30).stdout
        m = re.search(r"CUDA Version:\s*([\d.]+)", banner)
        out["driver_cuda"] = m.group(1) if m else None
    except (OSError, subprocess.SubprocessError) as e:
        out["nvidia_smi_error"] = f"{type(e).__name__}: {e}"
    try:
        import torch
    except ImportError as e:
        out["torch_error"] = str(e)
        return out
    out["torch"] = {"version": torch.__version__, "cuda_build": torch.version.cuda, "cuda_available": torch.cuda.is_available(),
                    "arch_list": torch.cuda.get_arch_list() if torch.cuda.is_available() else []}
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            pr = torch.cuda.get_device_properties(i)
            free, total = torch.cuda.mem_get_info(i)
            out["devices"].append({"index": i, "name": pr.name, "capability": f"{pr.major}.{pr.minor}",
                                   "total_bytes": total, "free_bytes": free})
        try:
            a = torch.randn(256, 256, device=f"cuda:{gpu}", dtype=torch.float16)
            out["fp16_matmul_ok"] = bool(torch.isfinite(a @ a).all().item())
        except Exception as e:  # noqa: BLE001
            out["fp16_matmul_ok"] = False
            out["fp16_error"] = f"{type(e).__name__}: {str(e)[:200]}"
        try:
            out["bf16_native"] = bool(torch.cuda.is_bf16_supported(including_emulation=False))
        except TypeError:
            out["bf16_native"] = bool(torch.cuda.is_bf16_supported())
    return out


# ===================================================================================== preflight
DEFAULT_ENDPOINT = "https://huggingface.co"


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *a, **k):
        return None


def _hf_request(url: str, method: str = "GET", form: dict | None = None, token: str = "", timeout: float = 30.0):
    """Returns (status, headers, body). Redirects are NOT followed (a HEAD on /resolve/ answers 302 with the
    metadata headers we need, so no file content is ever transferred)."""
    headers = {"User-Agent": "vidsgen-preflight/1"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    data = urllib.parse.urlencode(form).encode() if form else None
    req = urllib.request.Request(url, data=data, method=method, headers=headers)
    try:
        with urllib.request.build_opener(_NoRedirect).open(req, timeout=timeout) as r:
            return r.status, r.headers, r.read()
    except urllib.error.HTTPError as e:
        return e.code, e.headers, e.read() if method != "HEAD" else b""


def hf_metadata_check(endpoint: str, row: dict, token: str = "") -> tuple[str, str]:
    """Compare the live, authoritative metadata for one model file with the manifest.
    SHA-256 = the Git-LFS object id, read from two independent sources that must agree. The Xet hash and the
    Git blob id are different identifiers and are never used."""
    repo, rev, path = row["repo"], row["revision"], row["path"]
    name = Path(path).name
    try:
        st, _, body = _hf_request(f"{endpoint}/api/models/{repo}/revision/{rev}", token=token)
        if st != 200 or json.loads(body).get("sha") != rev:
            return "FAIL", f"{name}: pinned revision {rev[:12]} of {repo} was not found (HTTP {st})"
        st, _, body = _hf_request(f"{endpoint}/api/models/{repo}/paths-info/{rev}", "POST",
                                  {"paths": path, "expand": "true"}, token)
        items = json.loads(body) if st == 200 else []
        lfs = (items[0].get("lfs") or {}) if items else {}
        st2, hdrs, _ = _hf_request(f"{endpoint}/{repo}/resolve/{rev}/{urllib.parse.quote(path)}", "HEAD", token=token)
        etag = (hdrs.get("x-linked-etag") or "").strip('"')
        size_head = hdrs.get("x-linked-size")
    except (OSError, ValueError, urllib.error.URLError) as e:
        return "FAIL", f"{name}: could not read Hugging Face metadata ({type(e).__name__})"
    problems = []
    if lfs.get("oid") != row["sha256"]:
        problems.append(f"paths-info SHA-256 {lfs.get('oid')} != manifest {row['sha256']}")
    if etag != row["sha256"]:
        problems.append(f"x-linked-etag {etag or None} != manifest {row['sha256']}")
    if lfs.get("size") != row["size_bytes"] or (size_head and int(size_head) != row["size_bytes"]):
        problems.append(f"size {lfs.get('size')}/{size_head} != manifest {row['size_bytes']}")
    if problems:
        return "FAIL", f"{name}: authoritative metadata disagrees with the manifest: " + "; ".join(problems)
    return "PASS", f"{name}: revision {rev[:12]}, {row['size_bytes']:,} bytes, SHA-256 {row['sha256'][:16]}... confirmed by 2 sources"


def disk_requirement(manifest: dict, comfy_data: Path, overhead_gib: float) -> tuple[int, int]:
    """(bytes of model files still to download, total bytes needed incl. overhead for packages/outputs/temp)."""
    remaining = 0
    for f in manifest["files"]:
        dest = comfy_data / f["comfyui_dir"] / Path(f["path"]).name
        part = dest.with_name(dest.name + ".part")
        if dest.is_file() and dest.stat().st_size == f["size_bytes"]:
            continue
        remaining += f["size_bytes"] - (part.stat().st_size if part.is_file() else 0)
    return remaining, remaining + int(overhead_gib * GIB)


def gpu_checks(probe: dict, gpu: int, min_vram_gib: float, allow_no_gpu: bool) -> list[tuple[str, str, str]]:
    out: list[tuple[str, str, str]] = []
    devs = probe.get("devices") or []
    t = probe.get("torch") or {}
    if not devs:
        why = probe.get("torch_error") or "PyTorch sees no CUDA GPU (set the notebook Accelerator to GPU T4 x2)"
        return [("gpu_detected", "WARN" if allow_no_gpu else "FAIL", why)]
    names = "; ".join(f"GPU{d['index']}: {d['name']} (sm_{d['capability'].replace('.', '')}, {d['total_bytes'] / GIB:.1f} GiB)" for d in devs)
    out.append(("gpu_detected", "PASS", names))
    out.append(("gpu_selection", "INFO", f"{len(devs)} GPU(s) visible. VRAM is NOT pooled: ComfyUI will use GPU {gpu} only"
                                         + (" (the other GPU stays idle)" if len(devs) > 1 else "")))
    d = next((x for x in devs if x["index"] == gpu), None)
    if d is None:
        return out + [("gpu_selected", "FAIL", f"GPU {gpu} is not available (visible: {[x['index'] for x in devs]})")]
    vram = d["total_bytes"] / GIB
    out.append(("gpu_vram", "PASS" if vram >= min_vram_gib else "WARN" if vram >= min_vram_gib - 2 else "FAIL",
                f"GPU {gpu}: {vram:.1f} GiB total, {d['free_bytes'] / GIB:.1f} GiB free (need >= {min_vram_gib:g} GiB; the fp16 5B model alone is ~9.3 GiB)"))
    cap = d["capability"].replace(".", "")
    in_arch = any(a.endswith("_" + cap) for a in t.get("arch_list", []))
    out.append(("torch_supports_gpu", "PASS" if in_arch else "FAIL",
                f"sm_{cap} {'is' if in_arch else 'is NOT'} in this PyTorch build's architectures {t.get('arch_list')}"))
    out.append(("fp16_kernel", "PASS" if probe.get("fp16_matmul_ok") else "FAIL",
                "a small fp16 matmul ran on the GPU" if probe.get("fp16_matmul_ok") else probe.get("fp16_error", "fp16 matmul failed")))
    build, drv = t.get("cuda_build"), probe.get("driver_cuda")
    if build and drv:
        ok = tuple(map(int, build.split(".")[:2])) <= tuple(map(int, drv.split(".")[:2]))
        out.append(("cuda_driver", "PASS" if ok else "WARN", f"PyTorch {t.get('version')} built for CUDA {build}; driver supports CUDA {drv}"
                                                              + ("" if ok else " (older driver: relies on minor-version compatibility)")))
    else:
        out.append(("cuda_driver", "WARN", f"PyTorch CUDA build {build}, driver CUDA {drv}: cannot compare"))
    out.append(("bf16", "INFO", f"native bf16: {probe.get('bf16_native')} (without it ComfyUI uses fp16/fp32; black frames would indicate fp16 overflow)"))
    return out


def preflight(base: Path, manifest_path: Path, gpu: int, min_vram_gib: float = 14.0, min_ram_gib: float = 24.0,
              overhead_gib: float = 8.0, offline: bool = False, allow_no_gpu: bool = False,
              repos: tuple[str, ...] = (), probe: dict | None = None, endpoint: str | None = None,
              pypi_url: str = "https://pypi.org/simple/pip/") -> dict:
    import yaml

    checks: list[dict] = []

    def add(name: str, status: str, detail: str) -> None:
        checks.append({"name": name, "status": status, "detail": detail})

    manifest = yaml.safe_load(Path(manifest_path).read_text())
    base = Path(base)
    add("python", "PASS" if sys.version_info >= (3, 11) else "FAIL",
        f"Python {sys.version.split()[0]} (the VIDSGEN backend code needs >= 3.11)")
    resolved = str(base.resolve()) if base.exists() else str(base)
    if resolved.startswith("/kaggle/working"):
        add("install_location", "FAIL", f"{base} is inside /kaggle/working, which is saved as notebook output and size-capped. Use /kaggle/tmp.")
    else:
        anchor = base
        while not anchor.exists() and anchor != anchor.parent:
            anchor = anchor.parent
        writable = os.access(anchor, os.W_OK)
        add("install_location", "PASS" if writable else "FAIL", f"{base} (non-persistent; nearest existing folder {anchor} is {'writable' if writable else 'NOT writable'})")
        remaining, need = disk_requirement(manifest, base / "comfy-data", overhead_gib)
        free = shutil.disk_usage(anchor).free
        add("disk_space", "PASS" if free >= need else "FAIL",
            f"{free / GIB:.1f} GiB free at {anchor}; need {need / GIB:.1f} GiB = {remaining / GIB:.1f} GiB of models still to download + "
            f"{overhead_gib:g} GiB for packages, outputs, temp files and margin")
    for name, st, detail in gpu_checks(probe if probe is not None else gpu_probe(gpu), gpu, min_vram_gib, allow_no_gpu):
        add(name, st, detail)
    try:
        mem = dict((k, int(v.split()[0]) * 1024) for k, v in (l.split(":", 1) for l in Path("/proc/meminfo").read_text().splitlines()))
        total, avail = mem["MemTotal"] / GIB, mem["MemAvailable"] / GIB
        add("system_ram", "PASS" if total >= min_ram_gib else "WARN" if total >= 0.8 * min_ram_gib else "FAIL",
            f"{total:.1f} GiB total, {avail:.1f} GiB available (offloaded weights live in RAM; need >= {min_ram_gib:g} GiB)")
    except (OSError, KeyError, ValueError):
        add("system_ram", "WARN", "could not read /proc/meminfo")
    if offline:
        add("network", "INFO", "skipped (--offline)")
    else:
        token = os.environ.get("HF_TOKEN", "")
        ep = (endpoint or os.environ.get("HF_ENDPOINT", DEFAULT_ENDPOINT)).rstrip("/")
        for url in repos:
            p = subprocess.run(["git", "ls-remote", "--exit-code", url, "HEAD"], capture_output=True, text=True, timeout=90)
            add(f"github {url.rsplit('/', 1)[-1]}", "PASS" if p.returncode == 0 else "FAIL",
                "reachable" if p.returncode == 0 else redact(p.stderr.strip())[-160:])
        try:
            st, _, _ = _hf_request(pypi_url, timeout=20)
            add("pypi", "PASS" if st == 200 else "FAIL", f"HTTP {st}")
        except OSError as e:
            add("pypi", "FAIL", type(e).__name__)
        for row in manifest["files"]:
            add(f"hf_metadata {row['role']}", *hf_metadata_check(ep, row, token))
    counts = {k: sum(c["status"] == k for c in checks) for k in ("PASS", "WARN", "FAIL", "INFO")}
    return {"ok": counts["FAIL"] == 0, "counts": counts, "checks": checks}


# ===================================================================================== install
def git(args: list[str], cwd: Path, timeout: int = 600) -> str:
    p = subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True, timeout=timeout)
    if p.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)} failed: {redact(p.stderr.strip())[-400:]}")
    return p.stdout.strip()


def checkout_commit(repo_url: str, commit: str, dest: Path) -> str:
    """Make `dest` a checkout of exactly `commit` (fetching only that commit). Returns the verified HEAD."""
    if (dest / ".git").exists() and _safe_head(dest) == commit:
        return commit
    dest.mkdir(parents=True, exist_ok=True)
    if not (dest / ".git").exists():
        git(["init", "-q"], dest)
        git(["remote", "add", "origin", repo_url], dest)
    git(["fetch", "-q", "--depth", "1", "origin", commit], dest)
    git(["checkout", "-q", "--detach", "FETCH_HEAD"], dest)
    head = git(["rev-parse", "HEAD"], dest)
    if head != commit:
        raise RuntimeError(f"checked out {head}, expected {commit}")
    return head


def _safe_head(dest: Path) -> str | None:
    try:
        return git(["rev-parse", "HEAD"], dest)
    except (RuntimeError, OSError):
        return None


def comfyui_version(checkout: Path) -> str | None:
    m = re.search(r'__version__\s*=\s*"([^"]+)"', (checkout / "comfyui_version.py").read_text())
    return m.group(1) if m else None


def filter_requirements(text: str) -> tuple[list[str], list[str]]:
    """Split a requirements.txt into (kept, removed). PyTorch packages are removed because the host
    already provides a working, GPU-matched build that must not be replaced."""
    kept, removed = [], []
    for line in text.splitlines():
        name = re.split(r"[<>=!~;\[\s#]", line.strip(), maxsplit=1)[0].lower().replace("_", "-")
        if name in TORCH_PACKAGES:
            removed.append(line.strip())
        else:
            kept.append(line.rstrip())
    return kept, removed


def write_constraints(path: Path, versions: dict[str, str]) -> None:
    path.write_text("\n".join(f"{n}=={v}" for n, v in sorted(versions.items())) + "\n")


# --- The install virtual environment -------------------------------------------------------------------------------
# --system-site-packages  the venv can see the host's PyTorch/CUDA stack, which must NOT be reinstalled.
# --without-pip           Kaggle's Python 3.13 has no `ensurepip`, so a venv cannot bootstrap its own pip. The host's pip
#                         stays importable through system-site-packages, and `<venv>/bin/python -m pip` installs into the
#                         VENV (pip derives the target from sys.prefix), leaving the host's site-packages untouched.
VENV_FLAGS = ("--system-site-packages", "--without-pip")


def venv_python(venv: Path) -> Path:
    return venv / "bin" / "python"


def venv_includes_system_packages(venv: Path) -> bool:
    """True if pyvenv.cfg says the venv can see the host's site-packages."""
    try:
        for line in (venv / "pyvenv.cfg").read_text().splitlines():
            key, _, value = line.partition("=")
            if key.strip().lower() == "include-system-site-packages":
                return value.strip().lower() == "true"
    except OSError:
        pass
    return False


def create_venv(python: str, venv: Path, env: dict[str, str]) -> dict:
    """Create the install venv, or accept an existing one, so re-running the installer is safe.

    An existing venv is reused as it is (never modified or deleted), whether it has its own pip or not. It must be
    able to see the host's packages, otherwise it could never import the host's PyTorch."""
    if venv_python(venv).exists():
        if not venv_includes_system_packages(venv):
            raise RuntimeError(f"the existing virtual environment {venv} does not include the host's site-packages "
                               "(pyvenv.cfg: include-system-site-packages is not true), so it cannot see the host's PyTorch. "
                               "Nothing was changed: delete that folder and run the install again.")
        return {"created": False}
    cmd = [python, "-m", "venv", *VENV_FLAGS, str(venv)]
    p = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=300)
    if p.returncode != 0 or not venv_python(venv).exists():
        raise RuntimeError(f"could not create the virtual environment ({' '.join(cmd)}):\n" + redact(p.stdout + p.stderr)[-800:])
    return {"created": True, "command": cmd}


def pip_info(py: str, venv: Path, env: dict[str, str]) -> dict:
    """Which pip the venv will use. A venv made with --without-pip relies on the HOST's pip."""
    p = subprocess.run([py, "-m", "pip", "--version"], capture_output=True, text=True, env=env, timeout=120)
    m = re.match(r"pip (\S+) from (.+?) \(python", p.stdout) if p.returncode == 0 else None
    if not m:
        raise RuntimeError("pip is not available inside the virtual environment. It was created without its own pip "
                           "(Kaggle's Python 3.13 has no ensurepip), so it needs the HOST Python's pip, and none was found. "
                           "Install pip for the host Python, or create the venv with pip, then run the install again.\n"
                           + redact(p.stdout + p.stderr)[-400:])
    version, location = m.group(1), m.group(2)
    return {"version": version, "location": location, "source": "venv" if str(venv) in location else "host"}


# --- Proving that the host's environment is untouched ----------------------------------------------------------------
IMPORT_PROBE = (
    "import importlib,json,sys\n"
    "out={}\n"
    f"for n in {list(PINNED_PACKAGES)!r}:\n"
    " try:\n"
    "  m=importlib.import_module(n);out[n]=[str(getattr(m,'__version__',None)),getattr(m,'__file__',None)]\n"
    " except ImportError:\n"
    "  out[n]=None\n"
    " except Exception as e:\n"
    "  out[n]=['error',type(e).__name__]\n"
    "print(json.dumps({'python':sys.version.split()[0],'packages':out}))")

INVENTORY_PROBE = (
    "import importlib.metadata as m,json\n"
    "d={}\n"
    "for x in m.distributions():\n"
    " n=(x.metadata['Name'] or '').lower().replace('_','-')\n"
    " if n:d.setdefault(n,[]).append(x.version)\n"
    "print(json.dumps({k:sorted(v) for k,v in d.items()}))")


def import_probe(py: str, env: dict[str, str]) -> dict:
    """{'python': ..., 'packages': {name: [version, file] | None}} for the protected packages, as imported by `py`."""
    p = subprocess.run([py, "-c", IMPORT_PROBE], capture_output=True, text=True, env=env, timeout=300)
    if p.returncode != 0:
        raise RuntimeError(f"the protected packages cannot be probed by {py}:\n" + redact(p.stderr)[-600:])
    return json.loads(p.stdout.strip().splitlines()[-1])


def protected_packages_replaced(host: dict, seen: dict, venv: Path) -> list[str]:
    """Problems if the venv does not use EXACTLY the host's torch, torchvision, torchaudio and numpy. Compares like with
    like (the imported modules' versions and file locations), so 'same version, different copy' is caught too. A package
    the host does not have is not protected (there is nothing to replace)."""
    problems = []
    for name in PINNED_PACKAGES:
        h, s = host["packages"].get(name), seen["packages"].get(name)
        if h is None:
            continue
        if s is None:
            problems.append(f"{name} {h[0]} cannot be imported inside the venv")
            continue
        if s[0] != h[0]:
            problems.append(f"{name} changed: {h[0]} -> {s[0]}")
        if s[1] != h[1]:
            problems.append(f"the venv imports a different {name} ({s[1]}; host: {h[1]})")
        if s[1] and str(venv) in s[1]:
            problems.append(f"a second {name} was installed inside the venv")
    return problems


def host_inventory(python: str, env: dict[str, str], cwd: Path) -> dict[str, list[str]]:
    """Every distribution (name -> versions) the HOST Python can see, read by a separate process."""
    p = subprocess.run([python, "-c", INVENTORY_PROBE], capture_output=True, text=True, env=env, cwd=cwd, timeout=300)
    if p.returncode != 0:
        raise RuntimeError(f"cannot read the host's installed packages with {python}:\n" + redact(p.stderr)[-600:])
    return json.loads(p.stdout.strip().splitlines()[-1])


def inventory_changes(before: dict[str, list[str]], after: dict[str, list[str]]) -> list[str]:
    out = []
    for name in sorted(set(before) | set(after)):
        b, a = before.get(name), after.get(name)
        if b == a:
            continue
        if b is None:
            out.append(f"{name} was ADDED ({', '.join(a)})")
        elif a is None:
            out.append(f"{name} was REMOVED (was {', '.join(b)})")
        else:
            out.append(f"{name} changed: {', '.join(b)} -> {', '.join(a)}")
    return out


def install_comfyui(base: Path, repo_url: str, commit: str, expect_version: str | None, python: str = sys.executable) -> dict:
    base.mkdir(parents=True, exist_ok=True)
    cenv = cache_env(base)
    checkout = base / "ComfyUI"
    head = checkout_commit(repo_url, commit, checkout)
    version = comfyui_version(checkout)
    if expect_version and version != expect_version:
        raise RuntimeError(f"ComfyUI at {commit[:12]} reports version {version}, expected {expect_version}")
    dirty = git(["status", "--porcelain"], checkout)
    if dirty:
        raise RuntimeError("ComfyUI checkout has local modifications:\n" + dirty[:300])

    env = clean_env(None, **cenv)
    inventory_before = host_inventory(python, env, base)        # the host's packages, before we touch anything
    before = {n: inventory_before[n][0] for n in PINNED_PACKAGES if n in inventory_before}
    if "torch" not in before:
        raise RuntimeError("PyTorch is not installed on this host; refusing to install one blindly")
    constraints = base / "constraints.txt"
    write_constraints(constraints, before)                      # torch/torchvision/torchaudio/numpy stay exactly as they are
    kept, removed = filter_requirements((checkout / "requirements.txt").read_text())
    (base / "requirements-comfyui.txt").write_text("\n".join(kept) + "\n")
    (base / "requirements-extra.txt").write_text("\n".join(EXTRA_REQUIREMENTS) + "\n")

    venv = base / "venv"
    venv_info = create_venv(python, venv, env)
    pip = pip_info(str(venv_python(venv)), venv, env)
    cmd = [str(venv_python(venv)), "-m", "pip", "install", "--no-cache-dir", "--disable-pip-version-check", "-q",
           "-c", str(constraints), "-r", str(base / "requirements-comfyui.txt"), "-r", str(base / "requirements-extra.txt")]
    t0 = time.monotonic()
    p = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=3600)
    if p.returncode != 0:
        raise RuntimeError("pip install failed (PyTorch and NumPy were protected by the constraints file):\n" + redact(p.stdout + p.stderr)[-1500:])

    host_seen, seen = import_probe(python, env), import_probe(str(venv_python(venv)), env)
    problems = protected_packages_replaced(host_seen, seen, venv)
    if problems:
        raise RuntimeError("the host's PyTorch/NumPy stack was replaced: " + "; ".join(problems))
    changes = inventory_changes(inventory_before, host_inventory(python, env, base))
    if changes:
        more = f" (+{len(changes) - 12} more)" if len(changes) > 12 else ""
        raise RuntimeError("the host's Python environment was modified by the install (it must stay untouched): "
                           + "; ".join(changes[:12]) + more)
    chk = subprocess.run([str(venv_python(venv)), "-m", "pip", "check"], capture_output=True, text=True, env=env, timeout=300)
    protected = {n: {"version": v[0], "file": v[1], "identical_to_host": True}
                 for n, v in seen["packages"].items() if host_seen["packages"].get(n)}
    return {"commit": head, "comfyui_version": version, "checkout": str(checkout), "venv": str(venv),
            "venv_created": venv_info["created"], "venv_flags": list(VENV_FLAGS), "pip": pip,
            "python": seen["python"], "torch": seen["packages"]["torch"][0], "torch_file": seen["packages"]["torch"][1],
            "torch_replaced": False, "protected_packages": protected,
            "host_environment_unchanged": True, "host_packages_checked": len(inventory_before),
            "pinned_by_constraints": before, "requirements_removed": removed, "pip_seconds": round(time.monotonic() - t0),
            "pip_check": (chk.stdout + chk.stderr).strip()[-600:] or "No broken requirements found."}


# ===================================================================================== serve / stop
def listening_addresses(port: int, proc_root: Path = Path("/proc/net")) -> list[str]:
    """Local addresses that are in LISTEN state on `port` (parsed from /proc/net/tcp{,6})."""
    found = []
    for name, width in (("tcp", 8), ("tcp6", 32)):
        f = proc_root / name
        if not f.exists():
            continue
        for line in f.read_text().splitlines()[1:]:
            cols = line.split()
            if len(cols) < 4 or cols[3] != "0A":
                continue
            addr, hexport = cols[1].rsplit(":", 1)
            if int(hexport, 16) != port:
                continue
            if width == 8:
                found.append(".".join(str(int(addr[i:i + 2], 16)) for i in (6, 4, 2, 0)))
            else:
                found.append("::" if set(addr) == {"0"} else ("::1" if addr == "0" * 30 + "01" or addr == "00000000000000000000000001000000" else addr))
    return sorted(set(found))


def is_loopback(addr: str) -> bool:
    return addr.startswith("127.") or addr == "::1"


def http_get(url: str, timeout: float = 5.0) -> tuple[int, bytes]:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            return r.status, r.read()
    except urllib.error.HTTPError as e:
        return e.code, b""


def comfy_args(base: Path, port: int, gpu: str | None, extra: list[str]) -> list[str]:
    data = base / "comfy-data"
    args = ["--listen", "127.0.0.1", "--port", str(port), "--disable-auto-launch",
            "--base-directory", str(data), "--temp-directory", str(base / "comfy-temp")]
    if gpu not in (None, "", "none"):
        args += ["--cuda-device", str(gpu)]
    return args + list(extra)


def serve(base: Path, port: int, gpu: str | None, extra: list[str], wait: float) -> dict:
    checkout, venv = base / "ComfyUI", base / "venv"
    if not (checkout / "main.py").exists() or not venv_python(venv).exists():
        raise RuntimeError("ComfyUI is not installed in " + str(base) + " (run install-comfyui first)")
    for sub in ("models/diffusion_models", "models/text_encoders", "models/vae", "custom_nodes", "input", "output", "user"):
        (base / "comfy-data" / sub).mkdir(parents=True, exist_ok=True)
    (base / "comfy-temp").mkdir(parents=True, exist_ok=True)
    (base / "logs").mkdir(parents=True, exist_ok=True)
    pidfile, logfile = base / "logs" / "comfyui.pid", base / "logs" / "comfyui.log"
    stop(base, quiet=True)

    env = server_env(base, gpu)
    cmd = [str(venv_python(venv)), str(checkout / "main.py"), *comfy_args(base, port, gpu, extra)]
    with open(logfile, "wb") as log:   # the child keeps its own copy of the handle; the parent must not leak one
        proc = subprocess.Popen(cmd, cwd=checkout, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    pidfile.write_text(str(proc.pid))
    _CHILDREN[proc.pid] = proc
    deadline, ready = time.monotonic() + wait, False
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            stop(base, quiet=True)
            raise RuntimeError(f"ComfyUI exited with code {proc.returncode}:\n" + redact(tail(logfile, 40)))
        try:
            if http_get(f"http://127.0.0.1:{port}/system_stats")[0] == 200:
                ready = True
                break
        except OSError:
            pass
        time.sleep(1)
    if not ready:
        stop(base, quiet=True)
        raise RuntimeError(f"ComfyUI did not become ready within {wait:.0f}s:\n" + redact(tail(logfile, 40)))
    addrs = listening_addresses(port)
    if not addrs or not all(is_loopback(a) for a in addrs):
        stop(base, quiet=True)
        raise RuntimeError(f"ComfyUI is listening on {addrs}, not loopback only; stopped it")
    return {"pid": proc.pid, "port": port, "listening_on": addrs, "loopback_only": True, "log": str(logfile),
            "args": comfy_args(base, port, gpu, extra), "cuda_visible_devices": env.get("CUDA_VISIBLE_DEVICES")}


_CHILDREN: dict[int, subprocess.Popen] = {}   # servers started by this process, kept so they are reaped (no zombies)


def tail(path: Path, n: int) -> str:
    try:
        return "\n".join(path.read_text(errors="replace").splitlines()[-n:])
    except OSError:
        return ""


def stop(base: Path, quiet: bool = False) -> bool:
    """Stop the ComfyUI started by `serve` (SIGTERM to its process group, SIGKILL after ~15 s)."""
    pidfile = base / "logs" / "comfyui.pid"
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
            if (proc.poll() is not None) if proc is not None else not _alive(pid):
                break
        else:
            os.killpg(group, signal.SIGKILL)
    except OSError:
        pass                      # already gone
    if proc is not None:
        try:
            proc.wait(timeout=10)
        except subprocess.SubprocessError:
            pass
    pidfile.unlink(missing_ok=True)
    if not quiet:
        print("ComfyUI stopped")
    return True


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


# ===================================================================================== MP4 verification
def probe_mp4(path: Path, frames: int | None = None, size: tuple[int, int] | None = None, fps: float | None = None) -> dict:
    """Open and fully decode an MP4 with PyAV. `problems` is empty only for a real, non-blank video."""
    import numpy as np

    path = Path(path)
    r: dict = {"path": str(path), "exists": path.is_file(), "bytes": path.stat().st_size if path.is_file() else 0,
               "opened": False, "problems": []}
    if not r["exists"] or r["bytes"] < 1024:
        r["problems"].append("file is missing or implausibly small")
        return r
    r["ftyp"] = path.read_bytes()[4:8] == b"ftyp"
    if not r["ftyp"]:
        r["problems"].append("not an MP4 (no ftyp box)")
    try:
        import av

        with av.open(str(path)) as c:
            r["opened"] = True
            r["container"] = c.format.name
            v = c.streams.video[0]
            cc = v.codec_context
            r.update(codec=cc.name, width=cc.width, height=cc.height, pix_fmt=cc.pix_fmt,
                     fps=float(v.average_rate) if v.average_rate else None,
                     duration_s=round(c.duration / 1_000_000, 3) if c.duration else None)
            means, stds, first, last = [], [], None, None
            for frame in c.decode(video=0):
                g = frame.to_ndarray(format="gray")
                means.append(float(g.mean()))
                stds.append(float(g.std()))
                small = g[::4, ::4].astype(np.int16)
                first = small if first is None else first
                last = small
            r["decoded_frames"] = len(means)
            if means:
                r["luma_mean_min_max"] = [round(min(means), 2), round(max(means), 2)]
                r["luma_std_max"] = round(max(stds), 2)
                r["first_last_mean_abs_diff"] = round(float(np.abs(first - last).mean()), 2)
    except Exception as e:  # noqa: BLE001
        r["problems"].append(f"cannot be decoded: {type(e).__name__}: {str(e)[:200]}")
        return r
    if "mp4" not in r["container"]:
        r["problems"].append(f"container is {r['container']}, not MP4")
    if r["codec"] != "h264":
        r["problems"].append(f"video codec is {r['codec']}, expected h264")
    if frames is not None and r["decoded_frames"] != frames:
        r["problems"].append(f"decoded {r['decoded_frames']} frames, expected {frames}")
    if size is not None and (r["width"], r["height"]) != tuple(size):
        r["problems"].append(f"size is {r['width']}x{r['height']}, expected {size[0]}x{size[1]}")
    if fps is not None and r["fps"] is not None and abs(r["fps"] - fps) > 0.01:
        r["problems"].append(f"fps is {r['fps']}, expected {fps}")
    if r["decoded_frames"] and r["luma_mean_min_max"][1] < 3.0:
        r["problems"].append("every frame is black (suspected fp16 numerical failure)")
    elif r["decoded_frames"] and r["luma_std_max"] < 1.0:
        r["problems"].append("every frame is a flat single colour (no image content)")
    return r


# ===================================================================================== metrics
class Sampler(threading.Thread):
    """Samples GPU memory (nvidia-smi), host RAM and the ComfyUI process' RSS while a job runs."""

    def __init__(self, gpu: int = 0, pid: int | None = None, interval: float = 0.5):
        super().__init__(daemon=True)
        self.gpu, self.pid, self.interval = gpu, pid, interval
        self._stop = threading.Event()
        self.gpu_available = shutil.which("nvidia-smi") is not None
        self.samples = 0
        self.vram_used_mib: list[int] = []
        self.gpu_util: list[int] = []
        self.vram_total_mib: int | None = None
        self.ram_available_min: int | None = None
        self.ram_total: int | None = None
        self.rss_peak_kib: int | None = None
        self.baseline_vram_mib: int | None = None

    def sample_once(self) -> None:
        self.samples += 1
        if self.gpu_available:
            try:
                p = subprocess.run(["nvidia-smi", "-i", str(self.gpu), "--query-gpu=memory.used,memory.total,utilization.gpu",
                                    "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=5)
                used, total, util = [x.strip() for x in p.stdout.strip().split(",")[:3]]
                self.vram_used_mib.append(int(used))
                self.vram_total_mib = int(total)
                if util.isdigit():
                    self.gpu_util.append(int(util))
            except (OSError, ValueError, subprocess.SubprocessError):
                pass
        try:
            mem = dict((k, int(v.split()[0]) * 1024) for k, v in (l.split(":", 1) for l in Path("/proc/meminfo").read_text().splitlines()))
            self.ram_total = mem["MemTotal"]
            self.ram_available_min = min(self.ram_available_min or mem["MemAvailable"], mem["MemAvailable"])
        except (OSError, KeyError, ValueError):
            pass
        if self.pid:
            try:
                for line in Path(f"/proc/{self.pid}/status").read_text().splitlines():
                    if line.startswith("VmHWM:"):
                        self.rss_peak_kib = max(self.rss_peak_kib or 0, int(line.split()[1]))
            except (OSError, ValueError):
                pass

    def run(self) -> None:
        while not self._stop.is_set():
            self.sample_once()
            self._stop.wait(self.interval)

    def finish(self) -> dict:
        self._stop.set()
        self.join(timeout=10)
        self.sample_once()
        peak = max(self.vram_used_mib) if self.vram_used_mib else None
        return {
            "samples": self.samples,
            "gpu_sampling": "nvidia-smi" if self.vram_used_mib else "unavailable",
            "gpu_index": self.gpu,
            "vram_total_mib": self.vram_total_mib,
            "vram_baseline_mib": self.baseline_vram_mib,
            "vram_peak_used_mib": peak,
            "vram_peak_above_baseline_mib": (peak - self.baseline_vram_mib) if peak is not None and self.baseline_vram_mib is not None else None,
            "gpu_util_peak_pct": max(self.gpu_util) if self.gpu_util else None,
            "gpu_util_mean_pct": round(sum(self.gpu_util) / len(self.gpu_util)) if self.gpu_util else None,
            "ram_total_gib": round(self.ram_total / GIB, 2) if self.ram_total else None,
            "ram_peak_used_gib": round((self.ram_total - self.ram_available_min) / GIB, 2) if self.ram_total and self.ram_available_min is not None else None,
            "comfyui_rss_peak_gib": round(self.rss_peak_kib / 1024 / 1024, 2) if self.rss_peak_kib else None,
        }


# ===================================================================================== template check
def check_templates(url: str, backend_dir: Path, workflows_dir: Path) -> dict:
    """Reuse the Phase 4 code: WorkflowTemplates builds the real graphs, validate_graph checks them against
    the schemas that THIS server reports. Catches ComfyUI version drift and missing model files."""
    sys.path[:0] = [str(Path(backend_dir).resolve()), str(Path(backend_dir).resolve() / "tests")]
    from app.jobs.models import GenerationMode
    from app.workflows import GraphValues, WorkflowTemplates
    from comfyui_support import validate_graph

    wt = WorkflowTemplates(Path(workflows_dir))
    info: dict = {}
    missing: list[str] = []
    for cls in wt.node_classes:
        status, body = http_get(f"{url}/object_info/{cls}", timeout=30)
        data = json.loads(body) if status == 200 and body else {}
        if cls in data:
            info[cls] = data[cls]
        else:
            missing.append(cls)
    out: dict = {"server": url, "node_classes_required": wt.node_classes, "node_classes_missing": missing, "templates": {}}
    for mode in GenerationMode:
        values = GraphValues(prompt="check", negative_prompt="", width=832, height=480, num_frames=17, fps=24, steps=20,
                             guidance_scale=5.0, seed=1, filename_prefix="vidsgen/job_check",
                             image="vidsgen/job_check.png" if mode == GenerationMode.i2v else None)
        problems = validate_graph(wt.build(mode, values), info) if not missing else ["skipped: required node classes are missing"]
        out["templates"][mode.value] = {"problems": problems, "ok": not problems}
    out["ok"] = not missing and all(t["ok"] for t in out["templates"].values())
    return out


# ===================================================================================== smoke test
DEFAULT_PROMPT = ("A paper boat drifting down a rainy neon-lit city street at night, reflections rippling on wet asphalt, "
                  "slow cinematic camera movement")


async def run_provider(args, sampler: Sampler) -> dict:
    sys.path.insert(0, str(Path(args.backend_dir).resolve()))
    from app.jobs.models import GenerationMode
    from app.providers.base import GenerationRequest, ProviderError
    from app.providers.comfyui import ComfyUIProvider
    from app.workflows import WorkflowTemplates

    width, height = (int(x) for x in args.resolution.split("x"))
    prov = ComfyUIProvider(base_url=args.url, workflows=WorkflowTemplates(Path(args.workflows_dir)),
                           api_token=os.environ.get("COMFYUI_API_TOKEN", ""), timeout=60.0, poll_interval=2.0)
    res: dict = {"queued": False, "updates": [], "final_state": None, "error": None}
    try:
        h = await prov.health()
        res["health_before"] = {"ok": h.ok, "detail": h.detail, "checks": [{"name": c.name, "status": c.status, "detail": c.detail} for c in h.checks]}
        if not h.ok:
            res["error"] = {"code": "provider_not_ready", "message": h.detail}
            return res
        sampler.sample_once()
        sampler.baseline_vram_mib = sampler.vram_used_mib[-1] if sampler.vram_used_mib else None
        request = GenerationRequest(job_id="job_smoke_t2v", mode=GenerationMode.t2v, prompt=args.prompt, negative_prompt="",
                                    width=width, height=height, num_frames=args.frames, fps=args.fps, steps=args.steps,
                                    guidance_scale=args.cfg, seed=args.seed)
        t0 = time.monotonic()
        pid = await prov.submit(request)
        res["queued"], res["prompt_id"] = True, pid
        try:
            async with asyncio.timeout(args.timeout):
                last_beat = 0.0
                async for u in prov.progress(pid):
                    now = time.monotonic() - t0
                    res["updates"].append({"t": round(now, 2), "state": u.state, "percent": u.percent, "message": u.message})
                    if u.state == "failed":
                        res["error"] = {"code": u.error_code, "message": u.message}
                    if now - last_beat >= 10 or u.state in ("completed", "failed", "cancelled"):   # a heartbeat for long runs
                        last_beat = now
                        vram = f" | GPU memory {sampler.vram_used_mib[-1]} MiB" if sampler.vram_used_mib else ""
                        print(f"  [{now:6.0f}s] {u.state:<9} {u.percent:5.1f}%  {u.message}{vram}", file=sys.stderr, flush=True)
        except TimeoutError:
            await prov.cancel(pid)
            res["error"] = {"code": "timeout", "message": f"no result within {args.timeout}s; cancelled"}
        res["final_state"] = res["updates"][-1]["state"] if res["updates"] else None
        res["wall_seconds"] = round(time.monotonic() - t0, 1)
        if res["final_state"] == "completed":
            result = await prov.fetch_result(pid)
            Path(args.out).parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(result.path, args.out)
            result.cleanup()
            res["output"] = args.out
        await prov.release(pid)
        h2 = await prov.health()
        res["health_after"] = {"ok": h2.ok, "detail": h2.detail, "checks": [{"name": c.name, "status": c.status, "detail": c.detail} for c in h2.checks]}
    except ProviderError as e:
        res["error"] = {"code": e.code, "message": e.message}
    finally:
        await prov.aclose()
    return res


def step_timing(updates: list[dict]) -> dict:
    steps = [(u["t"], int(m.group(1)), int(m.group(2))) for u in updates if (m := re.match(r"step (\d+)/(\d+)", u["message"]))]
    out: dict = {"first_update_s": updates[0]["t"] if updates else None}
    if steps:
        out.update(sampler_steps_seen=len(steps), first_step_at_s=steps[0][0], last_step_at_s=steps[-1][0], steps_total=steps[-1][2])
        if steps[-1][1] > steps[0][1]:
            out["seconds_per_step"] = round((steps[-1][0] - steps[0][0]) / (steps[-1][1] - steps[0][1]), 2)
    return out


def evaluate(res: dict, probe: dict | None, sampler_stats: dict, require_gpu: bool) -> tuple[str, list[str]]:
    """PASS only if a real video came out. Queued/started/completed-without-a-file never counts."""
    reasons: list[str] = []
    if not res.get("queued"):
        reasons.append("the workflow was never queued: " + json.dumps(res.get("error")))
    elif res.get("final_state") != "completed":
        reasons.append(f"generation ended as {res.get('final_state')!r}: {json.dumps(res.get('error'))}")
    if res.get("final_state") == "completed":
        if probe is None:
            reasons.append("no output file was retrieved")
        else:
            reasons += probe["problems"]
    if require_gpu and sampler_stats["gpu_sampling"] == "unavailable":
        reasons.append("GPU memory could not be measured (nvidia-smi unavailable)")
    return ("PASS" if not reasons else "FAIL"), reasons


def smoke(args) -> int:
    """One smoke test; if it fails with an out-of-memory error and fallback arguments were given, restart
    ComfyUI once with them (e.g. --disable-dynamic-vram --disable-smart-memory) and try again."""
    code = smoke_once(args)
    fallback = getattr(args, "fallback_args", None) or []
    fallback = shlex.split(fallback) if isinstance(fallback, str) else list(fallback)
    if code == 0 or not fallback or not args.base:
        return code
    first = json.loads(Path(args.report).read_text())
    if (first.get("error") or {}).get("code") != "gpu_out_of_memory":
        return code
    print(f"\nOut of GPU memory with default settings. Restarting ComfyUI ONCE with {fallback} and retrying...", file=sys.stderr, flush=True)
    first_path = Path(args.report).with_name(Path(args.report).stem + "_attempt1.json")
    shutil.copyfile(args.report, first_path)
    stop(Path(args.base), quiet=True)
    info = serve(Path(args.base), int(getattr(args, "port", 8188)), str(args.gpu), fallback, wait=300)
    code = smoke_once(args)
    second = json.loads(Path(args.report).read_text())
    second["retry"] = {"first_attempt_report": str(first_path), "first_attempt_error": first.get("error"), "fallback_args": fallback,
                       "server_args": info["args"]}
    Path(args.report).write_text(json.dumps(second, indent=1, default=str))
    return code


def smoke_once(args) -> int:
    pid = None
    pidfile = Path(args.base) / "logs" / "comfyui.pid" if args.base else None
    if pidfile and pidfile.exists():
        pid = int(pidfile.read_text())
    sampler = Sampler(gpu=args.gpu, pid=pid)
    sampler.start()
    t_start = time.monotonic()
    try:
        res = asyncio.run(run_provider(args, sampler))
    except ImportError as e:   # e.g. run with the system python instead of the ComfyUI venv's python
        res = {"queued": False, "updates": [], "final_state": None, "error": {
            "code": "missing_dependency", "message": f"{e}. Run this command with the ComfyUI venv's python (<base>/venv/bin/python)."}}
    stats = sampler.finish()
    size = tuple(int(x) for x in args.resolution.split("x"))
    probe = probe_mp4(Path(args.out), frames=args.frames, size=size, fps=float(args.fps)) if res.get("output") else None
    verdict, reasons = evaluate(res, probe, stats, args.require_gpu)
    server_log = redact(tail(Path(args.base) / "logs" / "comfyui.log", 60)) if args.base and verdict == "FAIL" else None
    report = {
        "verdict": verdict, "reasons": reasons,
        "claims_checked": {"server_responded": res.get("health_before") is not None, "workflow_queued": res.get("queued", False),
                           "generation_completed": res.get("final_state") == "completed",
                           "mp4_exists": bool(probe and probe["exists"]), "mp4_decodes": bool(probe and probe["opened"]),
                           "frames_and_size_match": bool(probe and not any("frames" in p or "size" in p for p in probe["problems"])),
                           "not_blank": bool(probe and probe["opened"] and not any("black" in p or "flat" in p for p in probe["problems"]))},
        "request": {"mode": "t2v", "resolution": args.resolution, "frames": args.frames, "fps": args.fps, "steps": args.steps,
                    "cfg": args.cfg, "seed": args.seed, "prompt": args.prompt},
        "timing": {"total_wall_s": round(time.monotonic() - t_start, 1), "generation_wall_s": res.get("wall_seconds"), **step_timing(res.get("updates", []))},
        "resources": stats, "output": probe, "error": res.get("error"),
        "health_before": res.get("health_before"), "health_after": res.get("health_after"),
        "progress_updates": res.get("updates", [])[-25:], "server_log_tail": server_log,
    }
    text = json.dumps(report, indent=1, default=str)
    assert "Bearer " not in text.replace("Bearer [redacted]", "")
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text(text)
    emit({k: report[k] for k in ("verdict", "reasons", "claims_checked", "timing", "resources", "output", "error")})
    return 0 if verdict == "PASS" else 1


# ===================================================================================== CLI
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("preflight")
    p.add_argument("--base", required=True); p.add_argument("--manifest", required=True)
    p.add_argument("--gpu", type=int, default=0); p.add_argument("--min-vram-gib", type=float, default=14.0)
    p.add_argument("--min-ram-gib", type=float, default=24.0); p.add_argument("--overhead-gib", type=float, default=8.0)
    p.add_argument("--repo", action="append", default=[], help="Git URL to check for reachability (repeatable)")
    p.add_argument("--offline", action="store_true", help="skip internet checks (for tests)")
    p.add_argument("--allow-no-gpu", action="store_true", help="downgrade 'no GPU' to a warning (for tests on CPU machines)")
    p.add_argument("--report")
    p = sub.add_parser("gpu-probe"); p.add_argument("--gpu", type=int, default=0)
    p = sub.add_parser("install-comfyui")
    p.add_argument("--base", required=True); p.add_argument("--commit", required=True)
    p.add_argument("--expect-version"); p.add_argument("--repo-url", default="https://github.com/comfyanonymous/ComfyUI.git")
    p = sub.add_parser("serve")
    p.add_argument("--base", required=True); p.add_argument("--port", type=int, default=8188)
    p.add_argument("--gpu", default="0"); p.add_argument("--wait", type=float, default=300)
    p.add_argument("extra", nargs="*", help="extra ComfyUI arguments (after --)")
    p = sub.add_parser("stop"); p.add_argument("--base", required=True)
    p = sub.add_parser("check-templates")
    p.add_argument("--url", default="http://127.0.0.1:8188")
    p.add_argument("--backend-dir", required=True); p.add_argument("--workflows-dir", required=True)
    p = sub.add_parser("smoke-t2v")
    p.add_argument("--base"); p.add_argument("--url", default="http://127.0.0.1:8188")
    p.add_argument("--backend-dir", required=True); p.add_argument("--workflows-dir", required=True)
    p.add_argument("--resolution", default="832x480"); p.add_argument("--frames", type=int, default=17)
    p.add_argument("--fps", type=int, default=24); p.add_argument("--steps", type=int, default=20)
    p.add_argument("--cfg", type=float, default=5.0); p.add_argument("--seed", type=int, default=424242)
    p.add_argument("--prompt", default=DEFAULT_PROMPT); p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--timeout", type=float, default=2400); p.add_argument("--require-gpu", action="store_true")
    p.add_argument("--port", type=int, default=8188)
    p.add_argument("--fallback-args", default="", help='ComfyUI arguments for ONE retry after an out-of-memory failure, as one string; '
                   'use the = form: --fallback-args="--disable-dynamic-vram --disable-smart-memory"')
    p.add_argument("--out", required=True); p.add_argument("--report", required=True)
    p = sub.add_parser("verify-mp4")
    p.add_argument("file"); p.add_argument("--frames", type=int); p.add_argument("--size"); p.add_argument("--fps", type=float)
    a = ap.parse_args(argv)

    try:
        if a.cmd == "preflight":
            r = preflight(Path(a.base), Path(a.manifest), a.gpu, a.min_vram_gib, a.min_ram_gib, a.overhead_gib, a.offline,
                          a.allow_no_gpu, tuple(a.repo))
            for c in r["checks"]:
                print(f"[{c['status']:^4}] {c['name']}: {c['detail']}")
            print(f"\npreflight: {r['counts']['PASS']} PASS, {r['counts']['WARN']} WARN, {r['counts']['FAIL']} FAIL, {r['counts']['INFO']} INFO"
                  f" -> {'OK to continue' if r['ok'] else 'STOP: fix the FAIL items first'}")
            if a.report:
                Path(a.report).parent.mkdir(parents=True, exist_ok=True)
                Path(a.report).write_text(json.dumps(r, indent=1))
            return 0 if r["ok"] else 1
        elif a.cmd == "gpu-probe":
            emit(gpu_probe(a.gpu))
        elif a.cmd == "install-comfyui":
            emit(install_comfyui(Path(a.base), a.repo_url, a.commit, a.expect_version))
        elif a.cmd == "serve":
            emit(serve(Path(a.base), a.port, a.gpu, a.extra, a.wait))
        elif a.cmd == "stop":
            stop(Path(a.base))
        elif a.cmd == "check-templates":
            r = check_templates(a.url, Path(a.backend_dir), Path(a.workflows_dir))
            emit(r)
            return 0 if r["ok"] else 1
        elif a.cmd == "smoke-t2v":
            return smoke(a)
        elif a.cmd == "verify-mp4":
            size = tuple(int(x) for x in a.size.split("x")) if a.size else None
            r = probe_mp4(Path(a.file), a.frames, size, a.fps)
            emit(r)
            return 0 if not r["problems"] else 1
    except (RuntimeError, subprocess.SubprocessError, OSError) as e:
        print(f"FAILED: {redact(str(e))}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
