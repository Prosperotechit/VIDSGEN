#!/usr/bin/env python3
"""Download the three Wan 2.2 TI2V-5B model files for ComfyUI, from workflows/comfyui/models.yaml.

SAFE BY DEFAULT: without --execute this is a dry run. It only prints which files would be downloaded,
their sizes, pinned revisions and destinations. Nothing is downloaded and no network connection is made.

    python scripts/download_models.py --comfyui-dir DIR                 # dry run (default)
    python scripts/download_models.py --comfyui-dir DIR --execute       # download + verify (~16.9 GiB)
    python scripts/download_models.py --comfyui-dir DIR --verify-only   # hash existing files, no network

What --execute guarantees, for every file:
  * it downloads ONLY the three files listed in the manifest, each from its pinned repository revision
    (an immutable commit), never from a moving branch;
  * it refuses to start if the manifest lacks a valid revision or SHA-256 for any file
    (authoritative hashes are never guessed), or lists any 14B / LoRA / alternative-VAE style file;
  * the exact byte size AND the SHA-256 of the whole file are checked against the manifest;
  * on any mismatch the file is DELETED and the run STOPS (exit code 4);
  * interrupted downloads resume from a `.part` file via HTTP Range requests, and a resumed file is
    trusted only after the SHA-256 of the complete file matches (a corrupt prefix cannot survive).

Exit codes: 0 ok | 1 --verify-only found a problem | 2 refused before downloading (manifest, disk space)
            3 download incomplete (network; partial kept for resume) | 4 verification failed (file deleted)

Tokens: an optional Hugging Face token is read ONLY from the HF_TOKEN environment variable (e.g. a Kaggle
Secret), sent only to the Hugging Face host (never to CDN redirect targets) and never printed.
Uses only the Python standard library (+ PyYAML). Weights must never be committed to Git.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import socket
import sys
import time
from http.client import HTTPException
from pathlib import Path
from urllib import error, parse, request

REPO_ROOT = Path(__file__).resolve().parents[1]
MANIFEST = REPO_ROOT / "workflows" / "comfyui" / "models.yaml"
DEFAULT_ENDPOINT = "https://huggingface.co"   # override with HF_ENDPOINT (e.g. a local test server)

CHUNK = 8 * 1024 * 1024
HASH_CHUNK = 16 * 1024 * 1024
MAX_ATTEMPTS = 8
PROGRESS_SECONDS = 15.0
SOCKET_TIMEOUT = 60.0
FREE_SPACE_MARGIN = 1 << 30                       # keep 1 GiB free after the downloads
RETRY_BASE_DELAY = float(os.environ.get("VIDSGEN_DL_RETRY_DELAY", "5"))

EXPECTED_ROLES = ("diffusion_model", "vae", "text_encoder")
ALLOWED_DIRS = {"models/diffusion_models", "models/vae", "models/text_encoders"}
FORBIDDEN_PATH = re.compile(r"14B|high_noise|low_noise|fun_|lora|wan_2\.1_vae|animate|s2v|inpaint|camera", re.I)
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
REVISION_RE = re.compile(r"^[0-9a-f]{40}$")


class DownloaderError(Exception):
    exit_code = 2


class Refused(DownloaderError):
    exit_code = 2


class IncompleteDownload(DownloaderError):
    exit_code = 3


class VerificationFailed(DownloaderError):
    exit_code = 4


# ------------------------------------------------------------------------------------ manifest
def load_manifest(path: Path) -> dict:
    try:
        import yaml
    except ImportError:
        sys.exit("PyYAML is required: pip install pyyaml")
    data = yaml.safe_load(Path(path).read_text())
    files = data.get("files") or []
    for f in files:
        for key in ("role", "repo", "path", "comfyui_dir", "size_bytes"):
            if key not in f:
                sys.exit(f"Manifest entry is missing '{key}': {f}")
    return data


def validate_for_download(manifest: dict) -> list[str]:
    """Everything that must hold BEFORE any byte is downloaded. Returns a list of problems."""
    problems: list[str] = []
    files = manifest.get("files") or []
    roles = [f.get("role") for f in files]
    if sorted(roles) != sorted(EXPECTED_ROLES):
        problems.append(f"manifest must list exactly the roles {list(EXPECTED_ROLES)} once each, found {roles}")
    for f in files:
        name = f"{f.get('role')} ({Path(str(f.get('path'))).name})"
        if not REVISION_RE.match(str(f.get("revision", ""))):
            problems.append(f"{name}: missing or invalid pinned `revision` (needs a 40-character commit id)")
        if not SHA256_RE.match(str(f.get("sha256", ""))):
            problems.append(f"{name}: authoritative `sha256` is missing or invalid (64 lowercase hex characters); "
                            "refusing to download a file whose hash is unknown")
        if not str(f.get("path", "")).startswith("split_files/") or ".." in str(f.get("path", "")):
            problems.append(f"{name}: path must be under split_files/")
        if FORBIDDEN_PATH.search(str(f.get("path", ""))):
            problems.append(f"{name}: path matches a file type this project must not download (14B/LoRA/alternative VAE)")
        if f.get("comfyui_dir") not in ALLOWED_DIRS:
            problems.append(f"{name}: comfyui_dir must be one of {sorted(ALLOWED_DIRS)}")
        if not isinstance(f.get("size_bytes"), int) or f["size_bytes"] <= 0:
            problems.append(f"{name}: size_bytes must be a positive integer")
    return problems


def gb(n: int) -> str:
    return f"{n / 1e9:.1f} GB"


def gib(n: float) -> str:
    return f"{n / 2**30:.2f} GiB"


def plan(manifest: dict, comfyui_dir: Path) -> list[dict]:
    rows = []
    for f in manifest["files"]:
        dest = comfyui_dir / f["comfyui_dir"] / Path(f["path"]).name
        size = dest.stat().st_size if dest.is_file() else None
        status = "present" if size == f["size_bytes"] else ("wrong size" if size is not None else "missing")
        rows.append({**f, "dest": dest, "status": status})
    return rows


# ------------------------------------------------------------------------------------ hashing
def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while chunk := fh.read(HASH_CHUNK):
            h.update(chunk)
    return h.hexdigest()


def verify_file(path: Path, row: dict, label: str = "") -> tuple[bool, str]:
    """(ok, reason). Checks exact size first (cheap), then the SHA-256 of the whole file."""
    if not path.is_file():
        return False, "file does not exist"
    size = path.stat().st_size
    if size != row["size_bytes"]:
        return False, f"size {size:,} bytes != expected {row['size_bytes']:,}"
    t0 = time.monotonic()
    print(f"  {label}hashing {gib(size)} (SHA-256)...", flush=True)
    digest = sha256_file(path)
    took = time.monotonic() - t0
    if digest != row["sha256"]:
        return False, f"SHA-256 {digest} != expected {row['sha256']}"
    print(f"  {label}SHA-256 OK ({took:.0f}s)", flush=True)
    return True, "ok"


# ------------------------------------------------------------------------------------ network
class _NoRedirect(request.HTTPRedirectHandler):
    def redirect_request(self, *args, **kwargs):  # we follow redirects ourselves, to control headers
        return None


def _open_range(url: str, start: int, token: str, trusted_netloc: str):
    """GET url (optionally from byte `start`). Redirects are followed manually so the Authorization
    header is only ever sent to the Hugging Face host, never to a CDN host."""
    opener = request.build_opener(_NoRedirect)
    for _hop in range(6):
        headers = {"User-Agent": "vidsgen-model-downloader/2"}
        if start:
            headers["Range"] = f"bytes={start}-"
        if token and parse.urlsplit(url).netloc == trusted_netloc:
            headers["Authorization"] = f"Bearer {token}"
        try:
            return opener.open(request.Request(url, headers=headers), timeout=SOCKET_TIMEOUT)
        except error.HTTPError as e:
            location = e.headers.get("Location") if e.headers else None
            if e.code in (301, 302, 303, 307, 308) and location:
                url = parse.urljoin(url, location)
                e.close()
                continue
            raise
    raise IncompleteDownload("too many redirects")


def _retriable(e: BaseException) -> bool:
    if isinstance(e, error.HTTPError):
        return e.code in (408, 429) or e.code >= 500
    return isinstance(e, (error.URLError, socket.timeout, TimeoutError, ConnectionError, HTTPException, OSError))


class _Progress:
    def __init__(self, label: str, total: int, start: int):
        self.label, self.total, self.start = label, total, start
        self.t0 = self.last = time.monotonic()

    def tick(self, done: int, force: bool = False) -> None:
        now = time.monotonic()
        if not force and now - self.last < PROGRESS_SECONDS:
            return
        self.last = now
        rate = (done - self.start) / max(now - self.t0, 1e-6)
        eta = (self.total - done) / rate if rate > 0 else 0
        print(f"  {self.label}{gib(done)} / {gib(self.total)} ({100 * done / self.total:5.1f}%)  "
              f"{rate / 1e6:6.1f} MB/s  ETA {int(eta // 60)}m{int(eta % 60):02d}s", flush=True)


def _discard(*paths: Path) -> None:
    for p in paths:
        p.unlink(missing_ok=True)


def download_file(row: dict, endpoint: str, token: str, label: str = "") -> None:
    """Download one file into `<dest>.part`, resuming if possible, verify, then move it into place.
    Raises IncompleteDownload (network) or VerificationFailed (file deleted)."""
    dest: Path = row["dest"]
    part, meta = dest.with_name(dest.name + ".part"), dest.with_name(dest.name + ".part.json")
    expected = row["size_bytes"]
    want_meta = {"repo": row["repo"], "revision": row["revision"], "path": row["path"],
                 "size_bytes": expected, "sha256": row["sha256"]}
    url = f"{endpoint}/{row['repo']}/resolve/{row['revision']}/{parse.quote(row['path'])}"
    trusted = parse.urlsplit(endpoint).netloc

    # A partial file is resumed only if it provably belongs to this exact revision/hash and is not too big.
    if part.exists():
        try:
            same = json.loads(meta.read_text()) == want_meta
        except (OSError, ValueError):
            same = False
        if not same or part.stat().st_size > expected:
            print(f"  {label}discarding an unverifiable partial download", flush=True)
            _discard(part, meta)
    dest.parent.mkdir(parents=True, exist_ok=True)
    meta.write_text(json.dumps(want_meta))

    t0 = time.monotonic()
    attempt, last_have = 0, -1
    while True:
        have = part.stat().st_size if part.exists() else 0
        if have == expected:
            break
        if have > last_have:        # the previous try made progress: only consecutive failures count
            attempt = 0
        last_have = have
        attempt += 1
        if attempt > MAX_ATTEMPTS:
            raise IncompleteDownload(f"gave up after {MAX_ATTEMPTS} attempts without progress at "
                                     f"{have:,}/{expected:,} bytes (the partial file is kept; run again to resume)")
        try:
            resp = _open_range(url, have, token, trusted)
            status = resp.status
            length = resp.headers.get("Content-Length")
            if have and status == 206:
                m = re.match(r"bytes (\d+)-(\d+)/(\d+|\*)$", resp.headers.get("Content-Range", ""))
                if not m or int(m.group(1)) != have or (m.group(3) != "*" and int(m.group(3)) != expected):
                    raise VerificationFailed(f"server answered with an unexpected Content-Range "
                                             f"{resp.headers.get('Content-Range')!r}")
            if status == 200 and have:
                print(f"  {label}server ignored the Range request; restarting from byte 0", flush=True)
                have = 0
            if length is not None and int(length) != expected - have:
                raise VerificationFailed(f"server reports {int(length):,} bytes to send, expected {expected - have:,}: "
                                         f"this is not the file in the manifest")
            progress = _Progress(label, expected, have)
            if have:
                print(f"  {label}resuming at {gib(have)}", flush=True)
            with open(part, "ab" if have else "wb") as fh:
                done = have
                while chunk := resp.read(CHUNK):
                    done += len(chunk)
                    if done > expected:
                        raise VerificationFailed("server sent more bytes than the manifest size")
                    fh.write(chunk)
                    progress.tick(done)
                fh.flush()
                os.fsync(fh.fileno())
            resp.close()
            progress.tick(done, force=True)
            if done < expected:   # http.client returns silently when a server closes early
                print(f"  {label}connection closed early at {gib(done)}; retrying ({attempt}/{MAX_ATTEMPTS})", flush=True)
                time.sleep(min(RETRY_BASE_DELAY * 2 ** (attempt - 1), 60))
        except VerificationFailed:
            _discard(part, meta)
            raise
        except error.HTTPError as e:
            if e.code == 416:   # asked for bytes beyond the end: our partial file cannot be trusted
                _discard(part, meta)
                meta.write_text(json.dumps(want_meta))
                continue
            if _retriable(e):
                print(f"  {label}HTTP {e.code}; retrying ({attempt}/{MAX_ATTEMPTS})", flush=True)
                time.sleep(min(RETRY_BASE_DELAY * 2 ** (attempt - 1), 60))
                continue
            hint = " (check HF_TOKEN / access)" if e.code in (401, 403) else ""
            raise Refused(f"HTTP {e.code} for {row['repo']}@{row['revision'][:12]}/{row['path']}{hint}") from None
        except (OSError, HTTPException) as e:
            if isinstance(e, IncompleteDownload):
                raise
            print(f"  {label}connection problem ({type(e).__name__}); retrying ({attempt}/{MAX_ATTEMPTS})", flush=True)
            time.sleep(min(RETRY_BASE_DELAY * 2 ** (attempt - 1), 60))

    dl_time = time.monotonic() - t0
    size = part.stat().st_size
    if size != expected:
        _discard(part, meta)
        raise VerificationFailed(f"size {size:,} bytes != expected {expected:,}; file deleted")
    print(f"  {label}download complete ({dl_time:.0f}s); size {size:,} bytes OK", flush=True)
    ok, why = verify_file(part, row, label)
    if not ok:
        _discard(part, meta)
        raise VerificationFailed(f"{why}; the downloaded file was deleted")
    os.replace(part, dest)
    meta.unlink(missing_ok=True)


# ------------------------------------------------------------------------------------ commands
def _print_plan(manifest: dict, rows: list[dict], comfyui_dir: Path, manifest_path: Path) -> list[dict]:
    todo = [r for r in rows if r["status"] != "present"]
    print(f"Model: {manifest.get('model')}   manifest: {manifest_path}")
    print(f"ComfyUI directory: {comfyui_dir}\n")
    for r in rows:
        print(f"  [{r['status']:>10}] {r['role']:<16} {gb(r['size_bytes']):>8}  {r['repo']}/{r['path']}")
        if r.get("revision") or r.get("sha256"):
            print(f"               revision {str(r.get('revision', '?'))[:12]}  sha256 {str(r.get('sha256', '?'))[:16]}...")
        print(f"               -> {r['dest']}")
    print(f"\nTo download: {len(todo)} file(s), {gb(sum(r['size_bytes'] for r in todo))}")
    return todo


def cmd_verify_only(rows: list[dict]) -> int:
    bad = 0
    for i, r in enumerate(rows, 1):
        print(f"[{i}/{len(rows)}] {r['role']}: {r['dest']}")
        ok, why = verify_file(r["dest"], r, label="")
        print(f"  [{'PASS' if ok else 'FAIL'}] {why if not ok else 'size and SHA-256 match the manifest'}")
        bad += not ok
    print("\n" + ("ALL MODEL FILES VERIFIED (size + SHA-256)" if not bad else f"{bad} file(s) FAILED verification"))
    return 0 if not bad else 1


def cmd_execute(manifest: dict, rows: list[dict], comfyui_dir: Path) -> int:
    endpoint = os.environ.get("HF_ENDPOINT", DEFAULT_ENDPOINT).rstrip("/")
    token = os.environ.get("HF_TOKEN", "").strip()
    print(f"Source: {endpoint}   token: {'provided via HF_TOKEN (not shown)' if token else 'none (public repositories)'}")

    # Pre-flight: nothing is downloaded unless the whole set fits on disk.
    need = 0
    for r in rows:
        part = r["dest"].with_name(r["dest"].name + ".part")
        have = part.stat().st_size if part.exists() else 0
        if r["status"] != "present":
            need += max(r["size_bytes"] - have, 0)
    anchor = comfyui_dir
    while not anchor.exists():
        anchor = anchor.parent
    free = shutil.disk_usage(anchor).free
    print(f"Disk: {gib(free)} free at {anchor}; up to {gib(need)} still to download (+{gib(FREE_SPACE_MARGIN)} margin)")
    if free < need + FREE_SPACE_MARGIN:
        raise Refused(f"not enough free disk space ({gib(free)}) for {gib(need)} of downloads; nothing downloaded")

    for i, r in enumerate(rows, 1):
        label = f"[{i}/{len(rows)} {r['role']}] "
        print(f"\n{label}{r['repo']}@{r['revision'][:12]}/{r['path']}\n  destination: {r['dest']}", flush=True)
        if r["dest"].is_file():
            ok, why = verify_file(r["dest"], r, label)
            if ok:
                print(f"  {label}already present and verified; skipping", flush=True)
                continue
            print(f"  {label}existing file is not the expected one ({why}); deleting it and downloading again", flush=True)
            r["dest"].unlink()
        download_file(r, endpoint, token, label)
        print(f"  {label}VERIFIED and installed: {r['dest']}", flush=True)

    print(f"\nALL {len(rows)} MODEL FILES DOWNLOADED AND VERIFIED (exact size + SHA-256, pinned revisions)")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--comfyui-dir", type=Path, default=Path("ComfyUI"),
                    help="Directory whose models/ subfolders receive the files. Default: ./ComfyUI")
    ap.add_argument("--manifest", type=Path, default=MANIFEST)
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--execute", action="store_true", help="Actually download and verify. Without it: dry run only.")
    mode.add_argument("--verify-only", action="store_true", help="Hash the files already on disk; no network, no changes.")
    args = ap.parse_args(argv)

    manifest = load_manifest(args.manifest)
    comfyui_dir = args.comfyui_dir.resolve()
    rows = plan(manifest, comfyui_dir)
    try:
        if args.verify_only:
            return cmd_verify_only(rows)
        todo = _print_plan(manifest, rows, comfyui_dir, args.manifest)
        problems = validate_for_download(manifest)
        if not args.execute:
            if problems:
                print("\nNOTE: --execute would refuse to run until the manifest is fixed:")
                for p in problems:
                    print(f"  - {p}")
            print("\nDRY RUN: nothing was downloaded. Re-run with --execute to download.")
            return 0
        if problems:
            raise Refused("manifest not acceptable for download:\n  - " + "\n  - ".join(problems))
        if not todo:
            print("All model files are present; verifying them.")
        return cmd_execute(manifest, rows, comfyui_dir)
    except DownloaderError as e:
        print(f"\nSTOPPED: {e}", file=sys.stderr)
        return e.exit_code


if __name__ == "__main__":
    raise SystemExit(main())
