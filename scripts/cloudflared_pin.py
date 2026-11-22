#!/usr/bin/env python3
"""Version pin and SHA-256 verification for a FUTURE cloudflared binary (Phase 6C Stage 4A: prepared, never run).

This tool never starts a tunnel and never executes the binary. It exists so that, when a tunnel test is separately approved, the
binary used is exactly one that was pinned in review:

  status    Show whether the pin file is complete. (Today it is NOT: version and hashes are null, so everything below refuses.)
  download  Fetch the pinned release asset over https from an allow-listed host into a quarantine file, verify its size and SHA-256
            while streaming, and only then move it into place with the execute bit set. A mismatch deletes it.
  verify    Re-check an existing file against the pin (use this immediately before any launch).

`require_verified(path)` is the function any future launcher must call before executing the binary.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import stat
import sys
import urllib.request
from pathlib import Path
from urllib.parse import urlsplit

PINS_FILE = Path(__file__).resolve().parent / "cloudflared_pins.json"
VERSION_RE = re.compile(r"^\d{4}\.\d{1,2}\.\d{1,2}$")                    # cloudflared's calendar versions, e.g. 2026.9.3
SHA_RE = re.compile(r"^[0-9a-f]{64}$")
ASSET_RE = re.compile(r"^cloudflared-[a-z0-9._-]{1,40}$")
RELEASE_URL = "https://github.com/cloudflare/cloudflared/releases/download/{version}/{asset}"
ALLOWED_HOSTS = ("github.com", "objects.githubusercontent.com", "release-assets.githubusercontent.com")
CHUNK = 1024 * 1024


class PinError(Exception):
    """The pin file is incomplete, a download is refused, or a binary does not match its pin. Never contains a secret."""


def load_pins(path: Path = PINS_FILE) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def platform_key() -> str:
    machine = {"x86_64": "amd64", "amd64": "amd64", "aarch64": "arm64", "arm64": "arm64"}.get(platform.machine().lower(), platform.machine().lower())
    return f"{sys.platform.replace('linux2', 'linux')}-{machine}"


def pinned_asset(pins: dict, key: str | None = None) -> dict:
    """The validated pin for this platform: {'version', 'asset', 'sha256', 'max_bytes'}. Raises PinError unless fully pinned."""
    version = pins.get("version")
    if not isinstance(version, str) or not VERSION_RE.match(version):
        raise PinError("no cloudflared version is pinned (the pin file is deliberately empty until a tunnel test is approved)")
    entry = (pins.get("assets") or {}).get(key or platform_key())
    if not isinstance(entry, dict):
        raise PinError(f"no pin for platform {key or platform_key()}")
    asset, sha, size = entry.get("asset"), entry.get("sha256"), entry.get("max_bytes")
    if not isinstance(asset, str) or not ASSET_RE.match(asset) or "/" in asset or ".." in asset:
        raise PinError("the pinned asset name is invalid")
    if not isinstance(sha, str) or not SHA_RE.match(sha):
        raise PinError("the pinned SHA-256 is missing or malformed: a binary without a pinned hash is never accepted")
    if not isinstance(size, int) or not 1 <= size <= 512 * 1024 * 1024:
        raise PinError("the pinned size limit is invalid")
    return {"version": version, "asset": asset, "sha256": sha, "max_bytes": size}


def release_url(pin: dict) -> str:
    return RELEASE_URL.format(version=pin["version"], asset=pin["asset"])


def check_download_url(url: str) -> None:
    """Only https URLs on the allow-listed GitHub hosts (the release asset and its CDN redirect) may be fetched."""
    parts = urlsplit(url)
    if parts.scheme != "https" or (parts.hostname or "") not in ALLOWED_HOSTS or parts.username or parts.password:
        raise PinError("refusing a download URL outside the allow-listed https hosts")


class _CheckedRedirects(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        check_download_url(newurl)                                       # a redirect may not leave the allow-list
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def open_url(url: str):
    check_download_url(url)
    return urllib.request.build_opener(_CheckedRedirects).open(url, timeout=60)


def sha256_file(path: Path, max_bytes: int | None = None) -> tuple[str, int]:
    h, n = hashlib.sha256(), 0
    with open(path, "rb") as f:
        while chunk := f.read(CHUNK):
            n += len(chunk)
            if max_bytes is not None and n > max_bytes:
                raise PinError("the file is larger than the pinned size limit")
            h.update(chunk)
    return h.hexdigest(), n


def require_verified(path: Path, pins: dict | None = None, key: str | None = None) -> Path:
    """Return `path` only if it is a regular, non-symlink file whose SHA-256 equals the pin. Any launcher MUST call this first."""
    pin = pinned_asset(pins if pins is not None else load_pins(), key)
    p = Path(path)
    st = os.lstat(p) if p.exists() or p.is_symlink() else None
    if st is None or not stat.S_ISREG(st.st_mode):
        raise PinError("the binary is missing, or is not a regular file (symlinks are refused)")
    digest, _ = sha256_file(p, pin["max_bytes"])
    if digest != pin["sha256"]:
        raise PinError("the binary's SHA-256 does not match the pin: refusing to use it")
    return p


def download(dest_dir: Path, pins: dict | None = None, key: str | None = None, opener=open_url) -> Path:
    """Fetch the pinned asset into `dest_dir`. The bytes are quarantined (no execute bit, a .partial name) until size and SHA-256 match."""
    pin = pinned_asset(pins if pins is not None else load_pins(), key)
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    final, partial = dest_dir / pin["asset"], dest_dir / (pin["asset"] + ".partial")
    h, n = hashlib.sha256(), 0
    try:
        with opener(release_url(pin)) as resp, open(partial, "wb") as out:
            while chunk := resp.read(CHUNK):
                n += len(chunk)
                if n > pin["max_bytes"]:
                    raise PinError("the download exceeds the pinned size limit")
                h.update(chunk)
                out.write(chunk)
        if h.hexdigest() != pin["sha256"]:
            raise PinError("the downloaded file's SHA-256 does not match the pin: it was deleted")
        os.chmod(partial, 0o755)                                         # only now does it become executable
        os.replace(partial, final)
        return final
    finally:
        partial.unlink(missing_ok=True)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("status")
    p = sub.add_parser("download"); p.add_argument("--dest-dir", required=True)
    p = sub.add_parser("verify"); p.add_argument("--binary", required=True)
    a = ap.parse_args(argv)
    try:
        if a.cmd == "status":
            pin = pinned_asset(load_pins())
            print(f"pinned: cloudflared {pin['version']} {pin['asset']} sha256 {pin['sha256'][:12]}...")
        elif a.cmd == "download":
            print(f"verified and saved: {download(Path(a.dest_dir))}")
        elif a.cmd == "verify":
            print(f"verified: {require_verified(Path(a.binary))}")
    except PinError as e:
        print(f"REFUSED: {e}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
