"""scripts/cloudflared_pin.py: a future cloudflared binary must be version-pinned and SHA-256-verified before any use. The tool never runs it
and never touches the network in these tests (the downloader is given a fake opener)."""

from __future__ import annotations

import hashlib
import importlib.util
import io
import json
import os
import stat
import subprocess
import sys

import pytest

from app.core.config import REPO_ROOT

SCRIPT = REPO_ROOT / "scripts" / "cloudflared_pin.py"
_spec = importlib.util.spec_from_file_location("cloudflared_pin", SCRIPT)
cp = importlib.util.module_from_spec(_spec)
sys.modules["cloudflared_pin"] = cp
_spec.loader.exec_module(cp)

PAYLOAD = b"\x7fELF fake cloudflared binary " * 1000
SHA = hashlib.sha256(PAYLOAD).hexdigest()
KEY = "linux-amd64"


def pins(version="2026.9.3", sha=SHA, asset="cloudflared-linux-amd64", max_bytes=1024 * 1024):
    return {"version": version, "assets": {KEY: {"asset": asset, "sha256": sha, "max_bytes": max_bytes}}}


class Resp(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.close()
        return False


def opener_for(data: bytes, seen: list | None = None):
    def open_(url):
        if seen is not None:
            seen.append(url)
        return Resp(data)
    return open_


# ====================================================================== the repository's own pin file
def test_the_committed_pin_file_is_deliberately_empty_so_nothing_can_be_used_yet():
    data = cp.load_pins()
    assert data["version"] is None and all(a["sha256"] is None for a in data["assets"].values())
    for key in data["assets"]:
        with pytest.raises(cp.PinError, match="no cloudflared version is pinned"):
            cp.pinned_asset(data, key)


def test_the_cli_refuses_everything_while_unpinned(tmp_path):
    for args in (["status"], ["download", "--dest-dir", str(tmp_path / "d")], ["verify", "--binary", str(tmp_path / "x")]):
        r = subprocess.run([sys.executable, str(SCRIPT), *args], capture_output=True, text=True, timeout=30)
        assert r.returncode == 2 and "REFUSED" in r.stderr and "Traceback" not in r.stderr
    assert not (tmp_path / "d").exists() or not list((tmp_path / "d").iterdir())


def test_the_pin_file_structure_is_valid_json_with_the_expected_assets():
    data = json.loads((REPO_ROOT / "scripts" / "cloudflared_pins.json").read_text())
    assert set(data) == {"_comment", "version", "assets"} and "never pin a hash" in data["_comment"].lower()
    for entry in data["assets"].values():
        assert set(entry) == {"asset", "sha256", "max_bytes"} and cp.ASSET_RE.match(entry["asset"])


# ====================================================================== pin validation
@pytest.mark.parametrize("bad", [
    dict(version=None), dict(version="latest"), dict(version="2026.9"), dict(version="v2026.9.3"), dict(version="2026.9.3; rm -rf /"),
    dict(sha=None), dict(sha=SHA.upper()), dict(sha=SHA[:-1]), dict(sha="g" * 64), dict(sha=""),
    dict(asset="../cloudflared"), dict(asset="cloudflared/../../x"), dict(asset="other-binary"), dict(asset=None),
    dict(max_bytes=0), dict(max_bytes=10 ** 10), dict(max_bytes="big"),
])
def test_incomplete_or_malformed_pins_are_refused(bad):
    with pytest.raises(cp.PinError):
        cp.pinned_asset(pins(**bad), KEY)


def test_a_complete_pin_is_accepted_and_builds_the_exact_release_url():
    pin = cp.pinned_asset(pins(), KEY)
    assert pin["sha256"] == SHA
    assert cp.release_url(pin) == "https://github.com/cloudflare/cloudflared/releases/download/2026.9.3/cloudflared-linux-amd64"


def test_an_unknown_platform_has_no_pin():
    with pytest.raises(cp.PinError, match="no pin for platform"):
        cp.pinned_asset(pins(), "plan9-mips")


# ====================================================================== download URL allow-list
@pytest.mark.parametrize("url", ["https://github.com/cloudflare/cloudflared/releases/download/2026.9.3/cloudflared-linux-amd64",
                                 "https://objects.githubusercontent.com/x", "https://release-assets.githubusercontent.com/x"])
def test_only_allow_listed_https_hosts_may_be_fetched(url):
    cp.check_download_url(url)


@pytest.mark.parametrize("url", ["http://github.com/x", "https://evil.example.com/cloudflared", "https://github.com.evil.com/x",
                                 "https://user:pw@github.com/x", "ftp://github.com/x", "https://127.0.0.1/x", "file:///etc/passwd", ""])
def test_everything_else_is_refused(url):
    with pytest.raises(cp.PinError):
        cp.check_download_url(url)


def test_a_redirect_off_the_allow_list_is_refused():
    handler = cp._CheckedRedirects()
    import urllib.request
    req = urllib.request.Request("https://github.com/a")
    with pytest.raises(cp.PinError):
        handler.redirect_request(req, None, 302, "Found", {}, "https://evil.example.com/payload")


# ====================================================================== download + verification
def test_a_matching_download_is_installed_executable_and_verified(tmp_path):
    seen = []
    path = cp.download(tmp_path / "bin", pins(), KEY, opener=opener_for(PAYLOAD, seen))
    assert path.read_bytes() == PAYLOAD and path.name == "cloudflared-linux-amd64" and os.access(path, os.X_OK)
    assert seen == ["https://github.com/cloudflare/cloudflared/releases/download/2026.9.3/cloudflared-linux-amd64"]
    assert not list((tmp_path / "bin").glob("*.partial"))
    assert cp.require_verified(path, pins(), KEY) == path


def test_a_hash_mismatch_deletes_the_download_and_installs_nothing(tmp_path):
    with pytest.raises(cp.PinError, match="does not match the pin"):
        cp.download(tmp_path / "bin", pins(), KEY, opener=opener_for(PAYLOAD + b"tampered"))
    assert list((tmp_path / "bin").iterdir()) == []


def test_the_file_is_not_executable_until_it_has_been_verified(tmp_path):
    observed = {}

    class Spy(Resp):
        def read(self, n=-1):
            part = super().read(n)
            if not part:                                                          # the stream is finished: look at what is on disk now
                partial = tmp_path / "bin" / "cloudflared-linux-amd64.partial"
                observed["partial_mode"] = stat.S_IMODE(os.stat(partial).st_mode) if partial.exists() else None
                observed["final_exists"] = (tmp_path / "bin" / "cloudflared-linux-amd64").exists()
            return part
    cp.download(tmp_path / "bin", pins(), KEY, opener=lambda url: Spy(PAYLOAD))
    assert observed["final_exists"] is False and observed["partial_mode"] is not None and not observed["partial_mode"] & 0o111


def test_an_oversized_download_is_stopped_and_deleted(tmp_path):
    with pytest.raises(cp.PinError, match="size limit"):
        cp.download(tmp_path / "bin", pins(max_bytes=len(PAYLOAD) - 1), KEY, opener=opener_for(PAYLOAD))
    assert list((tmp_path / "bin").iterdir()) == []


def test_downloading_with_an_unpinned_file_never_opens_a_connection(tmp_path):
    called = []
    with pytest.raises(cp.PinError):
        cp.download(tmp_path / "bin", pins(sha=None), KEY, opener=lambda url: called.append(url))
    assert called == []


def test_a_failed_download_never_replaces_a_previously_verified_binary(tmp_path):
    good = cp.download(tmp_path / "bin", pins(), KEY, opener=opener_for(PAYLOAD))
    with pytest.raises(cp.PinError):
        cp.download(tmp_path / "bin", pins(), KEY, opener=opener_for(b"evil"))
    assert good.read_bytes() == PAYLOAD


# ====================================================================== require_verified: the gate any launcher must pass
def test_require_verified_accepts_only_the_exact_pinned_bytes(tmp_path):
    f = tmp_path / "cloudflared"
    f.write_bytes(PAYLOAD)
    assert cp.require_verified(f, pins(), KEY) == f
    f.write_bytes(PAYLOAD + b"x")
    with pytest.raises(cp.PinError, match="does not match"):
        cp.require_verified(f, pins(), KEY)


def test_require_verified_refuses_missing_files_symlinks_and_directories(tmp_path):
    real = tmp_path / "real"
    real.write_bytes(PAYLOAD)
    link = tmp_path / "link"
    link.symlink_to(real)
    for target in (tmp_path / "missing", link, tmp_path):
        with pytest.raises(cp.PinError, match="not a regular file|missing"):
            cp.require_verified(target, pins(), KEY)


def test_require_verified_refuses_when_nothing_is_pinned(tmp_path):
    f = tmp_path / "cloudflared"
    f.write_bytes(PAYLOAD)
    with pytest.raises(cp.PinError):
        cp.require_verified(f, pins(version=None), KEY)
    with pytest.raises(cp.PinError):
        cp.require_verified(f, pins(sha=None), KEY)


def test_require_verified_stops_reading_a_file_larger_than_the_limit(tmp_path):
    f = tmp_path / "cloudflared"
    f.write_bytes(PAYLOAD)
    with pytest.raises(cp.PinError, match="larger than the pinned size limit"):
        cp.require_verified(f, pins(max_bytes=10), KEY)


def test_the_tool_never_executes_the_binary():
    src = SCRIPT.read_text()
    for forbidden in ("subprocess", "os.system", "os.exec", "Popen", "check_call", "check_output"):
        assert forbidden not in src, forbidden


def test_platform_key_is_a_known_shape():
    assert cp.platform_key().count("-") == 1
