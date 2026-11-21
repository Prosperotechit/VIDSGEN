"""Tests for `scripts/download_models.py --execute` against LOCAL fake Hugging Face + CDN servers.

Nothing here touches the network or real model weights: the "models" are a few hundred KB of
deterministic random bytes, and the manifest is generated per test with their true SHA-256.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import random
import socket
import subprocess
import sys
import threading
from collections import namedtuple
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest
import yaml

from app.core.config import REPO_ROOT

SCRIPT = REPO_ROOT / "scripts" / "download_models.py"
TOKEN = "test-token-not-a-real-secret-123"

_spec = importlib.util.spec_from_file_location("download_models", SCRIPT)
dm = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(dm)

ROLES = [  # role, repo, path, comfyui_dir, size
    ("diffusion_model", "Comfy-Org/Wan-2.2", "split_files/diffusion_models/wan2.2_ti2v_5B_fp16.safetensors", "models/diffusion_models", 200_000),
    ("vae", "Comfy-Org/Wan-2.2", "split_files/vae/wan2.2_vae.safetensors", "models/vae", 60_000),
    ("text_encoder", "Comfy-Org/Wan-2.1", "split_files/text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors", "models/text_encoders", 90_000),
]
REVISIONS = {"Comfy-Org/Wan-2.2": "a" * 40, "Comfy-Org/Wan-2.1": "b" * 40}


class FakeHub:
    """A fake Hugging Face host (302 redirects) plus a fake CDN host (bytes, Range, fault injection)."""

    def __init__(self):
        self.blobs: dict[str, bytes] = {}
        self.hub_requests: list[dict] = []
        self.cdn_requests: list[dict] = []
        self.hub_status: int | None = None      # force an HTTP status from the hub
        self.cut_plan: list[int] = []           # per CDN request: drop the connection after N body bytes
        self.ignore_range = False
        self.override: dict[str, bytes] = {}    # blob id -> different bytes served instead
        hub = self

        class Hub(BaseHTTPRequestHandler):
            def log_message(self, *a): ...

            def do_GET(self):
                parts = self.path.split("/", 5)    # '', org, name, 'resolve', rev, path
                hub.hub_requests.append({"path": self.path, "auth": self.headers.get("Authorization")})
                if hub.hub_status:
                    self.send_response(hub.hub_status); self.send_header("Content-Length", "0"); self.end_headers(); return
                repo, rev, path = f"{parts[1]}/{parts[2]}", parts[4], parts[5] if len(parts) > 5 else ""
                key = f"{repo}|{path.replace('%2F', '/')}"
                if REVISIONS.get(repo) != rev or key not in hub.blobs:
                    self.send_response(404); self.send_header("Content-Length", "0"); self.end_headers(); return
                self.send_response(302)
                self.send_header("Location", f"http://127.0.0.1:{hub.cdn_port}/blob/{hashlib.md5(key.encode()).hexdigest()}?sig=x")
                self.send_header("Content-Length", "0")
                self.end_headers()

        class Cdn(BaseHTTPRequestHandler):
            def log_message(self, *a): ...

            def do_GET(self):
                bid = self.path.split("/blob/")[1].split("?")[0]
                data = hub.override.get(bid) or hub.by_id[bid]
                rng = self.headers.get("Range")
                hub.cdn_requests.append({"id": bid, "range": rng, "auth": self.headers.get("Authorization")})
                start = 0
                if rng and not hub.ignore_range:
                    start = int(rng.split("=")[1].rstrip("-"))
                    if start >= len(data):
                        self.send_response(416); self.send_header("Content-Length", "0"); self.end_headers(); return
                    self.send_response(206)
                    self.send_header("Content-Range", f"bytes {start}-{len(data) - 1}/{len(data)}")
                else:
                    self.send_response(200)
                body = data[start:]
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                if hub.cut_plan:
                    cut = hub.cut_plan.pop(0)
                    self.wfile.write(body[:cut]); self.wfile.flush()
                    self.request.shutdown(socket.SHUT_RDWR)
                    return
                self.wfile.write(body)

        self.by_id: dict[str, bytes] = {}
        self._hub, self._cdn = ThreadingHTTPServer(("127.0.0.1", 0), Hub), ThreadingHTTPServer(("127.0.0.1", 0), Cdn)
        self.cdn_port = self._cdn.server_address[1]
        self.url = f"http://127.0.0.1:{self._hub.server_address[1]}"
        for s in (self._hub, self._cdn):
            threading.Thread(target=s.serve_forever, daemon=True).start()

    def add(self, repo: str, path: str, data: bytes) -> None:
        key = f"{repo}|{path}"
        self.blobs[key] = data
        self.by_id[hashlib.md5(key.encode()).hexdigest()] = data

    def blob_id(self, repo: str, path: str) -> str:
        return hashlib.md5(f"{repo}|{path}".encode()).hexdigest()

    def close(self):
        for s in (self._hub, self._cdn):
            s.shutdown(); s.server_close()


@pytest.fixture
def env(tmp_path, monkeypatch):
    hub = FakeHub()
    rng = random.Random(1234)
    payloads, files = {}, []
    for role, repo, path, cdir, size in ROLES:
        data = rng.randbytes(size)
        payloads[role] = data
        hub.add(repo, path, data)
        files.append({"role": role, "repo": repo, "revision": REVISIONS[repo], "path": path, "comfyui_dir": cdir,
                      "size_bytes": size, "sha256": hashlib.sha256(data).hexdigest()})
    manifest = {"model": "test-model", "files": files}
    mpath = tmp_path / "models.yaml"
    mpath.write_text(yaml.safe_dump(manifest))
    comfy = tmp_path / "ComfyUI"

    monkeypatch.setenv("HF_ENDPOINT", hub.url)
    monkeypatch.setenv("NO_PROXY", "127.0.0.1,localhost")
    monkeypatch.setenv("no_proxy", "127.0.0.1,localhost")
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.setattr(dm, "CHUNK", 16 * 1024)
    monkeypatch.setattr(dm, "RETRY_BASE_DELAY", 0.0)
    monkeypatch.setattr(dm, "PROGRESS_SECONDS", 0.0)

    class E:
        pass
    e = E()
    e.hub, e.files, e.payloads, e.mpath, e.comfy, e.manifest = hub, files, payloads, mpath, comfy, manifest
    e.dest = lambda role: comfy / next(f["comfyui_dir"] for f in files if f["role"] == role) / Path(next(f["path"] for f in files if f["role"] == role)).name
    e.run = lambda *a: dm.main(["--comfyui-dir", str(comfy), "--manifest", str(mpath), *a])
    e.rewrite = lambda m: mpath.write_text(yaml.safe_dump(m))
    yield e
    hub.close()


def part_of(p: Path) -> Path:
    return p.with_name(p.name + ".part")


# ====================================================================== happy path
def test_downloads_verifies_and_installs_all_three(env, capsys):
    assert env.run("--execute") == 0
    out = capsys.readouterr().out
    for role, data in env.payloads.items():
        assert env.dest(role).read_bytes() == data
        assert not part_of(env.dest(role)).exists() and not part_of(env.dest(role)).with_suffix(".part.json").exists()
    assert "ALL 3 MODEL FILES DOWNLOADED AND VERIFIED" in out
    assert out.count("SHA-256 OK") == 3 and out.count("VERIFIED and installed") == 3
    assert "destination:" in out and "MB/s" in out and "ETA" in out        # progress + destination reported
    assert not any(env.comfy.rglob("*.part*"))


def test_downloads_only_the_three_pinned_files(env):
    assert env.run("--execute") == 0
    seen = [r["path"] for r in env.hub.hub_requests]
    assert len(seen) == 3
    assert sorted(p.split("/resolve/")[1].split("/", 1)[0] for p in seen) == ["a" * 40, "a" * 40, "b" * 40]   # pinned commits
    assert not any("/resolve/main/" in p for p in seen)
    assert sorted(f.name for f in env.comfy.rglob("*") if f.is_file()) == sorted(Path(f["path"]).name for f in env.files)


def test_already_verified_files_are_skipped_without_network(env, capsys):
    assert env.run("--execute") == 0
    env.hub.hub_requests.clear(); env.hub.cdn_requests.clear()
    assert env.run("--execute") == 0
    assert env.hub.hub_requests == [] and env.hub.cdn_requests == []
    assert capsys.readouterr().out.count("already present and verified") == 3


def test_zero_byte_placeholder_is_replaced(env):
    dest = env.dest("vae")
    dest.parent.mkdir(parents=True)
    dest.write_bytes(b"")
    assert env.run("--execute") == 0
    assert dest.read_bytes() == env.payloads["vae"]


def test_same_size_but_wrong_content_existing_file_is_replaced(env):
    dest = env.dest("vae")
    dest.parent.mkdir(parents=True)
    dest.write_bytes(b"\x00" * len(env.payloads["vae"]))      # right size, wrong bytes
    assert env.run("--execute") == 0
    assert dest.read_bytes() == env.payloads["vae"]


# ====================================================================== verification failures
def test_sha256_mismatch_deletes_file_and_stops(env, capsys):
    bid = env.hub.blob_id("Comfy-Org/Wan-2.2", ROLES[0][2])
    corrupt = bytearray(env.payloads["diffusion_model"]); corrupt[1000] ^= 0xFF
    env.hub.override[bid] = bytes(corrupt)                     # same size, one flipped byte
    assert env.run("--execute") == 4
    err = capsys.readouterr().err
    assert "SHA-256" in err and "deleted" in err
    assert not env.dest("diffusion_model").exists() and not part_of(env.dest("diffusion_model")).exists()
    assert not env.dest("vae").exists() and not env.dest("text_encoder").exists()   # stopped: nothing after it
    assert len(env.hub.hub_requests) == 1


def test_wrong_size_from_server_deletes_and_stops(env, capsys):
    bid = env.hub.blob_id("Comfy-Org/Wan-2.2", ROLES[0][2])
    env.hub.override[bid] = env.payloads["diffusion_model"][:-100]          # truncated file
    assert env.run("--execute") == 4
    assert "not the file in the manifest" in capsys.readouterr().err
    assert not env.dest("diffusion_model").exists() and not part_of(env.dest("diffusion_model")).exists()
    assert len(env.hub.hub_requests) == 1                      # stopped at the first bad file


def test_oversized_stream_deletes_and_stops(env):
    bid = env.hub.blob_id("Comfy-Org/Wan-2.2", ROLES[1][2])
    env.hub.override[bid] = env.payloads["vae"] + b"extra"
    assert env.run("--execute") == 4
    assert not env.dest("vae").exists() and not part_of(env.dest("vae")).exists()


# ====================================================================== resume
def test_resumes_after_dropped_connection(env, capsys):
    env.hub.cut_plan = [70_000, 70_000]                        # the first two CDN requests die mid-body
    assert env.run("--execute") == 0
    out = capsys.readouterr().out
    assert "connection closed early" in out and "retrying" in out and "resuming at" in out
    ranges = [r["range"] for r in env.hub.cdn_requests if r["range"]]
    assert ranges, "no Range request was sent"                            # real HTTP Range resumption
    starts = [int(x.split("=")[1].rstrip("-")) for x in ranges]
    assert all(0 < st < 200_000 for st in starts) and starts[0] >= 16 * 1024   # resumed mid-file, never from 0
    for role, data in env.payloads.items():
        assert env.dest(role).read_bytes() == data


def test_gives_up_keeps_partial_and_second_run_resumes(env, monkeypatch, capsys):
    monkeypatch.setattr(dm, "MAX_ATTEMPTS", 2)
    env.hub.cut_plan = [50_000] + [0] * 99                     # one burst of progress, then nothing
    assert env.run("--execute") == 3
    assert "STOPPED: gave up after 2 attempts without progress" in capsys.readouterr().err
    part = part_of(env.dest("diffusion_model"))
    assert 0 < part.stat().st_size < len(env.payloads["diffusion_model"]) and not env.dest("diffusion_model").exists()
    env.hub.cut_plan = []                                       # network recovers
    assert env.run("--execute") == 0
    assert env.dest("diffusion_model").read_bytes() == env.payloads["diffusion_model"]


def test_flaky_link_that_keeps_progressing_is_not_given_up_on(env, monkeypatch):
    monkeypatch.setattr(dm, "MAX_ATTEMPTS", 2)
    env.hub.cut_plan = [30_000] * 6                            # six drops in a row, each after real progress
    assert env.run("--execute") == 0
    assert env.dest("diffusion_model").read_bytes() == env.payloads["diffusion_model"]


def test_corrupt_partial_file_is_never_accepted(env, capsys):
    """A leftover .part with the right metadata but corrupt bytes: it is resumed, but the final
    SHA-256 of the complete file catches the corruption, deletes everything and stops."""
    dest = env.dest("vae")
    dest.parent.mkdir(parents=True)
    f = next(x for x in env.files if x["role"] == "vae")
    part_of(dest).write_bytes(b"\xAB" * 20_000)                 # corrupt prefix
    dest.with_name(dest.name + ".part.json").write_text(json.dumps({
        "repo": f["repo"], "revision": f["revision"], "path": f["path"], "size_bytes": f["size_bytes"], "sha256": f["sha256"]}))
    env.mpath.write_text(yaml.safe_dump({**env.manifest, "files": [x for x in env.files if x["role"] != "diffusion_model" or True]}))
    # run only the vae by pre-installing the others
    for role in ("diffusion_model", "text_encoder"):
        d = env.dest(role); d.parent.mkdir(parents=True, exist_ok=True); d.write_bytes(env.payloads[role])
    assert env.run("--execute") == 4
    assert "SHA-256" in capsys.readouterr().err
    assert not dest.exists() and not part_of(dest).exists()
    assert any(r["range"] == "bytes=20000-" for r in env.hub.cdn_requests)    # it really did resume the corrupt prefix


def test_partial_without_matching_metadata_is_discarded(env, capsys):
    dest = env.dest("vae")
    dest.parent.mkdir(parents=True)
    part_of(dest).write_bytes(b"\x01" * 5_000)                  # no .part.json: provenance unknown
    for role in ("diffusion_model", "text_encoder"):
        d = env.dest(role); d.parent.mkdir(parents=True, exist_ok=True); d.write_bytes(env.payloads[role])
    assert env.run("--execute") == 0
    assert "discarding an unverifiable partial download" in capsys.readouterr().out
    assert dest.read_bytes() == env.payloads["vae"]
    assert all(r["range"] is None for r in env.hub.cdn_requests)        # restarted from byte 0


def test_partial_from_another_revision_is_discarded(env, capsys):
    dest = env.dest("vae")
    dest.parent.mkdir(parents=True)
    f = next(x for x in env.files if x["role"] == "vae")
    part_of(dest).write_bytes(env.payloads["vae"][:10_000])
    dest.with_name(dest.name + ".part.json").write_text(json.dumps({
        "repo": f["repo"], "revision": "c" * 40, "path": f["path"], "size_bytes": f["size_bytes"], "sha256": f["sha256"]}))
    for role in ("diffusion_model", "text_encoder"):
        d = env.dest(role); d.parent.mkdir(parents=True, exist_ok=True); d.write_bytes(env.payloads[role])
    assert env.run("--execute") == 0
    assert "discarding an unverifiable partial" in capsys.readouterr().out
    assert dest.read_bytes() == env.payloads["vae"]


def test_server_ignoring_range_restarts_cleanly(env, capsys):
    env.hub.cut_plan, env.hub.ignore_range = [40_000], True
    assert env.run("--execute") == 0
    assert "ignored the Range request" in capsys.readouterr().out
    assert env.dest("diffusion_model").read_bytes() == env.payloads["diffusion_model"]


# ====================================================================== refusals (before any download)
def _refused(env, capsys, fragment):
    assert env.run("--execute") == 2
    assert fragment in capsys.readouterr().err
    assert env.hub.hub_requests == [] and env.hub.cdn_requests == []          # not one network request
    assert not env.comfy.exists()


def test_refuses_without_sha256(env, capsys):
    m = json.loads(json.dumps(env.manifest)); del m["files"][1]["sha256"]
    env.rewrite(m)
    _refused(env, capsys, "authoritative `sha256` is missing or invalid")


@pytest.mark.parametrize("bad", ["", "abc123", "G" * 64, "A" * 64, "0" * 63])
def test_refuses_malformed_sha256(env, capsys, bad):
    m = json.loads(json.dumps(env.manifest)); m["files"][0]["sha256"] = bad
    env.rewrite(m)
    _refused(env, capsys, "sha256")


def test_refuses_without_pinned_revision(env, capsys):
    m = json.loads(json.dumps(env.manifest)); m["files"][0]["revision"] = "main"
    env.rewrite(m)
    _refused(env, capsys, "pinned `revision`")


@pytest.mark.parametrize("path", [
    "split_files/diffusion_models/wan2.2_t2v_high_noise_14B_fp16.safetensors",
    "split_files/diffusion_models/wan2.2_fun_control_5B_bf16.safetensors",
    "split_files/loras/some_lora.safetensors",
    "split_files/vae/wan_2.1_vae.safetensors",
    "../escape/model.safetensors",
])
def test_refuses_unapproved_files(env, capsys, path):
    m = json.loads(json.dumps(env.manifest)); m["files"][1]["path"] = path
    env.rewrite(m)
    assert env.run("--execute") == 2
    assert env.hub.hub_requests == []


def test_refuses_extra_or_missing_roles(env, capsys):
    m = json.loads(json.dumps(env.manifest)); m["files"].append({**m["files"][1], "role": "lora"})
    env.rewrite(m)
    _refused(env, capsys, "exactly the roles")
    m2 = json.loads(json.dumps(env.manifest)); m2["files"].pop()
    env.rewrite(m2)
    assert env.run("--execute") == 2


def test_refuses_when_disk_is_too_small(env, monkeypatch, capsys):
    usage = namedtuple("usage", "total used free")
    monkeypatch.setattr(dm.shutil, "disk_usage", lambda p: usage(10**9, 10**9 - 100_000, 100_000))
    assert env.run("--execute") == 2
    assert "not enough free disk space" in capsys.readouterr().err
    assert env.hub.hub_requests == [] and not any(env.comfy.rglob("*.safetensors*"))


def test_http_404_and_auth_errors_are_clear(env, capsys):
    env.hub.hub_status = 404
    assert env.run("--execute") == 2
    assert "HTTP 404" in capsys.readouterr().err
    env.hub.hub_status = 401
    assert env.run("--execute") == 2
    assert "check HF_TOKEN" in capsys.readouterr().err


# ====================================================================== token handling
def test_token_goes_only_to_the_hub_and_is_never_printed(env, monkeypatch, capsys):
    monkeypatch.setenv("HF_TOKEN", TOKEN)
    assert env.run("--execute") == 0
    captured = capsys.readouterr()
    assert TOKEN not in captured.out and TOKEN not in captured.err
    assert "(not shown)" in captured.out
    assert env.hub.hub_requests and all(r["auth"] == f"Bearer {TOKEN}" for r in env.hub.hub_requests)
    assert env.hub.cdn_requests and all(r["auth"] is None for r in env.hub.cdn_requests)    # never forwarded to the CDN


# ====================================================================== verify-only and dry-run
def test_verify_only_passes_and_makes_no_requests(env, capsys):
    assert env.run("--execute") == 0
    env.hub.hub_requests.clear()
    capsys.readouterr()
    assert env.run("--verify-only") == 0
    out = capsys.readouterr().out
    assert out.count("[PASS]") == 3 and "ALL MODEL FILES VERIFIED" in out
    assert env.hub.hub_requests == []


def test_verify_only_reports_failures_without_deleting(env, capsys):
    assert env.run("--execute") == 0
    corrupt = bytearray(env.payloads["vae"]); corrupt[5] ^= 1
    env.dest("vae").write_bytes(bytes(corrupt))
    env.dest("text_encoder").write_bytes(b"")
    env.dest("diffusion_model").unlink()
    capsys.readouterr()
    assert env.run("--verify-only") == 1
    out = capsys.readouterr().out
    assert out.count("[FAIL]") == 3 and "SHA-256" in out and "does not exist" in out
    assert env.dest("vae").exists() and env.dest("text_encoder").exists()            # nothing was deleted


def test_dry_run_default_never_touches_the_network(env, capsys):
    assert env.run() == 0
    out = capsys.readouterr().out
    assert "DRY RUN: nothing was downloaded" in out
    assert env.hub.hub_requests == [] and not env.comfy.exists()


# ====================================================================== the real manifest and the CLI
def test_real_manifest_is_complete_and_pinned():
    m = yaml.safe_load((REPO_ROOT / "workflows/comfyui/models.yaml").read_text())
    assert dm.validate_for_download(m) == []
    assert sum(f["size_bytes"] for f in m["files"]) == m["total_bytes"] == 18_144_966_705
    assert {(f["repo"], Path(f["path"]).name) for f in m["files"]} == {
        ("Comfy-Org/Wan-2.2", "wan2.2_ti2v_5B_fp16.safetensors"),
        ("Comfy-Org/Wan-2.2", "wan2.2_vae.safetensors"),
        ("Comfy-Org/Wan-2.1", "umt5_xxl_fp8_e4m3fn_scaled.safetensors")}
    assert len({f["sha256"] for f in m["files"]}) == 3
    assert all(f["revision"] != "main" for f in m["files"])


def test_cli_end_to_end_exit_codes(env, tmp_path):
    cmd = [sys.executable, str(SCRIPT), "--comfyui-dir", str(env.comfy), "--manifest", str(env.mpath)]
    sub_env = {"HF_ENDPOINT": env.hub.url, "NO_PROXY": "127.0.0.1", "PATH": "/usr/bin:/bin", "VIDSGEN_DL_RETRY_DELAY": "0"}
    r = subprocess.run([*cmd, "--execute"], capture_output=True, text=True, env=sub_env, timeout=120)
    assert r.returncode == 0, r.stderr
    assert "ALL 3 MODEL FILES DOWNLOADED AND VERIFIED" in r.stdout
    r = subprocess.run([*cmd, "--verify-only"], capture_output=True, text=True, env=sub_env, timeout=120)
    assert r.returncode == 0 and "ALL MODEL FILES VERIFIED" in r.stdout
