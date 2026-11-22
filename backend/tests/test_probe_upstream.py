"""The dummy ComfyUI (scripts/probe_upstream.py): deterministic downloads, Range, routes, WebSocket events, controls, loopback and lifetime."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import subprocess
import sys
import time
import uuid

import pytest
from fastapi.testclient import TestClient

from app.core.config import REPO_ROOT

SCRIPT = REPO_ROOT / "scripts" / "probe_upstream.py"
_spec = importlib.util.spec_from_file_location("probe_upstream", SCRIPT)
pu = importlib.util.module_from_spec(_spec)
sys.modules["probe_upstream"] = pu
_spec.loader.exec_module(pu)

SIZE = 3 * 1024 * 1024 + 12345
NAME = f"job_{uuid.uuid4().hex[:24]}_00001_.mp4"
VIEW = {"filename": NAME, "subfolder": "vidsgen", "type": "output"}


@pytest.fixture
def app():
    return pu.create_app(SIZE, seed=5, event_delay=0.0, steps=6)


@pytest.fixture
def client(app):
    with TestClient(app) as c:
        yield c


def test_the_download_is_a_pure_function_of_size_and_seed():
    a, b = pu.expected_sha256(SIZE, 5), pu.expected_sha256(SIZE, 5)
    assert a == b and a != pu.expected_sha256(SIZE, 6) and a != pu.expected_sha256(SIZE + 1, 5)
    data = b"".join(pu.iter_stream(SIZE, 5))
    assert len(data) == SIZE and hashlib.sha256(data).hexdigest() == a and data[4:8] == b"ftyp"


@pytest.mark.parametrize("start,end", [(0, 0), (0, 1023), (5, 20), (pu.BLOCK - 3, pu.BLOCK + 3), (SIZE - 10, SIZE - 1), (2 * pu.BLOCK, 2 * pu.BLOCK + 99)])
def test_any_range_of_the_stream_matches_the_whole(start, end):
    whole = b"".join(pu.iter_stream(SIZE, 5))
    assert b"".join(pu.iter_stream(SIZE, 5, start, end)) == whole[start:end + 1]


def test_view_streams_the_exact_bytes_with_a_content_length(client):
    r = client.get("/view", params=VIEW)
    assert r.status_code == 200 and r.headers["content-length"] == str(SIZE) and r.headers["accept-ranges"] == "bytes"
    assert hashlib.sha256(r.content).hexdigest() == pu.expected_sha256(SIZE, 5)


def test_view_supports_ranges(client):
    whole = b"".join(pu.iter_stream(SIZE, 5))
    r = client.get("/view", params=VIEW, headers={"Range": "bytes=10-19"})
    assert r.status_code == 206 and r.content == whole[10:20] and r.headers["content-range"] == f"bytes 10-19/{SIZE}"
    r = client.get("/view", params=VIEW, headers={"Range": "bytes=-5"})
    assert r.status_code == 206 and r.content == whole[-5:]
    r = client.get("/view", params=VIEW, headers={"Range": f"bytes={SIZE - 4}-"})
    assert r.status_code == 206 and r.content == whole[-4:]
    assert client.get("/view", params=VIEW, headers={"Range": f"bytes={SIZE + 5}-"}).status_code == 416
    assert client.get("/view", params=VIEW, headers={"Range": "bytes=abc"}).status_code == 416
    assert client.get("/view", params=VIEW, headers={"Range": "bytes=-"}).status_code == 416


@pytest.mark.parametrize("params", [{**VIEW, "type": "input"}, {**VIEW, "subfolder": ""}, {**VIEW, "filename": "x.mp4"}, {**VIEW, "filename": "../../etc/passwd"}])
def test_view_serves_only_generated_looking_names(client, params):
    assert client.get("/view", params=params).status_code == 404


def test_a_prompt_runs_and_emits_progress_events_then_appears_in_history(client):
    pid, cid = str(uuid.uuid4()), "probe-abc"
    graph = {"58": {"inputs": {"filename_prefix": f"vidsgen/{NAME.split('_00001')[0]}"}}}
    with client.websocket_connect(f"/ws?clientId={cid}") as ws:
        assert json.loads(ws.receive_text())["type"] == "status"
        r = client.post("/prompt", json={"prompt": graph, "client_id": cid, "prompt_id": pid})
        assert r.status_code == 200 and r.json()["prompt_id"] == pid
        kinds = []
        while True:
            msg = json.loads(ws.receive_text())
            kinds.append(msg["type"])
            if msg["type"] == "execution_success":
                break
    assert kinds[0] == "execution_start" and kinds.count("progress") == 6
    entry = client.get(f"/history/{pid}").json()[pid]
    assert entry["status"]["status_str"] == "success" and entry["outputs"]["58"]["images"][0]["filename"] == NAME
    assert client.get("/__probe_stats").json()["prompts_received"] == 1


def test_queue_interrupt_delete_and_history_cleanup(client):
    pid = str(uuid.uuid4())
    assert client.get("/queue").json() == {"queue_running": [], "queue_pending": []}
    assert client.post("/queue", json={"delete": [pid]}).status_code == 200
    assert client.post("/interrupt", json={"prompt_id": pid}).status_code == 200
    assert client.post("/history", json={"delete": [pid]}).status_code == 200
    assert client.get(f"/history/{pid}").json() == {} and client.get("/__probe_stats").json()["deleted"] == [pid]


def test_system_stats_deliberately_contains_internals_for_the_proxy_to_strip(client):
    body = client.get("/system_stats").json()
    assert "argv" in body["system"] and "/probe" in json.dumps(body)               # so a test can prove the proxy removes them


def test_controls_can_simulate_an_outage(client):
    assert client.get("/system_stats").status_code == 200
    client.post("/__probe_control", json={"outage": True})
    assert client.get("/system_stats").status_code == 503
    client.post("/__probe_control", json={"outage": False, "unknown": 1})
    assert client.get("/system_stats").status_code == 200


def run_cli(*args, timeout=30):
    return subprocess.run([sys.executable, str(SCRIPT), *args], capture_output=True, text=True, timeout=timeout)


@pytest.mark.parametrize("host", ["0.0.0.0", "192.168.1.5", "example.com", ""])
def test_it_refuses_to_listen_anywhere_but_loopback(host):
    r = run_cli("--host", host, "--port", "1")
    assert r.returncode == 2 and "loopback only" in r.stderr


@pytest.mark.parametrize("args", [["--download-mib", "0"], ["--download-mib", "5000"], ["--max-lifetime-seconds", "-1"]])
def test_it_refuses_nonsensical_sizes_and_lifetimes(args):
    assert run_cli("--port", "1", *args).returncode == 2


def test_it_exits_by_itself_after_its_lifetime():
    from test_inference_proxy import free_port
    t0 = time.monotonic()
    r = run_cli("--port", str(free_port()), "--download-mib", "1", "--max-lifetime-seconds", "2", timeout=30)
    assert r.returncode == 0 and 1.5 <= time.monotonic() - t0 < 25
