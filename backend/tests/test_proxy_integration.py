"""Phase 6C Stage 3: the real FastAPI backend, through the authenticated inference proxy, to a fake ComfyUI. Everything on 127.0.0.1.

Covers: endpoint hardening (HTTPS + token for anything that is not this machine), the complete job lifecycle through the proxy
(HTTP, live WebSocket progress, streamed download, Range, preview, cancellation, health), failure modes (bad token, unreachable
proxy, interrupted connections, uncertain submissions), duplicate protection, no automatic resubmission, and clean shutdown.
No tunnel, no external network, no GPU.
"""

from __future__ import annotations

import asyncio
import json
import re
import signal
import socket
import subprocess
import sys
import threading
import time
import uuid

import httpx
import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

import test_wan22_gpu_test  # noqa: F401  (registers scripts/wan22_gpu_test.py: used for the /proc listening-address check)
from app.core.config import REPO_ROOT, Settings, check_comfyui_endpoint, is_loopback_host
from app.main import create_app
from app.providers import build_provider
from app.providers.comfyui import ComfyUIProvider
from app.workflows import WorkflowTemplates
from conftest import make_settings, submit, wait_for
from test_inference_proxy import (  # noqa: F401
    AUTH, TOKEN, WORKFLOWS, ProxyServer, e2e_client, fake, free_port, make_app, px, video_bytes,
)

gt = sys.modules["wan22_gpu_test"]
SCRIPT = REPO_ROOT / "scripts" / "inference_proxy.py"
GOOD_TOKEN = "t" * 40
PROVEN = dict(resolution="832x480", num_frames=17, steps=20, fps=24, guidance_scale=5.0)


def prompts(fake) -> list[dict]:
    return [r for r in fake.requests if r["path"] == "/prompt" and r["method"] == "POST"]


def provider_of(client):
    return client.app.state.manager.provider


async def wait_until(pred, timeout=5.0):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if pred():
            return True
        await asyncio.sleep(0.05)
    return False


def wait_sync(pred, timeout=30.0) -> bool:
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if pred():
            return True
        time.sleep(0.05)
    return False


# ====================================================================== endpoint hardening: HTTPS + token unless it is this machine
@pytest.mark.parametrize("host,loop", [("127.0.0.1", True), ("127.0.0.53", True), ("127.255.255.254", True), ("localhost", True),
                                       ("LOCALHOST", True), ("localhost.", True), ("::1", True), ("[::1]", True),
                                       ("0.0.0.0", False), ("192.168.1.5", False), ("10.0.0.1", False), ("example.com", False),
                                       ("localhost.example.com", False), ("127.0.0.1.evil.com", False), ("", False), (None, False)])
def test_loopback_detection(host, loop):
    assert is_loopback_host(host) is loop


@pytest.mark.parametrize("url", ["http://127.0.0.1:8189", "http://localhost:8189", "http://[::1]:8189", "http://127.0.0.2:9000/base",
                                 "https://127.0.0.1:8189"])
def test_local_endpoints_may_use_http_without_a_token(url):
    check_comfyui_endpoint(url, "")


@pytest.mark.parametrize("url", ["https://gpu.example.com", "https://abc-def.trycloudflare.com/", "https://203.0.113.9:8443", "https://gpu.example.com/prefix"])
def test_remote_https_with_a_strong_token_is_accepted(url):
    check_comfyui_endpoint(url, GOOD_TOKEN)


@pytest.mark.parametrize("url,token,fragment", [
    ("http://gpu.example.com", GOOD_TOKEN, "https://"), ("http://203.0.113.9:8188", GOOD_TOKEN, "https://"), ("http://0.0.0.0:8189", GOOD_TOKEN, "https://"),
    ("http://127.0.0.1.evil.com", GOOD_TOKEN, "https://"), ("http://localhost.example.com", GOOD_TOKEN, "https://"),
    ("https://gpu.example.com", "", "COMFYUI_API_TOKEN is required"), ("https://gpu.example.com", "x" * 31, "at least 32"),
    ("ftp://gpu.example.com", GOOD_TOKEN, "http:// or https://"), ("gpu.example.com:8188", GOOD_TOKEN, "http:// or https://"),
    ("https://", GOOD_TOKEN, "with a host"),
    ("https://user:pw@gpu.example.com", GOOD_TOKEN, "must not contain credentials"), ("https://user@gpu.example.com", GOOD_TOKEN, "credentials"),
    ("http://tok@127.0.0.1:8189", "", "credentials"),
    ("https://gpu.example.com/?token=abc", GOOD_TOKEN, "query string"), ("https://gpu.example.com/#frag", GOOD_TOKEN, "query string or fragment"),
])
def test_unsafe_endpoints_are_refused(url, token, fragment):
    with pytest.raises(ValueError, match=re.escape(fragment)) as e:
        check_comfyui_endpoint(url, token)
    for secret in ("pw", "tok@", "abc"):                                     # the message never repeats what it was given
        assert secret not in str(e.value).replace("COMFYUI_API_TOKEN", "").replace("must not contain credentials", "")


def settings(**kw):
    return Settings(_env_file=None, inference_provider="comfyui", **kw)


def test_settings_enforce_https_and_a_token_for_remote_endpoints():
    with pytest.raises(ValidationError, match="https://"):
        settings(comfyui_base_url="http://gpu.example.com", comfyui_api_token=GOOD_TOKEN)
    with pytest.raises(ValidationError, match="COMFYUI_API_TOKEN is required"):
        settings(comfyui_base_url="https://gpu.example.com")
    with pytest.raises(ValidationError, match="at least 32"):
        settings(comfyui_base_url="https://gpu.example.com", comfyui_api_token="short")
    assert settings(comfyui_base_url="https://gpu.example.com", comfyui_api_token=GOOD_TOKEN).comfyui_base_url == "https://gpu.example.com"


def test_localhost_development_behaviour_is_preserved():
    for url in ("http://127.0.0.1:8188", "http://localhost:8188", "http://[::1]:8188"):
        assert settings(comfyui_base_url=url).comfyui_base_url == url                      # no token, no https needed
    assert settings(comfyui_base_url="http://127.0.0.1:8188/", comfyui_api_token="dev").comfyui_base_url == "http://127.0.0.1:8188"


def test_the_mock_provider_is_unaffected_by_the_endpoint_rules():
    s = Settings(_env_file=None, inference_provider="mock", comfyui_base_url="http://anything.example.com")
    assert s.inference_provider == "mock"
    assert Settings(_env_file=None).comfyui_base_url == "" and Settings(_env_file=None).inference_provider == "mock"


def test_validation_errors_never_print_the_url_or_token_they_were_given():
    with pytest.raises(ValidationError) as e:
        settings(comfyui_base_url="https://user:SuperSecretPw@gpu.example.com", comfyui_api_token="TokenValueXYZ")
    text = str(e.value)
    assert "SuperSecretPw" not in text and "TokenValueXYZ" not in text and "gpu.example.com" not in text
    with pytest.raises(ValidationError) as e:
        settings(comfyui_base_url="https://gpu.example.com", comfyui_api_token="TokenValueXYZ")
    assert "TokenValueXYZ" not in str(e.value) and "gpu.example.com" not in str(e.value)
    with pytest.raises(ValidationError) as e:                                              # a field-level error also hides its input
        settings(comfyui_base_url="https://gpu.example.com", comfyui_api_token=987654321)
    assert "987654321" not in str(e.value)


def test_the_provider_itself_refuses_an_unsafe_endpoint_even_if_settings_are_bypassed():
    templates = WorkflowTemplates(WORKFLOWS)
    with pytest.raises(ValueError, match="https://"):
        ComfyUIProvider(base_url="http://gpu.example.com", workflows=templates, api_token=GOOD_TOKEN)
    with pytest.raises(ValueError, match="at least 32"):
        ComfyUIProvider(base_url="https://gpu.example.com", workflows=templates, api_token="x")
    for kwargs in (dict(base_url="http://127.0.0.1:8189"), dict(base_url="https://gpu.example.com", api_token=GOOD_TOKEN)):
        p = ComfyUIProvider(workflows=templates, **kwargs)
        asyncio.run(p.aclose())


def test_build_provider_applies_the_same_rules(tmp_path):
    s = make_settings(tmp_path, inference_provider="comfyui", comfyui_base_url="https://gpu.example.com", comfyui_api_token=GOOD_TOKEN)
    p = build_provider(s)
    assert p.name == "comfyui" and p._headers == {"Authorization": f"Bearer {GOOD_TOKEN}"}
    asyncio.run(p.aclose())


def test_a_429_from_the_proxy_is_reported_as_busy_and_never_resubmitted(make_client, fake):
    """No GPU-work slot is free: the proxy refuses before forwarding anything, the job fails clearly, nothing is retried or reconciled."""
    app = make_app(fake.url, px.Policy(max_heavy=0))
    with ProxyServer(app) as p:
        c = e2e_client(make_client, p)
        job = wait_for(c, submit(c, **PROVEN).json()["id"], timeout=30)
        time.sleep(1.5)
    assert job["status"] == "failed" and job["error"] == {"code": "comfyui_busy", "message": "The GPU server is busy; try again shortly."}
    assert [r for r in fake.requests if r["method"] == "POST"] == []                      # nothing forwarded (only the read-only existence check ran)
    assert app.state.registry.entries == {}                                               # and nothing remembered


def test_a_429_while_tracking_a_job_is_retried_within_the_grace_period(make_client, fake):
    fake.paused = True
    with ProxyServer(make_app(fake.url)) as p:
        c = e2e_client(make_client, p, comfyui_connection_grace_seconds=20)
        job_id = submit(c).json()["id"]
        wait_for(c, job_id, statuses={"processing"})
        fake.wait_pending()
        fake.outage, fake.outage_status = True, 429
        time.sleep(1.2)
        assert c.get(f"/api/jobs/{job_id}").json()["status"] == "processing"
        fake.outage = False
        fake.resume()
        assert wait_for(c, job_id, timeout=30)["status"] == "completed"


# ====================================================================== the complete lifecycle through the proxy
def test_full_lifecycle_through_the_authenticated_proxy(make_client, fake, video_bytes):
    fake.step_delay = 0.05
    app = make_app(fake.url)
    with ProxyServer(app) as p:
        c = e2e_client(make_client, p, comfyui_poll_interval=1.0)
        # --- health (HTTP): every probe route is allowlisted, nothing is claimed beyond what is known
        before = c.get("/api/health").json()
        assert before["provider"]["ok"] is True and before["status"] == "ok"
        names = {k["name"]: k["status"] for k in before["provider"]["checks"]}
        assert names["comfyui_reachable"] == "ok" and names["inference"] == "unverified"
        # --- submission and progress (live WebSocket through the proxy)
        job_id = submit(c, **PROVEN).json()["id"]
        seen, ws_connected_during = [], False
        deadline = time.monotonic() + 40
        while time.monotonic() < deadline:
            j = c.get(f"/api/jobs/{job_id}").json()
            seen.append(j["progress"])
            ws_connected_during |= provider_of(c)._ws_connected.is_set()
            if j["status"] in ("completed", "failed", "cancelled"):
                break
            time.sleep(0.05)
        assert j["status"] == "completed", j["error"]
        assert seen == sorted(seen) and len([x for x in seen if 5 < x < 90]) >= 3
        assert ws_connected_during and app.state.state["stats"]["ws_events"] >= 20           # the events really came over the WebSocket
        # --- retrieval: download, Range, inline preview
        dl = c.get(f"/api/jobs/{job_id}/download")
        assert dl.status_code == 200 and dl.content == video_bytes and dl.headers["content-type"] == "video/mp4"
        assert dl.headers["content-disposition"].startswith("attachment")
        rng = c.get(f"/api/jobs/{job_id}/download", headers={"Range": "bytes=0-1023"})
        assert rng.status_code == 206 and rng.content == video_bytes[:1024] and rng.headers["content-range"].startswith("bytes 0-1023/")
        inline = c.get(f"/api/jobs/{job_id}/download?inline=true")
        assert inline.status_code == 200 and inline.headers["content-disposition"].startswith("inline") and inline.content == video_bytes
        assert j["result"]["size_bytes"] == len(video_bytes)
        # --- health after a real generation
        after = {k["name"]: k["status"] for k in c.get("/api/health").json()["provider"]["checks"]}
        assert after["inference"] == "ok"
    # --- what ComfyUI actually saw
    assert all(r["auth"] is None for r in fake.requests) and fake.ws_auth_headers and all(h is None for h in fake.ws_auth_headers)
    assert len(prompts(fake)) == 1
    forbidden = [r for r in fake.requests if r["path"] in ("/free", "/upload/image", "/upload/mask", "/history") and r["method"] != "POST"]
    assert forbidden == []
    view = [r for r in fake.requests if r["path"] == "/view"]
    assert view and all(r["query"].get("subfolder") == "vidsgen" and r["query"].get("type") == "output" for r in view)


def test_the_proxy_accepts_the_backends_actual_graph_and_forwards_it_unchanged(make_client, fake):
    with ProxyServer(make_app(fake.url)) as p:
        c = e2e_client(make_client, p)
        job_id = submit(c, **PROVEN).json()["id"]
        assert wait_for(c, job_id)["status"] == "completed"
    sent = prompts(fake)[0]["body"]
    assert set(sent) == {"prompt", "client_id", "prompt_id"} and sent["client_id"].startswith("vidsgen-") and px.UUID_RE.match(sent["prompt_id"])
    assert sent["prompt"]["58"]["inputs"]["filename_prefix"] == f"vidsgen/{job_id}" and px.PREFIX_RE.match(sent["prompt"]["58"]["inputs"]["filename_prefix"])
    validator = px.GraphValidator(WORKFLOWS, px.Policy())
    validator.validate(sent["prompt"])                                                    # the proxy's validator accepts it as is


@pytest.mark.parametrize("over", [dict(steps=1), dict(steps=8, fps=8), dict(steps=20, fps=30, guidance_scale=15.0), dict(guidance_scale=1.0),
                                  dict(steps=13, fps=12, guidance_scale=7.5, seed=0), dict(seed=4294967295)])
def test_every_value_inside_the_envelope_works_end_to_end(make_client, fake, over):
    with ProxyServer(make_app(fake.url)) as p:
        c = e2e_client(make_client, p)
        job = wait_for(c, submit(c, **{**PROVEN, **over}).json()["id"])
    assert job["status"] == "completed", job["error"]


@pytest.mark.parametrize("over,reason", [(dict(resolution="1280x704"), "resolution 1280x704"), (dict(num_frames=49), "num_frames"),
                                         (dict(steps=30), "steps")])
def test_a_backend_without_limits_is_still_stopped_by_the_proxy(make_client, fake, over, reason):
    with ProxyServer(make_app(fake.url)) as p:
        c = e2e_client(make_client, p, generation_profile="unrestricted")
        job = wait_for(c, submit(c, **{**PROVEN, **over}).json()["id"])
    assert job["status"] == "failed" and job["error"]["code"] == "comfyui_rejected" and reason in job["error"]["message"]
    assert prompts(fake) == []


# ====================================================================== cancellation
def test_cancelling_a_running_job_sends_a_targeted_interrupt_through_the_proxy(make_client, fake):
    fake.default_scenario = "hang"
    app = make_app(fake.url)
    with ProxyServer(app) as p:
        c = e2e_client(make_client, p)
        job_id = submit(c, **PROVEN).json()["id"]
        wait_for(c, job_id, statuses={"processing"})
        fake.wait_until(lambda: fake._running is not None, what="a prompt to start running")
        assert c.post(f"/api/jobs/{job_id}/cancel").json()["status"] == "cancelled"
        fake.wait_interrupt()
        pid = prompts(fake)[0]["body"]["prompt_id"]
        assert fake.interrupts == [pid]                                                  # targeted, never a global interrupt
        assert app.state.registry.entries[pid].state == "cancelled"                       # tombstoned at the proxy
        assert wait_for(c, job_id)["status"] == "cancelled"
        replay = httpx.post(f"{p.url}/prompt", json=prompts(fake)[0]["body"], headers=AUTH)
        assert replay.status_code == 409                                                  # a cancelled prompt cannot be re-queued


def test_cancelling_a_queued_job_removes_it_from_the_comfyui_queue(make_client, fake):
    fake.paused = True
    app = make_app(fake.url)
    with ProxyServer(app) as p:
        c = e2e_client(make_client, p)
        job_id = submit(c, **PROVEN).json()["id"]
        wait_for(c, job_id, statuses={"processing"})
        fake.wait_pending()
        pid = prompts(fake)[0]["body"]["prompt_id"]
        assert c.post(f"/api/jobs/{job_id}/cancel").json()["status"] == "cancelled"
        fake.wait_until(lambda: pid in fake.deleted, what="the queue deletion")
        assert fake.interrupts == [] and [r for r in fake.requests if r["method"] == "POST" and r["path"] == "/queue"][0]["body"] == {"delete": [pid]}
        assert app.state.registry.entries[pid].state == "cancelled"


# ====================================================================== invalid tokens and unreachable proxy
def test_a_wrong_token_fails_the_job_and_health_clearly_and_nothing_reaches_comfyui(make_client, fake):
    bad = "wrong-" + uuid.uuid4().hex + uuid.uuid4().hex
    app = make_app(fake.url)
    with ProxyServer(app) as p:
        c = e2e_client(make_client, p, token=bad)
        health = c.get("/api/health").json()
        job = wait_for(c, submit(c, **PROVEN).json()["id"])
    assert health["status"] == "degraded" and health["provider"]["ok"] is False
    assert "token" in health["provider"]["detail"].lower()
    assert job["status"] == "failed" and job["error"]["code"] == "comfyui_auth_failed"
    assert fake.requests == [] and fake.ws_auth_headers == [] and app.state.state["stats"]["ws_events"] == 0
    for text in (json.dumps(health), json.dumps(job)):
        assert bad not in text and TOKEN not in text and str(p.port) not in text


def test_no_token_at_all_is_rejected_the_same_way(make_client, fake):
    with ProxyServer(make_app(fake.url)) as p:
        c = e2e_client(make_client, p, token="")
        job = wait_for(c, submit(c, **PROVEN).json()["id"])
    assert job["error"]["code"] == "comfyui_auth_failed" and fake.requests == []


def test_an_unreachable_proxy_fails_fast_with_a_clear_error_and_no_resubmission(make_client, fake):
    port = free_port()                                                                    # nothing listens here
    c = make_client(inference_provider="comfyui", comfyui_base_url=f"http://127.0.0.1:{port}", comfyui_api_token=TOKEN,
                    comfyui_poll_interval=0.2, comfyui_timeout=3, generation_profile="t4_verified")
    health = c.get("/api/health").json()
    t0 = time.monotonic()
    job = wait_for(c, submit(c, **PROVEN).json()["id"], timeout=30)
    assert health["status"] == "degraded" and health["provider"]["ok"] is False
    assert job["error"] == {"code": "comfyui_unreachable", "message": "The GPU server (ComfyUI) is unreachable."} and time.monotonic() - t0 < 10
    assert str(port) not in json.dumps(job) + json.dumps(health) and fake.requests == []


# ====================================================================== interrupted connections
def test_a_short_proxy_restart_during_a_job_is_survived(make_client, fake, video_bytes):
    """The proxy disappears for a moment mid-job (a tunnel/proxy blip) and comes back on the same port: the job completes."""
    fake.paused = True
    port = free_port()
    c = make_client(inference_provider="comfyui", comfyui_base_url=f"http://127.0.0.1:{port}", comfyui_api_token=TOKEN, comfyui_poll_interval=0.2,
                    comfyui_timeout=3, generation_profile="t4_verified", comfyui_connection_grace_seconds=30)
    with ProxyServer(make_app(fake.url), port):
        job_id = submit(c, **PROVEN).json()["id"]
        wait_for(c, job_id, statuses={"processing"})
        fake.wait_pending()
    time.sleep(1.5)                                                                       # proxy down: polls fail, the job waits
    assert c.get(f"/api/jobs/{job_id}").json()["status"] == "processing"
    with ProxyServer(make_app(fake.url), port):                                           # a fresh proxy: its idempotency memory is empty
        fake.resume()
        job = wait_for(c, job_id, timeout=40)
        assert job["status"] == "completed", job["error"]
        assert c.get(f"/api/jobs/{job_id}/download").content == video_bytes
    assert len(prompts(fake)) == 1


def test_an_outage_longer_than_the_grace_period_fails_the_job_without_resubmitting(make_client, fake):
    fake.paused = True
    with ProxyServer(make_app(fake.url)) as p:
        c = e2e_client(make_client, p, comfyui_connection_grace_seconds=2)
        job_id = submit(c, **PROVEN).json()["id"]
        wait_for(c, job_id, statuses={"processing"})
        fake.wait_pending()
        fake.outage = True                                                                # the proxy answers 503: ComfyUI is unreachable behind it
        job = wait_for(c, job_id, timeout=30)
        fake.outage = False
    assert job["status"] == "failed" and job["error"]["code"] == "comfyui_connection_lost" and "Please try again" in job["error"]["message"]
    assert len(prompts(fake)) == 1


def test_a_dropped_websocket_mid_job_falls_back_and_reconnects(make_client, fake, video_bytes):
    fake.default_scenario = "ws_drop"
    fake.step_delay = 0.05
    app = make_app(fake.url)
    with ProxyServer(app) as p:
        c = e2e_client(make_client, p, comfyui_poll_interval=0.3)
        job = wait_for(c, submit(c, **PROVEN).json()["id"], timeout=40)
        assert job["status"] == "completed", job["error"]
        assert c.get(f"/api/jobs/{job['id']}/download").content == video_bytes
        assert wait_sync(lambda: len(fake.ws_auth_headers) >= 2, 12)                       # the provider reconnected through the proxy
    assert len(prompts(fake)) == 1


# ====================================================================== uncertain submissions, duplicates, no automatic resubmission
def test_a_lost_response_is_reconciled_through_the_proxy_and_the_job_completes(make_client, fake):
    fake.prompt_mode = "queue_then_hang"                                                  # ComfyUI queued it; the answer never arrives
    app = make_app(fake.url, px.Policy(read_timeout=1.0))
    with ProxyServer(app) as p:
        c = e2e_client(make_client, p)
        job = wait_for(c, submit(c, **PROVEN).json()["id"], timeout=40)
    assert job["status"] == "completed", job["error"]
    assert len(prompts(fake)) == 1                                                        # sent exactly once, never resubmitted
    assert [r["method"] for r in fake.requests if r["path"] == "/prompt"] == ["POST"]


def test_a_5xx_after_queueing_is_reconciled_too(make_client, fake):
    fake.prompt_mode = "queue_then_500"
    with ProxyServer(make_app(fake.url)) as p:
        c = e2e_client(make_client, p)
        job = wait_for(c, submit(c, **PROVEN).json()["id"], timeout=40)
    assert job["status"] == "completed" and len(prompts(fake)) == 1


def test_an_uncertain_submission_that_never_reached_comfyui_fails_and_is_never_resubmitted(make_client, fake):
    fake.prompt_mode = "hang"                                                             # the request is lost for good
    with ProxyServer(make_app(fake.url, px.Policy(read_timeout=0.8))) as p:
        c = e2e_client(make_client, p, comfyui_connection_grace_seconds=3)
        job = wait_for(c, submit(c, **PROVEN).json()["id"], timeout=40)
        time.sleep(4)                                                                      # long enough for any (forbidden) retry to show up
    assert job["status"] == "failed" and job["error"]["code"] == "comfyui_server_error"
    assert len(prompts(fake)) == 1                                                        # exactly one POST /prompt, ever
    lookups = [r for r in fake.requests if r["path"].startswith("/history/") or r["path"] == "/queue"]
    assert lookups and all(r["method"] == "GET" for r in lookups)                         # reconciliation was read-only


def test_replaying_the_backends_own_submission_through_the_proxy_is_idempotent(make_client, fake):
    with ProxyServer(make_app(fake.url)) as p:
        c = e2e_client(make_client, p)
        assert wait_for(c, submit(c, **PROVEN).json()["id"])["status"] == "completed"
        body = prompts(fake)[0]["body"]
        again = httpx.post(f"{p.url}/prompt", json=body, headers=AUTH)
        assert again.status_code == 200 and again.headers["x-idempotent-replay"] == "1" and again.json()["prompt_id"] == body["prompt_id"]
        changed = json.loads(json.dumps(body))
        changed["prompt"]["3"]["inputs"]["seed"] += 1
        conflict = httpx.post(f"{p.url}/prompt", json=changed, headers=AUTH)
        assert conflict.status_code == 409 and "different request" in conflict.json()["error"]["message"]
    assert len(prompts(fake)) == 1


def test_two_jobs_never_share_or_reuse_a_prompt_id(make_client, fake):
    with ProxyServer(make_app(fake.url)) as p:
        c = e2e_client(make_client, p)
        ids = [submit(c, **PROVEN).json()["id"] for _ in range(3)]
        assert [wait_for(c, i)["status"] for i in ids] == ["completed"] * 3
    sent = [r["body"]["prompt_id"] for r in prompts(fake)]
    assert len(sent) == 3 and len(set(sent)) == 3


# ====================================================================== loopback binding and clean shutdown
def addrs(port):
    return gt.listening_addresses(port)


def refuses(port) -> bool:
    with socket.socket() as s:
        s.settimeout(1)
        return s.connect_ex(("127.0.0.1", port)) != 0


def test_everything_is_bound_to_loopback_only(make_client, fake):
    with ProxyServer(make_app(fake.url)) as p:
        for port in (p.port, fake.port):
            assert addrs(port) and all(gt.is_loopback(a) for a in addrs(port)), (port, addrs(port))


def test_all_connections_and_tasks_shut_down_cleanly_in_order(fake):
    baseline = {t.name for t in threading.enumerate()}
    app = make_app(fake.url)
    p = ProxyServer(app).__enter__()
    try:
        backend = TestClient(create_app(make_settings(_tmp(), inference_provider="comfyui", comfyui_base_url=p.url, comfyui_api_token=TOKEN,
                                                      comfyui_poll_interval=0.2, comfyui_timeout=3, generation_profile="t4_verified")))
        backend.__enter__()
        job = wait_for(backend, submit(backend, **PROVEN).json()["id"])
        assert job["status"] == "completed"
        assert wait_sync(lambda: app.state.state["ws"]), "the provider's WebSocket should be open while the backend runs"
        assert wait_sync(lambda: fake._sockets.get(next(iter(app.state.state["ws"]))))
        backend.__exit__(None, None, None)                                                # 1. the backend stops first
        assert wait_sync(lambda: not app.state.state["ws"] and not app.state.state["ws_tasks"]), "the relay was left running"
        assert wait_sync(lambda: not any(fake._sockets.values())), "the upstream WebSocket was left open"
        assert app.state.state["inflight"] == 0 and app.state.state["heavy"] == 0
    finally:
        p.__exit__(None, None, None)                                                      # 2. then the proxy
    assert wait_sync(lambda: refuses(p.port)) and not p.thread.is_alive()
    assert wait_sync(lambda: {t.name for t in threading.enumerate()} <= baseline | {t.name for t in threading.enumerate() if t.daemon})


def _tmp():
    import tempfile
    from pathlib import Path
    return Path(tempfile.mkdtemp(prefix="vidsgen-stage3-"))


def test_the_real_proxy_process_runs_a_job_and_stops_cleanly(make_client, fake):
    """The actual command-line program (not the in-process app): token from the environment, loopback only, clean SIGTERM."""
    port = free_port()
    env = {"PATH": "/usr/bin:/bin", "PROXY_API_TOKEN": TOKEN, "HOME": "/tmp", "PYTHONDONTWRITEBYTECODE": "1"}
    log = _tmp() / "proxy.log"
    with open(log, "wb") as out:
        proc = subprocess.Popen([sys.executable, str(SCRIPT), "--upstream", fake.url, "--port", str(port), "--workflows-dir", str(WORKFLOWS)],
                                env=env, stdout=out, stderr=subprocess.STDOUT)
    try:
        assert wait_sync(lambda: addrs(port), 30), log.read_text()
        assert all(gt.is_loopback(a) for a in addrs(port))
        assert httpx.get(f"http://127.0.0.1:{port}/system_stats").status_code == 401                # unauthenticated: refused
        c = make_client(inference_provider="comfyui", comfyui_base_url=f"http://127.0.0.1:{port}", comfyui_api_token=TOKEN,
                        comfyui_poll_interval=0.2, comfyui_timeout=3, generation_profile="t4_verified")
        job = wait_for(c, submit(c, **PROVEN).json()["id"], timeout=40)
        assert job["status"] == "completed", job["error"]
        assert c.get("/api/health").json()["provider"]["ok"] is True
    finally:
        proc.send_signal(signal.SIGTERM)
        try:
            code = proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            proc.kill()
            code = None
    assert code in (0, -signal.SIGTERM), code
    assert wait_sync(lambda: refuses(port))
    text = log.read_text(errors="replace")
    assert TOKEN not in text and "Authorization" not in text and "Traceback" not in text, text[-500:]


def test_the_proxy_program_refuses_to_start_without_a_strong_token(tmp_path):
    for token in ("", "short"):
        env = {"PATH": "/usr/bin:/bin", "PROXY_API_TOKEN": token}
        r = subprocess.run([sys.executable, str(SCRIPT), "--port", str(free_port()), "--workflows-dir", str(WORKFLOWS)], env=env,
                           capture_output=True, text=True, timeout=30)
        assert r.returncode == 2 and (token not in r.stderr or token == "") and "Traceback" not in r.stderr
