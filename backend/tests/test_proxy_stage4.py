"""Phase 6C Stage 4A: the proxy's read-only pre-forward existence check, the dead-man switch's safeguards, 416 handling and --max-lifetime-seconds."""

from __future__ import annotations

import signal
import subprocess
import sys
import time
import uuid

import httpx
import pytest
from fastapi.testclient import TestClient

from test_inference_proxy import AUTH, TOKEN, WORKFLOWS, JOB, fake, free_port, make_app, px, video_bytes  # noqa: F401
from test_inference_proxy_stage2 import Clock, body_for, dm_app, new_id, prompts, tick

SCRIPT = px.__file__


def methods_before_first_post(fake) -> list[tuple[str, str]]:
    seq = [(r["method"], r["path"]) for r in fake.requests]
    first = next((i for i, (m, p) in enumerate(seq) if (m, p) == ("POST", "/prompt")), len(seq))
    return seq[:first]


# ====================================================================== pre-forward existence check
def test_an_id_comfyui_already_knows_is_not_forwarded_again_even_after_a_proxy_restart(fake):
    pid = new_id()
    body = body_for(pid=pid)
    with TestClient(make_app(fake.url)) as c1:
        assert c1.post("/prompt", json=body, headers=AUTH).status_code == 200
    deadline = time.monotonic() + 10
    while pid not in fake.history and time.monotonic() < deadline:
        time.sleep(0.02)
    with TestClient(make_app(fake.url)) as c2:                                    # a restarted proxy: its memory is empty
        r = c2.post("/prompt", json=body, headers=AUTH)
    assert r.status_code == 200 and r.headers["x-idempotent-replay"] == "existing-unverified" and r.json()["prompt_id"] == pid
    assert len(prompts(fake)) == 1                                                # the GPU job was not started a second time


def test_a_prompt_that_is_still_queued_is_recognised_too(fake):
    fake.paused = True
    pid, body = new_id(), None
    body = body_for(pid=pid)
    with TestClient(make_app(fake.url)) as c1:
        assert c1.post("/prompt", json=body, headers=AUTH).status_code == 200
    with TestClient(make_app(fake.url)) as c2:
        r = c2.post("/prompt", json=body, headers=AUTH)
    assert r.status_code == 200 and r.headers["x-idempotent-replay"] == "existing-unverified" and len(prompts(fake)) == 1


def test_an_unknown_id_is_forwarded_exactly_once_after_read_only_lookups_only(fake):
    with TestClient(make_app(fake.url)) as c:
        pid = new_id()
        assert c.post("/prompt", json=body_for(pid=pid), headers=AUTH).status_code == 200
    before = methods_before_first_post(fake)
    assert before and all(m == "GET" for m, _ in before)                          # only reads happened before anything was forwarded
    assert {p for _, p in before} >= {f"/history/{pid}", "/queue"}
    assert len(prompts(fake)) == 1


def test_when_comfyui_cannot_be_asked_nothing_is_forwarded(fake):
    fake.outage = True
    app = make_app(fake.url)
    with TestClient(app) as c:
        r = c.post("/prompt", json=body_for(), headers=AUTH)
    assert r.status_code == 502 and r.json() == {"error": "upstream_error"}
    assert app.state.registry.entries == {}                                       # nothing remembered: a later, healthy attempt is fresh
    fake.outage = False
    assert [x for x in fake.requests if x["method"] == "POST"] == []


def test_the_check_can_be_disabled_and_then_forwards_without_lookups(fake):
    with TestClient(make_app(fake.url, px.Policy(preforward_check=False))) as c:
        assert c.post("/prompt", json=body_for(), headers=AUTH).status_code == 200
    assert methods_before_first_post(fake) == []
    assert px.Policy().preforward_check is True                                   # on by default


def test_the_check_never_resubmits_an_uncertain_prompt(fake):
    """The earlier uncertain-submission behaviour is unchanged: read-only reconciliation, never a second POST."""
    fake.prompt_mode = "hang"
    pid = new_id()
    with TestClient(make_app(fake.url, px.Policy(read_timeout=0.5))) as c:
        body = body_for(pid=pid)
        assert c.post("/prompt", json=body, headers=AUTH).status_code == 504
        fake.prompt_mode = "ok"
        assert c.post("/prompt", json=body, headers=AUTH).status_code == 409
    assert len(prompts(fake)) == 1


def test_documented_limitation_a_visibility_race_across_a_restart_can_run_a_prompt_twice():
    """NOT a guarantee: if ComfyUI has accepted a prompt but does not (yet) show it in /history or /queue, a restarted proxy cannot know and
    forwards again. This test pins the documented limitation down; if it is ever fixed, this test should be changed deliberately."""
    posts = []

    def handler(req: httpx.Request) -> httpx.Response:
        if req.method == "POST":
            posts.append(req.url.path)
            return httpx.Response(200, json={"prompt_id": str(uuid.uuid4()), "number": 1, "node_errors": {}})
        return httpx.Response(200, json={"queue_running": [], "queue_pending": []} if req.url.path == "/queue" else {})   # nothing visible
    body = body_for()
    for _ in range(2):                                                            # two proxy "lifetimes" (the second has an empty memory)
        with TestClient(px.create_app("http://127.0.0.1:8188", TOKEN, WORKFLOWS, transport=httpx.MockTransport(handler))) as c:
            assert c.post("/prompt", json=body, headers=AUTH).status_code == 200
    assert posts == ["/prompt", "/prompt"]


def test_within_one_proxy_lifetime_a_visibility_race_does_not_duplicate_the_prompt():
    """The in-memory registry covers the same proxy even when ComfyUI shows nothing: this is the part that IS protected."""
    posts = []

    def handler(req: httpx.Request) -> httpx.Response:
        if req.method == "POST":
            posts.append(req.url.path)
            return httpx.Response(200, json={"prompt_id": "x", "number": 1, "node_errors": {}})
        return httpx.Response(200, json={"queue_running": [], "queue_pending": []} if req.url.path == "/queue" else {})
    body = body_for()
    with TestClient(px.create_app("http://127.0.0.1:8188", TOKEN, WORKFLOWS, transport=httpx.MockTransport(handler))) as c:
        for _ in range(3):
            assert c.post("/prompt", json=body, headers=AUTH).status_code == 200
    assert posts == ["/prompt"]


def test_the_precheck_makes_only_get_requests():
    seen = []

    def handler(req: httpx.Request) -> httpx.Response:
        seen.append(req.method)
        if req.method == "POST":
            return httpx.Response(200, json={"prompt_id": "x", "number": 1, "node_errors": {}})
        return httpx.Response(200, json={"queue_running": [], "queue_pending": []} if req.url.path == "/queue" else {})
    with TestClient(px.create_app("http://127.0.0.1:8188", TOKEN, WORKFLOWS, transport=httpx.MockTransport(handler))) as c:
        c.post("/prompt", json=body_for(), headers=AUTH)
    assert seen[-1] == "POST" and seen[:-1] and set(seen[:-1]) == {"GET"}


# ====================================================================== 416 Range handling
def test_an_unsatisfiable_range_is_a_416_not_a_server_error():
    def handler(req):
        return httpx.Response(416, headers={"content-range": "bytes */10"})
    with TestClient(px.create_app("http://127.0.0.1:8188", TOKEN, WORKFLOWS, transport=httpx.MockTransport(handler))) as c:
        r = c.get(f"/view?filename={JOB}_00001_.mp4&subfolder=vidsgen&type=output", headers={**AUTH, "Range": "bytes=999-"})
    assert r.status_code == 416


# ====================================================================== the dead-man switch: elapsed time alone never triggers it
def test_the_age_of_a_job_alone_never_interrupts_it(fake):
    """A job that runs for ten hours with a live backend (it keeps polling) is never touched, however long it has been running."""
    fake.default_scenario = "hang"
    clock = Clock()
    app = dm_app(fake, clock, seconds=300.0)
    with TestClient(app) as c:
        pid = new_id()
        assert c.post("/prompt", json=body_for(pid=pid), headers=AUTH).status_code == 200
        for _ in range(240):                                                      # 240 x 150 s = 10 hours
            clock.t += 150
            assert c.get("/queue", headers=AUTH).status_code == 200
            assert tick(c, app) == 0
        assert fake.interrupts == [] and app.state.registry.entries[pid].state == "accepted"


def test_an_open_websocket_protects_every_tracked_job_however_long_it_has_run(fake):
    fake.default_scenario = "hang"
    clock = Clock()
    app = dm_app(fake, clock, seconds=300.0)
    with TestClient(app) as c:
        pid = new_id()
        c.post("/prompt", json=body_for(pid=pid), headers=AUTH)
        with c.websocket_connect("/ws?clientId=vidsgen-abcdef012345", headers=AUTH):
            for _ in range(50):
                clock.t += 3600                                                   # 50 hours with no HTTP request at all
                assert tick(c, app) == 0
        assert fake.interrupts == []


def test_silence_is_the_only_trigger_and_the_switch_is_off_unless_configured(fake):
    fake.default_scenario = "hang"
    clock = Clock()
    off = make_clock_app_off(fake, clock)
    with TestClient(off) as c:
        c.post("/prompt", json=body_for(), headers=AUTH)
        clock.t += 10 ** 7
        assert tick(c, off) == 0 and fake.interrupts == []                        # off: not even a day of silence matters
    assert px.Policy().dead_man_seconds == 0


def make_clock_app_off(fake, clock):
    return px.create_app(fake.url, TOKEN, WORKFLOWS, px.Policy(), clock=clock)


def test_activity_during_a_reap_stops_it_before_anything_is_interrupted():
    """The backend speaks while the proxy is looking at the queue: it is alive, so nothing is cancelled."""
    clock = Clock()
    pid = new_id()
    app_holder = {}
    calls = []

    def handler(req: httpx.Request) -> httpx.Response:
        calls.append((req.method, req.url.path))
        if req.url.path == "/queue" and req.method == "GET":
            app_holder["app"].state.state["last_activity"] = clock()             # the backend wakes up right now
            return httpx.Response(200, json={"queue_running": [[1, pid, {}, {}, ["58"]]], "queue_pending": []})
        return httpx.Response(200, json={})
    app = px.create_app("http://127.0.0.1:8188", TOKEN, WORKFLOWS, px.Policy(dead_man_seconds=300, dead_man_interval=3600),
                        transport=httpx.MockTransport(handler), clock=clock)
    app_holder["app"] = app
    with TestClient(app) as c:
        app.state.registry.begin(pid, "d", "c")
        app.state.registry.resolve(pid, 200, {})
        clock.t += 1000
        assert tick(c, app) == 0
    assert ("POST", "/interrupt") not in calls and ("POST", "/queue") not in calls and app.state.registry.entries[pid].state == "accepted"


def test_a_silent_backend_is_still_reaped_with_a_targeted_call(fake):
    fake.default_scenario = "hang"
    clock = Clock()
    app = dm_app(fake, clock, seconds=300.0)
    with TestClient(app) as c:
        pid = new_id()
        c.post("/prompt", json=body_for(pid=pid), headers=AUTH)
        deadline = time.monotonic() + 5
        while fake._running != pid and time.monotonic() < deadline:
            time.sleep(0.02)
        clock.t += 301
        assert tick(c, app) == 1
        assert wait_for_interrupt(fake) == [pid]


def wait_for_interrupt(fake):
    deadline = time.monotonic() + 5
    while not fake.interrupts and time.monotonic() < deadline:
        time.sleep(0.02)
    return fake.interrupts


# ====================================================================== --max-lifetime-seconds
def start_proxy(extra, fake_url="http://127.0.0.1:8188", token=TOKEN):
    env = {"PATH": "/usr/bin:/bin", "PROXY_API_TOKEN": token, "HOME": "/tmp"}
    return subprocess.Popen([sys.executable, SCRIPT, "--upstream", fake_url, "--port", str(free_port()), "--workflows-dir", str(WORKFLOWS), *extra],
                            env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)


def test_the_proxy_stops_itself_after_its_maximum_lifetime():
    proc = start_proxy(["--max-lifetime-seconds", "3"])
    t0 = time.monotonic()
    try:
        assert proc.wait(timeout=30) == 0
    finally:
        if proc.poll() is None:
            proc.kill()
        proc.stdout.close()
    assert 2 <= time.monotonic() - t0 < 25


@pytest.mark.parametrize("value", ["-1", "86401", "abc"])
def test_an_invalid_lifetime_is_refused(value):
    proc = start_proxy(["--max-lifetime-seconds", value])
    out = proc.communicate(timeout=30)[0].decode()
    assert proc.returncode == 2 and TOKEN not in out


def test_without_a_lifetime_the_proxy_keeps_running_until_stopped():
    proc = start_proxy([])
    try:
        time.sleep(3)
        assert proc.poll() is None
        proc.send_signal(signal.SIGTERM)
        assert proc.wait(timeout=15) in (0, -signal.SIGTERM)
    finally:
        if proc.poll() is None:
            proc.kill()
        proc.stdout.close()


def test_a_token_pasted_as_a_command_line_option_is_not_echoed_back():
    proc = subprocess.run([sys.executable, SCRIPT, "--workflows-dir", str(WORKFLOWS), "--token", TOKEN], capture_output=True, text=True,
                          env={"PATH": "/usr/bin:/bin", "PROXY_API_TOKEN": TOKEN}, timeout=30)
    assert proc.returncode == 2 and TOKEN not in proc.stderr + proc.stdout
