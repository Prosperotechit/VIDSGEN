"""Phase 6C Stage 2 tests: WebSocket relay, streaming, prompt_id idempotency, read-only reconciliation, optional dead-man switch.

Local only: FakeComfyUI or mocked upstreams, loopback sockets, no tunnel, no GPU.
"""

from __future__ import annotations

import asyncio
import json
import time
import tracemalloc
import uuid

import httpx
import pytest
from fastapi.testclient import TestClient
from websockets.asyncio.client import connect as ws_connect
from websockets.exceptions import ConnectionClosed, InvalidStatus

from test_inference_proxy import (  # noqa: F401  (shared helpers and fixtures)
    AUTH, CLIENT, JOB, PID, TOKEN, WORKFLOWS, ProxyServer, e2e_client, fake, free_port, graph, make_app, px, prompt_body, upstream_paths,
    video_bytes,
)
from conftest import submit, wait_for

CID = "vidsgen-abcdef012345"


def new_id() -> str:
    return str(uuid.uuid4())


def body_for(pid=None, cid=CID, **gv) -> dict:
    return prompt_body(graph(**gv), pid=pid or new_id(), client_id=cid)


def prompts(fake) -> list[dict]:
    return [r["body"] for r in fake.requests if r["path"] == "/prompt" and r["method"] == "POST"]


def run(coro):
    return asyncio.run(coro)


# ====================================================================== PromptRegistry (unit)
class Clock:
    def __init__(self):
        self.t = 1000.0

    def __call__(self):
        return self.t


def registry(max_entries=4, ttl=100.0):
    clock = Clock()
    return px.PromptRegistry(max_entries, ttl, clock), clock


def test_registry_states_new_pending_accepted():
    async def go():
        reg, _ = registry()
        assert reg.begin("a", "d1", "c")[0] == "new"
        assert reg.begin("a", "d1", "c")[0] == "pending"                  # the same request again while it is in flight
        reg.resolve("a", 200, {"x": 1})
        kind, e = reg.begin("a", "d1", "c")
        assert kind == "accepted" and (e.status, e.body) == (200, {"x": 1})
    run(go())


def test_registry_refuses_reuse_of_an_id_with_a_different_payload():
    async def go():
        reg, _ = registry()
        reg.begin("a", "d1", "c")
        assert reg.begin("a", "OTHER", "c")[0] == "conflict"             # while pending
        reg.resolve("a", 200, {})
        assert reg.begin("a", "OTHER", "c")[0] == "conflict"             # after acceptance
    run(go())


def test_registry_reject_frees_the_id_but_waiters_get_the_answer():
    async def go():
        reg, _ = registry()
        _, e = reg.begin("a", "d1", "c")
        reg.reject("a", 400, {"error": "x"})
        assert (await e.future) == (400, {"error": "x"}) and "a" not in reg.entries
        assert reg.begin("a", "d2", "c")[0] == "new"                      # nothing was queued: a corrected request may reuse it
    run(go())


def test_registry_uncertain_is_remembered_and_never_becomes_new():
    async def go():
        reg, _ = registry()
        reg.begin("a", "d1", "c")
        reg.uncertain("a", 504, {"error": "rejected"})
        assert reg.begin("a", "d1", "c")[0] == "uncertain" and reg.is_tracked("a")
    run(go())


def test_registry_tombstones_cannot_be_overridden():
    async def go():
        reg, _ = registry()
        reg.cancel("never-seen")                                          # an id the proxy never saw
        assert reg.begin("never-seen", "d", "c")[0] == "cancelled"
        reg.begin("a", "d1", "c")
        reg.cancel("a")
        reg.resolve("a", 200, {})                                          # a late ComfyUI answer must not resurrect it
        assert reg.begin("a", "d1", "c")[0] == "cancelled" and not reg.is_tracked("a")
    run(go())


def test_registry_is_bounded_and_never_evicts_pending_entries():
    async def go():
        reg, _ = registry(max_entries=3)
        for i in range(3):
            reg.begin(f"p{i}", "d", "c")
        assert reg.begin("x", "d", "c")[0] == "full"                      # everything is in flight: refuse, do not evict
        reg.resolve("p0", 200, {})
        assert reg.begin("x", "d", "c")[0] == "new" and "p0" not in reg.entries and len(reg.entries) == 3
        assert {"p1", "p2"} <= set(reg.entries)                            # the pending ones were kept
    run(go())


def test_registry_entries_expire_after_the_ttl_but_pending_ones_do_not():
    async def go():
        reg, clock = registry(ttl=100)
        reg.begin("done", "d", "c")
        reg.resolve("done", 200, {})
        reg.begin("flying", "d", "c")
        clock.t += 101
        assert reg.begin("other", "d", "c")[0] == "new"
        assert "done" not in reg.entries and "flying" in reg.entries
    run(go())


def test_registry_keeps_only_a_digest_and_the_response_not_the_graph():
    async def go():
        reg, _ = registry()
        d = px.payload_digest({"prompt": graph(), "client_id": CLIENT})
        reg.begin("a", d, CLIENT)
        reg.resolve("a", 200, {"prompt_id": "a"})
        e = reg.entries["a"]
        assert e.digest == d and len(d) == 64 and e.body == {"prompt_id": "a"}
        assert set(e.__slots__) == {"digest", "client_id", "state", "status", "body", "future", "created", "finished"}   # no graph is stored
    run(go())


def test_the_digest_is_stable_and_sensitive():
    g = graph()
    a = px.payload_digest({"prompt": g, "client_id": CLIENT})
    assert a == px.payload_digest({"client_id": CLIENT, "prompt": json.loads(json.dumps(g))})        # key order and round trips do not matter
    g2 = json.loads(json.dumps(g)); g2["3"]["inputs"]["seed"] += 1
    assert a != px.payload_digest({"prompt": g2, "client_id": CLIENT}) and a != px.payload_digest({"prompt": g, "client_id": "other"})


# ====================================================================== idempotent POST /prompt
def test_a_duplicate_submission_is_answered_again_and_never_forwarded_twice(fake):
    with TestClient(make_app(fake.url)) as c:
        body = body_for()
        first = c.post("/prompt", json=body, headers=AUTH)
        again = c.post("/prompt", json=body, headers=AUTH)
        third = c.post("/prompt", json=body, headers=AUTH)
    assert first.status_code == again.status_code == third.status_code == 200
    assert first.json() == again.json() == third.json() and "x-idempotent-replay" not in first.headers
    assert again.headers["x-idempotent-replay"] == "1"
    assert len(prompts(fake)) == 1


def test_concurrent_identical_submissions_produce_exactly_one_upstream_request(fake):
    async def go():
        with ProxyServer(make_app(fake.url)) as p:
            body = body_for()
            async with httpx.AsyncClient(base_url=p.url, timeout=20) as c:
                return await asyncio.gather(*[c.post("/prompt", json=body, headers=AUTH) for _ in range(30)])         # <= the 32 in-flight limit
    rs = run(go())
    assert [r.status_code for r in rs].count(200) == 30 and len({r.text for r in rs}) == 1
    assert len(prompts(fake)) == 1


def test_concurrent_submissions_of_one_id_with_different_payloads_forward_exactly_one(fake):
    pid = new_id()

    async def go():
        with ProxyServer(make_app(fake.url, px.Policy(max_heavy=40))) as p:
            async with httpx.AsyncClient(base_url=p.url, timeout=20) as c:
                bodies = [body_for(pid=pid, seed=i) for i in range(10)]
                return await asyncio.gather(*[c.post("/prompt", json=b, headers=AUTH) for b in bodies]), bodies
    rs, bodies = run(go())
    assert sorted(r.status_code for r in rs) == [200] + [409] * 9
    assert len(prompts(fake)) == 1
    winner = next(b for b, r in zip(bodies, rs) if r.status_code == 200)
    assert prompts(fake)[0]["prompt"] == winner["prompt"]


def test_reusing_an_id_with_a_different_payload_is_rejected_with_a_readable_reason(fake):
    pid = new_id()
    with TestClient(make_app(fake.url)) as c:
        assert c.post("/prompt", json=body_for(pid=pid, seed=1), headers=AUTH).status_code == 200
        r = c.post("/prompt", json=body_for(pid=pid, seed=2), headers=AUTH)
        r2 = c.post("/prompt", json=body_for(pid=pid, seed=1, cid="vidsgen-other0000000"), headers=AUTH)   # even just another client id
    for resp in (r, r2):
        assert resp.status_code == 409 and resp.json()["error"]["type"] == "proxy_rejected"
        assert "already used with a different request" in resp.json()["error"]["message"]
    assert len(prompts(fake)) == 1


def test_different_ids_with_the_same_payload_are_separate_submissions(fake):
    g = graph()
    with TestClient(make_app(fake.url)) as c:
        for _ in range(2):
            assert c.post("/prompt", json=prompt_body(g), headers=AUTH).status_code == 200
    assert len(prompts(fake)) == 2


def test_a_comfyui_validation_refusal_does_not_poison_the_id(fake):
    pid = new_id()
    with TestClient(make_app(fake.url)) as c:
        fake.prompt_mode = "reject_model"
        assert c.post("/prompt", json=body_for(pid=pid), headers=AUTH).status_code == 400
        fake.prompt_mode = "ok"
        assert c.post("/prompt", json=body_for(pid=pid), headers=AUTH).status_code == 200     # nothing had been queued
    assert len(fake.prompts) == 1                                         # only the second attempt was ever queued


def test_a_lost_response_is_reconciled_read_only_and_never_resubmitted(fake):
    """ComfyUI queued the prompt but the response never arrived. A duplicate request finds it by looking (GET only)."""
    fake.prompt_mode = "queue_then_hang"
    fake.paused = True                                              # keep it visibly queued
    pid, body = new_id(), None
    with TestClient(make_app(fake.url, px.Policy(read_timeout=0.5))) as c:
        body = body_for(pid=pid)
        first = c.post("/prompt", json=body, headers=AUTH)
        assert first.status_code == 504
        fake.prompt_mode = "ok"
        again = c.post("/prompt", json=body, headers=AUTH)
        later = c.post("/prompt", json=body, headers=AUTH)
    assert again.status_code == 200 and again.json()["prompt_id"] == pid and again.headers["x-idempotent-replay"] == "reconciled"
    assert later.status_code == 200 and later.json() == again.json()                       # now accepted: a plain replay
    assert len(prompts(fake)) == 1                                                         # exactly ONE POST /prompt ever reached ComfyUI
    posts = [(r["method"], r["path"]) for r in fake.requests if r["method"] == "POST"]
    assert posts == [("POST", "/prompt")]


def test_a_5xx_after_queueing_is_reconciled_the_same_way(fake):
    fake.prompt_mode = "queue_then_500"
    pid = new_id()
    with TestClient(make_app(fake.url)) as c:
        body = body_for(pid=pid)
        assert c.post("/prompt", json=body, headers=AUTH).status_code == 502
        fake.prompt_mode = "ok"
        again = c.post("/prompt", json=body, headers=AUTH)
    assert again.status_code == 200 and again.headers["x-idempotent-replay"] == "reconciled" and len(prompts(fake)) == 1


def test_an_uncertain_submission_that_comfyui_does_not_know_is_not_resubmitted(fake):
    """Nothing was queued (e.g. the request never arrived): still no automatic resubmission; the caller must decide."""
    fake.prompt_mode = "hang"
    pid = new_id()
    with TestClient(make_app(fake.url, px.Policy(read_timeout=0.5))) as c:
        body = body_for(pid=pid)
        assert c.post("/prompt", json=body, headers=AUTH).status_code == 504
        fake.prompt_mode = "ok"
        again = c.post("/prompt", json=body, headers=AUTH)
    assert again.status_code == 409 and "not resubmitted" in again.json()["error"]["message"]
    assert len(prompts(fake)) == 1                                        # only the original (hung) request; no second POST /prompt


def test_waiters_share_the_outcome_of_a_failed_first_attempt_without_a_second_forward(fake):
    fake.prompt_mode = "hang"
    pid = new_id()

    async def go():
        with ProxyServer(make_app(fake.url, px.Policy(read_timeout=0.8))) as p:
            body = body_for(pid=pid)
            async with httpx.AsyncClient(base_url=p.url, timeout=20) as c:
                return await asyncio.gather(*[c.post("/prompt", json=body, headers=AUTH) for _ in range(6)])
    rs = run(go())
    assert [r.status_code for r in rs] == [504] * 6
    assert len([r for r in fake.requests if r["path"] == "/prompt"]) == 1


def test_cancelling_an_id_leaves_a_tombstone_so_it_cannot_be_submitted_later(fake):
    with TestClient(make_app(fake.url)) as c:
        pid, other = new_id(), new_id()
        body = body_for(pid=pid)
        assert c.post("/prompt", json=body, headers=AUTH).status_code == 200
        assert c.post("/queue", json={"delete": [pid]}, headers=AUTH).status_code == 200
        r = c.post("/prompt", json=body, headers=AUTH)
        assert r.status_code == 409 and "cancelled" in r.json()["error"]["message"]
        # an interrupt for an id the proxy never saw also blocks that id (a late arrival cannot start after being cancelled)
        assert c.post("/interrupt", json={"prompt_id": other}, headers=AUTH).status_code == 200
        assert c.post("/prompt", json=body_for(pid=other), headers=AUTH).status_code == 409
        # deleting history (cleanup after a finished job) is NOT a cancellation
        third = new_id()
        assert c.post("/prompt", json=body_for(pid=third), headers=AUTH).status_code == 200
        assert c.post("/history", json={"delete": [third]}, headers=AUTH).status_code == 200
        again = c.post("/prompt", json=body_for(pid=third), headers=AUTH)
        assert again.status_code in (200, 409)


def test_the_registry_is_bounded_under_load(fake):
    app = make_app(fake.url, px.Policy(idempotency_max_entries=5))
    with TestClient(app) as c:
        for _ in range(12):
            assert c.post("/prompt", json=body_for(), headers=AUTH).status_code == 200
        assert len(app.state.registry.entries) <= 5


# ====================================================================== WebSocket relay
def wsconnect(p, cid=CID, headers=AUTH, query=None, **kw):
    q = query if query is not None else f"?clientId={cid}"
    return ws_connect(f"{p.ws}/ws{q}", additional_headers=headers, proxy=None, open_timeout=5, **kw)


async def collect(ws, until="execution_success", timeout=20):
    seen = []
    async with asyncio.timeout(timeout):
        async for raw in ws:
            msg = json.loads(raw)
            seen.append(msg)
            if msg["type"] == until:
                break
    return seen


def test_progress_events_are_relayed_for_prompts_submitted_through_the_proxy(fake):
    async def go():
        app = make_app(fake.url)
        with ProxyServer(app) as p:
            async with wsconnect(p) as ws, httpx.AsyncClient(base_url=p.url) as c:
                pid = new_id()
                assert (await c.post("/prompt", json=body_for(pid=pid), headers=AUTH)).status_code == 200
                return pid, await collect(ws), dict(app.state.state["stats"])
    pid, events, stats = run(go())
    kinds = [e["type"] for e in events]
    assert kinds[0] == "execution_start" and kinds[-1] == "execution_success" and "progress" in kinds and "executing" in kinds
    assert all(e["data"]["prompt_id"] == pid for e in events)
    progress = [e["data"] for e in events if e["type"] == "progress"]
    assert [x["value"] for x in progress] == list(range(1, 21)) and all(x["max"] == 20 and x["node"] == "3" for x in progress)
    assert stats["ws_events"] == len(events)


def test_only_allow_listed_event_types_and_fields_reach_the_client(fake):
    async def go():
        with ProxyServer(make_app(fake.url)) as p:
            async with wsconnect(p) as ws, httpx.AsyncClient(base_url=p.url) as c:
                pid = new_id()
                await c.post("/prompt", json=body_for(pid=pid), headers=AUTH)
                evs = await collect(ws)
                fake.inject(CID, json.dumps({"type": "executed", "data": {"prompt_id": pid, "node": "58", "output": {"secret": "x"}}}))
                fake.inject(CID, json.dumps({"type": "status", "data": {"status": {"exec_info": {"queue_remaining": 3}}}}))
                fake.inject(CID, json.dumps({"type": "execution_cached", "data": {"prompt_id": pid, "nodes": []}}))
                fake.inject(CID, json.dumps({"type": "progress", "data": {"prompt_id": pid, "value": 1, "max": 2, "node": "3", "evil": "x",
                                                                         "extra_field": {"a": 1}}}))
                fake.inject(CID, json.dumps({"type": "execution_success", "data": {"prompt_id": pid, "timestamp": 1}}))
                tail = await collect(ws, until="execution_success")
                return evs, tail
    evs, tail = run(go())
    assert [e["type"] for e in tail] == ["progress", "execution_success"]                 # executed/status/cached were dropped
    assert tail[0]["data"] == {"prompt_id": tail[0]["data"]["prompt_id"], "node": "3", "value": 1, "max": 2}   # unknown fields stripped
    for e in evs:
        assert set(e["data"]) <= set(px.WS_FIELDS[e["type"]])


def test_events_for_other_prompts_or_without_an_id_are_never_forwarded(fake):
    async def go():
        with ProxyServer(make_app(fake.url)) as p:
            async with wsconnect(p) as ws, httpx.AsyncClient(base_url=p.url) as c:
                pid = new_id()
                await c.post("/prompt", json=body_for(pid=pid), headers=AUTH)
                await collect(ws)
                stranger = new_id()
                for data in ({"prompt_id": stranger, "value": 1, "max": 2, "node": "3"}, {"value": 1, "max": 2, "node": "3"},
                             {"prompt_id": "../../etc", "value": 1}, {"prompt_id": 5}):
                    fake.inject(CID, json.dumps({"type": "progress", "data": data}))
                fake.inject(CID, "not json")
                fake.inject(CID, json.dumps([1, 2]))
                fake.inject(CID, json.dumps({"type": "progress", "data": {"prompt_id": pid, "value": 7, "max": 8, "node": "3"}}))
                return await collect(ws, until="progress")
    got = run(go())
    assert len(got) == 1 and got[0]["data"]["value"] == 7                                    # only the tracked prompt's event arrived


def test_execution_errors_are_rebuilt_without_file_paths_or_extra_data(fake):
    fake.default_scenario = "oom"

    async def go():
        with ProxyServer(make_app(fake.url)) as p:
            async with wsconnect(p) as ws, httpx.AsyncClient(base_url=p.url) as c:
                await c.post("/prompt", json=body_for(), headers=AUTH)
                return await collect(ws, until="execution_error")
    err = run(go())[-1]["data"]
    assert set(err) <= set(px.WS_FIELDS["execution_error"]) and "current_inputs" not in err
    assert "CUDA out of memory" in err["exception_message"] and "/srv/ComfyUI" not in json.dumps(err) and "<path>" in json.dumps(err)


def test_binary_and_oversized_frames_are_dropped(fake):
    async def go():
        app = make_app(fake.url, px.Policy(ws_frame_limit=2048))
        with ProxyServer(app) as p:
            async with wsconnect(p) as ws, httpx.AsyncClient(base_url=p.url) as c:
                pid = new_id()
                await c.post("/prompt", json=body_for(pid=pid), headers=AUTH)
                await collect(ws)
                fake.inject(CID, b"\x01\x00\x00\x00" + b"x" * 5000)                                   # a binary preview image
                fake.inject(CID, json.dumps({"type": "progress", "data": {"prompt_id": pid, "value": 1, "max": 2, "node": "3", "pad": "p" * 4000}}))
                fake.inject(CID, json.dumps({"type": "progress", "data": {"prompt_id": pid, "value": 9, "max": 9, "node": "3"}}))
                return await collect(ws, until="progress"), app.state.state["stats"]["ws_dropped"]
    got, dropped = run(go())
    assert [g["data"]["value"] for g in got] == [9] and dropped >= 2


@pytest.mark.parametrize("headers", [{}, {"Authorization": "Bearer wrong"}, {"Authorization": f"Basic {TOKEN}"}, {"Authorization": TOKEN},
                                     {"Authorization": f"Bearer {TOKEN}x"}])
def test_the_websocket_handshake_requires_the_bearer_token(fake, headers):
    async def go():
        with ProxyServer(make_app(fake.url)) as p:
            with pytest.raises(InvalidStatus) as e:
                async with wsconnect(p, headers=headers):
                    pass
            return e.value.response.status_code
    assert run(go()) == 403
    assert fake.ws_auth_headers == []                                                          # ComfyUI was never contacted


@pytest.mark.parametrize("query", [f"?clientId={CID}&token={TOKEN}", f"?token={TOKEN}", "", "?clientId=", "?clientId=bad id", "?clientId=../x",
                                   f"?clientId={'x' * 65}", f"?clientId={CID}&clientId={CID}", f"?clientId={CID}&other=1", "?clientid=" + CID])
def test_the_websocket_client_id_must_be_valid_and_nothing_else_is_accepted(fake, query):
    async def go():
        with ProxyServer(make_app(fake.url)) as p:
            with pytest.raises(InvalidStatus):
                async with wsconnect(p, query=query):
                    pass
    run(go())
    assert fake.ws_auth_headers == []


def test_a_browser_origin_is_refused_and_other_paths_are_not_websockets(fake):
    async def go():
        with ProxyServer(make_app(fake.url)) as p:
            with pytest.raises(InvalidStatus):
                async with ws_connect(f"{p.ws}/ws?clientId={CID}", additional_headers=AUTH, origin="https://evil.example", proxy=None):
                    pass
            for path in ("/ws/x", "/other", "/"):
                with pytest.raises(InvalidStatus):
                    async with ws_connect(f"{p.ws}{path}?clientId={CID}", additional_headers=AUTH, proxy=None):
                        pass
    run(go())
    assert fake.ws_auth_headers == []


def test_the_token_is_not_passed_to_comfyui_on_the_upstream_websocket(fake):
    async def go():
        with ProxyServer(make_app(fake.url)) as p:
            async with wsconnect(p):
                await asyncio.sleep(0.3)
    run(go())
    assert fake.ws_auth_headers == [None]


def test_only_one_socket_per_client_id_and_a_bounded_number_in_total(fake):
    async def go():
        app = make_app(fake.url, px.Policy(max_ws=2))
        with ProxyServer(app) as p:
            async with wsconnect(p, "vidsgen-aaaaaaaaaaaa"), wsconnect(p, "vidsgen-bbbbbbbbbbbb"):
                with pytest.raises(InvalidStatus):                                                 # same id again
                    async with wsconnect(p, "vidsgen-aaaaaaaaaaaa"):
                        pass
                with pytest.raises(InvalidStatus):                                                 # a third one
                    async with wsconnect(p, "vidsgen-cccccccccccc"):
                        pass
                return set(app.state.state["ws"])
    assert run(go()) == {"vidsgen-aaaaaaaaaaaa", "vidsgen-bbbbbbbbbbbb"}


async def wait_until(pred, timeout=5.0):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if pred():
            return True
        await asyncio.sleep(0.05)
    return False


def test_client_disconnect_closes_the_upstream_socket_and_frees_the_slot(fake):
    async def go():
        app = make_app(fake.url, px.Policy(max_ws=1))
        with ProxyServer(app) as p:
            async with wsconnect(p):
                assert await wait_until(lambda: len(fake._sockets.get(CID, [])) == 1)
            assert await wait_until(lambda: not fake._sockets.get(CID)), "the upstream socket was left open"
            assert await wait_until(lambda: not app.state.state["ws"]) and not app.state.state["ws_tasks"]
            async with wsconnect(p):                                                               # the slot is reusable
                pass
    run(go())


def test_upstream_disconnect_closes_the_client_socket_and_frees_the_slot(fake):
    async def go():
        app = make_app(fake.url, px.Policy(max_ws=1))
        with ProxyServer(app) as p:
            async with wsconnect(p) as ws:
                assert await wait_until(lambda: len(fake._sockets.get(CID, [])) == 1)
                fake.drop_websockets()
                with pytest.raises(ConnectionClosed):
                    async with asyncio.timeout(5):
                        await ws.recv()
            assert await wait_until(lambda: not app.state.state["ws"])
            async with wsconnect(p):
                pass
    run(go())


def test_the_websocket_is_refused_when_comfyui_is_down(fake):
    dead = f"http://127.0.0.1:{free_port()}"

    async def go():
        app = make_app(dead)
        with ProxyServer(app) as p:
            with pytest.raises(InvalidStatus):
                async with wsconnect(p):
                    pass
            assert not app.state.state["ws"]                                                       # the reserved slot was released
    run(go())


def test_client_frames_are_never_forwarded_and_flooding_closes_the_socket(fake):
    async def go():
        with ProxyServer(make_app(fake.url)) as p:
            async with wsconnect(p) as ws:
                assert await wait_until(lambda: len(fake._sockets.get(CID, [])) == 1)
                for i in range(3):
                    await ws.send(json.dumps({"type": "feature_flags", "data": {"x": i}}))
                await asyncio.sleep(0.4)
                assert fake.ws_received == []                                                      # nothing reached ComfyUI
                with pytest.raises(ConnectionClosed):
                    for i in range(60):
                        await ws.send("x")
                        await asyncio.sleep(0.01)
                    await asyncio.wait_for(ws.recv(), 3)
    run(go())


def test_an_oversized_client_frame_closes_the_socket(fake):
    async def go():
        with ProxyServer(make_app(fake.url)) as p:
            async with wsconnect(p) as ws:
                await ws.send("x" * 3000)
                with pytest.raises(ConnectionClosed):
                    async with asyncio.timeout(5):
                        await ws.recv()
    run(go())


def test_a_stalled_client_is_cut_off_instead_of_buffering_without_bound(fake, monkeypatch):
    """A client that cannot take a frame within ws_send_timeout is disconnected and everything is released. (uvicorn itself waits for
    the socket to become writable before accepting the next frame, so a slow reader makes send_text block: that block is what the
    timeout bounds. Here the slow reader is simulated deterministically.)"""
    real = px.WebSocket.send_text
    stall = {"on": False, "port": None}

    async def maybe_stalled(self, data):
        if stall["on"] and self.scope["server"][1] == stall["port"]:       # only the proxy's sockets (the fake ComfyUI shares the class)
            await asyncio.sleep(30)
        await real(self, data)
    monkeypatch.setattr(px.WebSocket, "send_text", maybe_stalled)

    async def go():
        app = make_app(fake.url, px.Policy(ws_send_timeout=0.5))
        with ProxyServer(app) as p:
            stall["port"] = p.port
            async with wsconnect(p) as ws, httpx.AsyncClient(base_url=p.url) as c:
                pid = new_id()
                await c.post("/prompt", json=body_for(pid=pid), headers=AUTH)
                await collect(ws)
                stall["on"] = True
                fake.inject(CID, json.dumps({"type": "progress", "data": {"prompt_id": pid, "value": 1, "max": 2, "node": "3"}}))
                released = await wait_until(lambda: not app.state.state["ws"] and not app.state.state["ws_tasks"], 8)
                stall["on"] = False
                assert await wait_until(lambda: not fake._sockets.get(CID), 5), "the upstream socket was left open"
                return released
    assert run(go()), "the proxy kept a stalled client connected"


# ====================================================================== the real provider and backend through the proxy (WebSocket live)
def test_the_provider_receives_live_progress_over_the_proxys_websocket(make_client, fake, video_bytes):
    app = make_app(fake.url)
    fake.step_delay = 0.05
    with ProxyServer(app) as p:
        c = e2e_client(make_client, p, comfyui_poll_interval=1.0)
        job_id = submit(c, resolution="832x480", num_frames=17, steps=20, fps=24, guidance_scale=5.0).json()["id"]
        seen = set()
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline:
            j = c.get(f"/api/jobs/{job_id}").json()
            seen.add(j["progress"])
            if j["status"] in ("completed", "failed", "cancelled"):
                break
            time.sleep(0.05)
        assert j["status"] == "completed", j["error"]
        assert c.get(f"/api/jobs/{job_id}/download").content == video_bytes
    assert app.state.state["stats"]["ws_events"] >= 20                                     # the events really came through the relay
    assert len([x for x in seen if 5 < x < 90]) >= 3                                       # live sampler steps, not just the end


def test_a_lost_submission_response_does_not_lose_the_job_through_the_proxy(make_client, fake):
    """End to end: ComfyUI queues the prompt, but the proxy times out waiting; the provider reconciles (GET only) and carries on."""
    fake.prompt_mode = "queue_then_hang"
    with ProxyServer(make_app(fake.url, px.Policy(read_timeout=1.0))) as p:
        c = e2e_client(make_client, p, comfyui_timeout=5)
        job = wait_for(c, submit(c).json()["id"], timeout=40)
    assert job["status"] == "completed", job["error"]
    assert len(prompts(fake)) == 1


# ====================================================================== streaming downloads
class Stream(httpx.AsyncByteStream):
    """A lazily generated upstream body that records how much was produced, whether it was closed, and can fail midway."""

    def __init__(self, total, chunk=1024 * 1024, fail_at=None, delay=0.0):
        self.total, self.chunk, self.fail_at, self.delay = total, chunk, fail_at, delay
        self.produced, self.closed = 0, False

    async def __aiter__(self):
        block = b"\x00" * self.chunk
        while self.produced < self.total:
            if self.fail_at is not None and self.produced >= self.fail_at:
                raise httpx.ReadError("upstream died")
            n = min(self.chunk, self.total - self.produced)
            self.produced += n
            yield block[:n]
            if self.delay:
                await asyncio.sleep(self.delay)

    async def aclose(self):
        self.closed = True


def video_upstream(stream, status=200, headers=None):
    def handler(request):
        h = {"content-type": "video/mp4", "content-length": str(stream.total), **(headers or {})}
        return httpx.Response(status, headers=h, stream=stream)
    return httpx.MockTransport(handler)


VIEW = f"/view?filename={JOB}_00001_.mp4&subfolder=vidsgen&type=output"


def test_a_large_video_streams_with_bounded_memory():
    total = 256 * 1024 * 1024
    stream = Stream(total)

    async def go():
        app = px.create_app("http://127.0.0.1:8188", TOKEN, WORKFLOWS, transport=video_upstream(stream))
        with ProxyServer(app) as p:
            async with httpx.AsyncClient(base_url=p.url, timeout=60) as c:
                tracemalloc.start()
                received = 0
                async with c.stream("GET", VIEW, headers=AUTH) as r:
                    assert r.status_code == 200 and int(r.headers["content-length"]) == total
                    async for chunk in r.aiter_bytes(1024 * 1024):
                        received += len(chunk)
                peak = tracemalloc.get_traced_memory()[1]
                tracemalloc.stop()
                return received, peak, app.state.state["inflight"], app.state.state["heavy"]
    received, peak, inflight, heavy = run(go())
    assert received == total and stream.closed
    assert peak < 48 * 1024 * 1024, f"peak traced memory {peak / 2**20:.0f} MiB for a {total / 2**20:.0f} MiB video"
    assert (inflight, heavy) == (0, 0)


def test_a_slow_client_does_not_make_the_proxy_read_ahead_without_bound():
    total = 128 * 1024 * 1024
    stream = Stream(total)

    async def go():
        app = px.create_app("http://127.0.0.1:8188", TOKEN, WORKFLOWS, transport=video_upstream(stream))
        lead = 0
        with ProxyServer(app) as p:
            async with httpx.AsyncClient(base_url=p.url, timeout=60) as c:
                async with c.stream("GET", VIEW, headers=AUTH) as r:
                    got = 0
                    async for chunk in r.aiter_bytes(1024 * 1024):
                        got += len(chunk)
                        if got in (8 * 1024 * 1024, 16 * 1024 * 1024, 24 * 1024 * 1024):
                            await asyncio.sleep(0.6)                                         # a slow reader
                            lead = max(lead, stream.produced - got)
                        if got >= 32 * 1024 * 1024:
                            break
        return lead
    lead = run(go())
    assert lead < 64 * 1024 * 1024, f"the proxy read {lead / 2**20:.0f} MiB ahead of a slow client"


def test_a_client_that_disconnects_mid_download_releases_the_upstream_and_the_slot():
    stream = Stream(512 * 1024 * 1024, delay=0.01)

    async def go():
        app = px.create_app("http://127.0.0.1:8188", TOKEN, WORKFLOWS, transport=video_upstream(stream))
        with ProxyServer(app) as p:
            async with httpx.AsyncClient(base_url=p.url, timeout=60) as c:
                async with c.stream("GET", VIEW, headers=AUTH) as r:
                    n = 0
                    async for chunk in r.aiter_bytes(1024 * 1024):
                        n += 1
                        if n == 3:
                            break                                                              # walk away
            ok = await wait_until(lambda: stream.closed and app.state.state["inflight"] == 0 and app.state.state["heavy"] == 0, 8)
            return ok, stream.produced
    ok, produced = run(go())
    assert ok, "the upstream stream or the concurrency slot leaked after the client left"
    assert produced < 100 * 1024 * 1024                                                        # and it stopped reading


def test_an_upstream_failure_midway_aborts_the_response_instead_of_looking_complete():
    stream = Stream(64 * 1024 * 1024, fail_at=8 * 1024 * 1024)

    async def go():
        app = px.create_app("http://127.0.0.1:8188", TOKEN, WORKFLOWS, transport=video_upstream(stream))
        with ProxyServer(app) as p:
            async with httpx.AsyncClient(base_url=p.url, timeout=30) as c:
                got = 0
                with pytest.raises(httpx.HTTPError):
                    async with c.stream("GET", VIEW, headers=AUTH) as r:
                        async for chunk in r.aiter_bytes(1024 * 1024):
                            got += len(chunk)
            ok = await wait_until(lambda: stream.closed and app.state.state["inflight"] == 0, 5)
            return got, ok
    got, ok = run(go())
    assert got < 64 * 1024 * 1024 and ok


def test_the_total_download_time_limit_aborts_rather_than_truncating():
    stream = Stream(64 * 1024 * 1024, delay=0.2)

    async def go():
        policy = px.Policy(view_total_timeout=0.5)
        app = px.create_app("http://127.0.0.1:8188", TOKEN, WORKFLOWS, policy, transport=video_upstream(stream))
        with ProxyServer(app) as p:
            async with httpx.AsyncClient(base_url=p.url, timeout=30) as c:
                with pytest.raises(httpx.HTTPError):
                    async with c.stream("GET", VIEW, headers=AUTH) as r:
                        async for _ in r.aiter_bytes(1024 * 1024):
                            pass
            return await wait_until(lambda: stream.closed and app.state.state["inflight"] == 0, 5)
    assert run(go())


def test_partial_content_and_range_headers_pass_through():
    stream = Stream(1024)
    with TestClient(px.create_app("http://127.0.0.1:8188", TOKEN, WORKFLOWS,
                                  transport=video_upstream(stream, 206, {"content-range": "bytes 0-1023/4096", "accept-ranges": "bytes"}))) as c:
        r = c.get(VIEW, headers={**AUTH, "Range": "bytes=0-1023"})
    assert r.status_code == 206 and r.headers["content-range"] == "bytes 0-1023/4096" and len(r.content) == 1024


def test_download_slots_are_enforced_and_released():
    stream = Stream(64 * 1024 * 1024, delay=0.05)

    async def go():
        app = px.create_app("http://127.0.0.1:8188", TOKEN, WORKFLOWS, px.Policy(max_heavy=1), transport=video_upstream(stream))
        with ProxyServer(app) as p:
            async with httpx.AsyncClient(base_url=p.url, timeout=30) as c:
                async with c.stream("GET", VIEW, headers=AUTH) as first:
                    await first.aiter_bytes(1024).__anext__()
                    second = await c.get(VIEW, headers=AUTH)
                    light = await c.get("/queue", headers=AUTH)                         # light calls are unaffected (upstream mock 200)
                ok = await wait_until(lambda: app.state.state["heavy"] == 0, 5)
                return second.status_code, light.status_code, ok
    second, light, ok = run(go())
    assert second == 429 and ok and light != 429


# ====================================================================== the optional dead-man switch
def tick(client, app):
    return client.portal.call(app.state.dead_man_tick)


def dm_app(fake, clock, **kw):
    policy = px.Policy(dead_man_seconds=kw.pop("seconds", 200.0), dead_man_interval=3600.0, **kw)
    return make_clock_app(fake, policy, clock)


def make_clock_app(fake, policy, clock):
    return px.create_app(fake.url, TOKEN, WORKFLOWS, policy, clock=clock)


def test_the_dead_man_switch_is_disabled_by_default(fake):
    clock = Clock()
    app = make_clock_app(fake, px.Policy(), clock)
    assert px.Policy().dead_man_seconds == 0
    with TestClient(app) as c:
        fake.default_scenario = "hang"
        assert c.post("/prompt", json=body_for(), headers=AUTH).status_code == 200
        clock.t += 10 ** 7
        assert tick(c, app) == 0 and app.state.state["reaper"] is None
    assert fake.interrupts == []


def test_enabling_it_requires_a_meaningful_inactivity_period():
    with pytest.raises(ValueError, match="at least 120"):
        px.create_app("http://127.0.0.1:8188", TOKEN, WORKFLOWS, px.Policy(dead_man_seconds=30))
    px.create_app("http://127.0.0.1:8188", TOKEN, WORKFLOWS, px.Policy(dead_man_seconds=120))


def test_the_cli_keeps_it_off_unless_asked_and_enforces_the_minimum(monkeypatch, capsys):
    monkeypatch.setenv(px.TOKEN_ENV, TOKEN)
    seen = {}
    monkeypatch.setattr(px, "create_app", lambda *a, **k: seen.setdefault("policy", a[3]) and (_ for _ in ()).throw(SystemExit(0)))
    with pytest.raises(SystemExit):
        px.main(["--workflows-dir", str(WORKFLOWS)])
    assert seen["policy"].dead_man_seconds == 0
    monkeypatch.undo()
    monkeypatch.setenv(px.TOKEN_ENV, TOKEN)
    assert px.main(["--workflows-dir", str(WORKFLOWS), "--dead-man-seconds", "30"]) == 2


def test_a_silent_backend_gets_its_running_prompt_interrupted_with_a_targeted_call(fake):
    fake.default_scenario = "hang"
    clock = Clock()
    app = dm_app(fake, clock)
    with TestClient(app) as c:
        pid = new_id()
        assert c.post("/prompt", json=body_for(pid=pid), headers=AUTH).status_code == 200
        deadline = time.monotonic() + 5
        while fake._running != pid and time.monotonic() < deadline:
            time.sleep(0.02)
        clock.t += 199
        assert tick(c, app) == 0                                    # not silent long enough
        clock.t += 2
        assert tick(c, app) == 1
        deadline = time.monotonic() + 5
        while not fake.interrupts and time.monotonic() < deadline:
            time.sleep(0.02)
        assert fake.interrupts == [pid]                              # targeted, never a global interrupt
        assert c.post("/prompt", json=body_for(pid=pid), headers=AUTH).status_code == 409     # and it cannot be resubmitted
        assert tick(c, app) == 0


def test_a_queued_prompt_is_removed_from_the_queue_not_interrupted(fake):
    fake.paused = True
    clock = Clock()
    app = dm_app(fake, clock)
    with TestClient(app) as c:
        pid = new_id()
        c.post("/prompt", json=body_for(pid=pid), headers=AUTH)
        clock.t += 500
        assert tick(c, app) == 1
    posts = [r["body"] for r in fake.requests if r["method"] == "POST" and r["path"] == "/queue"]
    assert posts == [{"delete": [pid]}] and fake.interrupts == []


def test_a_live_backend_is_never_interrupted_for_a_long_job(fake):
    """Any authenticated request counts as life, and so does an open WebSocket: a long job with a live backend is left alone."""
    fake.default_scenario = "hang"
    clock = Clock()
    app = dm_app(fake, clock, seconds=200.0)
    with TestClient(app) as c:
        pid = new_id()
        c.post("/prompt", json=body_for(pid=pid), headers=AUTH)
        for _ in range(30):                                          # 30 x 150 s = 75 minutes of job, polled every 150 s
            clock.t += 150
            assert c.get("/queue", headers=AUTH).status_code == 200  # the backend's progress poll
            assert tick(c, app) == 0
        assert fake.interrupts == []
        clock.t += 1000                                              # silence, BUT a WebSocket is open
        with c.websocket_connect(f"/ws?clientId={CID}", headers=AUTH):
            assert tick(c, app) == 0
            clock.t += 10 ** 6
            assert tick(c, app) == 0
        assert fake.interrupts == []
        assert tick(c, app) == 0                                      # just disconnected: the silence clock restarts now
        clock.t += 201
        assert tick(c, app) == 1 and fake.interrupts == [pid]


def test_unauthenticated_requests_do_not_count_as_a_live_backend(fake):
    fake.default_scenario = "hang"
    clock = Clock()
    app = dm_app(fake, clock)
    with TestClient(app) as c:
        pid = new_id()
        c.post("/prompt", json=body_for(pid=pid), headers=AUTH)
        clock.t += 150
        c.get("/queue", headers={"Authorization": "Bearer wrong"})
        clock.t += 100
        assert tick(c, app) == 1


def test_it_only_touches_prompts_that_this_proxy_forwarded(fake):
    fake.default_scenario = "hang"
    clock = Clock()
    app = dm_app(fake, clock)
    with TestClient(app) as c:
        # someone else's job, submitted straight to ComfyUI (not through the proxy)
        foreign = new_id()
        with httpx.Client(base_url=fake.url) as direct:
            direct.post("/prompt", json={"prompt": graph(), "client_id": "other", "prompt_id": foreign})
        deadline = time.monotonic() + 5
        while fake._running != foreign and time.monotonic() < deadline:
            time.sleep(0.02)
        clock.t += 10 ** 6
        assert tick(c, app) == 0 and fake.interrupts == []


def test_finished_prompts_are_left_alone(fake):
    clock = Clock()
    app = dm_app(fake, clock)
    with TestClient(app) as c:
        pid = new_id()
        c.post("/prompt", json=body_for(pid=pid), headers=AUTH)
        deadline = time.monotonic() + 10
        while pid not in fake.history and time.monotonic() < deadline:
            time.sleep(0.02)
        clock.t += 10_000                                            # long silence, but inside the registry's 6 h memory
        assert tick(c, app) == 0 and fake.interrupts == [] and app.state.registry.entries[pid].finished is True
        assert app.state.registry.begin(pid, app.state.registry.entries[pid].digest, CLIENT)[0] == "accepted"   # still idempotent


def test_an_unreachable_comfyui_during_a_reap_is_survived():
    clock = Clock()
    dead = f"http://127.0.0.1:{free_port()}"
    app = px.create_app(dead, TOKEN, WORKFLOWS, px.Policy(dead_man_seconds=200, dead_man_interval=3600), clock=clock)
    with TestClient(app) as c:
        app.state.registry.begin("x" * 0 + new_id(), "d", "c")
        pid = next(iter(app.state.registry.entries))
        app.state.registry.resolve(pid, 200, {})
        clock.t += 1000
        assert tick(c, app) == 0 and app.state.registry.entries[pid].state == "accepted"       # not marked cancelled: it will be retried


def test_the_background_loop_reaps_an_orphan_and_leaves_an_active_backend_alone(fake):
    """Real time: threshold 1.5 s, check every 0.2 s."""
    fake.default_scenario = "hang"
    policy = px.Policy(dead_man_seconds=1.5, dead_man_min=0.1, dead_man_interval=0.2)
    with TestClient(px.create_app(fake.url, TOKEN, WORKFLOWS, policy)) as c:
        pid = new_id()
        c.post("/prompt", json=body_for(pid=pid), headers=AUTH)
        deadline = time.monotonic() + 4                              # an active backend: polls every 0.3 s for 4 s
        while time.monotonic() < deadline:
            c.get("/queue", headers=AUTH)
            time.sleep(0.3)
        assert fake.interrupts == []
        deadline = time.monotonic() + 8                              # now it goes silent
        while not fake.interrupts and time.monotonic() < deadline:
            time.sleep(0.1)
    assert fake.interrupts == [pid]


# ====================================================================== static guarantees
def test_stage_two_adds_no_tunnel_and_no_new_listening_surface():
    from pathlib import Path
    src = Path(px.__file__).read_text()
    for forbidden in ("cloudflared", "trycloudflare", "ngrok", "0.0.0.0"):
        assert forbidden not in src.replace('"0.0.0.0"', "")
    assert 'proxy=None' in src and "ws://{host_port}" in src                                     # the upstream socket is loopback, no env proxy
