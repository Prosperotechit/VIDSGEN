"""Security tests for scripts/inference_proxy.py (Phase 6C Stage 1): the authenticated gate in front of ComfyUI.

Everything runs locally against FakeComfyUI or a mocked upstream. No tunnel, no external network, no GPU. The proxy is also
exercised end to end with the real ComfyUIProvider and the real backend, so the allowlist is checked against what the
successful Phase 6B workflow actually sends.
"""

from __future__ import annotations

import asyncio
import copy
import importlib.util
import json
import logging
import re
import socket
import sys
import threading
import time
import uuid
from pathlib import Path

import httpx
import pytest
import uvicorn
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from app.core.config import REPO_ROOT, Settings
from app.jobs.envelope import PROFILES
from app.jobs.models import GenerationMode, GenerationParams, new_job_id
from app.providers.mock_video import encode_mock_video
from app.workflows import GraphValues, WorkflowTemplates
from comfyui_support import FakeComfyUI
from conftest import submit, wait_for

SCRIPT = REPO_ROOT / "scripts" / "inference_proxy.py"
_spec = importlib.util.spec_from_file_location("inference_proxy", SCRIPT)
px = importlib.util.module_from_spec(_spec)
sys.modules["inference_proxy"] = px
_spec.loader.exec_module(px)

WORKFLOWS = REPO_ROOT / "workflows" / "comfyui"
TOKEN = "".join(["tok", uuid.uuid4().hex, uuid.uuid4().hex])               # built at runtime: never a literal secret in the repo
AUTH = {"Authorization": f"Bearer {TOKEN}"}
TEMPLATES = WorkflowTemplates(WORKFLOWS)
PID = str(uuid.uuid4())
CLIENT = "vidsgen-0123456789ab"


def graph(**over) -> dict:
    """Exactly what ComfyUIProvider sends for the verified configuration (832x480, 17 frames, 20 steps, 24 fps, CFG 5.0)."""
    v = dict(prompt="a red fox running through snow", negative_prompt="", width=832, height=480, num_frames=17, fps=24, steps=20,
             guidance_scale=5.0, seed=424242, filename_prefix=f"vidsgen/{new_job_id()}")
    v.update(over)
    return TEMPLATES.build(GenerationMode.t2v, GraphValues(**v))


def prompt_body(g=None, pid=None, **over) -> dict:
    """A valid submission. Each call gets a fresh prompt_id (the proxy is idempotent per id) unless one is given."""
    body = {"prompt": g if g is not None else graph(), "client_id": CLIENT, "prompt_id": pid or str(uuid.uuid4())}
    body.update(over)
    return body


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture(scope="module")
def video_bytes(tmp_path_factory) -> bytes:
    p = tmp_path_factory.mktemp("v") / "v.mp4"
    encode_mock_video(p, width=832, height=480, num_frames=17, fps=24, seed=3, prompt="proxy")
    return p.read_bytes()


@pytest.fixture
def fake(video_bytes):
    server = FakeComfyUI(video_bytes).start()
    yield server
    server.stop()


def make_app(upstream: str, policy=None, transport=None):
    return px.create_app(upstream, TOKEN, WORKFLOWS, policy, transport)


@pytest.fixture
def client(fake):
    with TestClient(make_app(fake.url)) as c:
        yield c


def upstream_paths(fake) -> list[tuple[str, str]]:
    return [(r["method"], r["path"]) for r in fake.requests]


class ProxyServer:
    """The proxy as a real server on a random loopback port (for the provider/backend end-to-end tests)."""

    def __init__(self, app, port=None):
        self.app = app
        self.port = port or free_port()
        self.server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=self.port, lifespan="on", timeout_graceful_shutdown=1,
                                                    **px.uvicorn_options()))
        self.thread = threading.Thread(target=self.server.run, daemon=True)

    def __enter__(self):
        self.thread.start()
        deadline = time.monotonic() + 10
        while not self.server.started:
            assert time.monotonic() < deadline, "proxy did not start"
            time.sleep(0.01)
        self.url = f"http://127.0.0.1:{self.port}"
        self.ws = f"ws://127.0.0.1:{self.port}"
        return self

    def __exit__(self, *a):
        self.server.should_exit = True
        self.thread.join(timeout=10)


# ====================================================================== startup / configuration
def test_a_short_token_is_refused(fake):
    with pytest.raises(ValueError, match="at least 32 characters"):
        px.create_app(fake.url, "x" * 31, WORKFLOWS)
    px.create_app(fake.url, "x" * 32, WORKFLOWS)


@pytest.mark.parametrize("url", ["https://abc.trycloudflare.com", "http://0.0.0.0:8188", "http://192.168.1.4:8188", "http://example.com:8188",
                                 "http://127.0.0.1", "https://127.0.0.1:8188", ""])
def test_only_a_loopback_upstream_is_accepted(url):
    with pytest.raises(ValueError, match="loopback"):
        px.create_app(url, TOKEN, WORKFLOWS)


def test_the_cli_refuses_a_non_loopback_listen_address(capsys, monkeypatch):
    monkeypatch.setenv(px.TOKEN_ENV, TOKEN)
    assert px.main(["--host", "0.0.0.0", "--workflows-dir", str(WORKFLOWS)]) == 2
    assert "loopback only" in capsys.readouterr().err


def test_the_token_comes_only_from_the_environment_and_is_never_echoed(capsys, monkeypatch):
    with pytest.raises(SystemExit):                                           # there is no --token option
        px.main(["--token", TOKEN, "--workflows-dir", str(WORKFLOWS)])
    capsys.readouterr()
    monkeypatch.delenv(px.TOKEN_ENV, raising=False)
    assert px.main(["--workflows-dir", str(WORKFLOWS)]) == 2
    monkeypatch.setenv(px.TOKEN_ENV, "short-secret-value")
    assert px.main(["--workflows-dir", str(WORKFLOWS)]) == 2
    err = capsys.readouterr().err
    assert "short-secret-value" not in err and "at least 32" in err or "invalid configuration" in err


def test_the_defaults_equal_the_backends_verified_t4_profile():
    p, e = px.Policy(), PROFILES["t4_verified"]
    assert {f"{w}x{h}" for w, h in p.resolutions} == set(e["resolutions"])
    assert (p.max_frames, p.max_steps) == (e["max_frames"], e["max_steps"])
    fields = GenerationParams.model_fields
    assert (p.fps[0], p.fps[1]) == (fields["fps"].metadata[0].ge, fields["fps"].metadata[1].le)
    assert p.cfg == (fields["guidance_scale"].metadata[0].ge, fields["guidance_scale"].metadata[1].le)
    assert p.max_text_chars == Settings(_env_file=None).max_prompt_chars and p.min_frames == 17


def test_the_patterns_match_what_the_backend_and_comfyui_really_produce():
    assert px.PREFIX_RE.match(f"vidsgen/{new_job_id()}")
    assert px.VIEW_NAME_RE.match(f"{new_job_id()}_00001_.mp4") and px.VIEW_NAME_RE.match(f"{new_job_id()}_00002.mp4")
    assert px.CLIENT_ID_RE.match(f"vidsgen-{uuid.uuid4().hex[:12]}") and px.UUID_RE.match(str(uuid.uuid4()))
    assert not px.UUID_RE.match(str(uuid.uuid4()).upper())


# ====================================================================== authentication
@pytest.mark.parametrize("headers", [
    {}, {"Authorization": ""}, {"Authorization": "Bearer"}, {"Authorization": "Bearer "}, {"Authorization": f"Bearer {TOKEN}x"},
    {"Authorization": f"Bearer {TOKEN[:-1]}"}, {"Authorization": f"Token {TOKEN}"}, {"Authorization": f"Basic {TOKEN}"},
    {"Authorization": TOKEN}, {"Authorization": "Bearer " + "A" * 600}, {"X-Api-Key": TOKEN}, {"Authorization": f"Bearer  {TOKEN} extra"},
])
def test_unauthorized_requests_get_an_identical_information_free_401(client, fake, headers):
    for path in ("/system_stats", "/prompt", "/does-not-exist", "/free", "/view"):
        r = client.get(path, headers=headers)
        assert r.status_code == 401 and r.json() == {"error": "unauthorized"}, path
        assert r.headers["www-authenticate"] == "Bearer" and "server" not in r.headers
    assert fake.requests == []                                                # nothing reached ComfyUI


def test_a_401_reveals_nothing_about_which_routes_exist(client):
    a, b = client.get("/system_stats"), client.get("/definitely/not/a/route")
    assert (a.status_code, a.content, dict(a.headers)) == (b.status_code, b.content, dict(b.headers))


def test_valid_tokens_work_with_any_scheme_case(client):
    assert client.get("/system_stats", headers=AUTH).status_code == 200
    assert client.get("/system_stats", headers={"Authorization": f"bearer {TOKEN}"}).status_code == 200


def test_the_comparison_is_constant_time_and_always_runs(client, monkeypatch):
    calls = []
    real = px.hmac.compare_digest
    monkeypatch.setattr(px.hmac, "compare_digest", lambda a, b: calls.append(1) or real(a, b))
    client.get("/system_stats")                                               # no header at all
    client.get("/system_stats", headers={"Authorization": "garbage"})
    client.get("/system_stats", headers=AUTH)
    assert len(calls) == 3


def test_failed_attempts_are_throttled_but_a_valid_token_is_never_delayed(fake):
    with TestClient(make_app(fake.url, px.Policy(auth_failures_per_minute=5))) as c:
        codes = [c.get("/system_stats", headers={"Authorization": "Bearer wrong"}).status_code for _ in range(8)]
        assert codes[:4] == [401] * 4 and codes[4:] == [429] * 4
        assert c.get("/system_stats", headers={"Authorization": "Bearer wrong"}).headers["retry-after"] == "60"
        assert c.get("/system_stats", headers=AUTH).status_code == 200


def test_the_token_never_appears_in_logs_or_responses(fake, caplog):
    caplog.set_level(logging.DEBUG)
    bad = "bad-" + uuid.uuid4().hex
    with TestClient(make_app(fake.url)) as c:
        rs = [c.get("/system_stats", headers={"Authorization": f"Bearer {bad}"}), c.get("/system_stats", headers=AUTH),
              c.post("/prompt", json=prompt_body(), headers=AUTH), c.get(f"/history/{PID}", headers=AUTH),
              c.post("/prompt", content=b"{not json", headers=AUTH)]
    logs = caplog.text
    assert TOKEN not in logs and bad not in logs
    assert not any(TOKEN in r.text or bad in r.text or TOKEN in str(r.headers) for r in rs)


def test_the_token_is_not_forwarded_to_comfyui(client, fake):
    client.get("/system_stats", headers=AUTH)
    client.post("/prompt", json=prompt_body(), headers=AUTH)
    assert fake.requests and all(r["auth"] is None for r in fake.requests)


# ====================================================================== the allowlist
@pytest.mark.parametrize("method,path,body", [
    ("GET", "/system_stats", None), ("GET", "/object_info/KSampler", None), ("GET", "/object_info/LoadImage", None),
    ("GET", "/models/vae", None), ("GET", "/models/diffusion_models", None), ("GET", "/models/text_encoders", None),
    ("GET", "/experiment/models/vae", None), ("GET", "/queue", None), ("GET", f"/history/{PID}", None),
    ("POST", "/queue", {"delete": [PID]}), ("POST", "/history", {"delete": [PID]}), ("POST", "/interrupt", {"prompt_id": PID}),
])
def test_every_route_the_provider_needs_is_allowed(client, method, path, body):
    r = client.request(method, path, json=body, headers=AUTH)
    assert r.status_code == 200, (r.status_code, r.text)


@pytest.mark.parametrize("method,path", [
    ("POST", "/free"), ("POST", "/upload/image"), ("POST", "/upload/mask"), ("GET", "/view_metadata/vae"), ("GET", "/extensions"),
    ("GET", "/embeddings"), ("GET", "/default_workflow"), ("GET", "/history"), ("GET", "/api/jobs"), ("GET", f"/api/jobs/{PID}"),
    ("POST", "/api/jobs/cancel"), ("POST", f"/api/jobs/{PID}/cancel"), ("GET", "/"), ("GET", "/prompt"), ("PUT", "/queue"),
    ("GET", "/object_info"), ("GET", "/object_info/NotAWanNode"), ("GET", "/object_info/KSampler?x=1"), ("GET", "/models"),
    ("GET", "/models/loras"), ("GET", "/models/checkpoints"), ("GET", "/experiment/models/loras"), ("GET", "/models/../prompt"),
    ("POST", "//prompt"), ("POST", "/prompt/"), ("GET", "/view/x"), ("GET", f"/history/{PID}/extra"), ("GET", "/history/..%2f..%2fetc"),
    ("GET", "/%2e%2e/system_stats"), ("GET", "/system_stats?x=1"), ("GET", "/queue?clear=1"), ("DELETE", "/queue"), ("PATCH", "/prompt"),
    ("HEAD", "/system_stats"), ("OPTIONS", "/prompt"), ("GET", "/features"), ("GET", "/userdata?dir=x"), ("GET", "/internal/logs"),
    ("GET", "/api/v2/anything"), ("POST", "/manager/reboot"),
])
def test_everything_else_is_404_and_never_reaches_comfyui(client, fake, method, path):
    r = client.request(method, path, headers=AUTH)
    assert r.status_code == 404, (r.status_code, r.text)
    assert fake.requests == []
    assert r.content in (b'{"error":"not_found"}', b"")


# ====================================================================== POST /prompt: strict workflow validation
def test_the_workflow_the_backend_really_sends_is_accepted_and_forwarded_minimally(client, fake):
    body = prompt_body(pid=PID)
    r = client.post("/prompt", json=body, headers=AUTH)
    assert r.status_code == 200 and set(r.json()) == {"prompt_id", "number", "node_errors"} and r.json()["prompt_id"] == PID
    sent = [p for p in fake.requests if p["path"] == "/prompt"]
    assert len(sent) == 1 and set(sent[0]["body"]) == {"prompt", "client_id", "prompt_id"}
    assert sent[0]["body"] == body


@pytest.mark.parametrize("frames", [17])
@pytest.mark.parametrize("steps", [1, 8, 20])
@pytest.mark.parametrize("fps", [8, 24, 30])
@pytest.mark.parametrize("cfg", [1.0, 5.0, 7, 15.0])
@pytest.mark.parametrize("text", ["a fox", "多语言 prompt\nwith a newline\tand tab", "x" * 2000])
def test_every_value_inside_the_envelope_is_accepted(client, frames, steps, fps, cfg, text):
    g = graph(num_frames=frames, steps=steps, fps=fps, guidance_scale=cfg, prompt=text, negative_prompt=text, seed=2**32 - 1)
    assert client.post("/prompt", json=prompt_body(g), headers=AUTH).status_code == 200


def test_an_empty_negative_prompt_and_seed_zero_are_accepted(client):
    assert client.post("/prompt", json=prompt_body(graph(negative_prompt="", seed=0)), headers=AUTH).status_code == 200


def assert_rejected(client, fake, body=None, raw=None, headers=AUTH, reason=None):
    r = client.post("/prompt", **({"json": body} if raw is None else {"content": raw}), headers=headers)
    assert r.status_code == 400, (r.status_code, r.text)
    err = r.json()["error"]
    assert err["type"] == "proxy_rejected" and r.json()["node_errors"] == {}          # shaped like ComfyUI's, so the provider shows it
    if reason:
        assert reason in err["message"], err["message"]
    assert [p for p in fake.requests if p["path"] == "/prompt"] == []                    # never queued
    return err["message"]


@pytest.mark.parametrize("over,reason", [
    (dict(width=1280, height=704), "resolution 1280x704"), (dict(width=640, height=640), "resolution 640x640"),
    (dict(width=704, height=1280), "resolution 704x1280"), (dict(width=832, height=832), "resolution 832x832"),
    (dict(width=480, height=832), "resolution 480x832"),
    (dict(num_frames=21), "num_frames"), (dict(num_frames=49), "num_frames"), (dict(num_frames=121), "num_frames"),
    (dict(num_frames=18), "num_frames"), (dict(num_frames=13), "num_frames"), (dict(num_frames=0), "num_frames"),
    (dict(steps=21), "steps"), (dict(steps=30), "steps"), (dict(steps=0), "steps"), (dict(steps=-1), "steps"),
    (dict(fps=7), "fps"), (dict(fps=31), "fps"), (dict(fps=60), "fps"),
    (dict(guidance_scale=0.5), "guidance_scale"), (dict(guidance_scale=15.5), "guidance_scale"), (dict(guidance_scale=-5), "guidance_scale"),
    (dict(seed=-1), "seed"), (dict(seed=2**32), "seed"), (dict(seed=2**64), "seed"),
    (dict(prompt=""), "prompt"), (dict(prompt="   "), "prompt"), (dict(prompt="x" * 2001), "prompt"), (dict(prompt="bad\x00text"), "prompt"),
    (dict(negative_prompt="y" * 2001), "negative_prompt"), (dict(negative_prompt="bad\x07"), "negative_prompt"),
    (dict(filename_prefix="video/ComfyUI"), "filename_prefix"), (dict(filename_prefix="vidsgen/../../etc/passwd"), "filename_prefix"),
    (dict(filename_prefix="/etc/cron.d/x"), "filename_prefix"), (dict(filename_prefix="vidsgen/job_" + "a" * 23), "filename_prefix"),
    (dict(filename_prefix="other/job_" + "a" * 24), "filename_prefix"), (dict(filename_prefix=f"vidsgen/{new_job_id()}/../x"), "filename_prefix"),
])
def test_values_outside_the_verified_envelope_are_rejected_without_reaching_comfyui(client, fake, over, reason):
    # build with the real template filler, then force the (possibly invalid) value in: validation must not depend on the filler
    g = graph()
    nodes = {"width": ("55", "width"), "height": ("55", "height"), "num_frames": ("55", "length"), "steps": ("3", "steps"),
             "fps": ("57", "fps"), "guidance_scale": ("3", "cfg"), "seed": ("3", "seed"), "prompt": ("6", "text"),
             "negative_prompt": ("7", "text"), "filename_prefix": ("58", "filename_prefix")}
    for k, v in over.items():
        n, i = nodes[k]
        g[n]["inputs"][i] = v
    assert_rejected(client, fake, prompt_body(g), reason=reason)


@pytest.mark.parametrize("node,field,value", [
    ("3", "steps", True), ("3", "steps", 20.0), ("3", "steps", "20"), ("3", "steps", None), ("3", "steps", [20]),
    ("3", "seed", 1.5), ("3", "seed", True), ("3", "cfg", True), ("3", "cfg", "5"), ("3", "cfg", float("nan")), ("3", "cfg", float("inf")),
    ("55", "length", True), ("55", "length", 17.0), ("55", "width", "832"), ("55", "height", 480.0), ("57", "fps", True), ("57", "fps", 24.0),
    ("6", "text", 5), ("6", "text", None), ("6", "text", ["a"]), ("7", "text", {"a": 1}), ("58", "filename_prefix", 7),
])
def test_wrongly_typed_values_are_rejected(client, fake, node, field, value):
    g = graph()
    g[node]["inputs"][field] = value
    body = json.dumps(prompt_body(g), allow_nan=True)
    assert_rejected(client, fake, raw=body.encode())


@pytest.mark.parametrize("tamper", [
    lambda g: g["37"]["inputs"].update(unet_name="evil_model.safetensors"),
    lambda g: g["37"]["inputs"].update(unet_name="../../../etc/passwd"),
    lambda g: g["37"]["inputs"].update(weight_dtype="fp8_e4m3fn"),
    lambda g: g["38"]["inputs"].update(clip_name="other.safetensors"),
    lambda g: g["38"]["inputs"].update(type="stable_diffusion"),
    lambda g: g["39"]["inputs"].update(vae_name="wan_2.1_vae.safetensors"),
    lambda g: g["48"]["inputs"].update(shift=1),
    lambda g: g["55"]["inputs"].update(batch_size=2),
    lambda g: g["3"]["inputs"].update(sampler_name="euler"),
    lambda g: g["3"]["inputs"].update(scheduler="karras"),
    lambda g: g["3"]["inputs"].update(denoise=0.5),
    lambda g: g["3"]["inputs"].update(positive=["7", 0]),
    lambda g: g["8"]["inputs"].update(samples=["55", 0]),
    lambda g: g["58"]["inputs"].update(format="webm"),
    lambda g: g["58"]["inputs"].update({"format.codec": "vp9"}),
    lambda g: g["58"]["inputs"].update(codec="vp9"),
    lambda g: g["57"]["inputs"].update(codec="h264"),
    lambda g: g["3"]["inputs"].update(extra_input=1),
    lambda g: g["3"]["inputs"].pop("denoise"),
    lambda g: g["58"].update(class_type="SaveImage"),
    lambda g: g["58"].update(class_type="PreviewImage"),
    lambda g: g["8"].update(class_type="VAEDecodeTiled"),
    lambda g: g["3"].update(extra="x"),
    lambda g: g["3"].pop("_meta"),
    lambda g: g["3"]["_meta"].update(title="x"),
    lambda g: g.update({"99": {"class_type": "SaveImage", "inputs": {"images": ["8", 0], "filename_prefix": "/etc/x"}, "_meta": {}}}),
    lambda g: g.update({"99": {"class_type": "LoadImage", "inputs": {"image": "../../x.png"}, "_meta": {}}}),
    lambda g: g.pop("8"),
    lambda g: g.pop("58"),
    lambda g: g.clear(),
    lambda g: g.update({str(100 + i): copy.deepcopy(g["6"]) for i in range(70)}),
])
def test_any_change_to_the_allowed_workflow_is_rejected(client, fake, tamper):
    g = graph()
    tamper(g)
    msg = assert_rejected(client, fake, prompt_body(g))
    assert "evil" not in msg and "passwd" not in msg and "/etc" not in msg                # the message never echoes the input


def test_image_to_video_graphs_are_not_accepted_in_this_stage(client, fake):
    i2v = TEMPLATES.build(GenerationMode.i2v, GraphValues(prompt="a", negative_prompt="", width=832, height=480, num_frames=17, fps=24, steps=20,
                                                          guidance_scale=5.0, seed=1, filename_prefix=f"vidsgen/{new_job_id()}", image="vidsgen/job_x.png"))
    assert_rejected(client, fake, prompt_body(i2v), reason="nodes differ")


@pytest.mark.parametrize("extra", [{"extra_data": {"x": 1}}, {"number": -1}, {"partial_execution_targets": ["58"]},
                                   {"front": True}, {"extra_data": {"api_key_comfy_org": "x"}}])
def test_extra_top_level_fields_are_rejected(client, fake, extra):
    msg = assert_rejected(client, fake, prompt_body(**extra))
    assert not any(k in msg for k in extra)


@pytest.mark.parametrize("body", [
    {"prompt": graph(), "client_id": CLIENT},                                             # no prompt_id
    {"prompt": graph(), "prompt_id": PID},                                                # no client_id
    {"client_id": CLIENT, "prompt_id": PID},                                              # no prompt
    {"prompt": graph(), "client_id": CLIENT, "prompt_id": PID.upper()},
    {"prompt": graph(), "client_id": CLIENT, "prompt_id": "not-a-uuid"},
    {"prompt": graph(), "client_id": CLIENT, "prompt_id": 5},
    {"prompt": graph(), "client_id": "bad id/..", "prompt_id": PID},
    {"prompt": graph(), "client_id": "x" * 65, "prompt_id": PID},
    {"prompt": graph(), "client_id": 7, "prompt_id": PID},
    {"prompt": "a string", "client_id": CLIENT, "prompt_id": PID}, {"prompt": [], "client_id": CLIENT, "prompt_id": PID},
    {"prompt": None, "client_id": CLIENT, "prompt_id": PID},
])
def test_malformed_requests_are_rejected(client, fake, body):
    assert_rejected(client, fake, body)


@pytest.mark.parametrize("raw", [b"", b"{not json", b"[1,2]", b'"text"', b"null", b"\xff\xfe\x00", b'{"prompt": NaN}', b'{"a": Infinity}'])
def test_bodies_that_are_not_a_json_object_are_rejected(client, fake, raw):
    assert_rejected(client, fake, raw=raw)


def test_the_rejection_message_is_readable_through_the_real_provider_error_mapping():
    from app.providers.comfyui import ComfyUIProvider
    body = {"error": {"type": "proxy_rejected", "message": "steps must be an integer between 1 and 20", "details": "", "extra_info": {}}, "node_errors": {}}
    err = ComfyUIProvider._rejection(type("P", (), {"workflows": TEMPLATES})(), body)
    assert err.code == "comfyui_rejected" and "steps must be an integer between 1 and 20" in err.message


# ====================================================================== dangerous operations
@pytest.mark.parametrize("body", [{"clear": True}, {"clear": False}, {"delete": [PID], "clear": True}, {}, {"delete": []},
                                  {"delete": ["not-a-uuid"]}, {"delete": [PID.upper()]}, {"delete": [PID] * 11}, {"delete": PID}, {"delete": [1]},
                                  {"delete": [PID], "extra": 1}, [PID], "clear"])
@pytest.mark.parametrize("path", ["/queue", "/history"])
def test_clearing_and_malformed_deletes_are_refused(client, fake, path, body):
    r = client.post(path, json=body, headers=AUTH)
    assert r.status_code == 400
    assert [x for x in fake.requests if x["method"] == "POST"] == []                      # ComfyUI never saw it


def test_a_valid_delete_forwards_only_the_ids(client, fake):
    other = str(uuid.uuid4())
    assert client.post("/queue", json={"delete": [PID, other]}, headers=AUTH).status_code == 200
    posted = [x for x in fake.requests if x["method"] == "POST" and x["path"] == "/queue"]
    assert posted[0]["body"] == {"delete": [PID, other]}


@pytest.mark.parametrize("body,raw", [({}, None), (None, b""), ({"prompt_id": None}, None), ({"prompt_id": ""}, None), ({"prompt_id": "x"}, None),
                                      ({"prompt_id": PID, "extra": 1}, None), ({"prompt_id": [PID]}, None), (None, b"{bad")])
def test_unrestricted_interrupts_are_refused(client, fake, body, raw):
    r = client.post("/interrupt", **({"json": body} if raw is None else {"content": raw}), headers=AUTH)
    assert r.status_code == 400
    assert fake.interrupts == [] and [x for x in fake.requests if x["path"] == "/interrupt"] == []


def test_a_targeted_interrupt_is_forwarded_with_its_id_only(client, fake):
    assert client.post("/interrupt", json={"prompt_id": PID}, headers=AUTH).status_code == 200
    assert fake.interrupts == [PID]


@pytest.mark.parametrize("query", [
    "filename={n}&subfolder=vidsgen&type=output",
])
def test_a_generated_video_can_be_downloaded_and_streams_unchanged(client, fake, video_bytes, query):
    name = f"{new_job_id()}_00001_.mp4"
    r = client.get("/view?" + query.format(n=name), headers=AUTH)
    assert r.status_code == 200 and r.content == video_bytes
    assert set(r.headers) <= {"content-type", "content-length", "cache-control", "x-content-type-options", "accept-ranges", "content-range",
                              "etag", "last-modified"}, set(r.headers)
    sent = [x for x in fake.requests if x["path"] == "/view"][0]
    assert sent["query"] == {"filename": name, "subfolder": "vidsgen", "type": "output"} and sent["auth"] is None


JOB = new_job_id()


@pytest.mark.parametrize("query", [
    f"filename={JOB}_00001_.mp4&subfolder=vidsgen&type=input", f"filename={JOB}_00001_.mp4&subfolder=vidsgen&type=temp",
    f"filename={JOB}_00001_.mp4&subfolder=&type=output", f"filename={JOB}_00001_.mp4&subfolder=..&type=output",
    f"filename={JOB}_00001_.mp4&subfolder=vidsgen/..&type=output", f"filename={JOB}_00001_.mp4&subfolder=other&type=output",
    "filename=../../etc/passwd&subfolder=vidsgen&type=output", "filename=..%2f..%2fetc%2fpasswd&subfolder=vidsgen&type=output",
    f"filename=sub/{JOB}_00001_.mp4&subfolder=vidsgen&type=output", "filename=notajob.mp4&subfolder=vidsgen&type=output",
    f"filename={JOB}.png&subfolder=vidsgen&type=output", f"filename={JOB}_00001_.mp4.png&subfolder=vidsgen&type=output",
    f"filename={JOB[:-2]}_00001_.mp4&subfolder=vidsgen&type=output", f"filename={JOB}_00001_.mp4&subfolder=vidsgen&type=output&preview=webp",
    f"filename={JOB}_00001_.mp4&subfolder=vidsgen&type=output&channel=rgb", f"filename={JOB}_00001_.mp4&subfolder=vidsgen",
    f"filename={JOB}_00001_.mp4&type=output", "", f"filename={JOB}_00001_.mp4&filename=x.mp4&subfolder=vidsgen&type=output",
    f"filename={JOB}_00001_.mp4&subfolder=vidsgen&subfolder=x&type=output",
])
def test_only_generated_videos_can_be_downloaded(client, fake, query):
    r = client.get("/view?" + query, headers=AUTH)
    assert r.status_code in (400, 404), r.status_code
    assert [x for x in fake.requests if x["path"] == "/view"] == []


def test_range_requests_are_forwarded_only_when_well_formed(client, fake):
    url = f"/view?filename={JOB}_00001_.mp4&subfolder=vidsgen&type=output"
    assert client.get(url, headers={**AUTH, "Range": "bytes=0-99"}).status_code == 200
    for bad in ("bytes=0-1,5-9", "items=0-9", "bytes=abc", "bytes=0-9\r\nX: y", "bytes=" + "9" * 40 + "-"):
        r = client.get(url, headers={**AUTH, "Range": bad})
        assert r.status_code in (400, 416)
    assert len([x for x in fake.requests if x["path"] == "/view"]) == 1


# ====================================================================== limits, timeouts, concurrency
def test_oversized_bodies_are_rejected_before_reaching_comfyui(client, fake):
    big = prompt_body(graph(prompt="x" * 2000))
    big["padding"] = "p" * (256 * 1024)
    assert client.post("/prompt", json=big, headers=AUTH).status_code == 413
    assert client.post("/queue", content=b'{"delete": ["' + b"a" * 5000 + b'"]}', headers=AUTH).status_code == 413
    assert client.post("/interrupt", content=b"x" * 5000, headers=AUTH).status_code == 413
    assert [x for x in fake.requests if x["method"] == "POST"] == []


def test_a_body_without_content_length_is_still_capped_while_streaming(client, fake):
    def chunks():
        for _ in range(100):
            yield b"x" * 8192
    r = client.post("/prompt", content=chunks(), headers=AUTH)
    assert r.status_code == 413 and fake.requests == []


def test_a_non_numeric_content_length_is_refused(client):
    r = client.post("/queue", content=b"{}", headers={**AUTH, "Content-Length": "9999999999999999"})
    assert r.status_code in (400, 413)


def test_a_slow_body_is_cut_off(fake):
    async def run():
        with ProxyServer(make_app(fake.url, px.Policy(body_read_timeout=0.5))) as p:
            async def slow():
                yield b'{"delete": ['
                await asyncio.sleep(3)
                yield b"]}"
            async with httpx.AsyncClient(base_url=p.url, timeout=10) as c:
                return await c.post("/queue", content=slow(), headers=AUTH)
    assert asyncio.run(run()).status_code == 408


def test_an_unresponsive_comfyui_gives_504_and_a_dead_one_502_without_internal_detail(fake):
    fake.prompt_mode = "hang"
    with TestClient(make_app(fake.url, px.Policy(read_timeout=0.5))) as c:
        r = c.post("/prompt", json=prompt_body(), headers=AUTH)
        assert r.status_code == 504 and r.json()["error"] == "rejected" and "did not respond" in r.json()["reason"]
    dead = f"http://127.0.0.1:{free_port()}"
    with TestClient(make_app(dead)) as c:
        r = c.get("/system_stats", headers=AUTH)
        assert r.status_code == 502 and "127.0.0.1" not in r.text and str(dead.rsplit(":", 1)[1]) not in r.text


def test_heavy_requests_are_limited_but_health_and_progress_calls_stay_responsive(fake):
    fake.prompt_mode = "hang"

    async def run():
        with ProxyServer(make_app(fake.url, px.Policy(max_heavy=2, read_timeout=1.5))) as p:
            async with httpx.AsyncClient(base_url=p.url, timeout=10) as c:
                hang = [asyncio.create_task(c.post("/prompt", json=prompt_body(), headers=AUTH)) for _ in range(2)]
                await asyncio.sleep(0.4)
                third = await c.post("/prompt", json=prompt_body(), headers=AUTH)
                view = await c.get(f"/view?filename={JOB}_00001_.mp4&subfolder=vidsgen&type=output", headers=AUTH)
                light = [await c.get(path, headers=AUTH) for path in ("/system_stats", "/queue", f"/history/{PID}", "/models/vae")]
                done = await asyncio.gather(*hang)
                after = await c.post("/prompt", json=prompt_body(), headers=AUTH)
                return third, view, light, done, after
    third, view, light, done, after = asyncio.run(run())
    assert third.status_code == 429 and third.json() == {"error": "busy"} and third.headers["retry-after"] == "2" and view.status_code == 429
    assert all(r.status_code == 200 for r in light)
    assert [d.status_code for d in done] == [504, 504] and after.status_code == 504          # slots were released (it reached ComfyUI again)


def test_the_total_number_of_requests_in_flight_is_limited_and_released(fake):
    fake.prompt_mode = "hang"

    async def run():
        with ProxyServer(make_app(fake.url, px.Policy(max_concurrent=2, max_heavy=2, read_timeout=1.5))) as p:
            async with httpx.AsyncClient(base_url=p.url, timeout=10) as c:
                hang = [asyncio.create_task(c.post("/prompt", json=prompt_body(), headers=AUTH)) for _ in range(2)]
                await asyncio.sleep(0.4)
                busy = await c.get("/system_stats", headers=AUTH)
                await asyncio.gather(*hang)
                return busy, [await c.get("/system_stats", headers=AUTH) for _ in range(4)]
    busy, after = asyncio.run(run())
    assert busy.status_code == 429 and all(a.status_code == 200 for a in after)


def test_the_real_providers_health_burst_fits_inside_the_default_limits(make_client, fake):
    """Regression: the provider's health check fires ~15 metadata GETs at once; an early limit of 8 made health fail with 429."""
    assert px.Policy().max_concurrent >= 32
    with ProxyServer(make_app(fake.url)) as p:
        c = e2e_client(make_client, p)
        for _ in range(6):
            c.app.state.manager.provider._health_cache = None
            c.app.state.manager.provider._nodes_checked_until = 0
            h = c.get("/api/health").json()["provider"]
            assert h["ok"] is True, h
            assert {c["name"]: c["status"] for c in h["checks"]}["workflow_nodes"] == "ok"      # the ~11 parallel /object_info calls got through


def test_slots_are_released_after_streamed_downloads_and_errors(fake):
    with TestClient(make_app(fake.url, px.Policy(max_concurrent=1))) as c:
        url = f"/view?filename={JOB}_00001_.mp4&subfolder=vidsgen&type=output"
        for _ in range(3):
            assert c.get(url, headers=AUTH).status_code == 200
            assert c.post("/prompt", content=b"{bad", headers=AUTH).status_code == 400
            assert c.get("/nope", headers=AUTH).status_code == 404


# ====================================================================== response hygiene (mocked upstream)
def mocked(handler):
    return TestClient(make_app("http://127.0.0.1:8188", transport=httpx.MockTransport(handler)))


def test_system_stats_exposes_only_versions_and_device_capacity():
    raw = {"system": {"os": "posix", "ram_total": 1, "python_version": "3.13", "pytorch_version": "2.11.0+cu130", "comfyui_version": "0.46.0",
                      "embedded_python": False, "argv": ["/kaggle/tmp/vidsgen-gpu/ComfyUI/main.py", "--base-directory", "/kaggle/tmp/x"],
                      "installed_templates_version": "1"},
           "devices": [{"name": "cuda:0 Tesla T4 : cudaMallocAsync", "type": "cuda", "index": 0, "vram_total": 15, "vram_free": 14,
                        "torch_vram_total": 13, "torch_vram_free": 12, "path": "/dev/nvidia0"}]}
    with mocked(lambda req: httpx.Response(200, json=raw)) as c:
        r = c.get("/system_stats", headers=AUTH).json()
    assert r == {"system": {"comfyui_version": "0.46.0", "pytorch_version": "2.11.0+cu130"},
                 "devices": [{"name": "cuda:0 Tesla T4 : cudaMallocAsync", "type": "cuda", "vram_total": 15, "vram_free": 14,
                              "torch_vram_total": 13, "torch_vram_free": 12}]}
    assert "kaggle" not in json.dumps(r) and "argv" not in json.dumps(r)


def test_the_queue_is_reduced_to_ids_and_never_returns_submitted_graphs():
    item = [3, PID, {"6": {"inputs": {"text": "SECRET PROMPT"}}}, {"client_id": "c", "create_time": 1}, ["58"]]
    with mocked(lambda req: httpx.Response(200, json={"queue_running": [item], "queue_pending": [item, "junk"], "other": 1})) as c:
        r = c.get("/queue", headers=AUTH).json()
    assert r == {"queue_running": [[3, PID]], "queue_pending": [[3, PID]]} and "SECRET" not in json.dumps(r)


def test_history_drops_the_graph_and_scrubs_internal_paths():
    entry = {"prompt": [1, PID, {"6": {"inputs": {"text": "SECRET PROMPT"}}}], "meta": {"x": "/kaggle/tmp/y"},
             "outputs": {"58": {"images": [{"filename": f"{JOB}_00001_.mp4", "subfolder": "vidsgen", "type": "output"}], "animated": [True]}},
             "status": {"status_str": "error", "completed": False, "messages": [
                 ["execution_error", {"exception_message": "CUDA out of memory", "node_type": "KSampler",
                                      "traceback": ['  File "/kaggle/tmp/vidsgen-gpu/ComfyUI/execution.py", line 5, in x\n',
                                                    "torch.OutOfMemoryError: CUDA out of memory. See /opt/conda/lib/python3.12/x.py"]}]],
                        "extra": "dropped"}}
    with mocked(lambda req: httpx.Response(200, json={PID: entry})) as c:
        r = c.get(f"/history/{PID}", headers=AUTH).json()
    text = json.dumps(r)
    assert set(r[PID]) == {"outputs", "status"} and set(r[PID]["status"]) == {"status_str", "completed", "messages"}
    assert "SECRET" not in text and "/kaggle" not in text and "/opt/conda" not in text and "<path>" in text
    assert "CUDA out of memory" in text and r[PID]["outputs"]["58"]["images"][0]["filename"].endswith("_00001_.mp4")    # what the provider needs


def test_an_unknown_history_id_returns_an_empty_object():
    with mocked(lambda req: httpx.Response(200, json={})) as c:
        assert c.get(f"/history/{PID}", headers=AUTH).json() == {}


def test_comfyuis_own_validation_error_is_passed_on_with_paths_removed():
    up = {"error": {"type": "prompt_outputs_failed_validation", "message": "Prompt outputs failed validation", "details": "", "extra_info": {}},
          "node_errors": {"39": {"errors": [{"type": "value_not_in_list", "details": "vae_name: x not in [/kaggle/tmp/vidsgen-gpu/comfy-data/models/vae/a.safetensors]"}],
                                 "class_type": "VAELoader"}}, "secret_extra": "dropped"}
    with mocked(lambda req: httpx.Response(400, json=up)) as c:
        r = c.post("/prompt", json=prompt_body(), headers=AUTH)
    assert r.status_code == 400 and set(r.json()) == {"error", "node_errors"} and "/kaggle" not in r.text and "<path>" in r.text


def test_upstream_server_errors_become_a_generic_502():
    with mocked(lambda req: httpx.Response(500, text="Traceback (most recent call last): File /srv/ComfyUI/x.py")) as c:
        for path in ("/system_stats", "/queue", f"/history/{PID}", "/object_info/KSampler", "/models/vae"):
            r = c.get(path, headers=AUTH)
            assert r.status_code == 502 and r.json() == {"error": "upstream_error"}
        assert c.post("/prompt", json=prompt_body(), headers=AUTH).json() == {"error": "upstream_error"}


def test_an_internal_failure_never_leaks_a_traceback_or_message():
    def boom(req):
        raise RuntimeError("/srv/secret/path exploded " + TOKEN)
    with mocked(boom) as c:
        r = c.get("/queue", headers=AUTH)
    assert r.status_code == 500 and r.json() == {"error": "internal_error"} and TOKEN not in r.text and "/srv" not in r.text


def test_an_oversized_upstream_json_response_is_refused():
    with mocked(lambda req: httpx.Response(200, content=b"[" + b"1," * (5 * 1024 * 1024) + b"1]")) as c:
        assert c.get("/models/vae", headers=AUTH).status_code == 502


def test_every_response_carries_no_store_and_nosniff(client):
    for r in (client.get("/system_stats", headers=AUTH), client.get("/nope", headers=AUTH), client.get("/system_stats")):
        assert r.headers["cache-control"] == "no-store" and r.headers["x-content-type-options"] == "nosniff"


@pytest.mark.parametrize("text,expected", [
    ('File "/kaggle/tmp/vidsgen-gpu/ComfyUI/execution.py", line 1', 'File "<path>", line 1'),
    ("see /opt/conda/lib/python3.12/site-packages/torch/x.py now", "see <path> now"),
    ("/home/user/.cache/x", "<path>"), ("/tmp/", "<path>"),
    ("vidsgen/job_abc", "vidsgen/job_abc"), ("a/b/c", "a/b/c"), ("https://example.com/a/b", "https://example.com/a/b"), ("/view", "/view"),
    ("1/2", "1/2"), ("plain text", "plain text"),
])
def test_path_scrubbing(text, expected):
    assert px.scrub_paths(text) == expected


# ====================================================================== end to end: the real provider and backend through the proxy
def e2e_client(make_client, proxy, token=TOKEN, **over):
    opts = dict(inference_provider="comfyui", comfyui_base_url=proxy.url, comfyui_api_token=token, comfyui_poll_interval=0.2,
                comfyui_timeout=5, generation_profile="t4_verified")
    opts.update(over)
    return make_client(**opts)


def test_a_real_job_runs_through_the_proxy_using_only_allowed_routes(make_client, fake, video_bytes):
    with ProxyServer(make_app(fake.url)) as p:
        c = e2e_client(make_client, p)
        assert c.get("/api/health").json()["provider"]["ok"] is True                     # health: every probe route is allowlisted
        job = wait_for(c, submit(c, resolution="832x480", num_frames=17, steps=20, fps=24, guidance_scale=5.0).json()["id"])
        assert job["status"] == "completed", job["error"]
        got = c.get(f"/api/jobs/{job['id']}/download")
        assert got.status_code == 200 and got.content == video_bytes
        assert c.get("/api/health").json()["provider"]["checks"]                          # still healthy afterwards
    routes = {(m, re.sub(r"/[0-9a-f-]{36}$", "/{id}", pth)) for m, pth in upstream_paths(fake)}
    assert ("POST", "/prompt") in routes and ("GET", "/view") in routes and ("GET", "/history/{id}") in routes
    assert all(r["auth"] is None for r in fake.requests)                                 # the proxy strips the bearer token
    assert not [r for r in fake.requests if r["path"] in ("/ws", "/free", "/upload/image")]


def test_the_provider_cancels_a_running_job_through_the_proxy_with_a_targeted_interrupt(make_client, fake):
    fake.default_scenario = "hang"
    with ProxyServer(make_app(fake.url)) as p:
        c = e2e_client(make_client, p)
        job_id = submit(c).json()["id"]
        wait_for(c, job_id, statuses={"processing"})
        deadline = time.monotonic() + 5
        while fake._running is None and time.monotonic() < deadline:
            time.sleep(0.05)
        assert c.post(f"/api/jobs/{job_id}/cancel").json()["status"] == "cancelled"
        deadline = time.monotonic() + 5
        while not fake.interrupts and time.monotonic() < deadline:
            time.sleep(0.05)
    assert len(fake.interrupts) == 1 and px.UUID_RE.match(fake.interrupts[0])


def test_a_wrong_backend_token_fails_clearly_and_nothing_reaches_comfyui(make_client, fake):
    with ProxyServer(make_app(fake.url)) as p:
        c = e2e_client(make_client, p, token="wrong-" + uuid.uuid4().hex)
        job = wait_for(c, submit(c).json()["id"])
    assert job["status"] == "failed" and job["error"]["code"] == "comfyui_auth_failed"
    assert fake.requests == []


@pytest.mark.parametrize("over,reason", [(dict(resolution="1280x704"), "resolution 1280x704"), (dict(num_frames=49), "num_frames"),
                                         (dict(steps=30), "steps")])
def test_requests_outside_the_gpu_envelope_are_stopped_at_the_proxy_even_if_the_backend_is_unrestricted(make_client, fake, over, reason):
    """Defence in depth: a backend with no limits (or a stolen token) still cannot run a larger job on the GPU host."""
    with ProxyServer(make_app(fake.url)) as p:
        c = e2e_client(make_client, p, generation_profile="unrestricted")
        job = wait_for(c, submit(c, **{"resolution": "832x480", "num_frames": 17, "steps": 20, **over}).json()["id"])
    assert job["status"] == "failed" and job["error"]["code"] == "comfyui_rejected" and reason in job["error"]["message"]
    assert [r for r in fake.requests if r["path"] == "/prompt"] == []


def test_responses_from_the_real_server_have_no_server_header(fake):
    async def run():
        with ProxyServer(make_app(fake.url)) as p:
            async with httpx.AsyncClient(base_url=p.url) as c:
                return await c.get("/system_stats", headers=AUTH), await c.get("/system_stats")
    ok, denied = asyncio.run(run())
    assert "server" not in ok.headers and "server" not in denied.headers and "date" not in ok.headers


# ====================================================================== static guarantees
def test_the_proxy_has_no_tunnel_no_public_binding_and_no_token_output():
    src = SCRIPT.read_text()
    for forbidden in ("cloudflared", "trycloudflare", "ngrok", "0.0.0.0", "os.environ.get(\"HF_", "print(token", "print(TOKEN"):
        assert forbidden not in src.replace("\"0.0.0.0\"", ""), forbidden
    assert 'a.host not in ("127.0.0.1", "localhost", "::1")' in src and "add_argument(\"--token\"" not in src
    assert not re.search(r"log\.\w+\([^)]*(token|authorization|headers|query|body)", src, re.I)
