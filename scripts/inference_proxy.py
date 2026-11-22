#!/usr/bin/env python3
"""VIDSGEN inference proxy (Phase 6C, Stage 1: the HTTP core).

A small authenticated gate that sits in front of ComfyUI so that ComfyUI itself (which has NO authentication and many
dangerous routes) is never exposed. It listens on 127.0.0.1 and forwards to a ComfyUI on 127.0.0.1. Nothing here opens a
tunnel or a public port; making it reachable from outside is a later, separately approved stage.

What it enforces
  * Bearer-token authentication (constant time). Unauthorized requests get an identical, information-free 401.
  * A strict allowlist of (method, path) pairs: only what the VIDSGEN backend's ComfyUIProvider uses. Everything else is 404.
  * Strict request validation. Most importantly POST /prompt: the submitted graph must be EXACTLY the Wan 2.2 TI2V-5B
    text-to-video template (same nodes, classes, links, model files and settings); only the inputs VIDSGEN varies (prompt text,
    size, frames, fps, steps, CFG, seed, output prefix) may differ, and only within the verified T4 envelope.
  * Refused outright: queue clearing, unrestricted (global) interrupts, model loading/unloading, uploads, arbitrary graphs.
  * Body-size limits, a concurrency limit, a failed-authentication throttle and bounded timeouts.
  * Responses are rebuilt, not relayed: allow-listed headers only, queue entries reduced to ids, /history stripped of the
    submitted graph, absolute file paths scrubbed from every text that can reach a client.

Stage 2 adds: an authenticated one-way WebSocket relay for progress events (allow-listed event types only, rebuilt and
path-scrubbed, only for prompts this proxy forwarded), streamed MP4 downloads with deterministic cleanup, bounded and
concurrency-safe prompt_id idempotency (a reused id with a different payload is refused, a duplicate is never forwarded twice),
read-only reconciliation of uncertain submissions (it never resubmits), and an optional dead-man switch (OFF by default).

Stage 4A adds a read-only PRE-FORWARD existence check: before a prompt_id the proxy has not seen is forwarded, ComfyUI's history and
queue are asked (GET only) whether it already knows that id; if so, the submission is answered as already accepted and is NOT forwarded
again. This narrows the double-run window after a proxy restart, but it can NOT guarantee exactly-once execution:
  * a prompt that ComfyUI has accepted but not yet made visible in /queue or /history (a visibility race) looks "unknown";
  * a first request that is still in flight on another (older) proxy instance cannot be seen at all;
  * "known" is judged by prompt_id alone: the payload behind an id remembered only by ComfyUI cannot be compared;
  * if ComfyUI cannot be asked, the proxy does not forward (it answers 502) rather than guess.
Nothing here ever resubmits an uncertain request automatically. A bounded --max-lifetime-seconds is also available for short, supervised runs.

Still not here: the tunnel (a later, separately approved stage).

    PROXY_API_TOKEN=<>=32 random chars>  python inference_proxy.py --upstream http://127.0.0.1:8188 --port 8189 \
        --workflows-dir <repo>/workflows/comfyui

The token is read from the environment only (never from the command line) and is never printed or logged.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import hmac
import json
import logging
import math
import os
import re
import sys
import time
import unicodedata
from collections import OrderedDict, deque
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import parse_qsl, urlsplit

import httpx
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, Response, StreamingResponse
from starlette.routing import Route, WebSocketRoute
from starlette.websockets import WebSocket, WebSocketDisconnect, WebSocketState
from websockets.asyncio.client import connect as ws_connect

log = logging.getLogger("vidsgen.proxy")

TOKEN_ENV = "PROXY_API_TOKEN"
MIN_TOKEN_CHARS = 32
UUID_RE = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$")      # ComfyUI wants canonical lowercase
CLIENT_ID_RE = re.compile(r"^[A-Za-z0-9._-]{1,64}$")
PREFIX_RE = re.compile(r"^vidsgen/job_[0-9a-f]{24}$")                                          # what ComfyUIProvider sends
VIEW_NAME_RE = re.compile(r"^job_[0-9a-f]{24}_\d{5,}_?\.mp4$")                                 # what SaveVideo writes ("..._00001_.mp4")
MODEL_FOLDERS = frozenset({"diffusion_models", "text_encoders", "vae"})                       # the three Wan 2.2 files
RANGE_RE = re.compile(r"^bytes=\d{0,12}-\d{0,12}$")
ABS_PATH_RE = re.compile(r"(?<![\w.:/-])/(?:[\w.+@~=-]+/)+[\w.+@~=-]*")                         # /a/b/c  ->  <path>
FORWARDED_VIEW_HEADERS = ("content-type", "content-length", "content-range", "accept-ranges", "etag", "last-modified")


# ===================================================================================== policy
@dataclass(frozen=True)
class Policy:
    """The limits. The defaults are the configuration verified on a Kaggle T4 (the backend's `t4_verified` profile)."""

    resolutions: frozenset = frozenset({(832, 480)})
    min_frames: int = 17
    max_frames: int = 17
    max_steps: int = 20
    fps: tuple = (8, 30)
    cfg: tuple = (1.0, 15.0)
    max_text_chars: int = 2000
    max_prompt_body: int = 256 * 1024
    max_small_body: int = 4096
    max_json_response: int = 8 * 1024 * 1024
    max_concurrent: int = 32        # all in-flight requests; the provider's health check alone fires ~15 metadata GETs at once
    max_heavy: int = 4              # POST /prompt and GET /view (GPU work and large downloads) so they cannot crowd out health/progress calls
    max_ids_per_call: int = 10
    auth_failures_per_minute: int = 10
    body_read_timeout: float = 30.0
    connect_timeout: float = 5.0
    read_timeout: float = 30.0
    view_read_timeout: float = 120.0
    view_total_timeout: float = 600.0
    idempotency_max_entries: int = 512
    idempotency_ttl: float = 6 * 3600.0
    max_ws: int = 2                     # one backend process uses one; a spare for a reconnect overlap
    ws_frame_limit: int = 64 * 1024     # larger upstream text frames are dropped
    ws_inbound_limit: int = 1024        # clients send nothing useful; anything bigger closes the socket
    ws_send_timeout: float = 10.0       # a client that cannot take a frame in this time is disconnected (no unbounded buffering)
    ws_open_timeout: float = 5.0
    ws_max_seconds: float = 8 * 3600.0
    dead_man_seconds: float = 0.0       # 0 = disabled (the default). When enabled, at least dead_man_min.
    dead_man_min: float = 120.0
    dead_man_interval: float = 15.0
    preforward_check: bool = True       # read-only "does ComfyUI already know this prompt_id?" before forwarding a new one


class Rejected(Exception):
    """A request the proxy refuses. `reason` is written for the (authenticated) client and never echoes its input."""

    def __init__(self, reason: str, status: int = 400):
        super().__init__(reason)
        self.reason, self.status = reason, status


# ===================================================================================== sanitising
def scrub_paths(text: str) -> str:
    """Replace absolute file paths (server internals) in text that may reach a client."""
    return ABS_PATH_RE.sub("<path>", text)


def scrub_json(value: Any) -> Any:
    if isinstance(value, str):
        return scrub_paths(value)
    if isinstance(value, list):
        return [scrub_json(v) for v in value]
    if isinstance(value, dict):
        return {k: scrub_json(v) for k, v in value.items()}
    return value


def sanitize_system_stats(body: Any) -> dict:
    """Only what the provider reads (version and device capacity): no host name, paths, argv or Python build."""
    system = body.get("system", {}) if isinstance(body, dict) else {}
    keep = ("name", "type", "vram_total", "vram_free", "torch_vram_total", "torch_vram_free")
    devices = [{k: d[k] for k in keep if k in d} for d in (body.get("devices", []) if isinstance(body, dict) else []) if isinstance(d, dict)]
    return {"system": {k: system[k] for k in ("comfyui_version", "pytorch_version") if k in system}, "devices": scrub_json(devices)}


def sanitize_queue(body: Any) -> dict:
    """Queue entries reduced to [number, prompt_id]: the submitted graphs are never sent back."""
    out: dict = {}
    for key in ("queue_running", "queue_pending"):
        items = body.get(key, []) if isinstance(body, dict) else []
        out[key] = [[i[0], i[1]] for i in items if isinstance(i, list) and len(i) > 1]
    return out


def sanitize_history(body: Any, prompt_id: str) -> dict:
    """{id: {outputs, status}} without the submitted graph or metadata; error messages are path-scrubbed."""
    entry = body.get(prompt_id) if isinstance(body, dict) else None
    if not isinstance(entry, dict):
        return {}
    status = entry.get("status") if isinstance(entry.get("status"), dict) else {}
    return {prompt_id: {"outputs": entry.get("outputs") if isinstance(entry.get("outputs"), dict) else {},
                        "status": scrub_json({k: status[k] for k in ("status_str", "completed", "messages") if k in status})}}


WS_FIELDS = {
    "execution_start": ("prompt_id", "timestamp"),
    "executing": ("prompt_id", "node", "display_node"),
    "progress": ("prompt_id", "node", "display_node", "value", "max"),
    "execution_success": ("prompt_id", "timestamp"),
    "execution_error": ("prompt_id", "node_id", "node_type", "executed", "exception_message", "exception_type", "traceback"),
    "execution_interrupted": ("prompt_id", "node_id", "node_type", "executed"),
}                                           # exactly the events ComfyUIProvider uses; everything else is dropped


def sanitize_ws_event(raw: str, is_tracked) -> str | None:
    """Rebuild one ComfyUI event for the client, or None to drop it. Only allow-listed event types, only the fields the provider
    reads, only for prompts this proxy forwarded, with file paths scrubbed and sizes bounded. Never a pass-through."""
    try:
        msg = json.loads(raw)
    except ValueError:
        return None
    kind = msg.get("type") if isinstance(msg, dict) else None
    data = msg.get("data") if isinstance(msg, dict) else None
    if kind not in WS_FIELDS or not isinstance(data, dict):
        return None
    pid = data.get("prompt_id")
    if not isinstance(pid, str) or not UUID_RE.match(pid) or not is_tracked(pid):
        return None
    out = {k: data[k] for k in WS_FIELDS[kind] if k in data}
    if isinstance(out.get("traceback"), list):
        out["traceback"] = [str(t)[:2000] for t in out["traceback"][:50]]
    if isinstance(out.get("exception_message"), str):
        out["exception_message"] = out["exception_message"][:2000]
    return json.dumps({"type": kind, "data": scrub_json(out)})


# ===================================================================================== prompt_id registry (idempotency)
class _Entry:
    __slots__ = ("digest", "client_id", "state", "status", "body", "future", "created", "finished")

    def __init__(self, digest, client_id, created, future):
        self.digest, self.client_id, self.created, self.future = digest, client_id, created, future
        self.state, self.status, self.body, self.finished = "pending", None, None, False


def payload_digest(body: dict) -> str:
    """Identity of a submission: the validated graph plus the client id, canonicalised."""
    return hashlib.sha256(json.dumps({"prompt": body["prompt"], "client_id": body["client_id"]}, sort_keys=True,
                                     separators=(",", ":"), ensure_ascii=True).encode()).hexdigest()


class PromptRegistry:
    """Remembers every prompt_id the proxy has seen, to make submissions idempotent.

    States: pending (being forwarded now), accepted (ComfyUI confirmed), uncertain (the forward failed in a way that does not
    show whether ComfyUI queued it), cancelled (a tombstone: the id was deleted or interrupted). It is bounded (max entries,
    TTL) and only ever evicts entries that are not pending. All methods are synchronous (no await), so on asyncio's single thread
    each call is atomic: concurrent duplicates cannot both become 'new'."""

    def __init__(self, max_entries: int, ttl: float, clock=time.monotonic):
        self.max, self.ttl, self.clock = max_entries, ttl, clock
        self.entries: OrderedDict[str, _Entry] = OrderedDict()

    def _purge(self) -> None:
        now = self.clock()
        for pid in [k for k, e in self.entries.items() if e.state != "pending" and now - e.created > self.ttl]:
            del self.entries[pid]

    def _make_room(self) -> bool:
        if len(self.entries) < self.max:
            return True
        for pid, e in self.entries.items():
            if e.state != "pending":
                del self.entries[pid]
                return True
        return False

    def begin(self, pid: str, digest: str, client_id: str):
        """-> (kind, entry). kind: new | pending | accepted | uncertain | cancelled | conflict | full."""
        self._purge()
        e = self.entries.get(pid)
        if e is None:
            if not self._make_room():
                return "full", None
            try:
                future = asyncio.get_running_loop().create_future()
            except RuntimeError:                                  # called outside the event loop (tests): nobody can be waiting
                future = None
            e = _Entry(digest, client_id, self.clock(), future)
            self.entries[pid] = e
            return "new", e
        if e.state == "cancelled":
            return "cancelled", e
        if e.digest != digest:
            return "conflict", e
        return e.state, e

    def _settle(self, e: _Entry, status: int, body: Any) -> None:
        e.status, e.body = status, body
        if e.future is not None and not e.future.done():
            e.future.set_result((status, body))

    def resolve(self, pid: str, status: int, body: Any) -> None:
        e = self.entries.get(pid)
        if e is not None:
            e.state = "cancelled" if e.state == "cancelled" else "accepted"
            self._settle(e, status, body)

    def reject(self, pid: str, status: int, body: Any) -> None:
        """ComfyUI refused it (nothing was queued): forget the id so a corrected request can use it."""
        e = self.entries.pop(pid, None)
        if e is not None:
            self._settle(e, status, body)

    def uncertain(self, pid: str, status: int, body: Any) -> None:
        e = self.entries.get(pid)
        if e is not None:
            e.state = "cancelled" if e.state == "cancelled" else "uncertain"
            self._settle(e, status, body)

    def cancel(self, pid: str) -> None:
        """Tombstone: a cancelled id can never be submitted again through this proxy."""
        e = self.entries.get(pid)
        if e is None:
            if not self._make_room():
                return
            e = self.entries[pid] = _Entry(None, None, self.clock(), None)
        e.state = "cancelled"

    def is_tracked(self, pid: str) -> bool:
        e = self.entries.get(pid)
        return e is not None and e.state in ("pending", "accepted", "uncertain")

    def active_ids(self) -> list[str]:
        """Prompts this proxy forwarded and has not seen finish (candidates for the optional dead-man switch)."""
        return [k for k, e in self.entries.items() if e.state in ("accepted", "uncertain") and not e.finished]

    def finish(self, pid: str) -> None:
        if pid in self.entries:
            self.entries[pid].finished = True


class ClosingStreamingResponse(StreamingResponse):
    """A StreamingResponse that ALWAYS closes its generator and runs `on_close`, also when the client disconnects mid-stream
    (where Starlette only cancels the task): upstream connections and concurrency slots are released deterministically."""

    def __init__(self, *args, on_close=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.on_close = on_close

    async def __call__(self, scope, receive, send):
        try:
            await super().__call__(scope, receive, send)
        finally:
            try:
                await self.body_iterator.aclose()
            finally:
                if self.on_close is not None:
                    self.on_close()


def uvicorn_options() -> dict:
    """Server settings shared by the CLI and the tests: small inbound WebSocket frames, no Server/Date headers, bounded headers."""
    return dict(log_level="warning", access_log=False, server_header=False, date_header=False, timeout_keep_alive=5,
                limit_concurrency=64, h11_max_incomplete_event_size=16 * 1024, ws_max_size=4096, ws_ping_interval=20, ws_ping_timeout=20)


# ===================================================================================== graph validation
def _strict_eq(a: Any, b: Any) -> bool:
    """JSON equality that does not treat True as 1 and does not let floats pass for ints in fixed fields."""
    if isinstance(a, bool) or isinstance(b, bool):
        return isinstance(a, bool) and isinstance(b, bool) and a == b
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return a == b
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(_strict_eq(a[k], b[k]) for k in a)
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(_strict_eq(x, y) for x, y in zip(a, b))
    return type(a) is type(b) and a == b


def _is_int(v: Any) -> bool:
    return isinstance(v, int) and not isinstance(v, bool)


def _is_number(v: Any) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


def _clean_text(v: Any, limit: int, allow_empty: bool) -> bool:
    if not isinstance(v, str) or len(v) > limit or (not v.strip() and not allow_empty):
        return False
    return not any(unicodedata.category(c) == "Cc" and c not in "\n\t" for c in v)


class GraphValidator:
    """Accepts only the Wan 2.2 TI2V-5B text-to-video workflow, with VIDSGEN's own inputs inside the verified envelope.

    The reference is the repository's template; the variable inputs are those named by its mapping file. Everything else
    (node ids, classes, links, the three model files, sampler/scheduler/shift, codec, batch size...) must match the template
    exactly, so the proxy cannot be used to run any other graph, load any other model or write anywhere but `vidsgen/`."""

    def __init__(self, workflows_dir: Path, policy: Policy):
        d = Path(workflows_dir)
        mapping = json.loads((d / "wan22_ti2v_5b.mapping.json").read_text(encoding="utf-8"))
        self.template: dict[str, Any] = json.loads((d / mapping["templates"]["t2v"]).read_text(encoding="utf-8"))
        i2v = json.loads((d / mapping["templates"]["i2v"]).read_text(encoding="utf-8"))
        self.policy = policy
        nodes = mapping["nodes"]
        self.variable: dict[tuple[str, str], str] = {}                       # (node id, input) -> parameter name
        for param, target in mapping["inputs"].items():
            if param == "image":                                             # image-to-video: not part of Stage 1
                continue
            self.variable[(nodes[target["node"]]["id"], target["input"])] = param
        self.object_info_classes = frozenset(n["class_type"] for t in (self.template, i2v) for n in t.values())   # what health checks ask for
        missing = [k for k in self.variable if k[0] not in self.template or k[1] not in self.template[k[0]]["inputs"]]
        if missing:
            raise ValueError(f"the mapping names inputs the template does not have: {missing}")

    def _check_param(self, param: str, v: Any) -> str | None:
        p = self.policy
        if param == "prompt":
            return None if _clean_text(v, p.max_text_chars, allow_empty=False) else "prompt must be non-empty text within the length limit"
        if param == "negative_prompt":
            return None if _clean_text(v, p.max_text_chars, allow_empty=True) else "negative_prompt must be text within the length limit"
        if param in ("width", "height"):
            return None if _is_int(v) else f"{param} must be an integer"          # the pair is checked together
        if param == "num_frames":
            ok = _is_int(v) and p.min_frames <= v <= p.max_frames and (v - 1) % 4 == 0
            return None if ok else f"num_frames must be 4k+1 between {p.min_frames} and {p.max_frames} (this GPU's verified limit)"
        if param == "fps":
            return None if _is_int(v) and p.fps[0] <= v <= p.fps[1] else f"fps must be an integer between {p.fps[0]} and {p.fps[1]}"
        if param == "steps":
            return None if _is_int(v) and 1 <= v <= p.max_steps else f"steps must be an integer between 1 and {p.max_steps} (this GPU's verified limit)"
        if param == "guidance_scale":
            return None if _is_number(v) and p.cfg[0] <= v <= p.cfg[1] else f"guidance_scale must be a number between {p.cfg[0]:g} and {p.cfg[1]:g}"
        if param == "seed":
            return None if _is_int(v) and 0 <= v <= 2**32 - 1 else "seed must be an integer between 0 and 4294967295"
        if param == "filename_prefix":
            return None if isinstance(v, str) and PREFIX_RE.match(v) else "filename_prefix must be vidsgen/job_<id>"
        return "unexpected parameter"

    def validate(self, graph: Any) -> None:
        """Raises Rejected unless `graph` is the allowed workflow."""
        if not isinstance(graph, dict) or len(graph) > 64:
            raise Rejected("the workflow is not an allowed graph")
        if graph.keys() != self.template.keys():
            raise Rejected("the workflow does not match the allowed Wan 2.2 text-to-video graph (nodes differ)")
        values: dict[str, Any] = {}
        for nid, ref in self.template.items():
            node = graph[nid]
            if not isinstance(node, dict) or node.keys() != ref.keys() or node.get("class_type") != ref["class_type"] \
                    or not _strict_eq(node.get("_meta"), ref.get("_meta")):
                raise Rejected(f"node {nid} does not match the allowed workflow")
            inputs = node.get("inputs")
            if not isinstance(inputs, dict) or inputs.keys() != ref["inputs"].keys():
                raise Rejected(f"node {nid} has unexpected inputs")
            for name, expected in ref["inputs"].items():
                param = self.variable.get((nid, name))
                if param is None:
                    if not _strict_eq(inputs[name], expected):
                        raise Rejected(f"node {nid} input {name} is not the allowed fixed value (models and settings cannot be changed)")
                    continue
                problem = self._check_param(param, inputs[name])
                if problem:
                    raise Rejected(problem)
                values[param] = inputs[name]
        if (values["width"], values["height"]) not in self.policy.resolutions:
            allowed = ", ".join(f"{w}x{h}" for w, h in sorted(self.policy.resolutions))
            raise Rejected(f"resolution {values['width']}x{values['height']} is not enabled on this GPU server; allowed: {allowed}")


# ===================================================================================== the app
class AuthThrottle:
    """Failed authentications are throttled globally (the proxy sits behind a tunnel, so client addresses are not trustworthy).
    Valid tokens are never delayed by it."""

    def __init__(self, per_minute: int):
        self.per_minute, self.fails = per_minute, deque()

    def blocked(self, now: float) -> bool:
        while self.fails and now - self.fails[0] > 60:
            self.fails.popleft()
        return len(self.fails) >= self.per_minute

    def record_failure(self, now: float) -> None:
        self.fails.append(now)


@dataclass
class Route_:
    methods: tuple
    pattern: re.Pattern
    name: str
    handler: str
    body_limit: str | None = None          # "prompt" | "small" | None
    params: dict = field(default_factory=dict)


def _json_response(body: Any, status: int = 200) -> JSONResponse:
    return JSONResponse(body, status_code=status)


def create_app(upstream_url: str, token: str, workflows_dir: Path, policy: Policy | None = None,
               transport: httpx.AsyncBaseTransport | None = None, clock=time.monotonic) -> Starlette:
    if len(token) < MIN_TOKEN_CHARS:
        raise ValueError(f"the proxy token must be at least {MIN_TOKEN_CHARS} characters")
    parts = urlsplit(upstream_url)
    if parts.scheme != "http" or (parts.hostname or "") not in ("127.0.0.1", "localhost", "::1") or not parts.port:
        raise ValueError("the upstream must be a loopback ComfyUI URL like http://127.0.0.1:8188")
    host_port = f"{parts.hostname if parts.hostname != '::1' else '[::1]'}:{parts.port}"
    logging.getLogger("websockets").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)        # its INFO lines carry full upstream URLs: keep them out of the logs
    policy = policy or Policy()
    if policy.dead_man_seconds and policy.dead_man_seconds < policy.dead_man_min:
        raise ValueError(f"the dead-man switch needs at least {policy.dead_man_min:g} s of inactivity (or 0 to keep it disabled)")
    validator = GraphValidator(Path(workflows_dir), policy)
    digest = hashlib.sha256(token.encode()).digest()
    throttle = AuthThrottle(policy.auth_failures_per_minute)
    registry = PromptRegistry(policy.idempotency_max_entries, policy.idempotency_ttl, clock)
    state = {"inflight": 0, "heavy": 0, "client": None, "last_activity": clock(), "ws": set(), "ws_tasks": set(), "reaper": None,
             "stats": {"ws_events": 0, "ws_dropped": 0}}

    def client() -> httpx.AsyncClient:
        return state["client"]

    def authorized(request: Request) -> bool:
        header = request.headers.get("authorization", "")
        scheme, _, presented = header.partition(" ")
        presented = presented.strip()
        ok = scheme.lower() == "bearer" and 0 < len(presented) <= 512
        # Always compare (fixed work) so timing does not reveal whether the header was well-formed.
        match = hmac.compare_digest(hashlib.sha256(presented.encode("utf-8", "ignore")).digest(), digest)
        return ok and match

    async def read_body(request: Request, limit: int) -> bytes:
        declared = request.headers.get("content-length")
        if declared is not None and (not declared.isdigit() or int(declared) > limit):
            raise Rejected("request body too large", 413)
        chunks, size = [], 0

        async def gather():
            nonlocal size
            async for chunk in request.stream():
                size += len(chunk)
                if size > limit:
                    raise Rejected("request body too large", 413)
                chunks.append(chunk)
        try:
            await asyncio.wait_for(gather(), policy.body_read_timeout)
        except TimeoutError:
            raise Rejected("request body not received in time", 408) from None
        return b"".join(chunks)

    def parse_json_object(raw: bytes) -> dict:
        def no_constants(c):
            raise ValueError(c)
        try:
            body = json.loads(raw.decode("utf-8"), parse_constant=no_constants)
        except (ValueError, UnicodeDecodeError):
            raise Rejected("the request body must be valid JSON") from None
        if not isinstance(body, dict):
            raise Rejected("the request body must be a JSON object")
        return body

    def only_keys(body: dict, allowed: set[str], required: set[str]) -> None:
        extra = set(body) - allowed
        if extra:
            raise Rejected("the request contains fields that are not allowed")      # field names are not echoed
        if required - set(body):
            raise Rejected("the request is missing required fields")

    def uuid_list(value: Any, p: Policy) -> list[str]:
        if not isinstance(value, list) or not 1 <= len(value) <= p.max_ids_per_call or not all(isinstance(v, str) and UUID_RE.match(v) for v in value):
            raise Rejected("ids must be a short list of canonical prompt ids")
        return value

    async def upstream_json(method: str, path: str, body: Any = None, *, params: dict | None = None) -> tuple[int, Any]:
        try:
            resp = await client().request(method, path, params=params, content=json.dumps(body).encode() if body is not None else None,
                                          headers={"Content-Type": "application/json"} if body is not None else None)
        except httpx.TimeoutException:
            raise Rejected("the GPU server did not respond in time", 504) from None
        except httpx.HTTPError:
            raise Rejected("the GPU server is unreachable", 502) from None
        if len(resp.content) > policy.max_json_response:
            raise Rejected("the GPU server response is too large", 502)
        try:
            return resp.status_code, (resp.json() if resp.content else None)
        except ValueError:
            return resp.status_code, None

    # ------------------------------------------------------------------ handlers
    async def h_system_stats(request, route, match):
        status, body = await upstream_json("GET", "/system_stats")
        if status != 200:
            return _json_response({"error": "upstream_error"}, 502)
        return _json_response(sanitize_system_stats(body))

    async def h_object_info(request, route, match):
        cls = match.group("cls")
        if cls not in validator.object_info_classes:
            raise Rejected("not found", 404)
        status, body = await upstream_json("GET", f"/object_info/{cls}")
        return _json_response(scrub_json(body), status) if status == 200 and isinstance(body, dict) else _json_response({"error": "upstream_error"}, 502)

    async def h_models(request, route, match):
        folder = match.group("folder")
        if folder not in MODEL_FOLDERS:
            raise Rejected("not found", 404)
        status, body = await upstream_json("GET", f"/{route.name}/{folder}")
        if status == 404:
            raise Rejected("not found", 404)
        return _json_response(scrub_json(body)) if status == 200 and isinstance(body, list) else _json_response({"error": "upstream_error"}, 502)

    def _reply(status: int, body: Any, *, replay: str | None = None) -> JSONResponse:
        resp = JSONResponse(body, status_code=status)
        if replay:
            resp.headers["X-Idempotent-Replay"] = replay
        return resp

    def _refusal(status: int, reason: str) -> JSONResponse:
        """A refusal shaped like ComfyUI's own, so ComfyUIProvider shows the reason."""
        return _reply(status, {"error": {"type": "proxy_rejected", "message": reason, "details": "", "extra_info": {}}, "node_errors": {}})

    async def reconcile(pid: str) -> tuple[bool | None, int]:
        """READ-ONLY: is this prompt known to ComfyUI? (found, queue number). True = it is queued/running/finished; False = nowhere
        right now (this does NOT prove it never will be: a late arrival is possible); None = ComfyUI could not be asked."""
        try:
            for _ in range(2):                              # history, queue, history: a prompt moving between them is not missed
                s_h, hist = await upstream_json("GET", f"/history/{pid}")
                if s_h == 200 and isinstance(hist, dict) and pid in hist:
                    return True, 0
                s_q, queue = await upstream_json("GET", "/queue")
                if s_q == 200 and isinstance(queue, dict):
                    for key in ("queue_running", "queue_pending"):
                        for item in queue.get(key, []):
                            if isinstance(item, list) and len(item) > 1 and item[1] == pid:
                                return True, item[0] if isinstance(item[0], (int, float)) else 0
                if not (s_h == 200 and s_q == 200):
                    return None, 0
            return False, 0
        except Rejected:
            return None, 0

    async def h_prompt(request, route, match):
        body = parse_json_object(await read_body(request, policy.max_prompt_body))
        only_keys(body, {"prompt", "client_id", "prompt_id"}, {"prompt", "client_id", "prompt_id"})
        pid = body["prompt_id"]
        if not isinstance(pid, str) or not UUID_RE.match(pid):
            raise Rejected("prompt_id must be a canonical lowercase UUID")
        if not isinstance(body["client_id"], str) or not CLIENT_ID_RE.match(body["client_id"]):
            raise Rejected("client_id has an invalid format")
        validator.validate(body["prompt"])
        kind, entry = registry.begin(pid, payload_digest(body), body["client_id"])

        if kind == "full":
            return _reply(429, {"error": "busy"})
        if kind == "conflict":
            return _refusal(409, "this prompt_id was already used with a different request")
        if kind == "cancelled":
            return _refusal(409, "this prompt_id was cancelled and cannot be submitted again")
        if kind == "accepted":                               # a duplicate: answer again, never forward it a second time
            return _reply(entry.status, entry.body, replay="1")
        if kind == "pending":                                # the same request is being forwarded right now: share its outcome
            try:
                status, out = await asyncio.wait_for(asyncio.shield(entry.future), policy.read_timeout + policy.connect_timeout + 5)
            except TimeoutError:
                raise Rejected("the GPU server did not respond in time", 504) from None
            return _reply(status, out, replay="1")
        if kind == "uncertain":
            # The first attempt's outcome is unknown. Look (read-only); NEVER resubmit.
            found, number = await reconcile(pid)
            if found:
                out = {"prompt_id": pid, "number": number, "node_errors": {}}
                registry.resolve(pid, 200, out)
                return _reply(200, out, replay="reconciled")
            return _refusal(409, "the outcome of the earlier submission is unknown and it was not resubmitted; "
                                 "check the job's status, or use a new prompt_id")

        # kind == "new": this proxy has not seen the id. Look (read-only) before forwarding: ComfyUI may already know it (e.g. this proxy
        # restarted). This narrows, but cannot close, the double-run window: see the module docstring for what it cannot guarantee.
        if policy.preforward_check:
            known, number = await reconcile(pid)
            if known:
                out = {"prompt_id": pid, "number": number, "node_errors": {}}
                registry.resolve(pid, 200, out)
                return _reply(200, out, replay="existing-unverified")
            if known is None:                                    # ComfyUI could not be asked: do not forward blind
                registry.reject(pid, 502, {"error": "upstream_error"})
                return _reply(502, {"error": "upstream_error"})
        # (known is False: nothing visible yet) this call forwards it; duplicates and replays above never take a GPU-work slot
        if state["heavy"] >= policy.max_heavy:
            registry.reject(pid, 429, {"error": "busy"})         # nothing was sent: forget the id
            return JSONResponse({"error": "busy"}, status_code=429, headers={"Retry-After": "2"})
        forwarded = {"prompt": body["prompt"], "client_id": body["client_id"], "prompt_id": pid}
        state["heavy"] += 1
        try:
            status, up = await upstream_json("POST", "/prompt", forwarded)
        except Rejected as e:                                # timeout / unreachable: ComfyUI may or may not have queued it
            registry.uncertain(pid, e.status, {"error": "rejected", "reason": e.reason})
            raise
        except BaseException:                                # e.g. the client disconnected mid-forward
            registry.uncertain(pid, 502, {"error": "upstream_error"})
            raise
        finally:
            state["heavy"] -= 1
        if status == 200 and isinstance(up, dict):
            out = {k: scrub_json(up[k]) for k in ("prompt_id", "number", "node_errors") if k in up}
            registry.resolve(pid, 200, out)
            return _reply(200, out)
        if status == 400 and isinstance(up, dict):           # ComfyUI's own validation message: nothing was queued
            out = {k: scrub_json(up[k]) for k in ("error", "node_errors") if k in up}
            registry.reject(pid, 400, out)
            return _reply(400, out)
        registry.uncertain(pid, 502, {"error": "upstream_error"})
        return _reply(502, {"error": "upstream_error"})

    async def h_queue_get(request, route, match):
        status, body = await upstream_json("GET", "/queue")
        return _json_response(sanitize_queue(body)) if status == 200 else _json_response({"error": "upstream_error"}, 502)

    async def h_delete_ids(request, route, match):
        """POST /queue and POST /history: {"delete": [ids]} and nothing else ("clear" and every other field is refused)."""
        body = parse_json_object(await read_body(request, policy.max_small_body))
        only_keys(body, {"delete"}, {"delete"})
        ids = uuid_list(body["delete"], policy)
        if route.name == "queue":                            # a deleted prompt can never be submitted again through this proxy
            for pid in ids:
                registry.cancel(pid)
        status, _ = await upstream_json("POST", f"/{route.name}", {"delete": ids})
        return Response(status_code=200) if status == 200 else _json_response({"error": "upstream_error"}, 502)

    async def h_interrupt(request, route, match):
        """Targeted only: a global interrupt (no prompt_id) would stop other people's work."""
        body = parse_json_object(await read_body(request, policy.max_small_body))
        only_keys(body, {"prompt_id"}, {"prompt_id"})
        if not isinstance(body["prompt_id"], str) or not UUID_RE.match(body["prompt_id"]):
            raise Rejected("prompt_id must be a canonical lowercase UUID")
        registry.cancel(body["prompt_id"])
        status, _ = await upstream_json("POST", "/interrupt", {"prompt_id": body["prompt_id"]})
        return Response(status_code=200) if status == 200 else _json_response({"error": "upstream_error"}, 502)

    async def h_history_get(request, route, match):
        pid = match.group("pid")
        status, body = await upstream_json("GET", f"/history/{pid}")
        return _json_response(sanitize_history(body, pid)) if status == 200 else _json_response({"error": "upstream_error"}, 502)

    async def h_view(request, route, match):
        q = dict(parse_qsl(request.url.query, keep_blank_values=True))
        raw_pairs = parse_qsl(request.url.query, keep_blank_values=True)
        if len(raw_pairs) != len(q) or set(q) != {"filename", "subfolder", "type"}:
            raise Rejected("invalid view parameters")
        if q["type"] != "output" or q["subfolder"] != "vidsgen" or not VIEW_NAME_RE.match(q["filename"]):
            raise Rejected("only generated videos can be downloaded")
        headers = {}
        rng = request.headers.get("range")
        if rng is not None:
            if not RANGE_RE.match(rng):
                raise Rejected("invalid Range header", 416)
            headers["Range"] = rng
        req = client().build_request("GET", "/view", params=q, headers=headers,
                                     timeout=httpx.Timeout(policy.connect_timeout, read=policy.view_read_timeout,
                                                           write=policy.read_timeout, pool=policy.connect_timeout))
        try:
            up = await client().send(req, stream=True)
        except httpx.TimeoutException:
            raise Rejected("the GPU server did not respond in time", 504) from None
        except httpx.HTTPError:
            raise Rejected("the GPU server is unreachable", 502) from None
        if up.status_code == 416:
            await up.aclose()
            raise Rejected("range not satisfiable", 416)
        if up.status_code not in (200, 206):
            await up.aclose()
            raise Rejected("not found" if up.status_code == 404 else "the GPU server returned an error", 404 if up.status_code == 404 else 502)
        out_headers = {k: up.headers[k] for k in FORWARDED_VIEW_HEADERS if k in up.headers}
        deadline = time.monotonic() + policy.view_total_timeout

        async def body():
            try:
                async for chunk in up.aiter_bytes(1024 * 1024):      # one chunk at a time: the video is never held in memory
                    if time.monotonic() > deadline:
                        raise RuntimeError("download time limit")      # abort the response: a short body must not look complete
                    yield chunk
            finally:
                await up.aclose()
        return ClosingStreamingResponse(body(), status_code=up.status_code, headers=out_headers)

    # ------------------------------------------------------------------ routing table: THE allowlist
    ROUTES = [
        Route_(("GET",), re.compile(r"^/system_stats$"), "system_stats", "system_stats"),
        Route_(("GET",), re.compile(r"^/object_info/(?P<cls>[A-Za-z0-9_]{1,64})$"), "object_info/{class}", "object_info"),
        Route_(("GET",), re.compile(r"^/models/(?P<folder>[a-z_]{1,32})$"), "models", "models"),
        Route_(("GET",), re.compile(r"^/experiment/models/(?P<folder>[a-z_]{1,32})$"), "experiment/models", "models"),
        Route_(("POST",), re.compile(r"^/prompt$"), "prompt", "prompt"),
        Route_(("GET",), re.compile(r"^/queue$"), "queue", "queue_get"),
        Route_(("POST",), re.compile(r"^/queue$"), "queue", "delete_ids"),
        Route_(("POST",), re.compile(r"^/interrupt$"), "interrupt", "interrupt"),
        Route_(("GET",), re.compile(r"^/history/(?P<pid>[0-9a-f-]{36})$"), "history/{id}", "history_get"),
        Route_(("POST",), re.compile(r"^/history$"), "history", "delete_ids"),
        Route_(("GET",), re.compile(r"^/view$"), "view", "view"),
    ]
    HANDLERS = {"system_stats": h_system_stats, "object_info": h_object_info, "models": h_models, "prompt": h_prompt,
                "queue_get": h_queue_get, "delete_ids": h_delete_ids, "interrupt": h_interrupt, "history_get": h_history_get,
                "view": h_view}
    NO_QUERY = {"system_stats", "object_info", "models", "prompt", "queue_get", "delete_ids", "interrupt", "history_get"}
    HEADERS = {"Cache-Control": "no-store", "X-Content-Type-Options": "nosniff"}

    def deny(status: int, error: str, reason: str | None = None, *, prompt_style: bool = False) -> JSONResponse:
        body: dict = {"error": error}
        if reason is not None:
            body["reason"] = reason
        if prompt_style:       # shaped like ComfyUI's own rejection so ComfyUIProvider shows the reason
            body = {"error": {"type": "proxy_rejected", "message": reason or error, "details": "", "extra_info": {}}, "node_errors": {}}
        return JSONResponse(body, status_code=status, headers=HEADERS)

    async def dispatch(request: Request):
        started = time.monotonic()
        name, status, size = "-", 500, 0
        slot, heavy = False, False
        try:
            now = time.monotonic()
            if authorized(request):
                state["last_activity"] = clock()
            else:
                throttle.record_failure(now)
                status = 429 if throttle.blocked(now) else 401
                return JSONResponse({"error": "unauthorized" if status == 401 else "too_many_attempts"}, status_code=status,
                                    headers={**HEADERS, "WWW-Authenticate": "Bearer"} if status == 401 else {**HEADERS, "Retry-After": "60"})
            path, route, match = request.url.path, None, None
            for r in ROUTES:
                m = r.pattern.match(path)
                if m and request.method in r.methods:
                    route, match = r, m
                    break
            if route is None or (route.handler in NO_QUERY and request.url.query):
                name, status = "unrouted", 404
                return deny(404, "not_found")
            name = route.name
            heavy = route.handler == "view"                       # (POST /prompt takes a heavy slot only when it really forwards)
            if state["inflight"] >= policy.max_concurrent or (heavy and state["heavy"] >= policy.max_heavy):
                status = 429
                return JSONResponse({"error": "busy"}, status_code=429, headers={**HEADERS, "Retry-After": "2"})
            state["inflight"] += 1
            state["heavy"] += heavy
            slot = True
            resp = await HANDLERS[route.handler](request, route, match)
            if isinstance(resp, ClosingStreamingResponse):               # the slot is held until the body is sent or the client leaves
                released = []

                def release_slot():
                    if not released:
                        released.append(1)
                        state["inflight"] -= 1
                        state["heavy"] -= heavy
                resp.on_close = release_slot
                slot = False
            resp.headers.update(HEADERS)
            status = resp.status_code
            return resp
        except Rejected as e:
            status = e.status
            return deny(e.status, "not_found" if e.status == 404 else "rejected", None if e.status == 404 else e.reason,
                        prompt_style=(name == "prompt" and e.status == 400))
        except Exception:                                                  # never a traceback or internal detail to the client
            log.exception("proxy error on %s", name)
            status = 500
            return deny(500, "internal_error")
        finally:
            if slot:
                state["inflight"] -= 1
                state["heavy"] -= heavy
            # Route template, status, time and size only: never headers, query strings, bodies or the token.
            log.info("%s %s -> %d %.0fms", request.method, name, status, (time.monotonic() - started) * 1000)

    async def ws_reject(ws: WebSocket):
        await ws.close(code=1008)          # any WebSocket path other than /ws

    def _ws_authorized(ws: WebSocket) -> bool:
        return authorized(ws)             # same bearer check on the handshake (never a token in the URL)

    async def ws_relay(ws: WebSocket):
        """One-way, authenticated relay of ComfyUI progress events to the backend. Rejected (before the handshake completes) unless:
        Bearer token valid, no Origin header, exactly ?clientId=<valid id>, room for another socket, and ComfyUI accepted the
        upstream connection. Client frames are never forwarded; events are rebuilt (see sanitize_ws_event)."""
        if not _ws_authorized(ws):
            throttle.record_failure(time.monotonic())
            await ws.close(code=1008)
            return
        state["last_activity"] = clock()
        pairs = parse_qsl(ws.url.query, keep_blank_values=True)
        if "origin" in ws.headers or len(pairs) != 1 or pairs[0][0] != "clientId" or not CLIENT_ID_RE.match(pairs[0][1]):
            await ws.close(code=1008)
            return
        cid = pairs[0][1]
        if len(state["ws"]) >= policy.max_ws or cid in state["ws"]:
            await ws.close(code=1013)
            return
        state["ws"].add(cid)                                  # reserved before any await: no race between two handshakes
        upstream = None
        tasks: list[asyncio.Task] = []
        try:
            try:
                upstream = await ws_connect(f"ws://{host_port}/ws?clientId={cid}", open_timeout=policy.ws_open_timeout,
                                            max_size=4 * 1024 * 1024, ping_interval=20, ping_timeout=20, proxy=None)
            except Exception:                                 # ComfyUI down/refusing: the handshake is refused, the provider polls
                await ws.close(code=1013)
                return
            await ws.accept()

            async def pump():
                async for raw in upstream:
                    if isinstance(raw, (bytes, bytearray)) or len(raw) > policy.ws_frame_limit:
                        state["stats"]["ws_dropped"] += 1
                        continue                              # binary previews and oversized frames are dropped
                    out = sanitize_ws_event(raw, registry.is_tracked)
                    if out is None:
                        state["stats"]["ws_dropped"] += 1
                        continue
                    await asyncio.wait_for(ws.send_text(out), policy.ws_send_timeout)       # a stalled client is cut off, not buffered
                    state["stats"]["ws_events"] += 1

            async def watch():
                frames = 0
                while True:
                    msg = await ws.receive()
                    if msg["type"] == "websocket.disconnect":
                        return
                    frames += 1
                    if frames > 20 or len(msg.get("text") or msg.get("bytes") or b"") > policy.ws_inbound_limit:
                        return                                # clients have nothing to say: flooding closes the socket

            tasks = [asyncio.create_task(pump()), asyncio.create_task(watch())]
            state["ws_tasks"].update(tasks)
            await asyncio.wait(tasks, timeout=policy.ws_max_seconds, return_when=asyncio.FIRST_COMPLETED)
        finally:
            # Teardown must complete even if THIS handler is cancelled while it runs (a server shutdown, or a test client that cancels the
            # app right after disconnecting): every step is in its own try/finally, the slot is released last (so "slot free" means
            # "everything closed"), and a socket that could not be closed gracefully is aborted rather than leaked.
            try:
                for t in tasks:
                    t.cancel()
                await asyncio.gather(*tasks, return_exceptions=True)
            finally:
                try:
                    if upstream is not None:
                        try:
                            await asyncio.wait_for(upstream.close(), 2.0)
                        except (Exception, asyncio.TimeoutError):
                            pass
                finally:
                    if upstream is not None and getattr(upstream, "transport", None) is not None:
                        upstream.transport.abort()               # no-op when already closed; never leaves the socket open
                    try:
                        if ws.client_state == WebSocketState.CONNECTED:
                            # A stalled client never drains its socket, so even a close frame can block: bound it.
                            await asyncio.wait_for(ws.close(code=1000), 2.0)
                    except (RuntimeError, WebSocketDisconnect, TimeoutError):
                        pass
                    finally:
                        state["ws_tasks"].difference_update(tasks)
                        state["ws"].discard(cid)
                        state["last_activity"] = clock()

    # ------------------------------------------------------------------ optional dead-man switch (disabled unless configured)
    async def reap_once() -> int:
        """Cancel prompts THIS proxy forwarded that are still queued/running. Targeted only (never a global interrupt or clear)."""
        ids = registry.active_ids()
        if not ids:
            return 0
        try:
            status, queue = await upstream_json("GET", "/queue")
            if status != 200 or not isinstance(queue, dict):
                return 0
            running = {i[1] for i in queue.get("queue_running", []) if isinstance(i, list) and len(i) > 1}
            pending = {i[1] for i in queue.get("queue_pending", []) if isinstance(i, list) and len(i) > 1}
            reaped = 0
            for pid in ids:
                if state["ws"] or clock() - state["last_activity"] < policy.dead_man_seconds:
                    break                                         # the backend spoke while we were looking: it is alive, stop at once
                if pid in pending:
                    registry.cancel(pid)
                    await upstream_json("POST", "/queue", {"delete": [pid]})
                    reaped += 1
                elif pid in running:
                    registry.cancel(pid)
                    await upstream_json("POST", "/interrupt", {"prompt_id": pid})
                    reaped += 1
                else:
                    registry.finish(pid)                      # finished or unknown to ComfyUI: nothing to stop, stop tracking it
            if reaped:
                log.warning("dead-man switch: cancelled %d orphaned prompt(s) after %.0fs without backend activity", reaped,
                            policy.dead_man_seconds)
            return reaped
        except Rejected:
            return 0

    async def dead_man_tick() -> int:
        """Acts only when the backend has been completely silent: no authenticated request AND no open WebSocket for the whole
        period. A live backend always generates activity (its progress loop polls at least every ~10 s and keeps its WebSocket
        open), so a legitimately long GPU job is never interrupted merely for taking long."""
        if not policy.dead_man_seconds or state["ws"]:
            return 0
        if clock() - state["last_activity"] < policy.dead_man_seconds:
            return 0
        return await reap_once()

    async def dead_man_loop():
        while True:
            await asyncio.sleep(policy.dead_man_interval)
            try:
                await dead_man_tick()
            except Exception:
                log.exception("dead-man check failed")

    @asynccontextmanager
    async def lifespan(app):
        state["client"] = httpx.AsyncClient(
            base_url=f"http://{host_port}", follow_redirects=False,
            timeout=httpx.Timeout(policy.connect_timeout, read=policy.read_timeout, write=policy.read_timeout, pool=policy.connect_timeout),
            transport=transport, trust_env=False)
        if policy.dead_man_seconds:
            state["reaper"] = asyncio.create_task(dead_man_loop())
        try:
            yield
        finally:
            if state["reaper"] is not None:
                state["reaper"].cancel()
                await asyncio.gather(state["reaper"], return_exceptions=True)
            for t in list(state["ws_tasks"]):                  # tear every relay down; each handler closes its sockets
                t.cancel()
            await state["client"].aclose()

    methods = ["GET", "POST", "PUT", "DELETE", "PATCH", "HEAD", "OPTIONS"]
    app = Starlette(routes=[Route("/{path:path}", dispatch, methods=methods), Route("/", dispatch, methods=methods),
                            WebSocketRoute("/ws", ws_relay), WebSocketRoute("/{path:path}", ws_reject)],
                    lifespan=lifespan)
    app.state.policy, app.state.validator, app.state.registry = policy, validator, registry
    app.state.state, app.state.reap_once, app.state.dead_man_tick = state, reap_once, dead_man_tick
    return app


class SafeParser(argparse.ArgumentParser):
    """argparse echoes unrecognised arguments in its error text; if someone pastes a token as an option, it must not be printed back."""

    def __init__(self, *args, **kwargs):
        kwargs.setdefault("allow_abbrev", False)            # "--token" must not silently become "--token-env"
        super().__init__(*args, **kwargs)

    def error(self, message):
        import re as _re
        super().error(_re.sub(r"[A-Za-z0-9_\-]{24,}", "[redacted]", message))


# ===================================================================================== CLI
def main(argv: list[str] | None = None) -> int:
    ap = SafeParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--upstream", default="http://127.0.0.1:8188")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8189)
    ap.add_argument("--workflows-dir", required=True)
    ap.add_argument("--dead-man-seconds", type=float, default=0.0,
                    help="OFF by default (0). If set (minimum 120): cancel prompts this proxy forwarded when the backend has been "
                         "completely silent (no request and no open WebSocket) for this long")
    ap.add_argument("--max-lifetime-seconds", type=float, default=0.0,
                    help="OFF by default (0). If set, the proxy shuts itself down gracefully after this many seconds (for short, supervised runs)")
    a = ap.parse_args(argv)
    if a.max_lifetime_seconds < 0 or a.max_lifetime_seconds > 86400:
        print("FAILED: --max-lifetime-seconds must be between 0 (off) and 86400.", file=sys.stderr)
        return 2
    if a.host not in ("127.0.0.1", "localhost", "::1"):
        print("FAILED: the proxy listens on loopback only in this stage (making it reachable is a separate, approved step).", file=sys.stderr)
        return 2
    token = os.environ.get(TOKEN_ENV, "")
    try:
        app = create_app(a.upstream, token, Path(a.workflows_dir), Policy(dead_man_seconds=a.dead_man_seconds))
    except (ValueError, OSError, KeyError) as e:
        print(f"FAILED: {e if TOKEN_ENV not in str(e) else 'invalid configuration'}", file=sys.stderr)
        return 2
    import uvicorn

    logging.basicConfig(level="INFO", format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    server = uvicorn.Server(uvicorn.Config(app, host=a.host, port=a.port, **uvicorn_options()))
    if a.max_lifetime_seconds > 0:
        import threading

        timer = threading.Timer(a.max_lifetime_seconds, lambda: setattr(server, "should_exit", True))
        timer.daemon = True
        timer.start()
    server.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
