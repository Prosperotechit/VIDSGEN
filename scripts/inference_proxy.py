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

Not in Stage 1 (later stages): the /ws WebSocket relay (the provider falls back to polling without it), prompt_id
idempotency, the optional dead-man switch, and the tunnel.

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
from collections import deque
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
from starlette.websockets import WebSocket

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
               transport: httpx.AsyncBaseTransport | None = None) -> Starlette:
    if len(token) < MIN_TOKEN_CHARS:
        raise ValueError(f"the proxy token must be at least {MIN_TOKEN_CHARS} characters")
    parts = urlsplit(upstream_url)
    if parts.scheme != "http" or (parts.hostname or "") not in ("127.0.0.1", "localhost", "::1") or not parts.port:
        raise ValueError("the upstream must be a loopback ComfyUI URL like http://127.0.0.1:8188")
    logging.getLogger("httpx").setLevel(logging.WARNING)        # its INFO lines carry full upstream URLs: keep them out of the logs
    policy = policy or Policy()
    validator = GraphValidator(Path(workflows_dir), policy)
    digest = hashlib.sha256(token.encode()).digest()
    throttle = AuthThrottle(policy.auth_failures_per_minute)
    state = {"inflight": 0, "heavy": 0, "client": None}

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

    async def h_prompt(request, route, match):
        body = parse_json_object(await read_body(request, policy.max_prompt_body))
        only_keys(body, {"prompt", "client_id", "prompt_id"}, {"prompt", "client_id", "prompt_id"})
        if not isinstance(body["prompt_id"], str) or not UUID_RE.match(body["prompt_id"]):
            raise Rejected("prompt_id must be a canonical lowercase UUID")
        if not isinstance(body["client_id"], str) or not CLIENT_ID_RE.match(body["client_id"]):
            raise Rejected("client_id has an invalid format")
        validator.validate(body["prompt"])
        status, up = await upstream_json("POST", "/prompt", {"prompt": body["prompt"], "client_id": body["client_id"],
                                                             "prompt_id": body["prompt_id"]})
        if status == 200 and isinstance(up, dict):
            return _json_response({k: scrub_json(up[k]) for k in ("prompt_id", "number", "node_errors") if k in up})
        if status == 400 and isinstance(up, dict):                              # ComfyUI's own validation message, scrubbed
            return _json_response({k: scrub_json(up[k]) for k in ("error", "node_errors") if k in up}, 400)
        return _json_response({"error": "upstream_error"}, 502)

    async def h_queue_get(request, route, match):
        status, body = await upstream_json("GET", "/queue")
        return _json_response(sanitize_queue(body)) if status == 200 else _json_response({"error": "upstream_error"}, 502)

    async def h_delete_ids(request, route, match):
        """POST /queue and POST /history: {"delete": [ids]} and nothing else ("clear" and every other field is refused)."""
        body = parse_json_object(await read_body(request, policy.max_small_body))
        only_keys(body, {"delete"}, {"delete"})
        ids = uuid_list(body["delete"], policy)
        status, _ = await upstream_json("POST", f"/{route.name}", {"delete": ids})
        return Response(status_code=200) if status == 200 else _json_response({"error": "upstream_error"}, 502)

    async def h_interrupt(request, route, match):
        """Targeted only: a global interrupt (no prompt_id) would stop other people's work."""
        body = parse_json_object(await read_body(request, policy.max_small_body))
        only_keys(body, {"prompt_id"}, {"prompt_id"})
        if not isinstance(body["prompt_id"], str) or not UUID_RE.match(body["prompt_id"]):
            raise Rejected("prompt_id must be a canonical lowercase UUID")
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
        if up.status_code not in (200, 206):
            await up.aclose()
            raise Rejected("not found" if up.status_code == 404 else "the GPU server returned an error", 404 if up.status_code == 404 else 502)
        out_headers = {k: up.headers[k] for k in FORWARDED_VIEW_HEADERS if k in up.headers}
        deadline = time.monotonic() + policy.view_total_timeout

        async def body():
            try:
                async for chunk in up.aiter_bytes(1024 * 1024):
                    if time.monotonic() > deadline:
                        break                                                    # bounded: the client sees a short body
                    yield chunk
            finally:
                await up.aclose()
        return StreamingResponse(body(), status_code=up.status_code, headers=out_headers)

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
            if not authorized(request):
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
            heavy = route.handler in ("prompt", "view")
            if state["inflight"] >= policy.max_concurrent or (heavy and state["heavy"] >= policy.max_heavy):
                status = 429
                return JSONResponse({"error": "busy"}, status_code=429, headers={**HEADERS, "Retry-After": "2"})
            state["inflight"] += 1
            state["heavy"] += heavy
            slot = True
            resp = await HANDLERS[route.handler](request, route, match)
            if isinstance(resp, StreamingResponse):                       # the slot is held until the body has been sent
                inner = resp.body_iterator

                async def held():
                    try:
                        async for chunk in inner:
                            yield chunk
                    finally:
                        state["inflight"] -= 1
                        state["heavy"] -= heavy
                resp.body_iterator = held()
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
        await ws.close(code=1008)          # Stage 1: no WebSocket relay yet (rejected before the handshake completes)

    @asynccontextmanager
    async def lifespan(app):
        state["client"] = httpx.AsyncClient(
            base_url=f"http://{parts.hostname if parts.hostname != '::1' else '[::1]'}:{parts.port}", follow_redirects=False,
            timeout=httpx.Timeout(policy.connect_timeout, read=policy.read_timeout, write=policy.read_timeout, pool=policy.connect_timeout),
            transport=transport, trust_env=False)
        try:
            yield
        finally:
            await state["client"].aclose()

    methods = ["GET", "POST", "PUT", "DELETE", "PATCH", "HEAD", "OPTIONS"]
    app = Starlette(routes=[Route("/{path:path}", dispatch, methods=methods), Route("/", dispatch, methods=methods),
                            WebSocketRoute("/{path:path}", ws_reject)],
                    lifespan=lifespan)
    app.state.policy, app.state.validator = policy, validator
    return app


# ===================================================================================== CLI
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--upstream", default="http://127.0.0.1:8188")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8189)
    ap.add_argument("--workflows-dir", required=True)
    a = ap.parse_args(argv)
    if a.host not in ("127.0.0.1", "localhost", "::1"):
        print("FAILED: the proxy listens on loopback only in this stage (making it reachable is a separate, approved step).", file=sys.stderr)
        return 2
    token = os.environ.get(TOKEN_ENV, "")
    try:
        app = create_app(a.upstream, token, Path(a.workflows_dir))
    except (ValueError, OSError, KeyError) as e:
        print(f"FAILED: {e if TOKEN_ENV not in str(e) else 'invalid configuration'}", file=sys.stderr)
        return 2
    import uvicorn

    logging.basicConfig(level="INFO", format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    uvicorn.run(app, host=a.host, port=a.port, log_level="warning", access_log=False, server_header=False, date_header=False,
                timeout_keep_alive=5, limit_concurrency=64, h11_max_incomplete_event_size=16 * 1024)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
