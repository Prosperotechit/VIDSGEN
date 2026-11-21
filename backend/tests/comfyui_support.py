"""Test support for the ComfyUI provider (no GPU, no model weights).

* ``validate_graph()``: checks an API-format graph against ComfyUI ``/object_info`` schemas
  (node classes, input names incl. nested DynamicCombo inputs, types, ranges, combo values
  and link types), the way ComfyUI's own validation does. The fixture
  ``fixtures/comfyui/object_info.json`` was captured from ComfyUI 0.46.0.
* ``FakeComfyUI``: a real HTTP + WebSocket server (Starlette on uvicorn, random port) that
  imitates ComfyUI's API. Every response and event shape is copied from a real ComfyUI
  0.46.0 recorded during Phase 4 (see the provider docstring). Scenarios are scriptable.
"""

from __future__ import annotations

import asyncio
import json
import threading
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import uvicorn
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, PlainTextResponse, Response
from starlette.routing import Route, WebSocketRoute
from starlette.websockets import WebSocket, WebSocketDisconnect

FIXTURES = Path(__file__).parent / "fixtures" / "comfyui"
OBJECT_INFO: dict[str, Any] = json.loads((FIXTURES / "object_info.json").read_text())


# ====================================================================== schema validation
def _expand(spec_inputs: dict[str, Any], values: dict[str, Any], prefix: str = "") -> dict[str, tuple[Any, bool]]:
    """Flatten an object_info input spec into {name: (spec, required)}, expanding
    DynamicCombo inputs according to the chosen values (names become dotted)."""
    out: dict[str, tuple[Any, bool]] = {}
    for section, required in (("required", True), ("optional", False)):
        for name, spec in (spec_inputs.get(section) or {}).items():
            full = f"{prefix}{name}"
            out[full] = (spec, required)
            if isinstance(spec, list) and spec and spec[0] == "COMFY_DYNAMICCOMBO_V3":
                chosen = values.get(full)
                for opt in spec[1].get("options", []):
                    if opt["key"] == chosen:
                        out.update(_expand(opt.get("inputs", {}), values, prefix=f"{full}."))
    return out


def validate_graph(graph: dict[str, Any], object_info: dict[str, Any] = OBJECT_INFO) -> list[str]:
    errors: list[str] = []
    for nid, node in graph.items():
        cls = node.get("class_type")
        info = object_info.get(cls)
        if info is None:
            errors.append(f"{nid}: unknown node class {cls}")
            continue
        values = node.get("inputs", {})
        spec = _expand(info.get("input", {}), values)
        for name, (s, required) in spec.items():
            if required and name not in values:
                errors.append(f"{nid} {cls}: missing required input {name}")
        for name, value in values.items():
            if name not in spec:
                errors.append(f"{nid} {cls}: unexpected input {name}")
                continue
            s = spec[name][0]
            kind, opts = (s[0], s[1] if len(s) > 1 else {}) if isinstance(s, list) else (s, {})
            if isinstance(value, list) and len(value) == 2 and isinstance(value[0], str):
                src = graph.get(value[0])
                if src is None:
                    errors.append(f"{nid} {cls}.{name}: link to missing node {value[0]}")
                    continue
                out_types = object_info[src["class_type"]]["output"]
                if value[1] >= len(out_types) or out_types[value[1]] != kind:
                    errors.append(f"{nid} {cls}.{name}: link type {out_types[value[1]] if value[1] < len(out_types) else '?'} != {kind}")
                continue
            if isinstance(kind, list):  # combo
                if kind and value not in kind:
                    errors.append(f"{nid} {cls}.{name}: {value!r} not in {kind}")
            elif kind == "COMBO":  # V3 node combo: ["COMBO", {"options": [...]}]
                if value not in opts.get("options", []):
                    errors.append(f"{nid} {cls}.{name}: {value!r} not in {opts.get('options')}")
            elif kind == "COMFY_DYNAMICCOMBO_V3":
                keys = [o["key"] for o in opts.get("options", [])]
                if value not in keys:
                    errors.append(f"{nid} {cls}.{name}: {value!r} not in {keys}")
            elif kind == "INT":
                if not isinstance(value, int) or isinstance(value, bool):
                    errors.append(f"{nid} {cls}.{name}: {value!r} is not an int")
                elif not (opts.get("min", value) <= value <= opts.get("max", value)):
                    errors.append(f"{nid} {cls}.{name}: {value} outside [{opts.get('min')}, {opts.get('max')}]")
                elif opts.get("step") and (value - opts.get("min", 0)) % opts["step"] != 0:
                    errors.append(f"{nid} {cls}.{name}: {value} not on step {opts['step']} from {opts.get('min', 0)}")
            elif kind == "FLOAT":
                if not isinstance(value, (int, float)) or isinstance(value, bool):
                    errors.append(f"{nid} {cls}.{name}: {value!r} is not a number")
                elif not (opts.get("min", value) <= value <= opts.get("max", value)):
                    errors.append(f"{nid} {cls}.{name}: {value} outside [{opts.get('min')}, {opts.get('max')}]")
            elif kind == "STRING":
                if not isinstance(value, str):
                    errors.append(f"{nid} {cls}.{name}: {value!r} is not a string")
            else:
                errors.append(f"{nid} {cls}.{name}: value given for link-only input of type {kind}")
    return errors


# ====================================================================== fake server
SYSTEM_STATS = {
    "system": {"os": "linux", "comfyui_version": "0.46.0", "python_version": "3.13", "pytorch_version": "2.15.0",
               "required_frontend_version": "1.59.6", "embedded_python": False, "argv": ["main.py"]},
    "devices": [{"name": "cuda:0 Tesla T4 : cudaMallocAsync", "type": "cuda", "index": 0,
                 "vram_total": 15828320256, "vram_free": 15500000000, "torch_vram_total": 0, "torch_vram_free": 0}],
}
# Exact official sizes (bytes), as in workflows/comfyui/models.yaml.
OFFICIAL_SIZES = {
    "wan2.2_ti2v_5B_fp16.safetensors": 9999658848,
    "umt5_xxl_fp8_e4m3fn_scaled.safetensors": 6735906897,
    "wan2.2_vae.safetensors": 1409400960,
}
MODELS = {
    "diffusion_models": ["wan2.2_ti2v_5B_fp16.safetensors"],
    "text_encoders": ["umt5_xxl_fp8_e4m3fn_scaled.safetensors"],
    "vae": ["wan2.2_vae.safetensors"],
}


class _Outage:
    """ASGI wrapper: while `fake.outage` is set, plain HTTP requests are answered 503 (WebSockets are untouched)."""

    def __init__(self, app, fake):
        self.app, self.fake = app, fake

    async def __call__(self, scope, receive, send):
        if scope["type"] == "http" and self.fake.outage:
            await PlainTextResponse("Service Unavailable", status_code=503)(scope, receive, send)
            return
        await self.app(scope, receive, send)


class FakeComfyUI:
    """Imitates a ComfyUI server. Runs prompts one at a time, like ComfyUI.

    Scenario per prompt (from ``next_scenarios``, else a ``[fake:<name>]`` tag in the positive
    prompt text, else ``default_scenario``):
      success, error, oom, no_output, hang (until interrupted), silent_success (no WS events),
      ws_drop (WebSocket closes mid-run, job still succeeds), webm (non-MP4 output).
    """

    def __init__(self, video_bytes: bytes, token: str | None = None):
        self.video_bytes = video_bytes
        self.token = token
        self.models = {k: list(v) for k, v in MODELS.items()}
        self.missing_nodes: set[str] = set()
        self.sizes: dict[str, int] = dict(OFFICIAL_SIZES)   # what /experiment/models reports
        self.sizes_route = True                             # False: route absent (404), like other ComfyUI versions
        self.default_scenario = "success"
        self.next_scenarios: list[str] = []
        self.step_delay = 0.005
        self.paused = False                  # True: keep prompts pending (queued) until resumed
        self.foreign_events = False          # inject events for unknown prompt ids
        self.ignore_client_prompt_id = False
        self.validate_prompts = True         # False: accept graphs that aren't the Wan workflow
        self.prompt_mode = "ok"              # ok | reject_model | reject_image | server_error | malformed | hang | no_id | queue_then_hang | queue_then_500
        self.upload_mode = "ok"              # ok | error
        self.view_override: tuple[int, bytes] | None = None
        self.history_delete_mode = "ok"
        self.outage = False                  # True: every HTTP request gets 503 (a proxy/tunnel with no upstream)
        self.view_failures = 0               # the next N downloads (/view) get 503, then work
        # observed
        self.requests: list[dict[str, Any]] = []
        self.prompts: dict[str, dict[str, Any]] = {}
        self.uploads: dict[str, bytes] = {}
        self.history: dict[str, dict[str, Any]] = {}
        self.interrupts: list[Any] = []
        self.deleted: list[str] = []
        self.ws_auth_headers: list[str | None] = []
        self.ws_received: list[Any] = []     # frames clients sent to /ws
        # internals
        self._pending: list[str] = []
        self._running: str | None = None
        self._interrupt: set[str] = set()
        self._sockets: dict[str, list[WebSocket]] = {}
        self._number = 0
        self._wake: asyncio.Event | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self.app = Starlette(
            routes=[
                Route("/system_stats", self.system_stats),
                Route("/object_info/{cls}", self.object_info),
                Route("/models/{folder}", self.models_list),
                Route("/experiment/models/{folder}", self.experiment_models),
                Route("/upload/image", self.upload_image, methods=["POST"]),
                Route("/prompt", self.post_prompt, methods=["POST"]),
                Route("/queue", self.get_queue, methods=["GET"]),
                Route("/queue", self.post_queue, methods=["POST"]),
                Route("/interrupt", self.post_interrupt, methods=["POST"]),
                Route("/history/{pid}", self.get_history),
                Route("/history", self.post_history, methods=["POST"]),
                Route("/view", self.view),
                WebSocketRoute("/ws", self.ws),
            ],
            lifespan=self._lifespan,
        )

    # ------------------------------------------------------------------ lifecycle
    def start(self) -> FakeComfyUI:
        config = uvicorn.Config(_Outage(self.app, self), host="127.0.0.1", port=0, log_level="warning", ws="websockets-sansio", lifespan="on",
                                timeout_graceful_shutdown=1)
        self._server = uvicorn.Server(config)
        self._thread = threading.Thread(target=self._server.run, daemon=True)
        self._thread.start()
        deadline = time.monotonic() + 10
        while not self._server.started:
            if time.monotonic() > deadline:
                raise RuntimeError("fake ComfyUI did not start")
            time.sleep(0.01)
        self.port = self._server.servers[0].sockets[0].getsockname()[1]
        self.url = f"http://127.0.0.1:{self.port}"
        return self

    def stop(self) -> None:
        self._server.should_exit = True
        self._thread.join(timeout=10)

    @asynccontextmanager
    async def _lifespan(self, app):
        self._loop = asyncio.get_running_loop()
        self._wake = asyncio.Event()
        worker = self._loop.create_task(self._worker())
        try:
            yield
        finally:
            worker.cancel()

    def resume(self) -> None:
        self.paused = False
        if self._loop:
            self._loop.call_soon_threadsafe(self._wake.set)

    def inject(self, client_id: str, payload: str | bytes) -> None:
        """Send a raw frame (text or binary) to every socket of `client_id`, as ComfyUI would."""
        async def go():
            for ws in list(self._sockets.get(client_id, [])):
                try:
                    await (ws.send_bytes(payload) if isinstance(payload, bytes) else ws.send_text(payload))
                except Exception:
                    pass
        asyncio.run_coroutine_threadsafe(go(), self._loop).result(5)

    def drop_websockets(self) -> None:
        if self._loop:
            asyncio.run_coroutine_threadsafe(self._close_all(), self._loop).result(5)

    # ------------------------------------------------------------------ helpers
    def _authorized(self, headers) -> bool:
        return self.token is None or headers.get("authorization") == f"Bearer {self.token}"

    def _log(self, request: Request, body: Any = None) -> None:
        self.requests.append({"method": request.method, "path": request.url.path, "query": dict(request.query_params),
                              "auth": request.headers.get("authorization"), "body": body})

    async def _send(self, client_id: str | None, kind: str, data: dict[str, Any]) -> None:
        targets = self._sockets.get(client_id, []) if client_id else [s for l in self._sockets.values() for s in l]
        for ws in list(targets):
            try:
                await ws.send_text(json.dumps({"type": kind, "data": data}))
            except Exception:
                pass

    async def _close_all(self) -> None:
        for sockets in list(self._sockets.values()):
            for ws in list(sockets):
                try:
                    await ws.close(code=1011)
                except Exception:
                    pass
        self._sockets.clear()

    def _scenario(self, graph: dict[str, Any]) -> str:
        if self.next_scenarios:
            return self.next_scenarios.pop(0)
        text = str(graph.get("6", {}).get("inputs", {}).get("text", ""))
        if "[fake:" in text:
            return text.split("[fake:", 1)[1].split("]", 1)[0]
        return self.default_scenario

    # ------------------------------------------------------------------ HTTP routes
    async def system_stats(self, request: Request):
        self._log(request)
        if not self._authorized(request.headers):
            return PlainTextResponse("Unauthorized", status_code=401)
        return JSONResponse(SYSTEM_STATS)

    async def object_info(self, request: Request):
        cls = request.path_params["cls"]
        if cls in self.missing_nodes or cls not in OBJECT_INFO:
            return JSONResponse({})
        return JSONResponse({cls: OBJECT_INFO[cls]})

    async def models_list(self, request: Request):
        return JSONResponse(self.models.get(request.path_params["folder"], []))

    async def experiment_models(self, request: Request):
        # Shape recorded from ComfyUI 0.46.0: [{name, pathIndex, modified, created, size}]
        if not self.sizes_route:
            return PlainTextResponse("404: Not Found", status_code=404)
        folder = request.path_params["folder"]
        if folder not in self.models:
            return PlainTextResponse("404: Not Found", status_code=404)
        return JSONResponse([{"name": n, "pathIndex": 0, "modified": 1795102829.98, "created": 1795102829.98,
                              "size": self.sizes.get(n, 0)} for n in self.models[folder]])

    async def upload_image(self, request: Request):
        async with request.form() as form:
            self._log(request, {k: (v if isinstance(v, str) else f"<file {v.filename}>") for k, v in form.items()})
            if not self._authorized(request.headers):
                return PlainTextResponse("Unauthorized", status_code=401)
            if self.upload_mode == "error":
                return PlainTextResponse("boom", status_code=500)
            f = form["image"]
            data = await f.read()
            sub = form.get("subfolder", "")
        self.uploads[f"{sub}/{f.filename}" if sub else f.filename] = data
        return JSONResponse({"name": f.filename, "subfolder": sub, "type": form.get("type", "input")})

    async def post_prompt(self, request: Request):
        body = await request.json()
        self._log(request, body)
        if not self._authorized(request.headers):
            return PlainTextResponse("Unauthorized", status_code=401)
        mode = self.prompt_mode
        if mode == "hang":
            await asyncio.sleep(3600)
        if mode == "server_error":
            return PlainTextResponse("Internal Server Error", status_code=500)
        if mode == "malformed":
            return PlainTextResponse("<html>not json</html>", status_code=200)
        if mode == "no_id":
            return JSONResponse({"number": 1, "node_errors": {}})
        graph = body["prompt"]
        if mode == "reject_model":   # recorded from ComfyUI 0.46.0
            return JSONResponse({"error": {"type": "prompt_outputs_failed_validation", "message": "Prompt outputs failed validation",
                                           "details": "", "extra_info": {}},
                                 "node_errors": {"39": {"errors": [{"type": "value_not_in_list", "message": "Value not in list",
                                     "details": "vae_name: 'wan2.2_vae.safetensors' not in []",
                                     "extra_info": {"input_name": "vae_name"}}], "dependent_outputs": ["58"], "class_type": "VAELoader"}}},
                                status_code=400)
        errors = validate_graph(graph) if self.validate_prompts else []
        image = graph.get("56", {}).get("inputs", {}).get("image")
        if image is not None and image not in self.uploads:
            errors.append(f"image - Invalid image file: {image}")
        if errors or mode == "reject_image":
            return JSONResponse({"error": {"type": "prompt_outputs_failed_validation", "message": "Prompt outputs failed validation",
                                           "details": "", "extra_info": {}},
                                 "node_errors": {"56": {"errors": [{"type": "custom_validation_failed",
                                     "message": "Custom validation failed for node", "details": "; ".join(errors) or "image - Invalid image file"}],
                                     "dependent_outputs": ["58"], "class_type": "LoadImage"}}}, status_code=400)
        pid = body.get("prompt_id") if not self.ignore_client_prompt_id else None
        pid = pid or str(uuid.uuid4())
        self._number += 1
        self.prompts[pid] = {"graph": graph, "client_id": body.get("client_id"), "scenario": self._scenario(graph),
                             "number": self._number}
        self._pending.append(pid)
        self._wake.set()
        if mode == "queue_then_hang":        # the prompt IS queued, but the response never arrives (a lost response)
            await asyncio.sleep(3600)
        if mode == "queue_then_500":
            return PlainTextResponse("Internal Server Error", status_code=500)
        return JSONResponse({"prompt_id": pid, "number": self._number, "node_errors": {}})

    async def get_queue(self, request: Request):
        self._log(request)
        if not self._authorized(request.headers):
            return PlainTextResponse("Unauthorized", status_code=401)
        item = lambda pid: [self.prompts[pid]["number"], pid, {}, {"client_id": self.prompts[pid]["client_id"]}, ["58"]]
        return JSONResponse({"queue_running": [item(self._running)] if self._running else [],
                             "queue_pending": [item(p) for p in self._pending]})

    async def post_queue(self, request: Request):
        body = await request.json()
        self._log(request, body)
        for pid in body.get("delete", []):
            if pid in self._pending:
                self._pending.remove(pid)
                self.deleted.append(pid)
        return Response(status_code=200)

    async def post_interrupt(self, request: Request):
        body = await request.json() if await request.body() else {}
        self._log(request, body)
        self.interrupts.append(body.get("prompt_id"))
        target = body.get("prompt_id")
        if self._running and (target is None or target == self._running):
            self._interrupt.add(self._running)
        return Response(status_code=200)

    async def get_history(self, request: Request):
        self._log(request)
        if not self._authorized(request.headers):
            return PlainTextResponse("Unauthorized", status_code=401)
        pid = request.path_params["pid"]
        return JSONResponse({pid: self.history[pid]} if pid in self.history else {})

    async def post_history(self, request: Request):
        body = await request.json()
        self._log(request, body)
        for pid in body.get("delete", []):
            self.history.pop(pid, None)
        return Response(status_code=200)

    async def view(self, request: Request):
        self._log(request)
        if not self._authorized(request.headers):
            return PlainTextResponse("Unauthorized", status_code=401)
        if self.view_failures > 0:
            self.view_failures -= 1
            return PlainTextResponse("Service Unavailable", status_code=503)
        if self.view_override:
            return Response(self.view_override[1], status_code=self.view_override[0], media_type="application/octet-stream")
        q = request.query_params
        if q.get("type") != "output" or q.get("subfolder") != "vidsgen" or not q.get("filename", "").startswith("job_"):
            return PlainTextResponse("404: Not Found", status_code=404)
        return Response(self.video_bytes, media_type="video/mp4")

    async def ws(self, websocket: WebSocket):
        self.ws_auth_headers.append(websocket.headers.get("authorization"))
        if not self._authorized(websocket.headers):
            await websocket.close(code=1008)
            return
        await websocket.accept()
        cid = websocket.query_params.get("clientId") or uuid.uuid4().hex
        self._sockets.setdefault(cid, []).append(websocket)
        await websocket.send_text(json.dumps({"type": "status", "data": {"status": {"exec_info": {"queue_remaining": len(self._pending)}}, "sid": cid}}))
        try:
            while True:
                self.ws_received.append(await websocket.receive_text())
        except (WebSocketDisconnect, Exception):
            pass
        finally:
            if websocket in self._sockets.get(cid, []):
                self._sockets[cid].remove(websocket)

    # ------------------------------------------------------------------ execution
    async def _worker(self) -> None:
        while True:
            while self.paused or not self._pending:
                self._wake.clear()
                await self._wake.wait()
            pid = self._pending.pop(0)
            self._running = pid
            try:
                await self._execute(pid)
            finally:
                self._running = None

    def _ts(self) -> int:
        return int(time.time() * 1000)

    async def _execute(self, pid: str) -> None:
        p = self.prompts[pid]
        cid, graph, scenario = p["client_id"], p["graph"], p["scenario"]
        silent = scenario in ("silent_success", "silent_oom")
        msgs: list[list[Any]] = []

        async def emit(kind: str, data: dict[str, Any], record: bool = False):
            data = {**data, "prompt_id": pid}
            if record:
                msgs.append([kind, {**data, "timestamp": self._ts()}])
            if not silent:
                await self._send(cid, kind, data)

        async def interrupted(node: str) -> bool:
            if pid in self._interrupt:
                self._interrupt.discard(pid)
                await emit("execution_interrupted", {"node_id": node, "node_type": graph[node]["class_type"], "executed": []}, record=True)
                self.history[pid] = {"prompt": [], "outputs": {}, "status": {"status_str": "error", "completed": False, "messages": msgs}, "meta": {}}
                return True
            return False

        await emit("execution_start", {"timestamp": self._ts()}, record=True)
        await emit("execution_cached", {"nodes": [], "timestamp": self._ts()}, record=True)
        for node in ("37", "38", "39", "48"):
            await emit("executing", {"node": node, "display_node": node})
            await asyncio.sleep(self.step_delay)
        if scenario in ("error", "oom", "oom_ram", "silent_oom"):
            message, etype = {
                "oom": ("CUDA out of memory. Tried to allocate 2.00 GiB", "torch.OutOfMemoryError"),
                "silent_oom": ("CUDA out of memory. Tried to allocate 2.00 GiB", "torch.OutOfMemoryError"),
                "oom_ram": ("[enforce fail at alloc_cpu.cpp] DefaultCPUAllocator: can't allocate memory", "RuntimeError"),
            }.get(scenario, ("Error while deserializing header: header too small", "safetensors._safetensors_rust.SafetensorError"))
            err = {"node_id": "39", "node_type": "VAELoader", "executed": ["37", "38"],
                   "exception_message": message, "exception_type": etype,
                   "traceback": ["  File \"/srv/ComfyUI/execution.py\", line 1, in x\n"], "current_inputs": {}, "current_outputs": {}}
            await emit("execution_error", err, record=True)
            self.history[pid] = {"prompt": [], "outputs": {}, "status": {"status_str": "error", "completed": False, "messages": msgs}, "meta": {}}
            return
        await emit("executing", {"node": "3", "display_node": "3"})
        steps = int(graph["3"]["inputs"]["steps"])
        i = 0
        while True:
            if await interrupted("3"):
                return
            if scenario != "hang":
                i += 1
                await emit("progress", {"value": i, "max": steps, "node": "3"})
                if self.foreign_events:
                    await self._send(cid, "progress", {"value": 1, "max": 99, "node": "3", "prompt_id": str(uuid.uuid4())})
                    await self._send(cid, "progress", {"value": 1, "max": 99, "node": "3"})  # no prompt_id at all
                if scenario == "ws_drop" and i == max(1, steps // 2):
                    await self._close_all()
                if i >= steps:
                    break
            await asyncio.sleep(self.step_delay)
        for node in ("8", "57", "58"):
            await emit("executing", {"node": node, "display_node": node})
            await asyncio.sleep(self.step_delay)
        prefix = graph["58"]["inputs"]["filename_prefix"]
        sub, name = prefix.split("/", 1) if "/" in prefix else ("", prefix)
        ext = ".webm" if scenario == "webm" else ".mp4"
        output = {"images": [{"filename": f"{name}_00001_{ext}", "subfolder": sub, "type": "output",
                              "metadata": {"kind": "video", "width": graph["55"]["inputs"]["width"],
                                           "height": graph["55"]["inputs"]["height"], "fps": float(graph["57"]["inputs"]["fps"]),
                                           "frame_count": graph["55"]["inputs"]["length"]}}], "animated": [True]}
        outputs = {} if scenario == "no_output" else {"58": output}
        if outputs:
            await emit("executed", {"node": "58", "display_node": "58", "output": output})
        self.history[pid] = {"prompt": [], "outputs": outputs, "status": {"status_str": "success", "completed": True, "messages": msgs}, "meta": {}}
        await emit("execution_success", {"timestamp": self._ts()}, record=True)
