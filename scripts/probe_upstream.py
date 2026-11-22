#!/usr/bin/env python3
"""A DUMMY ComfyUI for connectivity testing (Phase 6C Stage 4A). It runs on CPU, needs no models and no GPU, and binds to loopback only.

It imitates just the routes the VIDSGEN inference proxy forwards, with shapes copied from the recorded ComfyUI 0.46.0 responses:
  GET /system_stats, /object_info/<class>, /models/<f>, /experiment/models/<f>, /queue, /history/<id>, /view (a large, deterministic,
  Range-capable download), POST /prompt, /queue, /history, /interrupt, and a WebSocket /ws that emits progress events.

Test controls (talk to this server DIRECTLY on loopback; the proxy never forwards them):
  GET  /__probe_stats      counters: prompts received, views, websocket connections...
  POST /__probe_control    {"hang_prompt": bool, "outage": bool, "view_delay": seconds}

The downloaded bytes are a pure function of (size, seed), so a client can verify an exact SHA-256 without the server's help:
see `expected_sha256()`.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import re
import sys
import threading
import time
import uuid
from contextlib import asynccontextmanager

from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, PlainTextResponse, Response, StreamingResponse
from starlette.routing import Route, WebSocketRoute
from starlette.websockets import WebSocket, WebSocketDisconnect

BLOCK = 1024 * 1024
MP4_HEAD = b"\x00\x00\x00\x18ftypmp42\x00\x00\x00\x00mp42isom"
VIEW_NAME_RE = re.compile(r"^job_[0-9a-f]{24}_\d{5,}_?\.mp4$")
RANGE_RE = re.compile(r"^bytes=(\d*)-(\d*)$")
LOOPBACK = ("127.0.0.1", "localhost", "::1")


def block_bytes(index: int, seed: int) -> bytes:
    """Block `index` of the deterministic stream (1 MiB, pseudo-random; block 0 starts with an MP4 'ftyp' box)."""
    data = hashlib.shake_256(seed.to_bytes(8, "big") + index.to_bytes(8, "big")).digest(BLOCK)
    return MP4_HEAD + data[len(MP4_HEAD):] if index == 0 else data


def iter_stream(size: int, seed: int, start: int = 0, end: int | None = None):
    """The bytes [start, end] (inclusive; default to the end) of the deterministic stream of `size` bytes."""
    end = size - 1 if end is None else min(end, size - 1)
    pos = start
    while pos <= end:
        index, offset = divmod(pos, BLOCK)
        chunk = block_bytes(index, seed)[offset:offset + (end - pos + 1)]
        yield chunk
        pos += len(chunk)


def expected_sha256(size: int, seed: int) -> str:
    h = hashlib.sha256()
    for chunk in iter_stream(size, seed):
        h.update(chunk)
    return h.hexdigest()


def create_app(download_bytes: int, seed: int = 7, event_delay: float = 0.02, steps: int = 20) -> Starlette:
    state = {
        "prompts": {}, "history": {}, "running": None, "pending": [], "number": 0, "sockets": {}, "interrupts": [], "deleted": [],
        "stats": {"prompts_received": 0, "views": 0, "ws_connections": 0, "requests": 0}, "control": {"hang_prompt": False, "outage": False, "view_delay": 0.0},
    }

    def event(kind: str, data: dict) -> str:
        return json.dumps({"type": kind, "data": data})

    async def send(cid: str, kind: str, data: dict) -> None:
        for ws in list(state["sockets"].get(cid, [])):
            try:
                await ws.send_text(event(kind, data))
            except Exception:
                pass

    async def run_prompt(pid: str) -> None:
        p = state["prompts"][pid]
        cid = p["client_id"]
        state["running"] = pid
        try:
            await send(cid, "execution_start", {"prompt_id": pid, "timestamp": int(time.time() * 1000)})
            await send(cid, "executing", {"prompt_id": pid, "node": "3", "display_node": "3"})
            for i in range(1, steps + 1):
                if pid in state["interrupts"]:
                    await send(cid, "execution_interrupted", {"prompt_id": pid, "node_id": "3", "node_type": "KSampler", "executed": []})
                    state["history"][pid] = {"prompt": [], "outputs": {}, "status": {"status_str": "error", "completed": False, "messages": [
                        ["execution_interrupted", {"prompt_id": pid}]]}}
                    return
                await send(cid, "progress", {"prompt_id": pid, "value": i, "max": steps, "node": "3", "display_node": "3"})
                await asyncio.sleep(event_delay)
            name = p["prefix"].split("/")[-1]
            state["history"][pid] = {"prompt": [1, pid, {"6": {"inputs": {"text": "PROBE PROMPT"}}}], "meta": {"path": "/probe/secret/path"},
                                     "outputs": {"58": {"images": [{"filename": f"{name}_00001_.mp4", "subfolder": "vidsgen", "type": "output",
                                                                    "metadata": {"kind": "video"}}], "animated": [True]}},
                                     "status": {"status_str": "success", "completed": True, "messages": []}}
            await send(cid, "execution_success", {"prompt_id": pid, "timestamp": int(time.time() * 1000)})
        finally:
            state["running"] = None
            if pid in state["pending"]:
                state["pending"].remove(pid)

    def guard(request: Request):
        state["stats"]["requests"] += 1
        if state["control"]["outage"]:
            return PlainTextResponse("outage", status_code=503)
        return None

    async def system_stats(request: Request):
        return guard(request) or JSONResponse({
            "system": {"os": "posix", "python_version": "3.13", "pytorch_version": "probe", "comfyui_version": "0.0.0-probe",
                       "argv": ["/probe/ComfyUI/main.py", "--base-directory", "/probe/secret/data"], "embedded_python": False},
            "devices": [{"name": "cpu (dummy upstream)", "type": "cpu", "index": 0, "vram_total": 0, "vram_free": 0, "torch_vram_total": 0,
                         "torch_vram_free": 0, "path": "/dev/probe"}]})

    async def object_info(request: Request):
        cls = request.path_params["cls"]
        return guard(request) or JSONResponse({cls: {"input": {"required": {}}, "output": [], "python_module": "probe.nodes"}})

    async def models(request: Request):
        return guard(request) or JSONResponse([])

    async def post_prompt(request: Request):
        if (r := guard(request)):
            return r
        body = await request.json()
        if state["control"]["hang_prompt"]:
            await asyncio.sleep(3600)
        pid = body.get("prompt_id") or str(uuid.uuid4())
        state["stats"]["prompts_received"] += 1
        graph = body.get("prompt") or {}
        prefix = str(((graph.get("58") or {}).get("inputs") or {}).get("filename_prefix", "vidsgen/job_" + "0" * 24))
        state["number"] += 1
        state["prompts"][pid] = {"client_id": body.get("client_id", ""), "prefix": prefix, "number": state["number"]}
        state["pending"].append(pid)
        asyncio.get_running_loop().create_task(run_prompt(pid))
        return JSONResponse({"prompt_id": pid, "number": state["number"], "node_errors": {}})

    def item(pid):
        return [state["prompts"][pid]["number"], pid, {"6": {"inputs": {"text": "PROBE PROMPT"}}}, {"client_id": state["prompts"][pid]["client_id"]}, ["58"]]

    async def get_queue(request: Request):
        return guard(request) or JSONResponse({"queue_running": [item(state["running"])] if state["running"] else [],
                                               "queue_pending": [item(p) for p in state["pending"] if p != state["running"]]})

    async def post_queue(request: Request):
        body = await request.json()
        for pid in body.get("delete", []):
            state["deleted"].append(pid)
            if pid in state["pending"] and pid != state["running"]:
                state["pending"].remove(pid)
        return guard(request) or Response(status_code=200)

    async def post_interrupt(request: Request):
        body = await request.json() if await request.body() else {}
        if body.get("prompt_id") == state["running"]:
            state["interrupts"].append(body["prompt_id"])
        return guard(request) or Response(status_code=200)

    async def get_history(request: Request):
        pid = request.path_params["pid"]
        return guard(request) or JSONResponse({pid: state["history"][pid]} if pid in state["history"] else {})

    async def post_history(request: Request):
        body = await request.json()
        for pid in body.get("delete", []):
            state["history"].pop(pid, None)
        return guard(request) or Response(status_code=200)

    async def view(request: Request):
        if (r := guard(request)):
            return r
        q = request.query_params
        if q.get("type") != "output" or q.get("subfolder") != "vidsgen" or not VIEW_NAME_RE.match(q.get("filename", "")):
            return PlainTextResponse("404: Not Found", status_code=404)
        state["stats"]["views"] += 1
        size, delay = download_bytes, state["control"]["view_delay"]
        start, end, status = 0, size - 1, 200
        rng = request.headers.get("range")
        if rng:
            m = RANGE_RE.match(rng)
            if not m or (m.group(1) == "" and m.group(2) == ""):
                return PlainTextResponse("bad range", status_code=416)
            if m.group(1) == "":                                           # suffix: the last N bytes
                start, end = max(0, size - int(m.group(2))), size - 1
            else:
                start = int(m.group(1))
                end = min(int(m.group(2)), size - 1) if m.group(2) else size - 1
            if start >= size or start > end:
                return Response(status_code=416, headers={"content-range": f"bytes */{size}"})
            status = 206
        headers = {"content-type": "video/mp4", "content-length": str(end - start + 1), "accept-ranges": "bytes"}
        if status == 206:
            headers["content-range"] = f"bytes {start}-{end}/{size}"

        async def body():
            for chunk in iter_stream(size, seed, start, end):
                if delay:
                    await asyncio.sleep(delay)
                yield chunk
        return StreamingResponse(body(), status_code=status, headers=headers)

    async def ws(websocket: WebSocket):
        await websocket.accept()
        cid = websocket.query_params.get("clientId") or uuid.uuid4().hex
        state["sockets"].setdefault(cid, []).append(websocket)
        state["stats"]["ws_connections"] += 1
        await websocket.send_text(event("status", {"status": {"exec_info": {"queue_remaining": 0}}, "sid": cid}))
        try:
            while True:
                await websocket.receive_text()
        except (WebSocketDisconnect, Exception):
            pass
        finally:
            if websocket in state["sockets"].get(cid, []):
                state["sockets"][cid].remove(websocket)

    async def probe_stats(request: Request):
        return JSONResponse({**state["stats"], "interrupts": list(state["interrupts"]), "deleted": list(state["deleted"]),
                             "open_websockets": sum(len(v) for v in state["sockets"].values()), "download_bytes": download_bytes, "seed": seed})

    async def probe_control(request: Request):
        state["control"].update({k: v for k, v in (await request.json()).items() if k in state["control"]})
        return JSONResponse(state["control"])

    @asynccontextmanager
    async def lifespan(app):
        yield

    return Starlette(routes=[
        Route("/system_stats", system_stats), Route("/object_info/{cls}", object_info), Route("/models/{folder}", models),
        Route("/experiment/models/{folder}", models), Route("/prompt", post_prompt, methods=["POST"]),
        Route("/queue", get_queue, methods=["GET"]), Route("/queue", post_queue, methods=["POST"]), Route("/interrupt", post_interrupt, methods=["POST"]),
        Route("/history/{pid}", get_history), Route("/history", post_history, methods=["POST"]), Route("/view", view),
        Route("/__probe_stats", probe_stats), Route("/__probe_control", probe_control, methods=["POST"]), WebSocketRoute("/ws", ws),
    ], lifespan=lifespan)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8188)
    ap.add_argument("--download-mib", type=float, default=100.0, help="size of the /view download")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--max-lifetime-seconds", type=float, default=0.0, help="exit after this long (0 = never)")
    a = ap.parse_args(argv)
    if a.host not in LOOPBACK:
        print("FAILED: the dummy upstream listens on loopback only.", file=sys.stderr)
        return 2
    if not 0 < a.download_mib <= 4096 or a.max_lifetime_seconds < 0:
        print("FAILED: --download-mib must be in (0, 4096] and --max-lifetime-seconds >= 0.", file=sys.stderr)
        return 2
    import uvicorn

    server = uvicorn.Server(uvicorn.Config(create_app(int(a.download_mib * 1024 * 1024), a.seed), host=a.host, port=a.port, log_level="warning",
                                           access_log=False, server_header=False, date_header=False, ws_max_size=65536))
    if a.max_lifetime_seconds > 0:
        timer = threading.Timer(a.max_lifetime_seconds, lambda: setattr(server, "should_exit", True))
        timer.daemon = True
        timer.start()
    server.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
