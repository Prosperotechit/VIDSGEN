"""ComfyUIProvider: runs Wan 2.2 TI2V-5B generations on a ComfyUI server.

Implements the existing InferenceProvider contract (submit / progress / fetch_result /
cancel / health / release). It never touches VIDSGEN storage: like every provider it hands
the finished MP4 back as a temp file, and the job manager stores it.

How a job runs (all behaviour below was verified against a real ComfyUI 0.46.0):

1. submit(): for i2v, upload the start image (POST /upload/image) into the ComfyUI input
   folder as ``vidsgen/<job_id>.png``. Fill in the API-format template and queue it with
   POST /prompt using a prompt_id WE generate (a UUID), so the VIDSGEN job <-> ComfyUI prompt
   mapping is known even if the HTTP response is lost. The job manager stores that id as
   the job's provider_job_id.
2. progress(): one shared WebSocket per backend process (/ws?clientId=...) receives
   events. Every event is routed ONLY by its ``prompt_id``; events for unknown prompts and
   events without a prompt_id are ignored, so concurrent jobs never see each other's
   progress. Step progress is taken only from the sampler node. If the WebSocket is down,
   or quiet for a while, status is polled from GET /history/{id} and GET /queue.
3. Completion is never assumed from an HTTP 200 or a single event: a job is "completed"
   only when /history reports status_str == "success" AND a video output exists.
4. fetch_result(): download the video with GET /view into a temp file and check that it is
   an MP4 before handing it over.
5. cancel(): a pending prompt is removed with POST /queue {"delete": [id]} (ComfyUI sends no
   event for that, so cancellation is also tracked locally); a running prompt is stopped
   with a targeted POST /interrupt {"prompt_id": id}, which never affects other jobs.
6. release(): delete the prompt's /history entry. Models are NOT unloaded per job (/free):
   reloading ~18 GB for every job would make every generation slow.

Errors become ProviderError with stable codes and short, user-safe messages. URLs, tokens,
tracebacks and image bytes are never put in job errors, and tokens/images are never logged.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import shutil
import tempfile
import time
from datetime import UTC, datetime
import uuid
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import httpx
from websockets.asyncio.client import connect as ws_connect

from ..jobs.models import GenerationMode
from ..workflows import GraphValues, WorkflowError, WorkflowTemplates
from .base import (
    GenerationRequest, HealthCheck, InferenceProvider, ProviderError, ProviderHealth, ProviderProgress, ProviderResult,
)

log = logging.getLogger(__name__)
# The websockets library logs the full handshake headers (including "Authorization: Bearer
# <token>") at DEBUG level. Give it a dedicated logger that can never emit DEBUG records, so
# the token cannot reach logs even with LOG_LEVEL=DEBUG.
_ws_log = logging.getLogger(f"{__name__}.websocket")
_ws_log.setLevel(logging.INFO)

UPLOAD_SUBFOLDER = "vidsgen"
VIDEO_EXTENSIONS = (".mp4", ".webm", ".mov", ".mkv")
_TERMINAL = ("completed", "failed", "cancelled")
_MAX_MESSAGE = 300
_HEALTH_TTL = 10.0          # seconds a health result is reused (the UI polls /api/health)
_NODE_CHECK_TTL = 600.0     # node classes rarely change: re-check every 10 minutes
_LOST_AFTER_POLLS = 3       # consecutive polls with the prompt in neither queue nor history
# Failures that mean "could not talk to ComfyUI right now" (a tunnel blip, a restart), as opposed to an answer.
_TRANSIENT_CODES = frozenset({"comfyui_unreachable", "comfyui_timeout", "comfyui_server_error"})
_DOWNLOAD_ATTEMPTS = 3
_DOWNLOAD_RETRY_STATUSES = frozenset({502, 503, 504})
_GPU_OOM_MARKERS = ("out of memory", "allocation on device", "cublas_status_alloc_failed", "cudnn_status_alloc_failed")
_SYSTEM_OOM_MARKERS = ("defaultcpuallocator", "can't allocate memory", "std::bad_alloc", "cannot allocate memory")


def _short(text: Any, limit: int = _MAX_MESSAGE) -> str:
    s = " ".join(str(text).split())
    return s if len(s) <= limit else s[: limit - 1] + "…"


def _ws_url(base_url: str, client_id: str) -> str:
    parts = urlsplit(base_url)
    scheme = "wss" if parts.scheme == "https" else "ws"
    path = parts.path.rstrip("/")
    return f"{scheme}://{parts.netloc}{path}/ws?clientId={client_id}"


@dataclass
class _Run:
    prompt_id: str
    job_id: str
    events: asyncio.Queue = field(default_factory=asyncio.Queue)
    percent: float = 0.0
    cancel_requested: bool = False
    lost_polls: int = 0
    outputs: list[dict[str, Any]] | None = None   # video entries from /history, once confirmed
    uses_required_models: bool = False            # graph actually loads the Wan model files


class ComfyUIProvider(InferenceProvider):
    name = "comfyui"

    def __init__(
        self,
        *,
        base_url: str,
        workflows: WorkflowTemplates,
        api_token: str = "",
        timeout: float = 30.0,
        client_id: str = "",
        poll_interval: float = 2.0,
        connection_grace: float = 30.0,
        download_retry_delay: float = 1.0,
        transport: httpx.AsyncBaseTransport | None = None,
    ):
        if not base_url:
            raise ValueError("COMFYUI_BASE_URL must be set when INFERENCE_PROVIDER=comfyui")
        self.base_url = base_url.rstrip("/")
        self.workflows = workflows
        self.timeout = timeout
        self.client_id = client_id or f"vidsgen-{uuid.uuid4().hex[:12]}"
        self.poll_interval = poll_interval
        self.connection_grace = connection_grace          # seconds a running job tolerates a lost connection
        self.download_retry_delay = download_retry_delay
        self._headers = {"Authorization": f"Bearer {api_token}"} if api_token else {}
        self._http = httpx.AsyncClient(
            base_url=self.base_url, headers=self._headers, timeout=httpx.Timeout(timeout),
            follow_redirects=False, transport=transport,
        )
        self._runs: dict[str, _Run] = {}
        self._ws_task: asyncio.Task | None = None
        self._ws_connected = asyncio.Event()
        self._health_cache: tuple[float, ProviderHealth] | None = None
        # Set only when a generation that loaded the required Wan model files completed with
        # a video. Until then, weight integrity and inference readiness are "unverified".
        self._inference_verified_at: datetime | None = None
        self._nodes_checked_until = 0.0
        self._sampler = workflows.node_id("sampler")
        p = workflows.progress_config
        self._p_started = float(p.get("started", 2))
        self._p_sampling = tuple(p.get("sampling", [5, 90]))
        self._p_after = {workflows.node_id(role): float(v) for role, v in p.get("after_node", {}).items()}
        self._output_node = workflows.node_id(workflows.mapping["output_node"])
        # Host only (no path/query/token) for logs.
        self._log_host = urlsplit(self.base_url).netloc

    # ================================================================== HTTP helpers
    async def _request(self, method: str, path: str, *, timeout: float | None = None, **kw) -> httpx.Response:
        try:
            resp = await self._http.request(method, path, timeout=timeout or self.timeout, **kw)
        except (httpx.ConnectError, httpx.ConnectTimeout) as e:
            # No connection was ever made, so the request certainly did not arrive (submit() must not reconcile these).
            log.warning("ComfyUI %s %s could not connect: %s (host %s)", method, path, type(e).__name__, self._log_host)
            err = ProviderError("comfyui_timeout" if isinstance(e, httpx.ConnectTimeout) else "comfyui_unreachable",
                                "The GPU server (ComfyUI) did not respond in time." if isinstance(e, httpx.ConnectTimeout)
                                else "The GPU server (ComfyUI) is unreachable.")
            err.not_sent = True
            raise err from None
        except httpx.TimeoutException:
            log.warning("ComfyUI %s %s timed out (host %s)", method, path, self._log_host)
            raise ProviderError("comfyui_timeout", "The GPU server (ComfyUI) did not respond in time.") from None
        except httpx.HTTPError as e:
            log.warning("ComfyUI %s %s failed: %s (host %s)", method, path, type(e).__name__, self._log_host)
            raise ProviderError("comfyui_unreachable", "The GPU server (ComfyUI) is unreachable.") from None
        if resp.status_code in (401, 403):
            raise ProviderError("comfyui_auth_failed", "The GPU server rejected the API token (check COMFYUI_API_TOKEN).")
        if resp.status_code >= 500:
            log.warning("ComfyUI %s %s -> HTTP %s", method, path, resp.status_code)
            raise ProviderError("comfyui_server_error", f"The GPU server returned an error (HTTP {resp.status_code}).")
        return resp

    @staticmethod
    def _json(resp: httpx.Response, what: str) -> Any:
        try:
            return resp.json()
        except (json.JSONDecodeError, ValueError):
            raise ProviderError("comfyui_bad_response", f"The GPU server sent an invalid response ({what}).") from None

    async def _get_json(self, path: str, what: str, **kw) -> Any:
        resp = await self._request("GET", path, **kw)
        if resp.status_code != 200:
            raise ProviderError("comfyui_bad_response", f"Unexpected HTTP {resp.status_code} from the GPU server ({what}).")
        return self._json(resp, what)

    # ================================================================== WebSocket
    def _ensure_ws(self) -> None:
        if self._ws_task is None or self._ws_task.done():
            self._ws_task = asyncio.create_task(self._ws_loop(), name="comfyui-ws")

    async def _ws_loop(self) -> None:
        url = _ws_url(self.base_url, self.client_id)
        backoff = 1.0
        while True:
            try:
                async with ws_connect(url, additional_headers=self._headers, open_timeout=self.timeout,
                                      max_size=None, ping_interval=20, ping_timeout=20, logger=_ws_log) as ws:
                    self._ws_connected.set()
                    backoff = 1.0
                    log.info("ComfyUI WebSocket connected (host %s)", self._log_host)
                    async for message in ws:
                        if isinstance(message, (bytes, bytearray)):
                            continue  # binary preview frames are not used
                        self._dispatch(message)
            except asyncio.CancelledError:
                raise
            except Exception as e:  # connection refused, closed, handshake rejected, ...
                log.info("ComfyUI WebSocket unavailable (%s); falling back to polling", type(e).__name__)
            finally:
                was_connected = self._ws_connected.is_set()
                self._ws_connected.clear()
                if was_connected:  # wake waiting jobs so they switch to fast polling at once
                    for run in list(self._runs.values()):
                        run.events.put_nowait(("_ws_lost", {}))
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, 30.0)

    def _dispatch(self, raw: str) -> None:
        """Route an event to the run it belongs to, by prompt_id only."""
        try:
            msg = json.loads(raw)
            kind, data = msg["type"], msg.get("data") or {}
        except (ValueError, KeyError, TypeError):
            return
        if not isinstance(data, dict):
            return
        run = self._runs.get(data.get("prompt_id"))
        if run is not None:
            run.events.put_nowait((kind, data))

    # ================================================================== submit
    async def _upload_image(self, request: GenerationRequest) -> str:
        files = {"image": (f"{request.job_id}.png", request.image_png, "image/png")}
        data = {"subfolder": UPLOAD_SUBFOLDER, "type": "input", "overwrite": "true"}
        try:
            resp = await self._request("POST", "/upload/image", files=files, data=data)
        except ProviderError as e:
            raise ProviderError("image_upload_failed", f"Uploading the start image to the GPU server failed: {e.message}") from None
        if resp.status_code != 200:
            raise ProviderError("image_upload_failed", f"The GPU server refused the start image (HTTP {resp.status_code}).")
        body = self._json(resp, "image upload")
        name = body.get("name") if isinstance(body, dict) else None
        if not name:
            raise ProviderError("image_upload_failed", "The GPU server did not confirm the start image upload.")
        sub = body.get("subfolder") or ""
        return f"{sub}/{name}" if sub else name

    def _rejection(self, body: Any) -> ProviderError:
        """Turn a /prompt validation failure into a clear error."""
        if not isinstance(body, dict):
            return ProviderError("comfyui_rejected", "The GPU server rejected the workflow.")
        model_nodes = {m.node_id: m for m in self.workflows.required_models}
        node_errors = body.get("node_errors") or {}
        for node_id, info in node_errors.items():
            for err in (info or {}).get("errors", []):
                if node_id in model_nodes and err.get("type") == "value_not_in_list":
                    m = model_nodes[node_id]
                    return ProviderError("model_missing",
                                         f"Model file {m.file} is not installed on the GPU server (expected in models/{m.folder}/).")
        for node_id, info in node_errors.items():
            for err in (info or {}).get("errors", []):
                detail = err.get("details") or err.get("message") or "invalid input"
                return ProviderError("comfyui_rejected",
                                     _short(f"ComfyUI rejected the workflow ({info.get('class_type', 'node ' + node_id)}): {detail}"))
        top = body.get("error")
        if isinstance(top, dict):
            return ProviderError("comfyui_rejected", _short(f"ComfyUI rejected the workflow: {top.get('message') or top.get('type')}"))
        return ProviderError("comfyui_rejected", "The GPU server rejected the workflow.")

    async def submit(self, request: GenerationRequest) -> str:
        image_ref = None
        if request.mode == GenerationMode.i2v:
            if not request.image_png:
                raise ProviderError("missing_image", "Image-to-video needs a start image.")
            image_ref = await self._upload_image(request)
        try:
            graph = self.workflows.build(request.mode, GraphValues(
                prompt=request.prompt, negative_prompt=request.negative_prompt,
                width=request.width, height=request.height, num_frames=request.num_frames, fps=request.fps,
                steps=request.steps, guidance_scale=request.guidance_scale, seed=request.seed,
                filename_prefix=f"{UPLOAD_SUBFOLDER}/{request.job_id}", image=image_ref,
            ))
        except WorkflowError as e:
            raise ProviderError("workflow_error", _short(str(e))) from None

        prompt_id = str(uuid.uuid4())
        run = _Run(prompt_id=prompt_id, job_id=request.job_id,
                   uses_required_models=self._loads_required_models(graph))
        self._runs[prompt_id] = run          # register BEFORE queueing so no early event is missed
        self._ensure_ws()
        try:
            await asyncio.wait_for(self._ws_connected.wait(), timeout=2.0)
        except TimeoutError:
            pass  # progress() falls back to polling
        try:
            try:
                resp = await self._request("POST", "/prompt", json={
                    "prompt": graph, "client_id": self.client_id, "prompt_id": prompt_id,
                })
                body = self._json(resp, "queue prompt")
                if resp.status_code != 200 or (isinstance(body, dict) and body.get("node_errors")):
                    raise self._rejection(body)
                if not isinstance(body, dict) or not body.get("prompt_id"):
                    raise ProviderError("comfyui_bad_response", "The GPU server did not return a prompt id.")
            except ProviderError as e:
                # The outcome is UNCERTAIN when the connection failed or timed out (or ComfyUI answered 5xx): the prompt may
                # have been queued. Look, read-only. Never resubmit: a second POST could run the same generation twice.
                if e.code not in _TRANSIENT_CODES or getattr(e, "not_sent", False) or not await self._reconcile_submission(prompt_id):
                    raise
                log.info("Prompt %s was queued although its submission response was lost (found by read-only reconciliation)", prompt_id)
                body = {"prompt_id": prompt_id}
        except BaseException:
            self._runs.pop(prompt_id, None)
            raise
        returned = str(body["prompt_id"])
        if returned != prompt_id:   # older ComfyUI ignores client-chosen ids: follow the server's
            self._runs.pop(prompt_id, None)
            run.prompt_id = returned
            self._runs[returned] = run
        log.info("Job %s queued on ComfyUI as prompt %s (mode=%s)", request.job_id, run.prompt_id, request.mode.value)
        return run.prompt_id

    async def _reconcile_submission(self, prompt_id: str) -> bool:
        """READ-ONLY check (GET /history/<id>, GET /queue) after an uncertain submission: True if ComfyUI knows the prompt.

        Retries only while ComfyUI cannot be reached, for at most COMFYUI_CONNECTION_GRACE_SECONDS (0 disables reconciliation).
        False means "not found right now" (a very late arrival is still conceivable) or "could not be asked": the submission is
        then reported as failed, never retried."""
        if self.connection_grace <= 0:
            return False
        deadline = time.monotonic() + self.connection_grace
        while True:
            try:
                history = await self._get_json(f"/history/{prompt_id}", "history")
                if isinstance(history, dict) and prompt_id in history:
                    return True
                queue = await self._get_json("/queue", "queue")
                if isinstance(queue, dict):
                    for key in ("queue_running", "queue_pending"):
                        if any(isinstance(i, list) and len(i) > 1 and str(i[1]) == prompt_id for i in queue.get(key, [])):
                            return True
                return False                                    # ComfyUI answered both questions: it does not know the prompt
            except ProviderError as e:
                if e.code not in _TRANSIENT_CODES or time.monotonic() >= deadline:
                    return False
                await asyncio.sleep(self.poll_interval)

    # ================================================================== progress
    def _get_run(self, prompt_id: str) -> _Run:
        run = self._runs.get(prompt_id)
        if run is None:
            raise ProviderError("unknown_provider_job", f"Unknown ComfyUI prompt {prompt_id}")
        return run

    def _advance(self, run: _Run, percent: float, message: str) -> ProviderProgress:
        run.percent = max(run.percent, min(percent, 99.0))
        return ProviderProgress("running", round(run.percent, 1), message)

    def _from_event(self, run: _Run, kind: str, data: dict[str, Any]) -> ProviderProgress | str | None:
        """Map one WebSocket event to progress. Returns "verify" when history must be checked."""
        if kind == "execution_start":
            return self._advance(run, self._p_started, "started")
        if kind == "progress" and str(data.get("node")) == self._sampler:
            try:
                value, maximum = float(data["value"]), float(data["max"])
            except (KeyError, TypeError, ValueError):
                return None
            if maximum <= 0:
                return None
            lo, hi = self._p_sampling
            return self._advance(run, lo + (hi - lo) * min(value / maximum, 1.0), f"step {int(value)}/{int(maximum)}")
        if kind == "executing":
            node = data.get("node")
            if node is None:
                return "verify"      # legacy end-of-prompt signal
            node = str(node)
            if node == self._sampler:
                return self._advance(run, self._p_sampling[0], "sampling")
            if node in self._p_after:
                return self._advance(run, self._p_after[node], "decoding and encoding video")
            return None
        if kind == "execution_success":
            return "verify"
        if kind == "execution_error":
            return self._execution_error(data)
        if kind == "execution_interrupted":
            return ProviderProgress("cancelled", run.percent, "interrupted")
        return None

    @staticmethod
    def _execution_error(data: dict[str, Any]) -> ProviderProgress:
        etype = str(data.get("exception_type") or "")
        emsg = str(data.get("exception_message") or "Unknown error")
        node_type = data.get("node_type") or "workflow"
        # The message is usually enough, but some OOMs only show in the exception type or the traceback.
        trace = data.get("traceback")
        haystack = " ".join([etype, emsg, " ".join(map(str, trace)) if isinstance(trace, list) else ""]).lower()
        if "outofmemory" in etype.lower() or any(m in haystack for m in _GPU_OOM_MARKERS):
            return ProviderProgress("failed", 0, "The GPU ran out of memory. Try a smaller resolution or fewer frames.",
                                    error_code="gpu_out_of_memory")
        if etype.rsplit(".", 1)[-1] == "MemoryError" or any(m in haystack for m in _SYSTEM_OOM_MARKERS):
            return ProviderProgress("failed", 0, "The GPU server ran out of system memory (RAM). "
                                    "Try a smaller resolution or fewer frames.", error_code="system_out_of_memory")
        return ProviderProgress("failed", 0, _short(f"ComfyUI error in {node_type}: {emsg}"),
                                error_code="comfyui_execution_error")

    def _video_outputs(self, outputs: dict[str, Any]) -> list[dict[str, Any]]:
        """Find saved video files in a /history 'outputs' object.

        Verified on ComfyUI 0.46.0: SaveVideo reports under outputs[<node>]["images"] with
        "animated": [true], each entry {filename, subfolder, type: "output",
        metadata: {kind: "video", ...}}. Parsed defensively: any list in the output node,
        entries recognised by metadata.kind == "video" or a video file extension.
        """
        def scan(node_out: Any) -> list[dict[str, Any]]:
            found = []
            if isinstance(node_out, dict):
                for value in node_out.values():
                    if not isinstance(value, list):
                        continue
                    for item in value:
                        if not isinstance(item, dict) or not isinstance(item.get("filename"), str):
                            continue
                        kind = (item.get("metadata") or {}).get("kind") if isinstance(item.get("metadata"), dict) else None
                        if item.get("type", "output") == "output" and (
                                kind == "video" or item["filename"].lower().endswith(VIDEO_EXTENSIONS)):
                            found.append(item)
            return found

        if not isinstance(outputs, dict):
            return []
        primary = scan(outputs.get(self._output_node))
        if primary:
            return primary
        return [v for node_out in outputs.values() for v in scan(node_out)]

    async def _history_entry(self, prompt_id: str) -> dict[str, Any] | None:
        body = await self._get_json(f"/history/{prompt_id}", "history")
        if not isinstance(body, dict):
            raise ProviderError("comfyui_bad_response", "The GPU server sent an invalid history response.")
        entry = body.get(prompt_id)
        return entry if isinstance(entry, dict) else None

    def _from_history(self, run: _Run, entry: dict[str, Any]) -> ProviderProgress | None:
        status = entry.get("status") if isinstance(entry.get("status"), dict) else {}
        status_str = status.get("status_str")
        messages = status.get("messages") or []
        if status_str == "success" and status.get("completed", True):
            videos = self._video_outputs(entry.get("outputs") or {})
            if not videos:
                return ProviderProgress("failed", run.percent, "ComfyUI finished but produced no video.",
                                        error_code="no_video_output")
            run.outputs = videos
            if run.uses_required_models and self._inference_verified_at is None:
                self._inference_verified_at = datetime.now(UTC)
                self._health_cache = None   # reflect the new evidence immediately
                log.info("Inference verified: a Wan generation completed on ComfyUI (prompt %s)", run.prompt_id)
            return ProviderProgress("completed", 100.0, "done")
        if status_str == "error":
            for item in messages:
                if isinstance(item, list) and item and item[0] == "execution_interrupted":
                    return ProviderProgress("cancelled", run.percent, "interrupted")
            for item in messages:
                if isinstance(item, list) and len(item) > 1 and item[0] == "execution_error" and isinstance(item[1], dict):
                    return self._execution_error(item[1])
            return ProviderProgress("failed", run.percent, "ComfyUI reported an error.", error_code="comfyui_execution_error")
        return None  # still running (history entries can appear before completion)

    async def _verify(self, run: _Run) -> ProviderProgress:
        """Confirm completion from /history (it can lag the WebSocket event slightly)."""
        for attempt in range(10):
            entry = await self._history_entry(run.prompt_id)
            if entry is not None:
                update = self._from_history(run, entry)
                if update is not None:
                    return update
            await asyncio.sleep(min(0.25 * (attempt + 1), 1.0))
        raise ProviderError("comfyui_bad_response", "ComfyUI reported success but its history never confirmed it.")

    async def _poll(self, run: _Run) -> ProviderProgress | None:
        entry = await self._history_entry(run.prompt_id)
        if entry is not None:
            run.lost_polls = 0
            return self._from_history(run, entry)
        queue = await self._get_json("/queue", "queue")
        def ids(key: str) -> set[str]:
            items = queue.get(key, []) if isinstance(queue, dict) else []
            return {str(i[1]) for i in items if isinstance(i, list) and len(i) > 1}
        if run.prompt_id in ids("queue_running"):
            run.lost_polls = 0
            return self._advance(run, max(run.percent, self._p_started), "running")
        if run.prompt_id in ids("queue_pending"):
            run.lost_polls = 0
            return ProviderProgress("queued", run.percent, "waiting in the ComfyUI queue")
        if run.cancel_requested:
            return ProviderProgress("cancelled", run.percent, "removed from queue")
        run.lost_polls += 1
        if run.lost_polls >= _LOST_AFTER_POLLS:
            raise ProviderError("comfyui_job_lost",
                                "The GPU server no longer knows this job (it may have restarted).")
        return None

    async def progress(self, provider_job_id: str) -> AsyncIterator[ProviderProgress]:
        run = self._get_run(provider_job_id)
        yield ProviderProgress("queued", 0.0, "queued on GPU server")
        outage_since: float | None = None       # when the connection to ComfyUI was first lost (None = fine)
        while True:
            try:
                update = await self._next_update(run, outage_since is not None)
            except ProviderError as e:
                if e.code not in _TRANSIENT_CODES or self.connection_grace <= 0:
                    raise
                now = time.monotonic()
                outage_since = now if outage_since is None else outage_since
                lost_for = now - outage_since
                if lost_for >= self.connection_grace:
                    log.warning("ComfyUI connection lost for %.0fs while generating (%s)", lost_for, e.code)
                    raise ProviderError(
                        "comfyui_connection_lost",
                        f"The connection to the GPU server was lost for more than {self.connection_grace:g} s while "
                        "the video was being generated. The server may have restarted or crashed (for example from "
                        "running out of memory). Please try again.") from None
                log.warning("ComfyUI unreachable while generating (%s); retrying (%.0fs of %gs)", e.code, lost_for,
                            self.connection_grace)
                await asyncio.sleep(self.poll_interval)
                continue
            if outage_since is not None:
                log.info("ComfyUI connection restored after %.0fs", time.monotonic() - outage_since)
                outage_since = None
            if update is None:
                continue
            yield update
            if update.state in _TERMINAL:
                return

    async def _next_update(self, run: _Run, in_outage: bool) -> ProviderProgress | None:
        """Wait for the next event (or poll) and map it to progress. May raise a transient ProviderError."""
        if run.cancel_requested and run.events.empty():
            # A deleted pending prompt produces no event: confirm by polling.
            update = await self._poll(run)
            if update is not None and update.state in ("cancelled", *_TERMINAL):
                return update
        # With live events, poll only occasionally as a safety net; without them, after a poll could
        # not find the prompt (e.g. ComfyUI restarted), or while the connection is down, poll at the normal interval.
        fast = in_outage or not self._ws_connected.is_set() or run.lost_polls > 0
        wait = self.poll_interval if fast else max(self.poll_interval * 5, 10.0)
        try:
            kind, data = await asyncio.wait_for(run.events.get(), timeout=wait)
        except TimeoutError:
            return await self._poll(run)
        if kind == "_ws_lost":
            return await self._poll(run)
        if kind == "_cancelled":
            return ProviderProgress("cancelled", run.percent, "cancelled")
        mapped = self._from_event(run, kind, data)
        return await self._verify(run) if mapped == "verify" else mapped

    # ================================================================== result
    async def fetch_result(self, provider_job_id: str) -> ProviderResult:
        run = self._get_run(provider_job_id)
        if run.outputs is None:
            entry = await self._history_entry(run.prompt_id)
            update = self._from_history(run, entry) if entry else None
            if update is None or update.state != "completed":
                raise ProviderError("result_not_ready", "The video is not ready on the GPU server.")
        video = next((v for v in run.outputs if v["filename"].lower().endswith(".mp4")), None)
        if video is None:
            raise ProviderError("unsupported_output", "ComfyUI produced a video, but not in MP4 format.")
        params = {"filename": video["filename"], "subfolder": video.get("subfolder", ""), "type": video.get("type", "output")}
        workdir = Path(tempfile.mkdtemp(prefix="vidsgen-comfyui-"))
        path = workdir / "video.mp4"
        try:
            await self._download(params, path)
        except BaseException:
            shutil.rmtree(workdir, ignore_errors=True)
            raise
        return ProviderResult(path=path, metadata=dict(video.get("metadata") or {}))

    async def _download(self, params: dict[str, str], path: Path) -> None:
        """GET /view into `path`, retrying (the download is idempotent) when the connection drops or a proxy
        answers 502/503/504, so a short tunnel blip does not lose a video that was fully generated."""
        for attempt in range(1, _DOWNLOAD_ATTEMPTS + 1):
            last = attempt == _DOWNLOAD_ATTEMPTS
            try:
                async with self._http.stream("GET", "/view", params=params, timeout=max(self.timeout, 120)) as resp:
                    if resp.status_code in _DOWNLOAD_RETRY_STATUSES and not last:
                        log.warning("ComfyUI /view -> HTTP %s; retrying download (%d/%d)", resp.status_code, attempt,
                                    _DOWNLOAD_ATTEMPTS)
                        await asyncio.sleep(self.download_retry_delay)
                        continue
                    if resp.status_code != 200:
                        raise ProviderError("output_download_failed",
                                            f"Downloading the video failed (HTTP {resp.status_code}).")
                    with open(path, "wb") as f:
                        async for chunk in resp.aiter_bytes(1024 * 1024):
                            f.write(chunk)
                with open(path, "rb") as f:
                    head = f.read(12)
                if len(head) < 12 or head[4:8] != b"ftyp":
                    raise ProviderError("unsupported_output", "The file from the GPU server is not a valid MP4 video.")
                return
            except httpx.TimeoutException:
                if last:
                    raise ProviderError("comfyui_timeout", "Downloading the video from the GPU server timed out.") from None
            except httpx.HTTPError:
                if last:
                    raise ProviderError("comfyui_unreachable",
                                        "The GPU server became unreachable while downloading the video.") from None
            log.warning("ComfyUI /view download interrupted; retrying (%d/%d)", attempt, _DOWNLOAD_ATTEMPTS)
            await asyncio.sleep(self.download_retry_delay)

    # ================================================================== cancel / release
    async def cancel(self, provider_job_id: str) -> None:
        run = self._runs.get(provider_job_id)
        if run is None:
            return
        run.cancel_requested = True
        try:
            queue = await self._get_json("/queue", "queue")
            pending = {str(i[1]) for i in queue.get("queue_pending", []) if isinstance(i, list) and len(i) > 1}
            running = {str(i[1]) for i in queue.get("queue_running", []) if isinstance(i, list) and len(i) > 1}
            if provider_job_id in pending:
                await self._request("POST", "/queue", json={"delete": [provider_job_id]})
                run.events.put_nowait(("_cancelled", {}))
            elif provider_job_id in running:
                await self._request("POST", "/interrupt", json={"prompt_id": provider_job_id})
        except ProviderError as e:
            log.warning("Cancelling ComfyUI prompt %s failed: %s", provider_job_id, e.code)

    async def release(self, provider_job_id: str) -> None:
        self._runs.pop(provider_job_id, None)
        try:
            await self._request("POST", "/history", json={"delete": [provider_job_id]})
        except ProviderError:
            pass  # best effort housekeeping

    # ================================================================== health
    def _loads_required_models(self, graph: dict[str, Any]) -> bool:
        """True if the graph really loads every required model file (i.e. it is a Wan
        generation, not some other graph sent through this provider)."""
        for m in self.workflows.required_models:
            node = graph.get(m.node_id)
            if not isinstance(node, dict) or node.get("inputs", {}).get(m.input) != m.file:
                return False
        return True

    async def health(self) -> ProviderHealth:
        now = time.monotonic()
        if self._health_cache and now - self._health_cache[0] < _HEALTH_TTL:
            return self._health_cache[1]
        result = await self._check_health(now)
        self._health_cache = (time.monotonic(), result)
        return result

    async def _check_health(self, now: float) -> ProviderHealth:
        """Report each aspect separately and never claim more than can be established.

        What the ComfyUI API can and cannot tell us (verified on ComfyUI 0.46.0):
          * reachability and version/device: GET /system_stats
          * node classes: GET /object_info/<class>
          * model FILENAMES: GET /models/<folder> (names only)
          * model FILE SIZES: GET /experiment/models/<folder> (experimental route; if it is
            missing, sizes are "unverified")
          * model CONTENT INTEGRITY: no API exposes hashes, so it is "unverified" until a real
            generation succeeds. A matching size is not proof of integrity.
        """
        quick = min(self.timeout, 4.0)
        checks: list[HealthCheck] = []

        def summary(*parts: str) -> str:
            return " · ".join(p for p in parts if p)

        # 1. Reachability -------------------------------------------------------------
        try:
            stats = await self._get_json("/system_stats", "system stats", timeout=quick)
        except ProviderError as e:
            checks.append(HealthCheck("comfyui_reachable", "failed", e.message))
            for name in ("workflow_nodes", "model_files_detected", "model_file_sizes", "model_weights_integrity", "inference"):
                checks.append(HealthCheck(name, "unverified", "ComfyUI is not reachable."))
            return ProviderHealth(False, e.message, tuple(checks))
        system = stats.get("system", {}) if isinstance(stats, dict) else {}
        devices = stats.get("devices", []) if isinstance(stats, dict) else []
        gpu = next((d for d in devices if isinstance(d, dict) and d.get("type") != "cpu"), None)
        device = f"{gpu.get('name', 'GPU')}, {gpu.get('vram_total', 0) / 1024**3:.0f} GB VRAM" if gpu else "CPU only"
        version = system.get("comfyui_version", "?")
        checks.append(HealthCheck("comfyui_reachable", "ok", f"ComfyUI {version} ({device})"))

        try:
            # 2. Required node classes ------------------------------------------------
            if now >= self._nodes_checked_until:
                infos = await asyncio.gather(*(self._get_json(f"/object_info/{c}", "node info", timeout=quick)
                                               for c in self.workflows.node_classes))
                missing_nodes = [c for c, info in zip(self.workflows.node_classes, infos)
                                 if not (isinstance(info, dict) and c in info)]
                if missing_nodes:
                    msg = "ComfyUI is missing required nodes (update ComfyUI): " + ", ".join(missing_nodes)
                    checks.append(HealthCheck("workflow_nodes", "failed", msg))
                    return ProviderHealth(False, msg, tuple(checks))
                self._nodes_checked_until = now + _NODE_CHECK_TTL
            checks.append(HealthCheck("workflow_nodes", "ok", f"{len(self.workflows.node_classes)} node classes available"))

            # 3. Model filenames --------------------------------------------------------
            models = self.workflows.required_models
            folders = sorted({m.folder for m in models})
            listings = await asyncio.gather(*(self._get_json(f"/models/{f}", "model list", timeout=quick) for f in folders))
            names = {f: set(l) if isinstance(l, list) else set() for f, l in zip(folders, listings)}
            missing = [f"{m.folder}/{m.file}" for m in models if m.file not in names[m.folder]]
            if missing:
                msg = "Missing model files on the GPU server: " + ", ".join(missing)
                checks.append(HealthCheck("model_files_detected", "failed", msg))
                return ProviderHealth(False, msg, tuple(checks))
            checks.append(HealthCheck("model_files_detected", "ok", f"All {len(models)} required filenames are present"))

            # 4. Model file sizes (experimental ComfyUI route; optional) ----------------
            sizes = await self._model_sizes(folders, quick)
        except ProviderError as e:
            return ProviderHealth(False, e.message, tuple(checks))

        wrong = []
        unknown = []
        for m in models:
            expected = self.workflows.expected_sizes.get((m.folder, m.file))
            actual = sizes.get((m.folder, m.file)) if sizes is not None else None
            if expected is None or actual is None:
                unknown.append(m)
            elif actual != expected:
                wrong.append(f"{m.folder}/{m.file} ({actual:,} bytes, expected {expected:,})")
        if wrong:
            msg = "Model files have the wrong size on the GPU server (placeholder or incomplete download?): " + "; ".join(wrong)
            checks.append(HealthCheck("model_file_sizes", "failed", msg))
            checks.append(HealthCheck("model_weights_integrity", "failed", "File sizes do not match the official model files."))
            checks.append(HealthCheck("inference", "unverified", "No successful generation yet."))
            return ProviderHealth(False, msg, tuple(checks))
        if unknown:
            why = ("this ComfyUI does not report file sizes" if sizes is None
                   else "no expected size for " + ", ".join(m.file for m in unknown))
            checks.append(HealthCheck("model_file_sizes", "unverified", f"Sizes could not be checked: {why}."))
            weights_word = "weights unverified"
        else:
            checks.append(HealthCheck("model_file_sizes", "ok", "File sizes match the official model files exactly."))
            weights_word = "file sizes match"

        # 5 + 6. Integrity and inference: only a real generation proves them --------------
        if self._inference_verified_at is not None:
            when = self._inference_verified_at.strftime("%Y-%m-%d %H:%M UTC")
            checks.append(HealthCheck("model_weights_integrity", "ok", f"Weights loaded and generated a video ({when})."))
            checks.append(HealthCheck("inference", "ok", f"A Wan 2.2 generation completed successfully at {when}."))
            detail = summary(f"ComfyUI {version} reachable ({device})", "model files detected", weights_word,
                             f"inference verified by a successful generation at {when}")
        else:
            checks.append(HealthCheck("model_weights_integrity", "unverified",
                                      "ComfyUI's API cannot check file contents; only a successful generation proves the weights."))
            checks.append(HealthCheck("inference", "unverified", "No successful Wan 2.2 generation on this server yet."))
            detail = summary(f"ComfyUI {version} reachable ({device})", "model files detected", weights_word,
                             "weight integrity and inference not yet verified")
        return ProviderHealth(True, detail, tuple(checks))

    async def _model_sizes(self, folders: list[str], timeout: float) -> dict[tuple[str, str], int] | None:
        """File sizes from ComfyUI's /experiment/models/<folder>, or None if the route is
        unavailable or its response is not understood (sizes are then "unverified")."""
        out: dict[tuple[str, str], int] = {}
        for folder in folders:
            resp = await self._request("GET", f"/experiment/models/{folder}", timeout=timeout)
            if resp.status_code != 200:
                return None
            try:
                items = resp.json()
            except ValueError:
                return None
            if not isinstance(items, list):
                return None
            for item in items:
                if isinstance(item, dict) and isinstance(item.get("name"), str) and isinstance(item.get("size"), int):
                    out[(folder, item["name"])] = item["size"]
        return out

    async def aclose(self) -> None:
        if self._ws_task is not None:
            self._ws_task.cancel()
            try:
                await self._ws_task
            except (asyncio.CancelledError, Exception):
                pass
        self._runs.clear()
        await self._http.aclose()
