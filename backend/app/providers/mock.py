"""MockProvider: simulates Wan 2.2 generation without a GPU.

It behaves like a real asynchronous provider: `submit()` returns at once, work runs in
the background, progress advances one "denoising step" at a time, and the result is a
real, playable MP4 at the requested resolution, frame count and fps.

Test triggers (mock only): put ``[mock:fail]`` in the prompt to make the job fail
halfway through, so error handling can be exercised end to end.
"""

from __future__ import annotations

import asyncio
import logging
import shutil
import tempfile
import uuid
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from pathlib import Path

from .base import (
    GenerationRequest, InferenceProvider, ProviderError, ProviderHealth, ProviderProgress, ProviderResult,
)
from .mock_video import VideoEncodeError, encode_mock_video, ffmpeg_binary

log = logging.getLogger(__name__)

FAIL_TRIGGER = "[mock:fail]"
_TERMINAL = {"completed", "failed", "cancelled"}


@dataclass
class _Run:
    request: GenerationRequest
    workdir: Path
    state: str = "queued"
    percent: float = 0.0
    message: str = "queued"
    error_code: str | None = None
    cancel_requested: bool = False
    changed: asyncio.Event = field(default_factory=asyncio.Event)
    task: asyncio.Task | None = None

    def update(self, state: str, percent: float | None = None, message: str = "", error_code: str | None = None):
        self.state = state
        if percent is not None:
            self.percent = percent
        self.message = message
        self.error_code = error_code
        self.changed.set()

    def snapshot(self) -> ProviderProgress:
        return ProviderProgress(self.state, round(self.percent, 1), self.message, self.error_code)


class MockProvider(InferenceProvider):
    name = "mock"

    def __init__(self, step_delay_seconds: float = 0.15):
        self.step_delay = step_delay_seconds
        self._runs: dict[str, _Run] = {}

    # ------------------------------------------------------------ simulation
    async def _simulate(self, run: _Run) -> None:
        req = run.request
        try:
            run.update("running", 0.0, "starting")
            for step in range(1, req.steps + 1):
                await asyncio.sleep(self.step_delay)
                if run.cancel_requested:
                    run.update("cancelled", message="cancelled")
                    return
                if FAIL_TRIGGER in req.prompt and step >= max(1, req.steps // 2):
                    run.update("failed", message="Simulated provider failure", error_code="mock_failure")
                    return
                # Denoising = 0-90 %, encoding = last 10 %.
                run.update("running", 90.0 * step / req.steps, f"step {step}/{req.steps}")

            run.update("running", 90.0, "encoding video")
            out = run.workdir / "video.mp4"
            await asyncio.to_thread(
                encode_mock_video, out, width=req.width, height=req.height, num_frames=req.num_frames,
                fps=req.fps, seed=req.seed, prompt=req.prompt, image_png=req.image_png,
            )
            if run.cancel_requested:
                run.update("cancelled", message="cancelled")
                return
            run.update("completed", 100.0, "done")
        except asyncio.CancelledError:
            run.update("cancelled", message="cancelled")
            raise
        except VideoEncodeError as e:
            log.error("mock encode failed: %s", e)
            run.update("failed", message="Video encoding failed", error_code="encode_error")
        except Exception:
            log.exception("mock provider crashed")
            run.update("failed", message="Mock provider internal error", error_code="provider_error")

    def _get(self, pid: str) -> _Run:
        run = self._runs.get(pid)
        if run is None:
            raise ProviderError("unknown_provider_job", f"Unknown provider job {pid}")
        return run

    # ------------------------------------------------------------ interface
    async def submit(self, request: GenerationRequest) -> str:
        pid = f"mock-{uuid.uuid4().hex[:12]}"
        run = _Run(request=request, workdir=Path(tempfile.mkdtemp(prefix="vidsgen-mock-")))
        self._runs[pid] = run
        run.task = asyncio.create_task(self._simulate(run), name=f"mock-{pid}")
        return pid

    async def progress(self, provider_job_id: str) -> AsyncIterator[ProviderProgress]:
        run = self._get(provider_job_id)
        while True:
            run.changed.clear()
            snap = run.snapshot()
            yield snap
            if snap.state in _TERMINAL:
                return
            await run.changed.wait()

    async def fetch_result(self, provider_job_id: str) -> ProviderResult:
        run = self._get(provider_job_id)
        if run.state != "completed":
            raise ProviderError("result_not_ready", f"Job is {run.state}, not completed")
        self._runs.pop(provider_job_id, None)
        return ProviderResult(path=run.workdir / "video.mp4", metadata={
            "width": run.request.width, "height": run.request.height,
            "num_frames": run.request.num_frames, "fps": run.request.fps,
        })

    async def cancel(self, provider_job_id: str) -> None:
        run = self._runs.get(provider_job_id)
        if run is None or run.state in _TERMINAL:
            return
        run.cancel_requested = True
        run.changed.set()

    async def health(self) -> ProviderHealth:
        try:
            ffmpeg_binary()
        except VideoEncodeError as e:
            return ProviderHealth(False, str(e))
        return ProviderHealth(True, "mock provider ready (no GPU)")

    async def release(self, provider_job_id: str) -> None:
        """Forget a run and delete its temp files (safe after fetch_result too)."""
        run = self._runs.pop(provider_job_id, None)
        if run:
            if run.task and not run.task.done():
                run.task.cancel()
            shutil.rmtree(run.workdir, ignore_errors=True)

    async def aclose(self) -> None:
        for pid, run in list(self._runs.items()):
            if run.task and not run.task.done():
                run.task.cancel()
                try:
                    await run.task
                except (asyncio.CancelledError, Exception):
                    pass
            await self.release(pid)
