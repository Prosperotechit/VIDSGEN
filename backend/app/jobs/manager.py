"""JobManager: owns the job lifecycle and drives the inference provider.

    queued ──▶ processing ──▶ completed
       │            ├────────▶ failed      (provider error, timeout, crash, restart)
       └────────────┴────────▶ cancelled   (user request)

- Jobs are persisted (JobRepository) before anything else happens, so they survive restarts.
- A fixed pool of asyncio workers (INFERENCE_MAX_CONCURRENT_JOBS) takes jobs from a queue.
- Every status change is a conditional transition, so a cancel request and a worker
  can never overwrite each other's result.
- Finished videos are copied into our Storage immediately, so they don't depend on
  the GPU host staying alive.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field

from ..core.config import Settings
from ..providers.base import GenerationRequest, InferenceProvider, ProviderError
from ..storage import ObjectNotFound, Storage, output_key, upload_key
from .envelope import GenerationEnvelope
from .images import ValidatedImage
from .limits import build_limits
from .models import GenerationParams, Job, JobError, JobStatus, new_job_id, utcnow
from .repository import JobRepository

log = logging.getLogger(__name__)


class JobNotFound(LookupError):
    pass


class JobNotCancellable(Exception):
    def __init__(self, status: JobStatus):
        super().__init__(f"Job is already {status.value}")
        self.status = status


@dataclass
class _Active:
    cancel: asyncio.Event = field(default_factory=asyncio.Event)
    provider_job_id: str | None = None


class JobManager:
    def __init__(self, settings: Settings, repo: JobRepository, storage: Storage, provider: InferenceProvider):
        self.settings = settings
        self.repo = repo
        self.storage = storage
        self.provider = provider
        self.envelope = GenerationEnvelope.from_settings(settings)
        build_limits(self.envelope, settings)       # fail at startup (LimitsError) rather than ever advertise limits we cannot honour
        self._queue: asyncio.Queue[str] = asyncio.Queue()
        self._workers: list[asyncio.Task] = []
        self._active: dict[str, _Active] = {}

    # ------------------------------------------------------------ lifecycle
    async def start(self) -> None:
        self._recover()
        for i in range(self.settings.inference_max_concurrent_jobs):
            self._workers.append(asyncio.create_task(self._worker(), name=f"job-worker-{i}"))
        log.info("Job manager started: provider=%s workers=%d limits=%s", self.provider.name, len(self._workers),
                 self.envelope.summary())

    async def stop(self) -> None:
        for w in self._workers:
            w.cancel()
        await asyncio.gather(*self._workers, return_exceptions=True)
        self._workers.clear()
        await self.provider.aclose()

    def _recover(self) -> None:
        """After a restart: jobs that were mid-generation cannot be resumed, so mark them
        failed. Jobs that were still queued go back on the queue."""
        for job_id in self.repo.ids_with_status(JobStatus.processing):
            self.repo.transition(job_id, [JobStatus.processing], status=JobStatus.failed, completed_at=utcnow(),
                                 error=JobError(code="interrupted",
                                                message="The server restarted while this job was processing."))
        for job_id in self.repo.ids_with_status(JobStatus.queued):
            self._queue.put_nowait(job_id)

    # ------------------------------------------------------------ public API
    async def create_job(self, params: GenerationParams, image: ValidatedImage | None) -> Job:
        now = utcnow()
        job_id = new_job_id()
        image_key = None
        if image is not None:
            image_key = upload_key(job_id, "png")
            await self.storage.put_bytes(image_key, image.png_bytes, "image/png")
        job = Job(
            id=job_id, mode=params.mode, params=params, provider=self.provider.name,
            input_image_key=image_key, input_image_info=image.info() if image else None,
            created_at=now, updated_at=now,
        )
        self.repo.insert(job)
        await self._queue.put(job_id)
        log.info("Job %s queued (mode=%s, %s, %d frames)", job_id, params.mode, params.resolution, params.num_frames)
        return job

    def get(self, job_id: str) -> Job:
        job = self.repo.get(job_id)
        if job is None:
            raise JobNotFound(job_id)
        return job

    def list(self, *, limit: int, offset: int, status: JobStatus | None = None) -> tuple[list[Job], int]:
        return self.repo.list(limit=limit, offset=offset, status=status)

    def stats(self) -> dict[str, int]:
        return self.repo.count_by_status()

    async def cancel(self, job_id: str) -> Job:
        job = self.get(job_id)
        now = utcnow()
        if job.status == JobStatus.queued:
            if self.repo.transition(job_id, [JobStatus.queued], status=JobStatus.cancelled, completed_at=now):
                return self.get(job_id)
            job = self.get(job_id)  # changed under us; fall through with the fresh state
        if job.status == JobStatus.processing:
            if self.repo.transition(job_id, [JobStatus.processing], status=JobStatus.cancelled, completed_at=now):
                active = self._active.get(job_id)
                if active:
                    active.cancel.set()
                    if active.provider_job_id:
                        try:
                            await self.provider.cancel(active.provider_job_id)
                        except Exception:
                            log.warning("Provider cancel failed for %s", job_id, exc_info=True)
                return self.get(job_id)
            job = self.get(job_id)
        raise JobNotCancellable(job.status)

    # ------------------------------------------------------------ worker
    async def _worker(self) -> None:
        while True:
            job_id = await self._queue.get()
            try:
                await self._process(job_id)
            except asyncio.CancelledError:
                raise
            except Exception:
                log.exception("Unhandled error processing job %s", job_id)
            finally:
                self._queue.task_done()

    def _fail(self, job_id: str, code: str, message: str) -> None:
        if self.repo.transition(job_id, [JobStatus.processing], status=JobStatus.failed,
                                completed_at=utcnow(), error=JobError(code=code, message=message)):
            log.warning("Job %s failed: %s (%s)", job_id, code, message)

    async def _build_request(self, job: Job) -> GenerationRequest:
        p = job.params
        image = await self.storage.get_bytes(job.input_image_key) if job.input_image_key else None
        return GenerationRequest(
            job_id=job.id, mode=p.mode, prompt=p.prompt, negative_prompt=p.negative_prompt,
            width=p.width, height=p.height, num_frames=p.num_frames, fps=p.fps, steps=p.steps,
            guidance_scale=p.guidance_scale, seed=p.seed, image_png=image,
        )

    async def _process(self, job_id: str) -> None:
        job = self.repo.get(job_id)
        if job is None or job.status != JobStatus.queued:
            return  # cancelled while waiting, or already handled
        if not self.repo.transition(job_id, [JobStatus.queued], status=JobStatus.processing,
                                    started_at=utcnow(), progress=0.0):
            return

        active = self._active[job_id] = _Active()
        try:
            async with asyncio.timeout(self.settings.inference_timeout_seconds):
                await self._run(job, active)
        except TimeoutError:
            if active.provider_job_id:
                await self.provider.cancel(active.provider_job_id)
            self._fail(job_id, "timeout",
                       f"Generation exceeded the {self.settings.inference_timeout_seconds:g}s time limit.")
        except ProviderError as e:
            self._fail(job_id, e.code, e.message)
        except ObjectNotFound:
            self._fail(job_id, "input_missing", "The job's input image is no longer available.")
        except asyncio.CancelledError:
            self._fail(job_id, "interrupted", "The server shut down while this job was processing.")
            raise
        except Exception:
            log.exception("Job %s crashed", job_id)
            self._fail(job_id, "internal_error", "Unexpected error during generation.")
        finally:
            self._active.pop(job_id, None)
            if active.provider_job_id:
                try:
                    await self.provider.release(active.provider_job_id)
                except Exception:
                    log.warning("Provider release failed for %s", job_id, exc_info=True)

    async def _run(self, job: Job, active: _Active) -> None:
        request = await self._build_request(job)
        pid = await self.provider.submit(request)
        active.provider_job_id = pid
        self.repo.transition(job.id, [JobStatus.processing], provider_job_id=pid)
        if active.cancel.is_set():  # cancelled during submit
            await self.provider.cancel(pid)
            return

        final = None
        last_written = -1.0
        async for update in self.provider.progress(pid):
            if active.cancel.is_set():
                await self.provider.cancel(pid)
                return
            if update.state in ("completed", "failed", "cancelled"):
                final = update
                break
            pct = min(update.percent, 99.0)  # 100 % only once the video is safely stored
            if pct - last_written >= 0.5:
                if not self.repo.transition(job.id, [JobStatus.processing], progress=pct):
                    return  # status changed (cancelled)
                last_written = pct

        if active.cancel.is_set():
            return
        if final is None:
            raise ProviderError("provider_error", "Provider stopped reporting progress unexpectedly.")
        if final.state == "cancelled":
            self.repo.transition(job.id, [JobStatus.processing], status=JobStatus.cancelled,
                                 completed_at=utcnow())
            return
        if final.state == "failed":
            raise ProviderError(final.error_code or "provider_error", final.message or "Generation failed.")

        result = await self.provider.fetch_result(pid)
        key = output_key(job.id, result.extension)
        try:
            await self.storage.put_file(key, result.path, result.content_type)
        finally:
            result.cleanup()
        size = await self.storage.size(key)

        if not self.repo.transition(job.id, [JobStatus.processing], status=JobStatus.completed, progress=100.0,
                                    output_key=key, output_size_bytes=size, completed_at=utcnow()):
            await self.storage.delete(key)  # cancelled at the last moment: discard the output
            return
        log.info("Job %s completed (%d bytes)", job.id, size)
