"""The inference-provider contract: the single swap point for GPU backends.

The job manager only ever talks to this interface. Moving from the mock provider to
ComfyUI on Kaggle, and later to ComfyUI on another host or a serverless GPU API,
means adding or configuring a provider. The API, jobs and storage code don't change.

Lifecycle, driven by the job manager:

    pid = await provider.submit(request)
    async for p in provider.progress(pid):     # ends after a terminal state
        ...
    result = await provider.fetch_result(pid)  # only after state == "completed"
    ...
    await provider.cancel(pid)                 # any time; best effort
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from ..jobs.models import GenerationMode


@dataclass(frozen=True)
class GenerationRequest:
    """Provider-neutral request. Each provider translates it into its own format."""

    job_id: str
    mode: GenerationMode
    prompt: str
    negative_prompt: str
    width: int
    height: int
    num_frames: int
    fps: int
    steps: int
    guidance_scale: float
    seed: int
    image_png: bytes | None = None   # I2V start image, already validated and normalised


ProgressState = Literal["queued", "running", "completed", "failed", "cancelled"]


@dataclass(frozen=True)
class ProviderProgress:
    state: ProgressState
    percent: float = 0.0             # 0-100
    message: str = ""
    error_code: str | None = None    # set when state == "failed"


@dataclass
class ProviderResult:
    """A finished video, handed over as a local temp file. The caller copies it into
    storage and then calls `cleanup()`."""

    path: Path
    content_type: str = "video/mp4"
    extension: str = "mp4"
    metadata: dict = field(default_factory=dict)

    def cleanup(self) -> None:
        self.path.unlink(missing_ok=True)
        try:
            self.path.parent.rmdir()
        except OSError:
            pass


CheckStatus = Literal["ok", "failed", "unverified"]


@dataclass(frozen=True)
class HealthCheck:
    """One separately reported aspect of provider health.

    "unverified" means the provider could not establish it (not that it failed): for
    example, model weight integrity cannot be checked through ComfyUI's API.
    """

    name: str
    status: CheckStatus
    detail: str = ""


@dataclass(frozen=True)
class ProviderHealth:
    ok: bool                              # can the provider accept work right now?
    detail: str = ""                      # one-line, honest summary
    checks: tuple[HealthCheck, ...] = ()  # optional breakdown (providers may leave it empty)


class ProviderError(Exception):
    """Raised by providers for failures that should mark the job as failed."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code
        self.message = message


class InferenceProvider(ABC):
    name: str = "abstract"

    @abstractmethod
    async def submit(self, request: GenerationRequest) -> str:
        """Start generation and return the provider's own job id."""

    @abstractmethod
    def progress(self, provider_job_id: str) -> AsyncIterator[ProviderProgress]:
        """Yield progress updates until a terminal state (completed/failed/cancelled)."""

    @abstractmethod
    async def fetch_result(self, provider_job_id: str) -> ProviderResult:
        """Return the finished video. Only valid after a 'completed' progress update."""

    @abstractmethod
    async def cancel(self, provider_job_id: str) -> None:
        """Best-effort cancellation. Must not raise for unknown or finished jobs."""

    @abstractmethod
    async def health(self) -> ProviderHealth:
        """Whether the provider can currently accept work."""

    async def release(self, provider_job_id: str) -> None:
        """Called once the job manager is finished with a job (any outcome).
        Providers free per-job resources here. Default: nothing to free."""

    async def aclose(self) -> None:
        """Release resources (connections, temp files) on shutdown."""
