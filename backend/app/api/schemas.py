"""Public API response models. Internal details (storage keys, provider job ids)
are deliberately not exposed: clients only see URLs."""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field

from ..jobs.models import GenerationMode, Job, JobError, JobStatus


class JobParamsOut(BaseModel):
    prompt: str
    negative_prompt: str
    resolution: str
    width: int
    height: int
    num_frames: int
    fps: int
    duration_seconds: float
    steps: int
    guidance_scale: float
    seed: int


class InputImageOut(BaseModel):
    width: int
    height: int
    original_format: str
    original_size_bytes: int


class ResultOut(BaseModel):
    content_type: str = "video/mp4"
    size_bytes: int | None
    download_url: str = Field(description="Download as an attachment")
    preview_url: str = Field(description="Same file served inline, for <video src>")


class JobOut(BaseModel):
    id: str
    mode: GenerationMode
    status: JobStatus
    progress: float = Field(description="0-100")
    params: JobParamsOut
    input_image: InputImageOut | None
    result: ResultOut | None
    error: JobError | None
    provider: str
    created_at: datetime
    updated_at: datetime
    started_at: datetime | None
    completed_at: datetime | None

    @classmethod
    def from_job(cls, job: Job) -> JobOut:
        p = job.params
        result = None
        if job.status == JobStatus.completed and job.output_key:
            base = f"/api/jobs/{job.id}/download"
            result = ResultOut(size_bytes=job.output_size_bytes, download_url=base, preview_url=f"{base}?inline=true")
        return cls(
            id=job.id, mode=job.mode, status=job.status, progress=round(job.progress, 1),
            params=JobParamsOut(
                prompt=p.prompt, negative_prompt=p.negative_prompt, resolution=p.resolution,
                width=p.width, height=p.height, num_frames=p.num_frames, fps=p.fps,
                duration_seconds=p.duration_seconds, steps=p.steps, guidance_scale=p.guidance_scale, seed=p.seed,
            ),
            input_image=InputImageOut(**job.input_image_info) if job.input_image_info else None,
            result=result, error=job.error, provider=job.provider,
            created_at=job.created_at, updated_at=job.updated_at,
            started_at=job.started_at, completed_at=job.completed_at,
        )


class JobListOut(BaseModel):
    items: list[JobOut]
    total: int
    limit: int
    offset: int


class HealthCheckOut(BaseModel):
    name: str = Field(description="e.g. comfyui_reachable, model_files_detected, model_file_sizes, model_weights_integrity, inference")
    status: Literal["ok", "failed", "unverified"] = Field(description="'unverified' = could not be established, not a failure")
    detail: str = ""


class ComponentHealth(BaseModel):
    name: str
    ok: bool = Field(description="Whether the component can accept work now. For a GPU provider this does NOT mean inference is verified; see checks.")
    detail: str = ""
    checks: list[HealthCheckOut] = Field(default_factory=list, description="Separately reported aspects (may be empty)")


class HealthOut(BaseModel):
    status: Literal["ok", "degraded"]
    version: str
    environment: str
    provider: ComponentHealth
    storage: ComponentHealth
    jobs: dict[str, int]


class ErrorDetail(BaseModel):
    code: str
    message: str


class ErrorOut(BaseModel):
    detail: ErrorDetail


# ---------------------------------------------------------------- GET /api/limits
class ModeLimit(BaseModel):
    value: GenerationMode
    label: str
    enabled: bool = Field(description="False = this server does not offer the mode; show it disabled and explain why")
    reason: str | None = Field(description="Why it is disabled (None when enabled)")


class ResolutionLimit(BaseModel):
    value: str = Field(description="e.g. 832x480; send as the `resolution` form field")
    width: int
    height: int
    enabled: bool
    reason: str | None


class FramesLimit(BaseModel):
    min: int
    max: int = Field(description="The largest frame count this server accepts")
    step: int = Field(description="Valid counts are min, min+step, ... (always 4k+1)")
    allowed: list[int] = Field(description="Every valid frame count")
    limited: bool = Field(description="True if this server accepts fewer frames than the model supports")
    reason: str | None


class StepsLimit(BaseModel):
    min: int
    max: int
    limited: bool
    reason: str | None


class RangeLimit(BaseModel):
    min: float
    max: float


class IntRangeLimit(BaseModel):
    min: int
    max: int


class PromptLimit(BaseModel):
    max_chars: int


class ImageLimit(BaseModel):
    mime_types: list[str]
    max_bytes: int
    min_side: int
    max_side: int


class DefaultsOut(BaseModel):
    """Starting values for a form. Every one is valid under the limits in the same response."""

    mode: GenerationMode
    resolution: str
    num_frames: int
    steps: int
    fps: int
    guidance_scale: float


class LimitsOut(BaseModel):
    """What this server will accept for a generation request, so a client can offer only valid choices and explain the rest.

    Independent of GPU availability: it describes configuration, not provider state. POST /api/jobs enforces exactly these limits
    (omitted form fields still take the API's own defaults, which may exceed them on a restricted server: send explicit values)."""

    profile: str = Field(description="Active generation profile: `unrestricted` or a verified GPU profile such as `t4_verified`")
    restricted: bool = Field(description="True if any limit below is stricter than the model's own")
    max_concurrent_jobs: int = Field(description="Jobs run this many at a time; others wait in the queue")
    modes: list[ModeLimit]
    resolutions: list[ResolutionLimit]
    num_frames: FramesLimit
    steps: StepsLimit
    fps: IntRangeLimit
    guidance_scale: RangeLimit
    seed: IntRangeLimit
    prompt: PromptLimit
    image: ImageLimit
    defaults: DefaultsOut
