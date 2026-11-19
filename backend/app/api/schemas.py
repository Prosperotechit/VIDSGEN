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
