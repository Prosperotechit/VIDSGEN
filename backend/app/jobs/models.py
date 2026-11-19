"""Job domain model and generation-parameter validation."""

from __future__ import annotations

import secrets
import unicodedata
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class JobStatus(StrEnum):
    queued = "queued"
    processing = "processing"
    completed = "completed"
    failed = "failed"
    cancelled = "cancelled"

    @property
    def is_terminal(self) -> bool:
        return self in TERMINAL_STATUSES


TERMINAL_STATUSES = frozenset({JobStatus.completed, JobStatus.failed, JobStatus.cancelled})


class GenerationMode(StrEnum):
    t2v = "t2v"   # text-to-video
    i2v = "i2v"   # image(+text)-to-video


# Wan 2.2 TI2V-5B: native 720p is 1280x704 / 704x1280. Smaller sizes are kept for
# fast tests on limited GPUs (e.g. a Kaggle T4). All dimensions are multiples of 32.
SUPPORTED_RESOLUTIONS: dict[str, tuple[int, int]] = {
    "1280x704": (1280, 704),
    "704x1280": (704, 1280),
    "832x480": (832, 480),
    "480x832": (480, 832),
    "640x640": (640, 640),
}
MIN_FRAMES, MAX_FRAMES = 17, 121


def utcnow() -> datetime:
    return datetime.now(UTC)


def new_job_id() -> str:
    return f"job_{secrets.token_hex(12)}"


def _clean_text(v: str) -> str:
    v = unicodedata.normalize("NFC", v).strip()
    if any(unicodedata.category(c) == "Cc" and c not in "\n\t" for c in v):
        raise ValueError("must not contain control characters")
    return v


class GenerationParams(BaseModel):
    """Validated, provider-neutral generation parameters (stored with the job)."""

    model_config = ConfigDict(extra="forbid")

    mode: GenerationMode
    prompt: str = Field(min_length=1)
    negative_prompt: str = ""
    resolution: str = "832x480"
    num_frames: int = Field(49, ge=MIN_FRAMES, le=MAX_FRAMES,
                            description="Must be 4k+1 (17, 21, ..., 121). 121 frames at 24 fps is about 5 s.")
    fps: int = Field(24, ge=8, le=30)
    steps: int = Field(30, ge=1, le=60)
    guidance_scale: float = Field(5.0, ge=1.0, le=15.0)
    seed: int | None = Field(None, ge=0, le=2**32 - 1)

    @field_validator("prompt", "negative_prompt")
    @classmethod
    def _text(cls, v: str) -> str:
        return _clean_text(v)

    @field_validator("prompt")
    @classmethod
    def _prompt_not_blank(cls, v: str) -> str:
        if not v:
            raise ValueError("prompt must not be empty")
        return v

    @field_validator("resolution")
    @classmethod
    def _resolution(cls, v: str) -> str:
        v = v.strip().lower()
        if v not in SUPPORTED_RESOLUTIONS:
            raise ValueError(f"unsupported resolution; choose one of {sorted(SUPPORTED_RESOLUTIONS)}")
        return v

    @field_validator("num_frames")
    @classmethod
    def _frames(cls, v: int) -> int:
        if (v - 1) % 4 != 0:
            raise ValueError("num_frames must be 4k+1 (e.g. 17, 49, 81, 121)")
        return v

    @model_validator(mode="after")
    def _fill_seed(self) -> GenerationParams:
        if self.seed is None:
            self.seed = secrets.randbelow(2**32)
        return self

    @property
    def width(self) -> int:
        return SUPPORTED_RESOLUTIONS[self.resolution][0]

    @property
    def height(self) -> int:
        return SUPPORTED_RESOLUTIONS[self.resolution][1]

    @property
    def duration_seconds(self) -> float:
        return round(self.num_frames / self.fps, 2)


class JobError(BaseModel):
    code: str
    message: str


class Job(BaseModel):
    """A generation job as stored in the database."""

    id: str
    mode: GenerationMode
    status: JobStatus = JobStatus.queued
    progress: float = Field(0.0, ge=0, le=100)
    params: GenerationParams
    input_image_key: str | None = None
    input_image_info: dict[str, Any] | None = None
    output_key: str | None = None
    output_size_bytes: int | None = None
    provider: str
    provider_job_id: str | None = None
    error: JobError | None = None
    created_at: datetime
    updated_at: datetime
    started_at: datetime | None = None
    completed_at: datetime | None = None
