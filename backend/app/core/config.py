"""Application settings, loaded from environment variables (and `.env` if present).

This is the only module that reads the environment. Everything else receives a
`Settings` instance, which keeps the app testable and host-independent.
"""

from __future__ import annotations

import ipaddress
from functools import lru_cache
from pathlib import Path
from typing import Literal
from urllib.parse import urlsplit

from pydantic import Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

MIN_REMOTE_TOKEN_CHARS = 32     # the inference proxy refuses shorter tokens too


def is_loopback_host(host: str | None) -> bool:
    """True for localhost and loopback addresses (127.0.0.0/8, ::1): traffic that never leaves this machine."""
    if not host:
        return False
    host = host.strip("[]").rstrip(".").lower()
    if host == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def check_comfyui_endpoint(url: str, token: str) -> None:
    """Raises ValueError (never echoing the URL or the token) unless this is a safe place to send generation traffic.

    * Only plain http(s) URLs without credentials, query or fragment.
    * A loopback address (development, or a proxy on the same machine) may use http and needs no token.
    * ANYTHING ELSE must use https and carry a bearer token of at least 32 characters: prompts, images and videos
      must not cross a network in clear text or unauthenticated."""
    parts = urlsplit(url)
    if parts.scheme not in ("http", "https") or not parts.hostname:
        raise ValueError("COMFYUI_BASE_URL must be an http:// or https:// URL with a host")
    if parts.username is not None or parts.password is not None or "@" in parts.netloc:
        raise ValueError("COMFYUI_BASE_URL must not contain credentials (use COMFYUI_API_TOKEN)")
    if parts.query or parts.fragment:
        raise ValueError("COMFYUI_BASE_URL must not contain a query string or fragment")
    if not is_loopback_host(parts.hostname):
        if parts.scheme != "https":
            raise ValueError("COMFYUI_BASE_URL must use https:// unless it points to this machine (127.0.0.1 / localhost)")
        if len(token) < MIN_REMOTE_TOKEN_CHARS:
            raise ValueError(f"COMFYUI_API_TOKEN is required for a remote COMFYUI_BASE_URL and must be at least "
                             f"{MIN_REMOTE_TOKEN_CHARS} characters")


APP_DIR = Path(__file__).resolve().parents[1]          # backend/app
BACKEND_DIR = APP_DIR.parent                           # backend/
REPO_ROOT = BACKEND_DIR.parent                         # repository root


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=(REPO_ROOT / ".env", BACKEND_DIR / ".env"),
        env_file_encoding="utf-8",
        hide_input_in_errors=True,          # a validation error must never print a URL or token it was given
        extra="ignore",
        case_sensitive=False,
    )

    # ------------------------------------------------------------- app
    app_name: str = "VIDSGEN API"
    app_env: Literal["development", "test", "production"] = "development"
    log_level: str = "INFO"
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    # Comma-separated list, e.g. "http://localhost:5173,https://app.example.com"
    cors_origins: str = "http://localhost:5173"
    # Empty = <storage root>/vidsgen.db. Only sqlite:/// URLs are supported for now.
    database_url: str = ""

    # ------------------------------------------------------------- inference
    inference_provider: Literal["mock", "comfyui"] = "mock"
    inference_timeout_seconds: float = Field(1800, gt=0)
    inference_max_concurrent_jobs: int = Field(1, ge=1, le=16)

    # ComfyUI provider (INFERENCE_PROVIDER=comfyui). Server-side only: never sent to the frontend.
    comfyui_base_url: str = ""                       # e.g. https://<tunnel-host> or http://gpu-box:8188
    comfyui_api_token: SecretStr = SecretStr("")     # sent as "Authorization: Bearer"; checked by the token proxy
    comfyui_timeout: float = Field(30, gt=0, le=600)  # per HTTP request, seconds (whole job: INFERENCE_TIMEOUT_SECONDS)
    comfyui_client_id: str = ""                      # empty = "vidsgen-<random>" per backend process
    comfyui_poll_interval: float = Field(2.0, gt=0, le=60)  # status polling when WebSocket events are unavailable
    comfyui_workflows_dir: Path = REPO_ROOT / "workflows" / "comfyui"
    # How long a job survives a lost connection to ComfyUI (tunnel blip, restart) while it is being generated,
    # retrying every poll interval, before it fails with "comfyui_connection_lost". 0 = fail on the first error.
    comfyui_connection_grace_seconds: float = Field(30, ge=0, le=600)

    # ------------------------------------------------------------- GPU generation envelope (see app/jobs/envelope.py)
    # "unrestricted" (default, used with the mock provider) or a verified preset such as "t4_verified".
    generation_profile: Literal["unrestricted", "t4_verified"] = "unrestricted"
    # Each of these overrides the preset's value when set. Empty = use the preset (or no limit).
    generation_allowed_modes: str = ""            # e.g. "t2v" or "t2v,i2v"
    generation_allowed_resolutions: str = ""      # e.g. "832x480"
    generation_max_frames: int | None = Field(None, ge=17, le=121)
    generation_max_steps: int | None = Field(None, ge=1, le=60)

    # Mock provider tuning (development/tests only)
    mock_step_delay_seconds: float = Field(0.15, ge=0)

    # ------------------------------------------------------------- storage
    storage_backend: Literal["local", "s3"] = "local"
    # Default: <repo>/storage (git-ignored, outside source directories)
    storage_local_root: Path = REPO_ROOT / "storage"
    storage_public_base_url: str = ""
    s3_endpoint_url: str = ""
    s3_bucket: str = ""
    s3_region: str = "auto"
    s3_access_key_id: SecretStr = SecretStr("")
    s3_secret_access_key: SecretStr = SecretStr("")

    # ------------------------------------------------------------- limits
    max_upload_mb: float = Field(10, gt=0, le=100)
    max_prompt_chars: int = Field(2000, ge=10, le=10000)
    min_image_side: int = Field(256, ge=16)
    max_image_side: int = Field(4096, ge=256)

    @field_validator("generation_max_frames", "generation_max_steps", mode="before")
    @classmethod
    def _blank_limit_means_unset(cls, v):
        return None if isinstance(v, str) and not v.strip() else v

    @model_validator(mode="after")
    def _check_gpu_envelope(self) -> Settings:
        if self.generation_profile == "t4_verified" and self.inference_max_concurrent_jobs != 1:
            raise ValueError("GENERATION_PROFILE=t4_verified requires INFERENCE_MAX_CONCURRENT_JOBS=1 "
                             "(one GPU, proven only for one job at a time)")
        from ..jobs.envelope import GenerationEnvelope   # validates the mode/resolution/frame overrides
        GenerationEnvelope.from_settings(self)
        if self.inference_provider == "comfyui" and self.comfyui_base_url:
            check_comfyui_endpoint(self.comfyui_base_url, self.comfyui_api_token.get_secret_value())
        return self

    @field_validator("comfyui_base_url", mode="after")
    @classmethod
    def _check_comfyui_url(cls, v: str) -> str:
        v = v.strip().rstrip("/")
        if v and not v.startswith(("http://", "https://")):
            raise ValueError("COMFYUI_BASE_URL must start with http:// or https://")
        return v

    @field_validator("comfyui_workflows_dir", mode="before")
    @classmethod
    def _empty_workflows_dir(cls, v):
        if v is None or (isinstance(v, str) and not v.strip()):
            return REPO_ROOT / "workflows" / "comfyui"
        return v

    @field_validator("storage_local_root", mode="before")
    @classmethod
    def _empty_means_default(cls, v):
        # `STORAGE_LOCAL_ROOT=` (as in .env.example) must mean "default", not the current directory.
        if v is None or (isinstance(v, str) and not v.strip()):
            return REPO_ROOT / "storage"
        return v

    @field_validator("storage_local_root", mode="after")
    @classmethod
    def _resolve_storage_root(cls, v: Path) -> Path:
        root = (v if v.is_absolute() else (Path.cwd() / v)).resolve()
        if root == APP_DIR or APP_DIR in root.parents:
            raise ValueError("STORAGE_LOCAL_ROOT must not be inside the application source directory")
        return root

    # ------------------------------------------------------------- derived
    @property
    def cors_origin_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]

    @property
    def database_path(self) -> Path:
        if not self.database_url:
            return self.storage_local_root / "vidsgen.db"
        prefix = "sqlite:///"
        if not self.database_url.startswith(prefix):
            raise ValueError("Only sqlite:/// DATABASE_URL values are supported in this version")
        p = Path(self.database_url[len(prefix):])
        return (p if p.is_absolute() else Path.cwd() / p).resolve()

    @property
    def max_upload_bytes(self) -> int:
        return int(self.max_upload_mb * 1024 * 1024)


@lru_cache
def get_settings() -> Settings:
    return Settings()
