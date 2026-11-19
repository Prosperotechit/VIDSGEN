"""Application settings, loaded from environment variables (and `.env` if present).

This is the only module that reads the environment. Everything else receives a
`Settings` instance, which keeps the app testable and host-independent.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

APP_DIR = Path(__file__).resolve().parents[1]          # backend/app
BACKEND_DIR = APP_DIR.parent                           # backend/
REPO_ROOT = BACKEND_DIR.parent                         # repository root


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=(REPO_ROOT / ".env", BACKEND_DIR / ".env"),
        env_file_encoding="utf-8",
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
    comfyui_url: str = ""
    comfyui_api_token: SecretStr = SecretStr("")

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
