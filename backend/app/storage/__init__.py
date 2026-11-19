from __future__ import annotations

from ..core.config import Settings
from .base import InvalidStorageKey, ObjectNotFound, Storage, StorageError, output_key, upload_key
from .local import LocalStorage

__all__ = [
    "InvalidStorageKey", "LocalStorage", "ObjectNotFound", "Storage", "StorageError",
    "build_storage", "output_key", "upload_key",
]


def build_storage(settings: Settings) -> Storage:
    if settings.storage_backend == "local":
        return LocalStorage(settings.storage_local_root)
    if settings.storage_backend == "s3":
        raise NotImplementedError(
            "STORAGE_BACKEND=s3 is planned but not implemented yet; use 'local'."
        )
    raise ValueError(f"Unknown storage backend {settings.storage_backend!r}")
