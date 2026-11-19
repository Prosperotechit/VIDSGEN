"""Storage interface.

Job records store *keys* (e.g. ``outputs/<job_id>/video.mp4``), never filesystem paths.
Any backend that implements this interface (local disk today, S3/R2/B2/MinIO later)
can be swapped in without changing job data or API code.
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from pathlib import Path

_KEY_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*(/[A-Za-z0-9][A-Za-z0-9._-]*)*$")


class StorageError(Exception):
    pass


class InvalidStorageKey(StorageError, ValueError):
    pass


class ObjectNotFound(StorageError, KeyError):
    pass


def validate_key(key: str) -> str:
    """Keys are relative, slash-separated, with no '..', empty or hidden segments."""
    if not isinstance(key, str) or len(key) > 512 or not _KEY_RE.match(key):
        raise InvalidStorageKey(f"Invalid storage key: {key!r}")
    if any(part in (".", "..") for part in key.split("/")):
        raise InvalidStorageKey(f"Invalid storage key: {key!r}")
    return key


def upload_key(job_id: str, ext: str) -> str:
    return validate_key(f"uploads/{job_id}/input.{ext}")


def output_key(job_id: str, ext: str = "mp4") -> str:
    return validate_key(f"outputs/{job_id}/video.{ext}")


class Storage(ABC):
    backend_name: str = "abstract"

    @abstractmethod
    async def put_bytes(self, key: str, data: bytes, content_type: str) -> None: ...

    @abstractmethod
    async def put_file(self, key: str, src: Path, content_type: str) -> None:
        """Copy a local file into storage (the source file is left in place)."""

    @abstractmethod
    async def get_bytes(self, key: str) -> bytes: ...

    @abstractmethod
    def iter_chunks(self, key: str, chunk_size: int = 1024 * 1024) -> AsyncIterator[bytes]: ...

    @abstractmethod
    async def exists(self, key: str) -> bool: ...

    @abstractmethod
    async def size(self, key: str) -> int: ...

    @abstractmethod
    async def delete(self, key: str) -> None:
        """Delete an object. Deleting a missing key is not an error."""

    def local_path(self, key: str) -> Path | None:
        """Filesystem path if the object is on local disk (enables efficient Range serving)."""
        return None

    async def url(self, key: str, expires_seconds: int = 3600) -> str | None:
        """Direct (e.g. presigned/CDN) URL if the backend supports it; None otherwise."""
        return None

    async def healthy(self) -> bool:
        return True
