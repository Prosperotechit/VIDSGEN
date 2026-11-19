"""Local-disk storage backend.

Objects live under a single root directory (default: <repo>/storage, git-ignored).
Writes are atomic (write to a temp file, then rename), and every key is resolved
and checked to stay inside the root, so a key can never escape it.
"""

from __future__ import annotations

import asyncio
import os
import shutil
import tempfile
from collections.abc import AsyncIterator
from pathlib import Path

from .base import InvalidStorageKey, ObjectNotFound, Storage, validate_key


class LocalStorage(Storage):
    backend_name = "local"

    def __init__(self, root: Path):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------ helpers
    def _path(self, key: str) -> Path:
        validate_key(key)
        p = (self.root / key).resolve()
        if self.root not in p.parents:
            raise InvalidStorageKey(f"Key escapes storage root: {key!r}")
        return p

    def _atomic_write(self, dest: Path, writer) -> None:
        dest.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=dest.parent, prefix=".tmp-", suffix=dest.suffix)
        try:
            with os.fdopen(fd, "wb") as f:
                writer(f)
            os.replace(tmp, dest)
        except BaseException:
            Path(tmp).unlink(missing_ok=True)
            raise

    # ------------------------------------------------------------ interface
    async def put_bytes(self, key: str, data: bytes, content_type: str) -> None:
        dest = self._path(key)
        await asyncio.to_thread(self._atomic_write, dest, lambda f: f.write(data))

    async def put_file(self, key: str, src: Path, content_type: str) -> None:
        dest = self._path(key)

        def copy(f):
            with open(src, "rb") as s:
                shutil.copyfileobj(s, f, 1024 * 1024)

        await asyncio.to_thread(self._atomic_write, dest, copy)

    async def get_bytes(self, key: str) -> bytes:
        p = self._path(key)
        try:
            return await asyncio.to_thread(p.read_bytes)
        except FileNotFoundError as e:
            raise ObjectNotFound(key) from e

    async def iter_chunks(self, key: str, chunk_size: int = 1024 * 1024) -> AsyncIterator[bytes]:
        p = self._path(key)
        if not p.is_file():
            raise ObjectNotFound(key)
        with open(p, "rb") as f:
            while chunk := await asyncio.to_thread(f.read, chunk_size):
                yield chunk

    async def exists(self, key: str) -> bool:
        return self._path(key).is_file()

    async def size(self, key: str) -> int:
        p = self._path(key)
        try:
            return p.stat().st_size
        except FileNotFoundError as e:
            raise ObjectNotFound(key) from e

    async def delete(self, key: str) -> None:
        p = self._path(key)
        p.unlink(missing_ok=True)
        # Tidy up empty parent directories up to the root.
        parent = p.parent
        while parent != self.root:
            try:
                parent.rmdir()
            except OSError:
                break
            parent = parent.parent

    def local_path(self, key: str) -> Path | None:
        p = self._path(key)
        return p if p.is_file() else None

    async def healthy(self) -> bool:
        return self.root.is_dir() and os.access(self.root, os.W_OK)
