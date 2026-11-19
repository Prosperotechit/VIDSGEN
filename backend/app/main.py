"""FastAPI application entry point.

Run (development):  uvicorn app.main:app --reload            (from backend/)
API docs:           http://localhost:8000/api/docs           (Swagger UI)

`create_app()` wires the components together:

    Settings ─▶ JobRepository (SQLite) ─┐
             ─▶ Storage (local / s3)  ──┼─▶ JobManager ─▶ API routes
             ─▶ InferenceProvider     ──┘
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, RedirectResponse

from . import __version__
from .api import health, jobs
from .core.config import Settings, get_settings
from .jobs.manager import JobManager
from .jobs.repository import JobRepository
from .providers import build_provider
from .storage import build_storage

log = logging.getLogger("vidsgen")


class UploadSizeLimitMiddleware:
    """Rejects oversized job submissions from the Content-Length header, before the body
    is read. The per-file limit is still enforced again while reading."""

    def __init__(self, app, max_body_bytes: int):
        self.app = app
        self.max_body_bytes = max_body_bytes

    async def __call__(self, scope, receive, send):
        if scope["type"] == "http" and scope["method"] == "POST" and scope["path"] == "/api/jobs":
            length = dict(scope["headers"]).get(b"content-length")
            if length is not None and length.isdigit() and int(length) > self.max_body_bytes:
                resp = JSONResponse(status_code=413, content={"detail": {
                    "code": "request_too_large", "message": "Request body is too large"}})
                await resp(scope, receive, send)
                return
        await self.app(scope, receive, send)


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()
    logging.basicConfig(level=settings.log_level.upper(),
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        repo = JobRepository(settings.database_path)
        storage = build_storage(settings)
        provider = build_provider(settings)
        manager = JobManager(settings, repo, storage, provider)
        app.state.manager = manager
        await manager.start()
        log.info("VIDSGEN API %s ready (env=%s, provider=%s, storage=%s at %s)", __version__, settings.app_env,
                 provider.name, storage.backend_name, settings.storage_local_root)
        try:
            yield
        finally:
            await manager.stop()
            repo.close()

    app = FastAPI(
        title=settings.app_name,
        version=__version__,
        description=(
            "Video generation API for VIDSGEN (Wan 2.2 TI2V-5B).\n\n"
            "Submit a text-to-video or image-to-video job, poll it for progress, then preview or download "
            "the MP4. The GPU backend is pluggable; this instance uses the provider shown by `/api/health`."
        ),
        lifespan=lifespan,
        docs_url="/api/docs",
        redoc_url="/api/redoc",
        openapi_url="/api/openapi.json",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origin_list,
        allow_credentials=False,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["Content-Type", "Authorization", "Range"],
        expose_headers=["Content-Disposition", "Content-Length", "Content-Range", "Accept-Ranges", "Location"],
        max_age=600,
    )
    # Whole request may hold the image plus form fields: allow 1 MB of overhead.
    app.add_middleware(UploadSizeLimitMiddleware, max_body_bytes=settings.max_upload_bytes + 1024 * 1024)

    app.include_router(health.router)
    app.include_router(jobs.router)

    @app.get("/", include_in_schema=False)
    async def root():
        return RedirectResponse("/api/docs")

    return app


app = create_app()
