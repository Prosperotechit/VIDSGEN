from __future__ import annotations

import asyncio

from fastapi import APIRouter, Request

from .. import __version__
from ..jobs.manager import JobManager
from .schemas import ComponentHealth, HealthCheckOut, HealthOut

router = APIRouter(tags=["health"])


@router.get("/api/health", response_model=HealthOut, summary="Service health")
async def health(request: Request) -> HealthOut:
    """Always returns 200 while the API is up. `status` is `degraded` when the inference
    provider or storage cannot accept work (e.g. the GPU host is offline, or its model files
    are missing or have the wrong size). `provider.checks` reports each aspect separately;
    for a GPU provider, weight integrity and inference stay "unverified" until a real
    generation has succeeded, so `ok` never claims inference readiness."""
    manager: JobManager = request.app.state.manager
    try:
        ph = await asyncio.wait_for(manager.provider.health(), timeout=5)
        provider = ComponentHealth(
            name=manager.provider.name, ok=ph.ok, detail=ph.detail,
            checks=[HealthCheckOut(name=c.name, status=c.status, detail=c.detail) for c in ph.checks],
        )
    except Exception:
        provider = ComponentHealth(name=manager.provider.name, ok=False, detail="health check failed or timed out")
    storage = ComponentHealth(name=manager.storage.backend_name, ok=await manager.storage.healthy())
    return HealthOut(
        status="ok" if provider.ok and storage.ok else "degraded",
        version=__version__, environment=manager.settings.app_env,
        provider=provider, storage=storage, jobs=manager.stats(),
    )
