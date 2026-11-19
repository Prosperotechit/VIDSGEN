from __future__ import annotations

import asyncio

from fastapi import APIRouter, Request

from .. import __version__
from ..jobs.manager import JobManager
from .schemas import ComponentHealth, HealthOut

router = APIRouter(tags=["health"])


@router.get("/api/health", response_model=HealthOut, summary="Service health")
async def health(request: Request) -> HealthOut:
    """Always returns 200 while the API is up. `status` is `degraded` when the
    inference provider or storage is unavailable (e.g. the GPU host is offline);
    jobs can still be submitted and will wait in the queue."""
    manager: JobManager = request.app.state.manager
    try:
        ph = await asyncio.wait_for(manager.provider.health(), timeout=5)
        provider = ComponentHealth(name=manager.provider.name, ok=ph.ok, detail=ph.detail)
    except Exception:
        provider = ComponentHealth(name=manager.provider.name, ok=False, detail="health check failed or timed out")
    storage = ComponentHealth(name=manager.storage.backend_name, ok=await manager.storage.healthy())
    return HealthOut(
        status="ok" if provider.ok and storage.ok else "degraded",
        version=__version__, environment=manager.settings.app_env,
        provider=provider, storage=storage, jobs=manager.stats(),
    )
