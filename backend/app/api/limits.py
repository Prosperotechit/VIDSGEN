from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException, Request

from ..jobs.limits import LimitsError, build_limits
from ..jobs.manager import JobManager
from .schemas import ErrorOut, LimitsOut

router = APIRouter(tags=["limits"])
log = logging.getLogger(__name__)


@router.get("/api/limits", response_model=LimitsOut, summary="Generation limits and defaults for this server",
            responses={500: {"model": ErrorOut, "description": "The server's generation limits are misconfigured"}})
async def limits(request: Request) -> LimitsOut:
    """The active generation capabilities: which modes and resolutions are offered, valid frame counts, step/fps/guidance ranges, upload
    and prompt limits, and starting values. Computed from the same configuration that validates `POST /api/jobs`; it never contacts
    the GPU provider and contains no URLs, credentials or paths."""
    manager: JobManager = request.app.state.manager
    try:
        return LimitsOut(**build_limits(manager.envelope, manager.settings))
    except LimitsError:
        # Never fall back to permissive numbers: say the limits cannot be determined.
        log.exception("generation limits are inconsistent")
        raise HTTPException(status_code=500, detail={"code": "limits_unavailable",
                                                     "message": "The server's generation limits are misconfigured."}) from None
