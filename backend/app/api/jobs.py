from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, File, Form, HTTPException, Query, Request, UploadFile, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, RedirectResponse, StreamingResponse
from pydantic import ValidationError

from ..jobs.images import ImageValidationError, validate_image
from ..jobs.manager import JobManager, JobNotCancellable, JobNotFound
from ..jobs.models import GenerationMode, GenerationParams, JobStatus
from .schemas import ErrorOut, JobListOut, JobOut

router = APIRouter(prefix="/api/jobs", tags=["jobs"])

NOT_FOUND = {404: {"model": ErrorOut, "description": "Job not found"}}


def _manager(request: Request) -> JobManager:
    return request.app.state.manager


def _error(code: int, err: str, message: str) -> HTTPException:
    return HTTPException(status_code=code, detail={"code": err, "message": message})


def _field_error(field: str, msg: str, value=None) -> RequestValidationError:
    return RequestValidationError([{"type": "value_error", "loc": ("body", field), "msg": msg, "input": value}])


def _get_or_404(manager: JobManager, job_id: str):
    try:
        return manager.get(job_id)
    except JobNotFound:
        raise _error(404, "job_not_found", f"No job with id {job_id}") from None


@router.post(
    "", status_code=status.HTTP_202_ACCEPTED, response_model=JobOut,
    summary="Submit a text-to-video or image-to-video job",
    responses={413: {"model": ErrorOut}, 415: {"model": ErrorOut}},
)
async def create_job(
    request: Request,
    mode: Annotated[GenerationMode, Form(description="t2v = text-to-video, i2v = image-to-video")],
    prompt: Annotated[str, Form()],
    negative_prompt: Annotated[str, Form()] = "",
    resolution: Annotated[str, Form(description="1280x704, 704x1280, 832x480, 480x832 or 640x640")] = "832x480",
    num_frames: Annotated[int, Form(description="4k+1, between 17 and 121")] = 49,
    fps: Annotated[int, Form()] = 24,
    steps: Annotated[int, Form()] = 30,
    guidance_scale: Annotated[float, Form()] = 5.0,
    seed: Annotated[int | None, Form()] = None,
    image: Annotated[UploadFile | None, File(description="Required for i2v (JPEG/PNG/WEBP); not allowed for t2v")] = None,
) -> JobOut:
    """Validates the request, stores it, and queues it. Returns **202** immediately;
    poll `GET /api/jobs/{id}` for status and progress."""
    manager = _manager(request)
    settings = manager.settings

    if len(prompt) > settings.max_prompt_chars:
        raise _field_error("prompt", f"prompt must be at most {settings.max_prompt_chars} characters")
    if len(negative_prompt) > settings.max_prompt_chars:
        raise _field_error("negative_prompt", f"negative_prompt must be at most {settings.max_prompt_chars} characters")

    try:
        params = GenerationParams(
            mode=mode, prompt=prompt, negative_prompt=negative_prompt, resolution=resolution,
            num_frames=num_frames, fps=fps, steps=steps, guidance_scale=guidance_scale, seed=seed,
        )
    except ValidationError as e:
        raise RequestValidationError(
            [{**err, "loc": ("body", *err["loc"])} for err in e.errors(include_url=False)]
        ) from None

    has_image = image is not None and bool(image.filename)
    if mode == GenerationMode.i2v and not has_image:
        raise _field_error("image", "an image is required for image-to-video (mode=i2v)")
    if mode == GenerationMode.t2v and has_image:
        raise _field_error("image", "images are not accepted for text-to-video; use mode=i2v")

    validated = None
    if has_image:
        data = await image.read(settings.max_upload_bytes + 1)
        try:
            validated = validate_image(data, max_bytes=settings.max_upload_bytes,
                                       min_side=settings.min_image_side, max_side=settings.max_image_side)
        except ImageValidationError as e:
            if e.status_code == 422:
                raise _field_error("image", str(e)) from None
            raise _error(e.status_code, "invalid_image", str(e)) from None

    job = await manager.create_job(params, validated)
    return JobOut.from_job(job)


@router.get("", response_model=JobListOut, summary="Generation history (newest first)")
async def list_jobs(
    request: Request,
    limit: Annotated[int, Query(ge=1, le=100)] = 20,
    offset: Annotated[int, Query(ge=0)] = 0,
    status_filter: Annotated[JobStatus | None, Query(alias="status")] = None,
) -> JobListOut:
    jobs, total = _manager(request).list(limit=limit, offset=offset, status=status_filter)
    return JobListOut(items=[JobOut.from_job(j) for j in jobs], total=total, limit=limit, offset=offset)


@router.get("/{job_id}", response_model=JobOut, responses=NOT_FOUND, summary="Job status, progress and details")
async def get_job(request: Request, job_id: str) -> JobOut:
    return JobOut.from_job(_get_or_404(_manager(request), job_id))


@router.post(
    "/{job_id}/cancel", response_model=JobOut, summary="Cancel a queued or processing job",
    responses={**NOT_FOUND, 409: {"model": ErrorOut, "description": "Job already finished"}},
)
async def cancel_job(request: Request, job_id: str) -> JobOut:
    manager = _manager(request)
    _get_or_404(manager, job_id)
    try:
        return JobOut.from_job(await manager.cancel(job_id))
    except JobNotCancellable as e:
        raise _error(409, "job_not_cancellable", f"Job is already {e.status.value} and cannot be cancelled") from None


@router.get(
    "/{job_id}/download", summary="Download (or preview) the generated video",
    response_class=FileResponse,
    responses={
        200: {"content": {"video/mp4": {}}, "description": "The MP4 file (supports HTTP Range requests)"},
        **NOT_FOUND,
        409: {"model": ErrorOut, "description": "Job is not completed"},
        410: {"model": ErrorOut, "description": "Result file is no longer available"},
    },
)
async def download(request: Request, job_id: str, inline: bool = False):
    """`inline=false` (default) sends `Content-Disposition: attachment` for a download.
    `inline=true` serves the same file for in-browser preview. Range requests are
    supported for seeking."""
    manager = _manager(request)
    job = _get_or_404(manager, job_id)
    if job.status != JobStatus.completed or not job.output_key:
        raise _error(409, "job_not_completed", f"Job status is '{job.status.value}'; a video is only available for completed jobs")

    filename = f"vidsgen-{job.id}.mp4"
    disposition = "inline" if inline else "attachment"
    storage = manager.storage

    path = storage.local_path(job.output_key)
    if path is not None:
        return FileResponse(path, media_type="video/mp4", filename=filename, content_disposition_type=disposition)
    if not await storage.exists(job.output_key):
        raise _error(410, "result_missing", "The video file is no longer available")
    url = await storage.url(job.output_key)
    if url:
        return RedirectResponse(url, status_code=307)
    return StreamingResponse(
        storage.iter_chunks(job.output_key), media_type="video/mp4",
        headers={"Content-Disposition": f'{disposition}; filename="{filename}"'},
    )
