# Architecture

## Goals

1. Text-to-video (T2V) and image-to-video (I2V) with Wan 2.2 TI2V-5B.
2. Live generation status and progress, in-browser preview, download.
3. **Replaceable GPU provider**: Kaggle today, something else tomorrow, with no
   frontend changes and no rewrite of the job or storage code.
4. **Replaceable storage**: local disk today, persistent cloud object storage later.

## Components

```
┌──────────────┐   REST + SSE    ┌───────────────────────────────────────────────┐
│ React (Vite) │ ──────────────▶ │ FastAPI backend                                │
│  frontend/   │ ◀────────────── │                                                │
└──────────────┘  video stream   │  api/ ──▶ jobs/ (DB + worker) ──▶ providers/   │
                                 │                 │                    │         │
                                 │                 ▼                    │         │
                                 │             storage/                 │         │
                                 └──────────────────┬───────────────────┼─────────┘
                                                    │                   │ HTTPS + WS
                                       local disk / S3 / R2             ▼  (bearer token)
                                                              ┌───────────────────────┐
                                                              │ GPU host              │
                                                              │  token proxy          │
                                                              │  └─ ComfyUI :8188     │
                                                              │      └─ Wan 2.2 5B    │
                                                              └───────────────────────┘
```

## Public API (frontend ↔ backend)

The frontend depends only on this contract. Interactive docs: `/api/docs` (Swagger UI).

| Method | Path | Purpose |
|---|---|---|
| `POST` | `/api/jobs` | Submit a job (multipart form): `mode` (`t2v`/`i2v`), `prompt`, `negative_prompt?`, `resolution`, `num_frames` (4k+1, 17–121), `fps`, `steps`, `guidance_scale`, `seed?`, `image` (required for i2v, rejected for t2v). Returns **202** immediately. |
| `GET` | `/api/jobs` | History, newest first. `limit` (1–100), `offset`, optional `status` filter |
| `GET` | `/api/jobs/{id}` | Status, progress (0–100), parameters, timestamps, error, result URLs |
| `GET` | `/api/jobs/{id}/download` | The MP4 as an attachment; `?inline=true` serves it for `<video>` preview. Supports HTTP Range (seeking). |
| `POST` | `/api/jobs/{id}/cancel` | Cancel a queued or processing job (409 if already finished) |
| `GET` | `/api/health` | API health plus provider and storage status; `provider.checks` lists each aspect (reachability, model files, sizes, weight integrity, inference) as ok / failed / unverified |

Errors use `{"detail": {"code": "...", "message": "..."}}` (validation errors use FastAPI's
standard 422 list format). Storage keys and provider job IDs are never exposed.

The frontend polls `GET /api/jobs/{id}` adaptively (1.5 s while progress moves, up to 6 s
when it stalls, 15 s in a background tab, and not at all once the job finishes). Planned
later: `GET /api/jobs/{id}/events` (Server-Sent Events) to replace polling.

### Contract between backend and frontend

The backend is the source of truth. `backend/scripts/export_contract.py` writes
`backend/openapi.json` (from which `frontend/src/api/schema.d.ts` is generated) and
`frontend/src/api/constraints.json` (allowed resolutions, frame counts, ranges and upload
limits). `backend/tests/test_contract.py` fails if either is stale.

### Job lifecycle

```
queued ──▶ processing ──▶ completed
   │            ├────────▶ failed      (provider error, timeout, crash, server restart)
   └────────────┴────────▶ cancelled   (user request)
```

- Progress reaches 100 % only after the video is safely stored.
- Every status change is a conditional database update, so a cancellation can never be
  overwritten by a worker that is finishing at the same moment.
- After a restart, `processing` jobs are marked `failed` (code `interrupted`) and `queued`
  jobs are queued again.
- If the provider is unreachable, `/api/health` reports `degraded`, and a submitted job **fails promptly with a
  clear error** (e.g. `comfyui_unreachable`). VIDSGEN never substitutes a mock video for a real one.

## Inference provider layer (the GPU swap point)

```python
class InferenceProvider(Protocol):
    name: str
    async def health(self) -> ProviderHealth: ...
    async def submit(self, req: GenerationRequest) -> str: ...          # provider job id
    def progress(self, provider_job_id: str) -> AsyncIterator[Progress]: ...
    async def fetch_result(self, provider_job_id: str) -> ResultFile: ...
    async def cancel(self, provider_job_id: str) -> None: ...
    async def release(self, provider_job_id: str) -> None: ...   # optional cleanup hook
```

`GenerationRequest` is provider-neutral (mode, prompt, image bytes, width, height,
frames, fps, steps, cfg, seed). Each provider translates it into its own format.

| Provider | When | Notes |
|---|---|---|
| `mock` | Development / CI | No GPU. Simulates progress and returns a small sample clip. |
| `comfyui` | Kaggle testing, then any GPU host running ComfyUI (RunPod, Vast.ai, own server) | Same class everywhere; only `COMFYUI_BASE_URL` / `COMFYUI_API_TOKEN` change. |
| future | Serverless GPU (Modal, Replicate, fal, …) | One new class implementing the same interface. |

Selected with `INFERENCE_PROVIDER` in `.env`. See [PROVIDERS.md](PROVIDERS.md).

### How the ComfyUI provider talks to ComfyUI

Implemented in `backend/app/providers/comfyui.py`; every behaviour below was verified against a real ComfyUI 0.46.0.

1. Pick the API-format template for the mode (`workflows/comfyui/wan22_ti2v_5b_{t2v,i2v}.json`) and fill it in
   via `wan22_ti2v_5b.mapping.json` (see [workflows/comfyui/README.md](../workflows/comfyui/README.md)).
2. I2V only: upload the start image with `POST /upload/image` (input folder, `vidsgen/<job_id>.png`).
3. Queue with `POST /prompt`, passing a **prompt_id generated by VIDSGEN** (UUID). It is stored as the job's
   `provider_job_id`, so the VIDSGEN job ↔ ComfyUI prompt mapping is known even if the response is lost.
4. Follow progress on one shared WebSocket per backend process (`/ws?clientId=…`). Events are routed to jobs
   **only by `prompt_id`**, so concurrent jobs never see each other's progress. Step progress comes from the
   sampler node. If the WebSocket is down or quiet, status is polled from `GET /history/{id}` and `GET /queue`.
5. A job is completed only when `/history` says `status_str == "success"` **and** a video output exists. The file
   is downloaded with `GET /view`, checked to be an MP4, and handed to the job manager, which stores it.
6. Cancel: pending → `POST /queue {"delete": [id]}`; running → targeted `POST /interrupt {"prompt_id": id}`.
7. Release: the prompt's `/history` entry is deleted. Models stay loaded (no per-job `/free`).

Every request carries `Authorization: Bearer $COMFYUI_API_TOKEN`. ComfyUI itself has no authentication, so it
only listens on `127.0.0.1` behind the token proxy. The frontend never sees the ComfyUI URL or token.

## Storage layer

```python
class Storage(Protocol):
    async def put_bytes(self, key: str, data: bytes, content_type: str) -> None: ...
    async def put_file(self, key: str, src: Path, content_type: str) -> None: ...
    async def get_bytes(self, key: str) -> bytes: ...
    def iter_chunks(self, key: str) -> AsyncIterator[bytes]: ...
    async def exists(self, key: str) -> bool: ...
    async def size(self, key: str) -> int: ...
    async def delete(self, key: str) -> None: ...
    def local_path(self, key: str) -> Path | None: ...   # local disk only: efficient Range serving
    async def url(self, key: str, expires_seconds: int = 3600) -> str | None: ...  # presigned/CDN URL (S3)
```

- Keys are stable and provider-independent: `uploads/{job_id}/input.png` (normalised,
  EXIF stripped) and `outputs/{job_id}/video.mp4`.
- The DB stores **keys, not file paths**, so moving from `local` to `s3` (AWS S3,
  Cloudflare R2, Backblaze B2, MinIO) needs no data-model change.
- Results are always copied off the GPU host as soon as they finish.

## Configuration

All settings come from environment variables (see `.env.example`). There is no
host-specific logic in code.

## Security

- Secrets only in `.env` (local) or the platform's secret store (Kaggle Secrets,
  hosting env vars). Never in Git, never in notebooks.
- ComfyUI is never exposed directly; only through the token proxy.
- Uploads are size- and type-checked; prompts are length-limited.
- The Kaggle tunnel is short-lived and for testing only.
