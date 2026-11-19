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

The frontend depends only on this contract.

| Method | Path | Purpose |
|---|---|---|
| `POST` | `/api/jobs` | Create a job. Multipart: `mode` (`t2v`/`i2v`), `prompt`, `negative_prompt?`, `image?` (I2V), `resolution`, `duration_frames`, `seed?` |
| `GET` | `/api/jobs/{id}` | Job record: status, progress (0–100), timestamps, error, result URLs |
| `GET` | `/api/jobs/{id}/events` | Server-Sent Events: `status`, `progress`, `completed`, `failed` |
| `GET` | `/api/jobs/{id}/video` | Stream the result for preview (HTTP Range support for seeking) |
| `GET` | `/api/jobs/{id}/download` | Same file with `Content-Disposition: attachment` |
| `POST` | `/api/jobs/{id}/cancel` | Cancel a queued or running job |
| `GET` | `/api/health` | API health plus inference-provider reachability |

### Job lifecycle

```
queued ──▶ submitting ──▶ running ──▶ saving ──▶ completed
   │            │             │          │
   └────────────┴─────────────┴──────────┴──▶ failed / cancelled
```

If the provider is unreachable (e.g. the Kaggle session ended), jobs stay `queued`
and `/api/health` reports the provider as offline, rather than failing silently.

## Inference provider layer (the GPU swap point)

```python
class InferenceProvider(Protocol):
    name: str
    async def health(self) -> ProviderHealth: ...
    async def submit(self, req: GenerationRequest) -> str: ...          # provider job id
    def progress(self, provider_job_id: str) -> AsyncIterator[Progress]: ...
    async def fetch_result(self, provider_job_id: str) -> ResultFile: ...
    async def cancel(self, provider_job_id: str) -> None: ...
```

`GenerationRequest` is provider-neutral (mode, prompt, image bytes, width, height,
frames, fps, steps, cfg, seed). Each provider translates it into its own format.

| Provider | When | Notes |
|---|---|---|
| `mock` | Development / CI | No GPU. Simulates progress and returns a small sample clip. |
| `comfyui` | Kaggle testing, then any GPU host running ComfyUI (RunPod, Vast.ai, own server) | Same class everywhere; only `COMFYUI_URL` / `COMFYUI_API_TOKEN` change. |
| future | Serverless GPU (Modal, Replicate, fal, …) | One new class implementing the same interface. |

Selected with `INFERENCE_PROVIDER` in `.env`. See [PROVIDERS.md](PROVIDERS.md).

### How the ComfyUI provider talks to ComfyUI

1. Load the API-format workflow template for the mode
   (`workflows/comfyui/wan22_ti2v_5b_{t2v,i2v}.api.json`) and fill in parameters
   using `workflows/comfyui/params.yaml` (field → node id / input name).
2. I2V only: upload the input image with `POST /upload/image`.
3. Queue the graph with `POST /prompt` → `prompt_id`.
4. Follow progress on the WebSocket `GET /ws?clientId=…` (`progress`, `executing`,
   `execution_error` events), mapped to 0–100 %.
5. When done, read `GET /history/{prompt_id}` to find the output file, download it
   with `GET /view?filename=…`, and hand it to storage.

Every request carries `Authorization: Bearer $COMFYUI_API_TOKEN`. ComfyUI itself has
no authentication, so it only listens on `127.0.0.1` behind the token proxy.

## Storage layer

```python
class Storage(Protocol):
    async def put(self, key: str, data: BinaryIO, content_type: str) -> None: ...
    async def open(self, key: str, byte_range: tuple[int, int] | None = None) -> AsyncIterator[bytes]: ...
    async def url(self, key: str, expires_s: int = 3600) -> str | None: ...   # presigned/CDN URL if supported
    async def delete(self, key: str) -> None: ...
```

- Keys are stable and provider-independent: `uploads/{job_id}/input.{ext}`,
  `outputs/{job_id}/video.mp4`, `outputs/{job_id}/thumb.jpg`.
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
