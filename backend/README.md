# VIDSGEN backend

Python + FastAPI service that owns generation jobs, progress, storage and the connection
to the GPU provider. The frontend only ever talks to this service.

## Quick start (no GPU needed)

```bash
cd backend
python3 -m venv .venv
.venv/bin/pip install -e ".[dev]"
.venv/bin/uvicorn app.main:app --reload
```

- Swagger UI: http://localhost:8000/api/docs
- Health: http://localhost:8000/api/health

By default it uses the **mock provider** (no GPU) and stores videos, uploads and the SQLite
database in `<repo>/storage/` (git-ignored). Configure it through environment variables or a
`.env` file at the repo root (see `../.env.example`).

## Tests

```bash
.venv/bin/python -m pytest
```

Every test runs against an isolated temp storage dir and DB with the mock provider.
`.env` files are ignored during tests. Leaked files and processes (ResourceWarning) fail the suite.

## Layout

```
app/
├── main.py              App factory: wires Settings → repo, storage, provider → JobManager → routes
├── core/config.py       Settings from env vars (the only place that reads the environment)
├── api/
│   ├── health.py        GET /api/health
│   ├── jobs.py          /api/jobs endpoints (submit, list, get, cancel, download)
│   └── schemas.py       Public response models (no storage keys or provider ids leak out)
├── jobs/
│   ├── models.py        Job, JobStatus, GenerationParams (+ all parameter validation)
│   ├── images.py        I2V upload validation and normalisation (PNG, EXIF stripped)
│   ├── repository.py    SQLite persistence with atomic conditional status transitions
│   └── manager.py       Lifecycle, worker pool, progress, cancellation, timeouts, restart recovery
├── providers/
│   ├── base.py          InferenceProvider interface (submit/progress/fetch_result/cancel/health)
│   ├── mock.py          MockProvider: simulated steps, real MP4 output, failure trigger
│   ├── mock_video.py    Synthesises H.264 MP4s with ffmpeg (bundled via imageio-ffmpeg)
│   └── __init__.py      build_provider(): picks provider from INFERENCE_PROVIDER
├── storage/
│   ├── base.py          Storage interface + key validation (keys, never paths, in job records)
│   ├── local.py         LocalStorage: atomic writes, path-traversal safe
│   └── __init__.py      build_storage(): picks backend from STORAGE_BACKEND
└── workflows/           ComfyUI template loading (Phase 4)
tests/                   91 tests: API, lifecycle, validation, failures, cancellation, storage
```

## How the pieces relate

```
HTTP request ─▶ api/jobs.py ─▶ JobManager ─┬─▶ JobRepository (SQLite)   job records, status, progress
                                            ├─▶ Storage (local → S3)     uploads/<id>/input.png, outputs/<id>/video.mp4
                                            └─▶ InferenceProvider        mock now, ComfyUI in Phase 4
```

1. `POST /api/jobs` validates input, stores the image (I2V), inserts the job (`queued`) and
   returns **202** at once.
2. A worker picks it up (`processing`), builds a provider-neutral `GenerationRequest`, calls
   `provider.submit()`, then streams `provider.progress()` into the job's `progress`.
3. On success the video is copied from the provider into Storage, and only then does the job
   become `completed` at 100 %. Errors become `failed` with an error code. A cancel request
   becomes `cancelled` and the provider is told to stop.

## Mock provider test hooks

- `MOCK_STEP_DELAY_SECONDS`: time per simulated denoising step (default 0.15).
- A prompt containing `[mock:fail]` fails halfway through, to exercise error handling.
