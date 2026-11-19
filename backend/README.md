# Backend (Phase 2, not yet implemented)

Python + FastAPI service that owns the jobs, progress, storage and the connection to the
GPU provider. The frontend only ever talks to this service.

Planned layout:

| Path | Responsibility |
|---|---|
| `app/main.py` | FastAPI app factory, CORS, router registration |
| `app/api/` | HTTP routes: `/api/jobs`, `/api/jobs/{id}/events` (SSE), video/download, `/api/health` |
| `app/core/` | Settings loaded from environment variables, logging, auth helpers |
| `app/jobs/` | Job model + DB, status machine, background worker that drives providers |
| `app/providers/` | `InferenceProvider` interface, `mock` and `comfyui` implementations, factory |
| `app/storage/` | `Storage` interface, `local` implementation (S3/R2 later) |
| `app/workflows/` | Loads ComfyUI templates from `/workflows/comfyui` and fills parameters |
| `tests/` | API, job lifecycle and provider contract tests (run against `mock`) |

See [../docs/ARCHITECTURE.md](../docs/ARCHITECTURE.md).
