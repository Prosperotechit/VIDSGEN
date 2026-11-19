# Frontend (Phase 3, not yet implemented)

React + Vite + TypeScript web app. It talks **only** to the VIDSGEN backend API and
never to a GPU provider directly.

Planned layout:

| Path | Responsibility |
|---|---|
| `src/api/` | Typed client for `/api/jobs` and the SSE progress stream |
| `src/pages/` | Create (T2V / I2V), job status, library/history |
| `src/components/` | Prompt form, image upload, progress bar, video player, download button |

Development will run against the backend's `mock` provider (no GPU needed).
