# VIDSGEN frontend

React + Vite + TypeScript web app for VIDSGEN. It talks **only** to the VIDSGEN FastAPI
backend: never to ComfyUI, Hugging Face or a GPU provider directly.

## Quick start

```bash
# terminal 1: backend (mock provider, no GPU)
cd backend && .venv/bin/uvicorn app.main:app --reload

# terminal 2: frontend
cd frontend
npm install
npm run dev          # http://localhost:5173
```

The dev server proxies `/api` to the backend (`VITE_DEV_API_PROXY`, default
`http://127.0.0.1:8000`), so no CORS setup is needed during development.

## Scripts

| Command | What it does |
|---|---|
| `npm run dev` | Vite dev server with `/api` proxy |
| `npm run build` | Typecheck + production build to `dist/` |
| `npm run typecheck` | TypeScript only |
| `npm test` | Unit tests (Vitest): API client, form encoding, constraints, polling backoff |
| `npm run e2e` | End-to-end: starts the backend (mock) and Vite, drives Chromium on desktop and mobile. Set `SCREENSHOT_DIR=…` to save screenshots. |
| `npm run gen:api` | Regenerate `src/api/schema.d.ts` from `../backend/openapi.json` |

## Configuration

| Variable | Default | Purpose |
|---|---|---|
| `VITE_API_BASE_URL` | empty (same origin) | Public URL of the API for production builds, e.g. `https://api.example.com`. The backend's `CORS_ORIGINS` must then include the site's origin. |
| `VITE_DEV_API_PROXY` | `http://127.0.0.1:8000` | Dev/preview server only: where `/api` is proxied |

Everything in the frontend bundle is public, so never put secrets in `VITE_*` variables.

## How it stays in sync with the backend

The backend owns the contract. Two generated files are committed:

- `src/api/schema.d.ts`: TypeScript types generated from `backend/openapi.json`.
- `src/api/constraints.json`: the allowed resolutions, frame counts, ranges and upload limits,
  exported from the backend's own validation code.

After changing the API or its validation rules:

```bash
cd backend && .venv/bin/python scripts/export_contract.py
cd ../frontend && npm run gen:api
```

`backend/tests/test_contract.py` fails if either file is stale, so the UI can never offer
values the API would reject.

## Layout

```
src/
├── main.tsx              Router + app shell
├── api/                  Everything that touches the network (no UI code)
│   ├── client.ts         fetch wrapper, base URL, ApiError normalisation, timeouts
│   ├── jobs.ts           Typed endpoint functions + multipart encoding
│   ├── types.ts          Aliases over the generated OpenAPI types
│   ├── constraints.ts    Typed access to constraints.json + client-side image checks
│   ├── schema.d.ts       GENERATED from backend/openapi.json
│   └── constraints.json  GENERATED from backend validation code
├── hooks/
│   ├── useJobPolling.ts  Adaptive polling (1.5–6 s, 15 s in background, backoff on errors; stops when done)
│   └── useHealth.ts      API/provider health
├── components/           Reusable UI: AppShell, JobStage, VideoPlayer, ImageDropzone, ResolutionPicker, …
├── pages/                Create, My Generations, Generation detail, Settings
├── lib/format.ts         Dates, durations, labels
└── styles/               Design tokens + layout/component/page CSS (dark theme)
e2e/run-e2e.mjs           Browser tests against the real backend
```

## Production notes

- `npm run build` outputs a static site in `dist/`. Any static host works; configure it to
  serve `index.html` for unknown paths, because the app uses client-side routes such as
  `/generations/<id>`.
- Either serve the API under the same domain at `/api`, or set `VITE_API_BASE_URL` and the
  backend's `CORS_ORIGINS`.
