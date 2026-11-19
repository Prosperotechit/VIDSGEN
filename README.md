# VIDSGEN

An AI video-generation application built on the pretrained **Wan 2.2 TI2V-5B** model.
Users can generate videos from **text** or from an **image + text**, watch generation
progress live, preview the result in the browser, and download it.

> VIDSGEN does **not** train a model. It wraps a pretrained open model (Apache-2.0)
> in a product: web UI, API, job queue, storage and a replaceable GPU backend.

## Architecture at a glance

```
 React frontend ──HTTP/SSE──▶ FastAPI backend ──▶ InferenceProvider ──▶ GPU host
   (frontend/)                 (backend/)          (swappable)          ComfyUI + Wan 2.2
                                   │                                    (Kaggle for testing,
                                   ▼                                     any GPU host later)
                              Storage (local → S3/R2)
```

- The **frontend only talks to our API**. It never knows which GPU provider is used.
- The **backend owns jobs and storage**. Finished videos are copied into our storage,
  so they survive when a GPU session (e.g. Kaggle) ends.
- The **inference provider is a plug-in** (`mock` for development, `comfyui` for real
  generation). Moving from Kaggle to RunPod, Modal or our own server is a config change
  or one new provider class, not a rewrite.

Details: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

## Repository layout

| Path | Purpose |
|---|---|
| `frontend/` | React + Vite + TypeScript web app |
| `backend/` | Python FastAPI API, job queue, providers, storage |
| `workflows/comfyui/` | ComfyUI workflow templates and the exact model manifest |
| `notebooks/` | Kaggle notebook that runs ComfyUI for GPU testing (Phase 5) |
| `scripts/` | Repo checks and model download helpers |
| `docs/` | Architecture, Kaggle/T4 setup and validation plan, provider guide |

## Model

Wan 2.2 TI2V-5B: one 5B model for both text-to-video and image-to-video, native
1280×704 / 704×1280 at 24 fps. Exact files are pinned in
[`workflows/comfyui/models.yaml`](workflows/comfyui/models.yaml) (~18 GB total).

## Project rules

- **Never commit** model weights, secrets (`.env`, tokens) or generated media.
  `.gitignore` blocks them and `scripts/check_repo_hygiene.sh` verifies it.
- Kaggle is for **temporary experimentation only**. Production must not depend on a
  running notebook.
- Build and test against the **mock provider** before using GPU time.

## Status / roadmap

| Phase | Scope | Status |
|---|---|---|
| 1 | Repository scaffolding, docs, model manifest | ✅ |
| 2 | Backend: jobs API, progress, provider interface, mock provider, local storage, tests | ✅ |
| 3 | Frontend: T2V/I2V forms, progress, preview, download, history | ✅ |
| 4 | ComfyUI provider, workflow templates, model download script | ✅ |
| 5 | Kaggle notebook: ComfyUI + token proxy + Cloudflare quick tunnel, T4 verification | ⏳ |
| 6 | End-to-end T2V and I2V validation | ⏳ |

## Getting started

Run the backend with the mock provider (no GPU needed), then the web app:
see [backend/README.md](backend/README.md) and [frontend/README.md](frontend/README.md).

```bash
cp .env.example .env
bash scripts/check_repo_hygiene.sh
```
