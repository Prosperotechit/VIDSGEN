# Inference providers

The backend never calls a GPU host directly. It calls an `InferenceProvider`
(see [ARCHITECTURE.md](ARCHITECTURE.md#inference-provider-layer-the-gpu-swap-point)).

## Choosing a provider

```dotenv
INFERENCE_PROVIDER=mock        # default; or: comfyui
COMFYUI_BASE_URL=https://...   # comfyui only (required)
COMFYUI_API_TOKEN=...          # comfyui only; sent as "Authorization: Bearer", never logged
COMFYUI_TIMEOUT=30             # seconds per HTTP request
COMFYUI_CLIENT_ID=             # empty = vidsgen-<random> per backend process
COMFYUI_POLL_INTERVAL=2        # status polling when WebSocket events are unavailable
```

Restart the backend after changing the provider. With `comfyui`, startup fails immediately (with a clear message)
if `COMFYUI_BASE_URL` is missing or the workflow templates don't match their mapping.

## `mock`

- No GPU, no network. Default for development and CI.
- Simulates queue delay and step-by-step progress, then returns a small generated
  sample clip, so the whole UI flow (progress, preview, download) can be tested.
- Can be configured to fail on purpose so error handling is tested too.

## `comfyui`

Runs Wan 2.2 TI2V-5B on any reachable ComfyUI that has the three model files from
[`workflows/comfyui/models.yaml`](../workflows/comfyui/models.yaml) and sits behind the token proxy. How it talks to
ComfyUI is described in [ARCHITECTURE.md](ARCHITECTURE.md#how-the-comfyui-provider-talks-to-comfyui); the workflow
templates in [workflows/comfyui/README.md](../workflows/comfyui/README.md).

**Health** (`/api/health` → `provider.checks`): each aspect is reported separately as `ok`, `failed` or
`unverified` ("could not be established", not a failure). Nothing claims more than the ComfyUI API can show:

| Check | How it is established | Status values |
|---|---|---|
| `comfyui_reachable` | `GET /system_stats` (version, GPU, VRAM) | ok / failed |
| `workflow_nodes` | `GET /object_info/<class>` for all 11 node classes | ok / failed |
| `model_files_detected` | `GET /models/<folder>`: **filenames only** | ok / failed |
| `model_file_sizes` | `GET /experiment/models/<folder>` (built-in, experimental route) compared with the exact byte sizes in `models.yaml`. Catches 0-byte placeholders and truncated downloads. If the route is unavailable: `unverified` | ok / failed / unverified |
| `model_weights_integrity` | **No ComfyUI API exposes file hashes**, so content integrity cannot be checked remotely. A matching size is not proof. `unverified` until a real generation succeeds | ok / failed / unverified |
| `inference` | `ok` only after a generation that actually loaded the three Wan model files completed with a video on this server (since the backend started) | ok / unverified |

`provider.ok` means "can accept work": reachable, nodes present, files present, and sizes not known to be wrong.
It **does not** mean inference is ready; that is only ever claimed by `inference: ok`. Example with 0-byte
placeholders: `ok=false`, "Model files have the wrong size on the GPU server … (0 bytes, expected 9,999,658,848)".

**Job errors** (shown to users; never containing URLs, tokens or tracebacks):

| Code | Meaning |
|---|---|
| `comfyui_unreachable` / `comfyui_timeout` | GPU server down, tunnel closed, or not answering |
| `comfyui_auth_failed` | Token rejected by the proxy |
| `model_missing` | A required model file is not installed on the GPU server |
| `image_upload_failed` | The I2V start image could not be uploaded |
| `comfyui_rejected` | ComfyUI's validation refused the workflow (details included) |
| `comfyui_execution_error` | A node failed while running (node type + ComfyUI's message) |
| `gpu_out_of_memory` | CUDA out of memory: try a smaller resolution or fewer frames |
| `no_video_output` / `unsupported_output` | Finished without a video, or not an MP4 |
| `comfyui_job_lost` | ComfyUI no longer knows the job (e.g. it restarted) |
| `comfyui_server_error` / `comfyui_bad_response` | HTTP 5xx or a malformed response |

| Host | Use | Notes |
|---|---|---|
| Kaggle (T4) | Temporary testing only | Free, about 30 GPU-h/week, sessions max 12 h, URL changes every session. See [KAGGLE_SETUP.md](KAGGLE_SETUP.md). |
| RunPod / Vast.ai / Lambda | Low-cost persistent GPU | Same ComfyUI + proxy setup; stable URL. |
| Own GPU server | Long-term | Same setup. |

**Testing without a GPU:** `backend/tests/test_comfyui_provider.py` runs the real provider against a fake ComfyUI
(real HTTP + WebSocket, shapes copied from ComfyUI 0.46.0). `tests/test_comfyui_live.py` runs against a real
ComfyUI when `VIDSGEN_COMFYUI_LIVE_URL` is set.

## Adding a new provider

1. Create `backend/app/providers/<name>.py` implementing `InferenceProvider`.
2. Translate `GenerationRequest` to the provider's API; map its progress to 0–100.
3. Register it in the provider factory and document its env vars in `.env.example`.
4. Add contract tests (the same test suite runs against every provider using fakes).

Nothing in `frontend/`, `backend/app/api/`, `backend/app/jobs/` or
`backend/app/storage/` changes.
