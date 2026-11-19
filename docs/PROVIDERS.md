# Inference providers

The backend never calls a GPU host directly. It calls an `InferenceProvider`
(see [ARCHITECTURE.md](ARCHITECTURE.md#inference-provider-layer-the-gpu-swap-point)).

## Choosing a provider

```dotenv
INFERENCE_PROVIDER=mock        # or: comfyui
COMFYUI_URL=https://...        # comfyui only
COMFYUI_API_TOKEN=...          # comfyui only
```

## `mock`

- No GPU, no network. Default for development and CI.
- Simulates queue delay and step-by-step progress, then returns a small generated
  sample clip, so the whole UI flow (progress, preview, download) can be tested.
- Can be configured to fail on purpose so error handling is tested too.

## `comfyui`

Works with any reachable ComfyUI instance that has the files in
[`workflows/comfyui/models.yaml`](../workflows/comfyui/models.yaml) installed and sits
behind the token proxy.

| Host | Use | Notes |
|---|---|---|
| Kaggle (T4) | Temporary testing only | Free, about 30 GPU-h/week, sessions max 12 h, URL changes every session. See [KAGGLE_SETUP.md](KAGGLE_SETUP.md). |
| RunPod / Vast.ai / Lambda | Low-cost persistent GPU | Same ComfyUI + proxy setup; stable URL. |
| Own GPU server | Long-term | Same setup. |

## Adding a new provider

1. Create `backend/app/providers/<name>.py` implementing `InferenceProvider`.
2. Translate `GenerationRequest` to the provider's API; map its progress to 0–100.
3. Register it in the provider factory and document its env vars in `.env.example`.
4. Add contract tests (the same test suite runs against every provider using fakes).

Nothing in `frontend/`, `backend/app/api/`, `backend/app/jobs/` or
`backend/app/storage/` changes.
