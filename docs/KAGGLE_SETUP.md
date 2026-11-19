# Kaggle setup and T4 validation plan

Kaggle is the **temporary** GPU environment for experiments. Production must not
depend on it. The notebook itself is built in Phase 5; this document defines what it
must do and how T4 compatibility is verified (not assumed).

## Notebook settings

- **Accelerator: GPU T4** (single T4 is enough; ComfyUI uses one GPU).
  Avoid P100: current PyTorch CUDA builds have dropped its architecture (sm_60).
- **Internet: On** (requires a phone-verified Kaggle account).
- **Secrets** (Add-ons → Secrets): `HF_TOKEN` (optional for these public repos, but
  avoids rate limits) and `COMFYUI_API_TOKEN` (random string, same value as in the
  backend's `.env`). Never paste tokens into notebook cells.

## What the notebook will do

1. Record the environment: `nvidia-smi`, GPU name, compute capability, VRAM, system
   RAM, free disk, Python/PyTorch/CUDA versions.
2. Install ComfyUI (pinned commit) using Kaggle's preinstalled PyTorch.
3. Download the three files in `workflows/comfyui/models.yaml` (about 18 GB) with
   `huggingface_hub`, check byte sizes, and place them in ComfyUI's model folders.
   Store weights outside `/kaggle/working` (that folder is saved with the notebook
   and size-capped).
4. Start ComfyUI on `127.0.0.1:8188`.
5. Run the **T4 verification** and **workflow validation** below inside the notebook,
   before any remote access is opened.
6. Start the token proxy in front of ComfyUI and a Cloudflare quick tunnel; print
   the URL for `COMFYUI_URL`.
7. Stop the tunnel when testing ends. Nothing stays exposed after the session.

## T4 compatibility: facts and risks

| Fact | Consequence |
|---|---|
| T4 = Turing, compute capability 7.5, 16 GB (≈15 GB usable) | Fits the 10 GB fp16 diffusion model, with little room to spare |
| No native bf16 on Turing | Run in fp16. Must check for fp16 overflow (NaNs / black frames). |
| No native fp8 on Turing | The fp8 text encoder is upcast at load; it saves RAM/disk, not compute |
| Text encoder (6.7 GB) + diffusion model (10 GB) > 15 GB | ComfyUI must offload the text encoder after encoding; verify it does |
| Wan 2.2 VAE decode at 1280×704×121 is memory-heavy | Likely needs tiled decode (`VAEDecodeTiled`) on T4 |
| Official guidance: 720p TI2V-5B needs about 24 GB with offloading | Native 720p on T4 is **unproven**; validate step by step |
| Kaggle system RAM is about 29 GB | Must hold offloaded weights; check peak RAM too |

## T4 verification matrix

Run each row with a fixed seed. Record peak VRAM (`torch.cuda.max_memory_allocated`
and `nvidia-smi`), peak system RAM, total time, sec/step, and whether the output
is valid (no NaN, no black or frozen frames). Stop going up at the first failure.

| # | Resolution | Frames | Steps | Decode | Purpose |
|---|---|---|---|---|---|
| 1 | 480×272 | 17 | 10 | normal | Smoke test: model loads, fp16 is numerically OK |
| 2 | 832×480 | 49 | 20 | normal | Practical test size |
| 3 | 832×480 | 49 | 20 | tiled | Compare decode memory |
| 4 | 1280×704 | 49 | 20 | tiled | Native resolution, short clip |
| 5 | 1280×704 | 121 | 30 | tiled | Full native 5 s clip (may exceed T4/time limits) |

The results go into `docs/T4_RESULTS.md` and decide the **default and maximum
resolution/duration** the backend allows when the provider is a T4.

## Workflow validation (T2V and I2V tested separately)

Each mode has its own workflow template and is validated on its own:

**T2V** (`wan22_ti2v_5b_t2v.api.json`, no `start_image`)
- Same prompt + seed twice → identical output (determinism).
- Output: expected width/height, frame count `4k+1`, 24 fps, playable H.264 MP4.
- Prompt adherence spot-check on 3 fixed prompts.

**I2V** (`wan22_ti2v_5b_i2v.api.json`, `start_image` connected)
- First output frame visually matches the input image.
- Landscape, portrait and square inputs: aspect handling and resizing are correct.
- Missing or invalid image → clear error, not a silent fall-back to T2V.

Both modes are first run inside the notebook through ComfyUI's API on `127.0.0.1`,
then again through the tunnel from the backend.

## Limits to respect

- About 30 GPU-hours/week; sessions max 12 h; idle sessions get stopped.
- Tunnel URL changes every session, so update `COMFYUI_URL` each time.
- Kaggle discourages using notebooks as long-running public servers: open the tunnel
  only during active tests.
