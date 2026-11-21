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
   the URL for `COMFYUI_BASE_URL`.
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

**T2V** (`wan22_ti2v_5b_t2v.json`, no `start_image`)
- Same prompt + seed twice → identical output (determinism).
- Output: expected width/height, frame count `4k+1`, 24 fps, playable H.264 MP4.
- Prompt adherence spot-check on 3 fixed prompts.

**I2V** (`wan22_ti2v_5b_i2v.json`, `start_image` connected)
- First output frame visually matches the input image.
- Landscape, portrait and square inputs: aspect handling and resizing are correct.
- Missing or invalid image → clear error, not a silent fall-back to T2V.

Both modes are first run inside the notebook through ComfyUI's API on `127.0.0.1`,
then again through the tunnel from the backend.

## Limits to respect

- About 30 GPU-hours/week; sessions max 12 h; idle sessions get stopped.
- Tunnel URL changes every session, so update `COMFYUI_BASE_URL` each time.
- Kaggle discourages using notebooks as long-running public servers: open the tunnel
  only during active tests.

## Ready from Phase 4

- **Models:** `python scripts/download_models.py --comfyui-dir <ComfyUI>` prints the plan; add `--execute` in the
  notebook to download (~18 GB). Set `HF_TOKEN` as a Kaggle Secret if needed.
- **Backend:** `INFERENCE_PROVIDER=comfyui`, `COMFYUI_BASE_URL=<tunnel URL>`, `COMFYUI_API_TOKEN=<secret>`.
- **First check against the GPU server** (read-only: health, models, templates vs that server's `/object_info`):

  ```bash
  cd backend && VIDSGEN_COMFYUI_LIVE_URL=<tunnel URL> VIDSGEN_COMFYUI_LIVE_TOKEN=<secret> .venv/bin/python -m pytest tests/test_comfyui_live.py
  ```

  Adding `VIDSGEN_COMFYUI_LIVE_SUBMIT=1` also runs two cheap, model-free graphs (output path + interrupt).

## Still to prove in Phase 5 (not tested in Phase 4)

- After the downloads, `/api/health` must show `model_file_sizes: ok`. `model_weights_integrity` and `inference`
  stay `unverified` until the first real Wan generation completes; only that turns them `ok`.

- ~~Wan 2.2 TI2V-5B loads and samples on a T4~~: **done in Phase 5B** for a 17-frame 832x480 text-to-video job (see below).
- Real I2V on a GPU, output quality, timings at other sizes, and the T4 verification matrix above.
- Whether full-size decode needs `VAEDecodeTiled` (the template currently uses the official `VAEDecode`).
- Real progress cadence from the sampler, and behaviour through the Cloudflare tunnel and token proxy (WebSocket
  included).

## Phase 5A: preflight and plan (read-only)

Run `notebooks/kaggle_phase5a_preflight.ipynb` on Kaggle (GPU T4 x2, Internet on). It changes nothing: no
installs, no weights, no ComfyUI, no tunnel. Its report decides the 5B setup below.

### One GPU, not two

Kaggle's "T4 x2" is two separate 16 GB GPUs. VRAM is **not pooled** and ComfyUI runs a workflow on one device.
Phase 5B pins ComfyUI to GPU 0 (`--cuda-device 0` and `CUDA_VISIBLE_DEVICES=0`). GPU 1 stays unused.

### Storage budget

| Item | Size |
|---|---|
| Model files (`models.yaml`, exact) | 18,144,966,705 bytes = 16.9 GiB |
| ComfyUI checkout (pinned commit) | ~0.07 GiB measured (budget 0.2) |
| Python packages ComfyUI needs that Kaggle lacks (`--no-cache-dir`, in a venv) | 1.5 GiB measured on a bare host (budget 3.0; Kaggle already has many) |
| cloudflared binary (later phase) | budget 0.1 GiB |
| Test outputs + I2V uploads | budget 1.0 GiB |
| ComfyUI temp/cache | budget 1.0 GiB |
| Safety margin (10 %, min 2 GiB) | ~2.2 GiB |
| **Required** | **~24.4 GiB** |

`/kaggle/working` (about 20 GB, saved as notebook output) is **not** large enough and must not hold weights.
The preflight measures each writable non-persistent location (e.g. `/kaggle/tmp`, `/tmp`), detects which
share a filesystem, and recommends `<location>/vidsgen-gpu`. The downloader moves each file from a staging
folder on the same filesystem, so a file never needs space twice.

### Model downloads and SHA-256 verification

- **Authoritative hashes (done in Phase 5B):** for Git-LFS files on Hugging Face, the SHA-256 of the content is the
  LFS object id. It was read from two independent metadata sources that agreed exactly: the `paths-info` API
  (`lfs.oid`) and the `X-Linked-Etag` header of a `HEAD` on the pinned resolve URL (redirects off, so no content
  is transferred). `models.yaml` now records, per file, the pinned `revision` (an immutable commit), `size_bytes`
  and `sha256`. The Xet hash and the Git blob id that Hugging Face also returns are different identifiers and are
  never used. The preflight re-reads and re-checks all of this live before every run.
- **Repository names:** `Comfy-Org/Wan-2.2` (diffusion model + VAE, revision `5df3a4a3…`) and `Comfy-Org/Wan-2.1`
  (text encoder, revision `d6258323…`). The older `…_ComfyUI_Repackaged` names redirect to the same files.
- **Downloader (`scripts/download_models.py --execute`):** stdlib only; downloads only the three manifest files from
  their pinned revisions; refuses to start if any revision/SHA-256 is missing or any 14B/LoRA/alternative-VAE path
  is listed; checks free disk first; resumes interrupted downloads from a `.part` file with HTTP Range (a partial
  is reused only if it provably belongs to the same revision/hash); verifies exact size **and** the SHA-256 of the
  whole file; on any mismatch deletes the file and stops; never sends the Hugging Face token to CDN redirect hosts.
  `--verify-only` re-hashes files already on disk with no network.

### ComfyUI version

Pinned to the **exact commit Phase 4 validated: `995456d0188ffb51032c28f41527cfa82d87d8a9`** (ComfyUI 0.46.0;
three commits after the `v0.46.0` tag, which only added a cloud "partner" node and CI configuration). The
installer fetches only that commit, checks `HEAD` and `comfyui_version.py`, and refuses a modified checkout.
Dependencies go into a venv created with `--system-site-packages --without-pip` (Kaggle's Python 3.13 has no
`ensurepip`; the venv uses the host's own pip, which installs into the venv), installed under a **constraints file that pins
`torch`, `torchvision`, `torchaudio` and `numpy` to what the host already has**, with those packages removed from
ComfyUI's requirements. Afterwards the venv must import the host's *identical PyTorch file* (a second copy, even at
the same version, is rejected). The host's installed packages are inventoried before and after and must not change;
an existing venv (with or without its own pip) is reused. ComfyUI's own pins (frontend 1.59.6, templates 0.11.111, …) are installed as
specified. After start, `check-templates` validates both workflow templates against that server's `/object_info`.

### Secure connection (5B)

```
backend ──HTTPS + WSS──▶ Cloudflare quick tunnel ──▶ auth proxy 127.0.0.1:8189 ──▶ ComfyUI 127.0.0.1:8188
          Authorization: Bearer <token>              checks token, strips it,       never exposed
                                                     forwards HTTP + WebSocket
```

- ComfyUI listens on `127.0.0.1` only. Only the proxy is published through the tunnel.
- The proxy compares `Authorization: Bearer` in constant time and returns 401 otherwise. It forwards HTTP
  (streamed bodies) and the `/ws` WebSocket, and **strips the Authorization header** before ComfyUI sees it.
  It allows only the routes the provider uses (`/system_stats`, `/object_info/*`, `/models/*`,
  `/experiment/models/*`, `/upload/image`, `/prompt`, `/queue`, `/interrupt`, `/history*`, `/view`, `/ws`) and
  never logs headers or tokens.
- The token comes from a Kaggle Secret `COMFYUI_API_TOKEN` (the same value goes into the backend's `.env`). It
  is never printed. The tunnel URL is printed once, for `COMFYUI_BASE_URL`.
- The tunnel is opened only while testing and closed when the session ends.


## Phase 5B: first real inference (text-to-video smoke test PASSED on a Kaggle T4)

### Result of the first real run

| Item | Result |
|---|---|
| GPU | NVIDIA Tesla T4, GPU 0 |
| ComfyUI | v0.46.0 |
| Model | Wan 2.2 TI2V-5B (all three files downloaded and verified by exact size and SHA-256) |
| Smoke-test verdict | **PASS**: workflow queued and completed; MP4 decoding, frame count, dimensions and non-blank checks all passed |
| Output | valid H.264 MP4, 832 x 480, 17 frames, 24 fps, 0.708 s |
| Generation time | 105.4 s (total wall time 106.5 s) |
| Peak GPU memory | 14,885 MiB of 15,360 MiB (about 97%) |
| Peak system RAM | 20.58 GiB |

These figures were reported from the Kaggle run; the report JSON itself is not stored in the repository.

**What this proves:** the pinned ComfyUI and Wan 2.2 TI2V-5B can load, sample and decode a real text-to-video job on one
T4, and produce a decodable, non-blank H.264 MP4 through the Phase 4 workflow.

**What it does not prove** (still open):
- **Image-to-video** has not been run on a GPU.
- **Frontend to backend to real inference** has not been exercised: the smoke test ran on the Kaggle host, with no
  tunnel and no token proxy (Phase 5C).
- **Production readiness**: this is one short, small clip. Peak VRAM left only about 470 MiB of headroom, so larger
  resolutions or longer clips may run out of memory on a T4. Output quality, longer or 1280x704 generations, repeated
  runs, concurrency and real progress cadence are untested. Kaggle remains a temporary test host.

### The notebook

`notebooks/kaggle_comfyui_wan22.ipynb` (generated by `notebooks/tools/build_kaggle_notebook.py`). Settings: GPU T4 x2,
Internet on, optional `HF_TOKEN` secret. **Run All.** Every stage is gated: a failure stops the notebook.

| Stage | Command (all in `scripts/wan22_gpu_test.py` unless noted) | Gate |
|---|---|---|
| 1 Preflight | `preflight` | Python >= 3.11; install location not under `/kaggle/working`; free disk >= remaining models + 8 GiB; GPU 0 present, VRAM >= 14 GiB, PyTorch build supports its architecture, fp16 kernel runs; RAM >= 24 GiB (warn >= 19); GitHub/PyPI reachable; Hugging Face revision + size + SHA-256 match the manifest |
| 2 Install | `install-comfyui` | exact commit, version 0.46.0, PyTorch untouched, `pip check` clean |
| 3 Download | `download_models.py --execute`, then `--verify-only` | exact size + SHA-256, twice |
| 4 Server | `serve` | exits cleanly, listening on **loopback only**, one GPU, credential-free environment |
| 5 Verify | model discovery via `/experiment/models`; `check-templates` | three files discovered with exact sizes; both templates valid on this server |
| 6 Smoke | `smoke-t2v` | a real MP4 (below) |

Everything large (ComfyUI, venv, weights, pip/HF/torch caches, `TMPDIR`) lives under `/kaggle/tmp/vidsgen-gpu`;
only the JSON reports and the test MP4 go to `/kaggle/working`.

**What counts as success.** The smoke test (text-to-video, 832x480, 17 frames, 20 steps, seed 424242) runs through
the real `ComfyUIProvider` and the Phase 4 workflow. It passes only if ComfyUI completed the job **and** the MP4 is
retrieved, opens, is H.264 in an MP4 container, decodes to exactly 17 frames at 832x480 / 24 fps, and is neither
all-black nor a flat colour (a black result would indicate fp16 overflow on the T4). "The server started" and "the
workflow queued" are never reported as success, and any failure is shown with its reason.

**Measurements recorded** (in `vidsgen_phase5b_smoke_report.json`): peak GPU memory (nvidia-smi, GPU 0, with the idle
baseline), peak system RAM, the ComfyUI process's peak RSS, time to first sampler step, seconds per step, total time,
output size and decode details, the provider's honest health before and after (`inference` turns `ok` only after this
real generation), and, on failure, a redacted log tail.

**Memory behaviour.** ComfyUI at this commit uses "dynamic VRAM" by default when supported (PyTorch >= 2.8); with it
on, `--lowvram` does nothing. So the first attempt uses ComfyUI's defaults. Only if it fails with an out-of-memory
error is ComfyUI restarted **once** with `--disable-dynamic-vram --disable-smart-memory` and the test repeated
(both reports are kept). It is never retried again.

**Security.** ComfyUI listens on `127.0.0.1` only (checked from `/proc/net/tcp`, and it is stopped if it is not). It is
started with an **allow-listed** environment, so no token or other credential reaches it. `HF_TOKEN` (optional) is
read from a Kaggle Secret, sent only to Hugging Face, and never printed or saved; reports and logs are redacted
and checked for token patterns. No tunnel is started.

**Still untested after the first GPU run:** image-to-video on a GPU, output quality beyond the automated checks, other
sizes and frame counts, the tunnel and token proxy, and the backend driving a real ComfyUI end to end.

## Phase 6A/6B: the backend with limits, then backend + ComfyUI on one Kaggle GPU (localhost only)

**6A (done, on GitHub):** the backend can enforce a GPU *generation envelope* (`GENERATION_PROFILE=t4_verified`: text-to-video,
832x480, at most 17 frames, at most 20 steps, one job at a time) and reports ComfyUI out-of-memory failures and lost connections
clearly (`gpu_out_of_memory`, `system_out_of_memory`, `comfyui_connection_lost`; a running job survives a connection loss of up to
`COMFYUI_CONNECTION_GRACE_SECONDS`, default 30). The limits are enforced at the API boundary (`POST /api/jobs`), not inside
`JobManager`. Uncertain ComfyUI submissions are never retried automatically. Image-to-video stays disabled in `t4_verified` until it
has been tested on a GPU. The API defaults (49 frames, 30 steps) are unchanged, so the test sends every value explicitly.

**6B (PASSED on a Kaggle T4):** `notebooks/kaggle_backend_e2e_phase6b.ipynb` runs the real FastAPI backend and ComfyUI together on the
Kaggle machine, both on `127.0.0.1` only. No tunnel, no proxy, nothing exposed. The Phase 5B notebook's stages are unchanged.

### Result of the real run (Kaggle, GPU 0)

| Item | Result |
|---|---|
| Notebook | all 8 stages passed; backend-to-ComfyUI end-to-end verdict **PASS** |
| GPU / software | NVIDIA Tesla T4; ComfyUI v0.46.0; Wan 2.2 TI2V-5B; `GENERATION_PROFILE=t4_verified`, one job at a time |
| Request (sent explicitly) | text-to-video, 832 x 480, 17 frames, 20 steps, 24 fps, CFG 5.0, through the backend's `POST /api/jobs` |
| Output | valid H.264 MP4, 227,056 bytes; decoding, frame count, dimensions and non-blank checks passed |
| Durations | job 97.31 s; total end-to-end 99.8 s |
| Peak GPU memory | 14,885 MiB of 15,360 MiB (about 97%) |
| Peak system RAM | 20.56 GiB |
| Also verified | live progress tracking and health (`inference: ok` after the generation); download, HTTP Range request and inline preview; four out-of-envelope requests (resolution, frames, steps, image-to-video) each returned HTTP 422; backend and ComfyUI both bound to localhost only |

These figures were reported from the Kaggle run; the report JSON files are not stored in the repository.

**What this shows:** the existing FastAPI backend, `ComfyUIProvider` and job manager run one real Wan 2.2 text-to-video job on a T4 over
localhost, with the generation limits enforced and the video retrieved and verified.

**What it does not show (still open):** real GPU **cancellation** was not tested (`RUN_CANCEL_TEST` stays off by default); **image-to-video**
has not been run on a GPU; no browser **frontend** has been used with a real GPU backend; the token proxy and a tunnel are not built; only this
one small size was tested, one job at a time. **No production readiness is claimed.** Peak VRAM left about 470 MiB of headroom.

**An installation problem found during this run (fixed).** On Kaggle the host has NumPy 2.1.3 plus a stale second NumPy metadata record
(1.26.4). The installer used the first of the sorted versions, so it pinned `numpy==1.26.4` and installed a second NumPy into the venv; the
protected-package verification (correctly) failed, and a hand-edited `constraints.txt` was overwritten on every run. The installer now pins the
version of the package the host actually **imports** (matched to its metadata record, never guessing), reports any host package with several
records as `host_metadata_conflicts`, and still verifies that the venv imports the host's identical torch, torchvision, torchaudio and numpy. The fix
is in `scripts/wan22_gpu_test.py` and both regenerated notebooks, with regression tests. Kaggle's system packages are never modified.

### How to run it on Kaggle
1. Download `notebooks/kaggle_backend_e2e_phase6b.ipynb` and import it (*File -> Import Notebook*). The repository is public: no GitHub
   credential is needed. (If it ever becomes private, add the token as a Kaggle Secret; never put it in a cell.)
2. *Settings -> Accelerator:* **GPU T4 x2**; *Internet:* **On**. Optional secret `HF_TOKEN` (not needed for these public models).
3. **Run All.** A new session starts empty, so the notebook installs the pinned ComfyUI and downloads the three model files (16.9 GiB, size
   and SHA-256 verified) again. Expect roughly 15-30 minutes in total; the first generation also loads the models.
4. Read the verdict printed by stage 7. It lists each check as PASS/FAIL. `BACKEND END-TO-END VERDICT: PASS` appears only if **all** pass.
5. Send back `/kaggle/working/vidsgen_phase6b_report.json` and `/kaggle/working/vidsgen_phase6b_e2e_report.json` (no secrets), and the
   `STOP`/`[FAIL]` lines if a stage stopped the run.
6. Optional: set `RUN_CANCEL_TEST = True` in the second code cell to also cancel a second job during processing (extra GPU time).

### What stage 7 checks (every one must pass)
Backend and ComfyUI listen on loopback only; `provider` is `comfyui` and ready; four out-of-envelope requests (1280x704, 49 frames, 30 steps,
image-to-video) are rejected with 422 and create no job; one real job (832x480, 17 frames, 20 steps, 24 fps, CFG 5.0) is accepted, its
status moves queued -> processing -> completed with live, non-decreasing progress including intermediate values; the job record matches the
request; the video downloads (attachment), supports an HTTP Range request and an inline preview; the MP4 fully decodes as H.264 with 17
frames at 832x480 and 24 fps and is not blank; `/api/health` reports `inference: ok` afterwards; the backend stops cleanly.

### What the PASS shows and does not show
It shows: the existing backend, `ComfyUIProvider` and job manager can run one real Wan 2.2 text-to-video job on a T4 over localhost.
It does **not** show: real GPU cancellation, image-to-video on a GPU, a browser/frontend in front of the backend, the token proxy or a tunnel,
other sizes or frame counts, repeated or concurrent jobs, or any production readiness.

**Troubleshooting**

| Symptom | Likely cause / action |
|---|---|
| `[FAIL] disk_space` | `/kaggle/tmp` is smaller than ~25 GiB: report the numbers; do not move weights to `/kaggle/working` |
| `[FAIL] torch_supports_gpu` | Kaggle's PyTorch lacks `sm_75`: needs a matching PyTorch (a decision, not automatic) |
| `[FAIL] python` | Kaggle Python < 3.11: the backend code needs 3.11+ |
| `could not create the virtual environment` / `ensurepip` | the venv must be created with `--without-pip` (the installer does this); an older venv that lacks `--system-site-packages` is refused, not modified: delete `/kaggle/tmp/vidsgen-gpu/venv` and re-run |
| `STOP at stage 'backend_install'` | the backend's dependencies could not be installed in the ComfyUI venv, or they would have replaced Kaggle's PyTorch/NumPy (the message says which); nothing was changed on the host |
| `backend_e2e` verdict FAIL | each failed check is named with its reason; `gpu_out_of_memory` means the GPU ran out of memory even at the verified size (report it, do not raise the limits) |
| `numpy changed: 2.1.3 -> 1.26.4` / `a second numpy was installed inside the venv` | the host has a stale second NumPy metadata record; fixed: the installer now pins the version the host actually **imports** (and lists any host package with several records under `host_metadata_conflicts`). If it still fails, delete `/kaggle/tmp/vidsgen-gpu/venv` and `constraints.txt` and re-run; never edit Kaggle's system packages |
| `pip install failed` | PyTorch was protected by the constraints; the message names the conflicting requirement |
| download `exit code 3` | network drop: re-run the cell; it resumes |
| download `exit code 4` | a file failed size/SHA-256 and was **deleted**: do not retry blindly; report it |
| smoke `every frame is black` | fp16 numerical failure on the T4: report it; do not use the output |
| smoke `gpu_out_of_memory` twice | try 640x640 or fewer frames; report peak VRAM from the report |
