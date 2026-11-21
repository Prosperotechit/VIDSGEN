# ComfyUI workflows (Wan 2.2 TI2V-5B)

| File | Purpose |
|---|---|
| `wan22_ti2v_5b_t2v.json` | Text-to-video graph, **API format** (Load Image bypassed: no image needed) |
| `wan22_ti2v_5b_i2v.json` | Image-to-video graph, **API format** (Load Image feeds `start_image`) |
| `wan22_ti2v_5b.mapping.json` | Which node/input each VIDSGEN parameter goes to; required models; progress ranges; default negative prompt |
| `export_provenance.json` | Exactly what the templates were exported from (official file hash, ComfyUI and frontend versions) |
| `models.yaml` | The three model files, their Hugging Face source, size and ComfyUI folder |
| `tools/export_api_workflows.mjs` | Regenerates the two templates and the test fixture from the official workflow |

## Where the templates come from

1. **Source:** the official Comfy-Org template `templates/video_wan2_2_5B_ti2v.json` from
   [`Comfy-Org/workflow_templates`](https://github.com/Comfy-Org/workflow_templates), as linked from the official
   [Wan 2.2 tutorial](https://docs.comfy.org/tutorials/video/wan/wan2_2) ("Wan2.2 TI2V 5B Hybrid Version Workflow
   Example"). This is the dedicated **5B** workflow, not any of the 14B or "fun" ones.
2. **It is a UI-format graph** (`nodes` + `links`), saved with frontend 1.27.10. ComfyUI's `/prompt` endpoint needs
   **API format**, so it has to be converted.
3. **Conversion is done by ComfyUI's own frontend.** The node schemas changed after the template was saved:
   `SaveVideo.format` became a nested "DynamicCombo" (`format`, `format.codec`, `format.codec.encoding`), and
   `CreateVideo` gained new optional inputs. Hand-conversion would be guesswork. `tools/export_api_workflows.mjs`
   loads the official file into a running ComfyUI's web UI in headless Chromium, sets the mode and output format
   through the real widgets, and calls the frontend's own `graphToPrompt()` (the code behind "Export (API)").
4. **Two changes from the official workflow, both through the UI's own controls:**
   - Node 56 `LoadImage` is bypassed for T2V and enabled for I2V, as the official docs describe ("Ctrl+B to enable
     the Load image node").
   - Node 58 `SaveVideo` uses `format=mp4`, `codec=h264` instead of `auto`, so the result always plays in browsers.

Current export: official file sha256 `8f761d02…`, **ComfyUI 0.46.0**, **frontend 1.59.6** (see `export_provenance.json`).

### Regenerating (e.g. after a ComfyUI upgrade)

No model weights are needed. Empty placeholder files with the official names are enough:

```bash
# 1) a ComfyUI checkout with its requirements installed, plus placeholder model files:
mkdir -p base/models/{diffusion_models,text_encoders,vae} base/custom_nodes base/input base/output
touch base/models/diffusion_models/wan2.2_ti2v_5B_fp16.safetensors \
      base/models/text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors \
      base/models/vae/wan2.2_vae.safetensors
python ComfyUI/main.py --cpu --listen 127.0.0.1 --port 8188 --base-directory base
```

```bash
# 2) from the VIDSGEN repo root (needs `npm install` in frontend/ for Playwright):
node workflows/comfyui/tools/export_api_workflows.mjs --official path/to/video_wan2_2_5B_ti2v.json
```

```bash
# 3) re-run the tests; the backend also refuses to start if the mapped nodes no longer match:
cd backend && .venv/bin/python -m pytest
```

## Node map (both templates; node 56 only in I2V)

| Node | Class | VIDSGEN sets | Fixed (official) |
|---|---|---|---|
| 6 | `CLIPTextEncode` | `text` ← prompt | |
| 7 | `CLIPTextEncode` | `text` ← negative prompt (or the default below) | |
| 55 | `Wan22ImageToVideoLatent` | `width`, `height`, `length` ← frames | `batch_size=1`; I2V: `start_image` ← node 56 |
| 3 | `KSampler` | `seed`, `steps`, `cfg` ← guidance scale | `uni_pc`, `simple`, `denoise=1` |
| 57 | `CreateVideo` | `fps` | other inputs at defaults |
| 58 | `SaveVideo` | `filename_prefix` ← `vidsgen/<job_id>` | `format=mp4`, `format.codec=h264` |
| 56 | `LoadImage` (I2V) | `image` ← `vidsgen/<job_id>.png` (uploaded) | |
| 37 | `UNETLoader` | | `wan2.2_ti2v_5B_fp16.safetensors` |
| 38 | `CLIPLoader` | | `umt5_xxl_fp8_e4m3fn_scaled.safetensors`, `type=wan` |
| 39 | `VAELoader` | | `wan2.2_vae.safetensors` (the Wan **2.2** VAE) |
| 48 | `ModelSamplingSD3` | | `shift=8` |
| 8 | `VAEDecode` | | |

**Node ids are not assumed to be stable across exports.** The mapping file names each node with its expected
`class_type`. When the backend starts with `INFERENCE_PROVIDER=comfyui`, it checks every mapped id, class, input,
model filename and the T2V/I2V wiring, and refuses to start with a clear list of problems if anything moved.

## Text-to-video vs image-to-video

- **T2V** uses `wan22_ti2v_5b_t2v.json`. It has no `LoadImage` node and nothing connected to `start_image`, so no
  image is needed.
- **I2V**: the backend uploads the job's validated start image to ComfyUI (`POST /upload/image`, input folder,
  subfolder `vidsgen`, name `<job_id>.png`) and puts the returned reference into node 56. ComfyUI's
  `Wan22ImageToVideoLatent` scales and centre-crops it to the requested size and encodes it as the first frame.

## Default negative prompt

When the user leaves the negative prompt empty, VIDSGEN uses the official Wan negative prompt from the template
(node 7). **A negative prompt typed by the user replaces it entirely**; the two are never combined.

> 色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走

English meaning: *garish colours, overexposed, static, blurry details, subtitles, style, artwork, painting, picture,
still, greyish overall, worst quality, low quality, JPEG compression artefacts, ugly, mutilated, extra fingers,
poorly drawn hands, poorly drawn face, deformed, disfigured, malformed limbs, fused fingers, motionless frame,
cluttered background, three legs, many people in the background, walking backwards.*

## Required models

| File | ComfyUI folder | Size | Source (pinned) |
|---|---|---|---|
| `wan2.2_ti2v_5B_fp16.safetensors` | `models/diffusion_models/` | 9,999,658,848 B | `Comfy-Org/Wan-2.2` @ `5df3a4a3ed06` |
| `wan2.2_vae.safetensors` | `models/vae/` | 1,409,400,960 B | `Comfy-Org/Wan-2.2` @ `5df3a4a3ed06` |
| `umt5_xxl_fp8_e4m3fn_scaled.safetensors` | `models/text_encoders/` | 6,735,906,897 B | `Comfy-Org/Wan-2.1` @ `d6258323e18f` |

Total 18,144,966,705 bytes (16.9 GiB). `models.yaml` also records each file's SHA-256 (the Git-LFS object id, confirmed by two
independent Hugging Face metadata sources; the Xet hash is a different identifier and is not used) and its pinned
revision. `scripts/download_models.py --comfyui-dir <ComfyUI data dir>` is a dry run unless `--execute` is given; with it,
every file is verified by exact size **and** SHA-256, and a mismatch deletes the file and stops. `--verify-only` re-checks
files already on disk. `/api/health` reports "degraded" if ComfyUI lacks any file or a file's size differs (a 0-byte
placeholder, an interrupted download); file contents cannot be hashed through ComfyUI's API, so weight integrity and
inference readiness stay **unverified** until a real generation succeeds (see `docs/PROVIDERS.md`).

## What has been validated, and how

| Level | What | Status |
|---|---|---|
| Schema | Both templates, and graphs built for **every** VIDSGEN-allowed value, against `/object_info` of ComfyUI 0.46.0 (`backend/tests/test_workflows.py`) | ✅ offline tests |
| Real server validation | Both templates accepted by a real ComfyUI 0.46.0 `POST /prompt` (CPU, placeholder weights) | ✅ done in Phase 4 |
| Real output path | `CreateVideo`→`SaveVideo` with these exact settings on real ComfyUI → MP4, H.264 High, yuv420p, 24 fps | ✅ done in Phase 4 |
| **Actual Wan generation (text-to-video)** | Model loading, sampling and decoding, 832×480 × 17 frames on a Kaggle T4: valid H.264 MP4, 105.4 s, peak 14,885 MiB VRAM | ✅ Phase 5B smoke test passed (see `docs/KAGGLE_SETUP.md`) |
| Image-to-video on a GPU; quality; larger sizes; backend-to-ComfyUI end to end | | ❌ Not tested yet |
