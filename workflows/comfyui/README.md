# ComfyUI workflows

| File | Status | Purpose |
|---|---|---|
| `models.yaml` | ✅ | Exact Wan 2.2 TI2V-5B files (repo, path, size, ComfyUI folder) |
| `wan22_ti2v_5b_t2v.api.json` | Phase 4 | Text-to-video graph, **API format** |
| `wan22_ti2v_5b_i2v.api.json` | Phase 4 | Image-to-video graph, **API format** |
| `params.yaml` | Phase 4 | Maps API request fields → `node_id.inputs.<name>` |

Graph (shared by both modes; I2V connects `start_image`):

```
UNETLoader(wan2.2_ti2v_5B_fp16) → ModelSamplingSD3(shift) ─┐
CLIPLoader(umt5_xxl_fp8, type=wan) → CLIPTextEncode(+ / −) ─┤
VAELoader(wan2.2_vae) → Wan22ImageToVideoLatent             ├→ KSampler → VAEDecode(Tiled) → CreateVideo(24 fps) → SaveVideo
      (width, height, length, start_image?) ────────────────┘
```

Templates are exported from a running ComfyUI with "Export (API)", and node names are
checked against that instance, not copied from memory. T2V and I2V are kept as
**separate files** so each can be validated on its own (see `docs/KAGGLE_SETUP.md`).
