# Notebooks

| Notebook | Phase | What it does |
|---|---|---|
| `kaggle_phase5a_preflight.ipynb` | 5A | **Read-only** Kaggle GPU preflight: GPU model and VRAM per GPU, CUDA/PyTorch compatibility, RAM, disk and storage budget, internet reachability, Hugging Face metadata (sizes + SHA-256, no downloads), and a dry run of `scripts/download_models.py`. Writes `/kaggle/working/vidsgen_preflight_report.json` (no secrets). Unchanged. |
| `kaggle_comfyui_wan22.ipynb` | 5B | **Downloads ~16.9 GiB and runs inference.** Gated stages: preflight, pinned ComfyUI install (reusing Kaggle's PyTorch), verified model download (exact size + SHA-256), loopback-only ComfyUI on GPU 0, live template validation, and a 17-frame text-to-video smoke test through the real `ComfyUIProvider` that passes only for a decodable, non-blank MP4. No tunnel. |
| `kaggle_backend_e2e_phase6b.ipynb` | 6B | **Downloads ~16.9 GiB and runs inference.** Self-contained: the verified 5B stages (pinned ComfyUI, verified download, loopback server), then installs the backend's dependencies into the same venv (PyTorch/NumPy protected) and runs the real FastAPI backend (`GENERATION_PROFILE=t4_verified`, one job at a time) against ComfyUI over `127.0.0.1`. One real 832x480 / 17-frame job is submitted, tracked, downloaded and fully verified. Localhost only: no tunnel, no proxy. **Passed on a Kaggle T4** (all 8 stages; end-to-end verdict PASS). Real GPU cancellation (opt-in `RUN_CANCEL_TEST`), image-to-video and the frontend are not covered. |

`tools/build_kaggle_notebook.py` generates the 5B notebook and embeds `scripts/download_models.py`,
`scripts/wan22_gpu_test.py` and `workflows/comfyui/models.yaml` verbatim (they are not on GitHub yet). Regenerate with
`python notebooks/tools/build_kaggle_notebook.py`; a test fails if the notebook is out of date.
`tools/build_kaggle_e2e_notebook.py` generates the 6B notebook the same way (it also embeds `scripts/backend_e2e_test.py`; `python notebooks/tools/build_kaggle_e2e_notebook.py --check`).

Plans, the T4 verification matrix and the troubleshooting table are in [../docs/KAGGLE_SETUP.md](../docs/KAGGLE_SETUP.md).

Rules: no tokens in cells (use Kaggle Secrets), clear outputs before committing,
no weights or videos saved into the repo.
