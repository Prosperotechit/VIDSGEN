"""Opt-in checks against a REAL ComfyUI server. Skipped unless configured:

    VIDSGEN_COMFYUI_LIVE_URL=http://127.0.0.1:8188   (+ VIDSGEN_COMFYUI_LIVE_TOKEN if behind the token proxy)

Always safe (read-only): health, required model files, and validation of both workflow
templates against THAT server's own /object_info, which catches ComfyUI version drift.

Model weights: set VIDSGEN_COMFYUI_LIVE_PLACEHOLDERS=1 for a test server that only has
0-byte placeholder files (it must then be reported as NOT ready); otherwise the real
weights are expected and their exact sizes must match the manifest.

With VIDSGEN_COMFYUI_LIVE_SUBMIT=1 it also submits work, but only cheap graphs that need
no model weights (EmptyImage -> the template's CreateVideo/SaveVideo nodes, and a slow blur
for interruption). This proves the real event/history/download/cancel path end to end. It
never runs Wan inference: that is Phase 5 GPU testing.
"""

from __future__ import annotations

import asyncio
import copy
import os
import subprocess

import httpx
import pytest

from app.core.config import REPO_ROOT
from app.jobs.models import GenerationMode
from app.providers.base import GenerationRequest
from app.providers.comfyui import ComfyUIProvider
from app.providers.mock_video import ffmpeg_binary
from app.workflows import WorkflowTemplates
from comfyui_support import validate_graph

URL = os.environ.get("VIDSGEN_COMFYUI_LIVE_URL", "")
TOKEN = os.environ.get("VIDSGEN_COMFYUI_LIVE_TOKEN", "")
SUBMIT = os.environ.get("VIDSGEN_COMFYUI_LIVE_SUBMIT") == "1"
PLACEHOLDERS = os.environ.get("VIDSGEN_COMFYUI_LIVE_PLACEHOLDERS") == "1"

pytestmark = pytest.mark.skipif(not URL, reason="set VIDSGEN_COMFYUI_LIVE_URL to run live ComfyUI checks")
WF = WorkflowTemplates(REPO_ROOT / "workflows" / "comfyui")


def _headers():
    return {"Authorization": f"Bearer {TOKEN}"} if TOKEN else {}


def _health():
    async def go():
        p = ComfyUIProvider(base_url=URL, api_token=TOKEN, workflows=WF)
        try:
            return await p.health()
        finally:
            await p.aclose()
    return asyncio.run(go())


def test_live_health_is_reported_separately_and_honestly():
    h = _health()
    checks = {c.name: c.status for c in h.checks}
    assert checks["comfyui_reachable"] == "ok", h.detail
    assert checks["workflow_nodes"] == "ok", h.detail
    assert checks["model_files_detected"] == "ok", h.detail
    # A fresh provider has never seen a successful generation: nothing may claim readiness.
    assert checks["inference"] == "unverified"
    assert "ready" not in h.detail.lower()


def test_live_model_weights():
    h = _health()
    checks = {c.name: c.status for c in h.checks}
    if PLACEHOLDERS:
        assert h.ok is False
        assert checks["model_file_sizes"] == "failed" and checks["model_weights_integrity"] == "failed"
        assert "0 bytes" in h.detail
    else:
        assert checks["model_file_sizes"] in ("ok", "unverified"), h.detail
        assert checks["model_weights_integrity"] == "unverified"   # only a real generation can prove it
        assert h.ok, h.detail


def test_live_templates_match_this_servers_schema():
    with httpx.Client(base_url=URL, headers=_headers(), timeout=30) as c:
        info = {}
        for cls in WF.node_classes:
            data = c.get(f"/object_info/{cls}").json()
            assert cls in data, f"node class {cls} not available on this ComfyUI"
            info.update(data)
    for mode in GenerationMode:
        assert validate_graph(WF.templates[mode], info) == [], mode


# ---------------------------------------------------------------- submits (opt-in)
class _CheapWorkflow(WorkflowTemplates):
    """Same CreateVideo/SaveVideo nodes as the real template, fed by EmptyImage instead of the
    model: exercises the real output path without any weights."""

    def __init__(self, base: WorkflowTemplates, slow: bool = False):
        self.__dict__.update(base.__dict__)
        self._slow = slow

    def build(self, mode, values):
        real = super().build(GenerationMode.t2v, values)
        if self._slow:
            return {"1": {"class_type": "EmptyImage", "inputs": {"width": 2048, "height": 2048, "batch_size": 64, "color": 255}},
                    "2": {"class_type": "ImageBlur", "inputs": {"image": ["1", 0], "blur_radius": 31, "sigma": 8.0}},
                    "57": {**copy.deepcopy(real["57"]), "inputs": {**real["57"]["inputs"], "images": ["2", 0]}},
                    "58": copy.deepcopy(real["58"])}
        return {"1": {"class_type": "EmptyImage", "inputs": {"width": values.width, "height": values.height,
                                                             "batch_size": values.num_frames, "color": 0xF0B45A}},
                "57": {**copy.deepcopy(real["57"]), "inputs": {**real["57"]["inputs"], "images": ["1", 0]}},
                "58": copy.deepcopy(real["58"])}


def _req(job_id="job_livecheck"):
    return GenerationRequest(job_id=job_id, mode=GenerationMode.t2v, prompt="live", negative_prompt="", width=832, height=480,
                             num_frames=17, fps=24, steps=1, guidance_scale=5.0, seed=1)


@pytest.mark.skipif(not SUBMIT, reason="set VIDSGEN_COMFYUI_LIVE_SUBMIT=1 to submit cheap test graphs")
def test_live_output_path_produces_browser_playable_mp4(tmp_path):
    async def go():
        p = ComfyUIProvider(base_url=URL, api_token=TOKEN, workflows=_CheapWorkflow(WF), poll_interval=0.5)
        try:
            pid = await p.submit(_req())
            updates = [u async for u in p.progress(pid)]
            result = await p.fetch_result(pid)
            data = result.path.read_bytes()
            result.cleanup()
            await p.release(pid)
            return updates, data
        finally:
            await p.aclose()
    updates, data = asyncio.run(go())
    assert updates[-1].state == "completed"
    (tmp_path / "v.mp4").write_bytes(data)
    probe = subprocess.run([ffmpeg_binary(), "-hide_banner", "-i", str(tmp_path / "v.mp4")], capture_output=True, text=True).stderr
    assert data[4:8] == b"ftyp" and "Video: h264" in probe and "yuv420p" in probe and "832x480" in probe and "24 fps" in probe


@pytest.mark.skipif(not SUBMIT, reason="set VIDSGEN_COMFYUI_LIVE_SUBMIT=1 to submit cheap test graphs")
def test_live_targeted_interrupt():
    async def go():
        p = ComfyUIProvider(base_url=URL, api_token=TOKEN, workflows=_CheapWorkflow(WF, slow=True), poll_interval=0.5)
        try:
            pid = await p.submit(_req("job_liveinterrupt"))
            states = []
            async for u in p.progress(pid):
                states.append(u.state)
                if u.state == "running" and "cancel" not in states:
                    states.append("cancel")
                    await p.cancel(pid)
            await p.release(pid)
            return states
        finally:
            await p.aclose()
    states = asyncio.run(go())
    assert states[-1] == "cancelled", states
