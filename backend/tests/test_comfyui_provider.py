"""ComfyUIProvider tests against a fake ComfyUI server (real HTTP + WebSocket, no GPU).

Most tests go through the public VIDSGEN API, so the job manager, storage and download
paths are exercised exactly as in production; a few target the provider directly.
"""

from __future__ import annotations

import asyncio
import io
import logging
import socket
import time
from pathlib import Path

import pytest
from PIL import Image

from app.core.config import REPO_ROOT, Settings
from app.jobs.models import GenerationMode
from app.providers import build_provider
from app.providers.base import GenerationRequest, ProviderError
from app.providers.comfyui import ComfyUIProvider
from app.providers.mock_video import encode_mock_video
from app.workflows import WorkflowError, WorkflowTemplates
from comfyui_support import FakeComfyUI, validate_graph
from conftest import submit, wait_for

TOKEN = "test-secret-token-123"


@pytest.fixture(scope="session")
def video_bytes(tmp_path_factory) -> bytes:
    path = tmp_path_factory.mktemp("vid") / "clip.mp4"
    encode_mock_video(path, width=64, height=64, num_frames=17, fps=24, seed=1, prompt="fake comfy")
    return path.read_bytes()


@pytest.fixture
def fake(video_bytes):
    server = FakeComfyUI(video_bytes, token=TOKEN).start()
    yield server
    server.stop()


@pytest.fixture
def comfy(make_client, fake):
    """API client whose backend uses the ComfyUI provider pointed at the fake server."""
    def factory(**overrides):
        opts = dict(inference_provider="comfyui", comfyui_base_url=fake.url, comfyui_api_token=TOKEN,
                    comfyui_poll_interval=0.2, comfyui_timeout=5)
        opts.update(overrides)
        return make_client(**opts)
    return factory


def png(size=(640, 480)) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", size, (30, 120, 200)).save(buf, "PNG")
    return buf.getvalue()


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def prompts_posted(fake: FakeComfyUI) -> list[dict]:
    return [r["body"] for r in fake.requests if r["path"] == "/prompt" and r["method"] == "POST"]


# ====================================================================== selection / config
def test_mock_remains_the_default(client):
    assert client.get("/api/health").json()["provider"]["name"] == "mock"


def test_comfyui_requires_base_url():
    with pytest.raises(ValueError, match="COMFYUI_BASE_URL"):
        build_provider(Settings(_env_file=None, inference_provider="comfyui"))


def test_invalid_base_url_rejected():
    with pytest.raises(ValueError, match="must start with http"):
        Settings(_env_file=None, comfyui_base_url="ftp://gpu-box")


def test_broken_workflows_fail_at_startup(tmp_path):
    with pytest.raises(WorkflowError):
        build_provider(Settings(_env_file=None, inference_provider="comfyui", comfyui_base_url="http://x:1",
                                comfyui_workflows_dir=tmp_path))


# ====================================================================== health
CHECK_NAMES = ["comfyui_reachable", "workflow_nodes", "model_files_detected", "model_file_sizes",
               "model_weights_integrity", "inference"]


def checks_of(health: dict) -> dict[str, str]:
    checks = health["provider"]["checks"]
    assert [c["name"] for c in checks] == CHECK_NAMES[: len(checks)]
    return {c["name"]: c["status"] for c in checks}


def test_health_ok_does_not_claim_inference_readiness(comfy):
    """Correct official sizes and everything reachable: the provider can accept work, but
    weight integrity and inference are still unverified until a real generation succeeds."""
    h = comfy().get("/api/health").json()
    assert h["status"] == "ok" and h["provider"]["ok"] is True
    assert checks_of(h) == {"comfyui_reachable": "ok", "workflow_nodes": "ok", "model_files_detected": "ok",
                            "model_file_sizes": "ok", "model_weights_integrity": "unverified", "inference": "unverified"}
    detail = h["provider"]["detail"]
    assert detail == ("ComfyUI 0.46.0 reachable (cuda:0 Tesla T4 : cudaMallocAsync, 15 GB VRAM) · model files detected · "
                      "file sizes match · weight integrity and inference not yet verified")
    assert "ready" not in detail.lower()


def test_health_reports_missing_model(comfy, fake):
    fake.models["vae"] = []
    h = comfy().get("/api/health").json()
    assert h["status"] == "degraded"
    assert h["provider"]["detail"] == "Missing model files on the GPU server: vae/wan2.2_vae.safetensors"
    assert checks_of(h) == {"comfyui_reachable": "ok", "workflow_nodes": "ok", "model_files_detected": "failed"}


def test_health_reports_missing_node_class(comfy, fake):
    fake.missing_nodes = {"Wan22ImageToVideoLatent"}
    h = comfy().get("/api/health").json()
    assert h["status"] == "degraded" and "Wan22ImageToVideoLatent" in h["provider"]["detail"]


def test_health_unreachable(make_client):
    c = make_client(inference_provider="comfyui", comfyui_base_url=f"http://127.0.0.1:{free_port()}", comfyui_timeout=2)
    h = c.get("/api/health").json()
    assert h["status"] == "degraded"
    assert h["provider"]["detail"] == "The GPU server (ComfyUI) is unreachable."
    assert checks_of(h) == {"comfyui_reachable": "failed", "workflow_nodes": "unverified", "model_files_detected": "unverified",
                            "model_file_sizes": "unverified", "model_weights_integrity": "unverified", "inference": "unverified"}


def test_health_wrong_token(comfy):
    h = comfy(comfyui_api_token="wrong").get("/api/health").json()
    assert h["status"] == "degraded" and "rejected the API token" in h["provider"]["detail"]


# ====================================================================== text-to-video
def test_t2v_end_to_end(comfy, fake, video_bytes):
    c = comfy()
    job_id = submit(c, prompt="A drone shot over a coral reef", seed=1234, resolution="1280x704",
                    num_frames=49, steps=12, guidance_scale=6.5, fps=16).json()["id"]
    job = wait_for(c, job_id)
    assert job["status"] == "completed", job["error"]
    assert job["provider"] == "comfyui" and job["progress"] == 100.0

    (body,) = prompts_posted(fake)
    graph = body["prompt"]
    assert validate_graph(graph) == []
    assert "56" not in graph and "start_image" not in graph["55"]["inputs"]       # t2v: no image at all
    assert graph["6"]["inputs"]["text"] == "A drone shot over a coral reef"
    assert graph["7"]["inputs"]["text"] == WorkflowTemplates(REPO_ROOT / "workflows/comfyui").default_negative_prompt
    assert (graph["55"]["inputs"]["width"], graph["55"]["inputs"]["height"], graph["55"]["inputs"]["length"]) == (1280, 704, 49)
    assert graph["57"]["inputs"]["fps"] == 16
    assert (graph["3"]["inputs"]["steps"], graph["3"]["inputs"]["cfg"], graph["3"]["inputs"]["seed"]) == (12, 6.5, 1234)
    assert graph["58"]["inputs"]["filename_prefix"] == f"vidsgen/{job_id}"

    # The VIDSGEN job <-> ComfyUI prompt mapping is the id we chose and ComfyUI accepted.
    pid = c.app.state.manager.get(job_id).provider_job_id
    assert pid == body["prompt_id"] and pid in fake.prompts and body["client_id"].startswith("vidsgen-")

    # The video went through VIDSGEN storage and is served by the existing endpoints.
    manager = c.app.state.manager
    assert manager.get(job_id).output_key == f"outputs/{job_id}/video.mp4"
    assert manager.storage.local_path(manager.get(job_id).output_key).read_bytes() == video_bytes
    dl = c.get(f"/api/jobs/{job_id}/download")
    assert dl.status_code == 200 and dl.content == video_bytes and dl.headers["content-type"] == "video/mp4"
    assert c.get(f"/api/jobs/{job_id}/download", params={"inline": "true"}).headers["content-disposition"].startswith("inline")

    # Housekeeping: the ComfyUI history entry is deleted once the video is stored.
    assert pid not in fake.history


def test_every_request_is_authenticated(comfy, fake):
    c = comfy()
    wait_for(c, submit(c).json()["id"])
    assert fake.requests and all(r["auth"] == f"Bearer {TOKEN}" for r in fake.requests)
    assert fake.ws_auth_headers and all(h == f"Bearer {TOKEN}" for h in fake.ws_auth_headers)


def test_user_negative_prompt_replaces_default(comfy, fake):
    c = comfy()
    wait_for(c, submit(c, negative_prompt="watermark, text").json()["id"])
    assert prompts_posted(fake)[0]["prompt"]["7"]["inputs"]["text"] == "watermark, text"


def test_progress_is_monotonic_with_intermediate_values(comfy, fake):
    fake.step_delay = 0.03
    c = comfy()
    job_id = submit(c, steps=30).json()["id"]
    seen = []
    deadline = time.monotonic() + 20
    while time.monotonic() < deadline:
        j = c.get(f"/api/jobs/{job_id}").json()
        seen.append(j["progress"])
        if j["status"] in ("completed", "failed", "cancelled"):
            break
        time.sleep(0.02)
    assert j["status"] == "completed"
    assert seen == sorted(seen)
    assert any(5 < p < 90 for p in seen), seen


# ====================================================================== image-to-video
def test_i2v_uploads_image_and_references_it(comfy, fake):
    c = comfy()
    job_id = submit(c, image=png(), mode="i2v", prompt="the scene comes alive").json()["id"]
    assert wait_for(c, job_id)["status"] == "completed"
    manager = c.app.state.manager
    stored_png = manager.storage.local_path(manager.get(job_id).input_image_key).read_bytes()
    assert fake.uploads == {f"vidsgen/{job_id}.png": stored_png}   # the normalised PNG, nothing else
    upload_req = next(r for r in fake.requests if r["path"] == "/upload/image")
    assert upload_req["body"]["type"] == "input" and upload_req["body"]["overwrite"] == "true"
    graph = prompts_posted(fake)[0]["prompt"]
    assert graph["56"] == {**graph["56"], "class_type": "LoadImage", "inputs": {"image": f"vidsgen/{job_id}.png"}}
    assert graph["55"]["inputs"]["start_image"] == ["56", 0]
    assert validate_graph(graph) == []


def test_i2v_upload_failure_is_clear(comfy, fake):
    fake.upload_mode = "error"
    c = comfy()
    job = wait_for(c, submit(c, image=png(), mode="i2v").json()["id"])
    assert job["status"] == "failed" and job["error"]["code"] == "image_upload_failed"
    assert prompts_posted(fake) == []


def test_i2v_image_rejected_by_comfyui(comfy, fake):
    fake.prompt_mode = "reject_image"
    c = comfy()
    job = wait_for(c, submit(c, image=png(), mode="i2v").json()["id"])
    assert job["status"] == "failed" and job["error"]["code"] == "comfyui_rejected"
    assert "LoadImage" in job["error"]["message"]


# ====================================================================== failures
@pytest.mark.parametrize("scenario,code,text", [
    ("error", "comfyui_execution_error", "ComfyUI error in VAELoader: Error while deserializing header: header too small"),
    ("oom", "gpu_out_of_memory", "The GPU ran out of memory. Try a smaller resolution or fewer frames."),
    ("no_output", "no_video_output", "ComfyUI finished but produced no video."),
    ("webm", "unsupported_output", "ComfyUI produced a video, but not in MP4 format."),
])
def test_generation_failures(comfy, fake, scenario, code, text):
    fake.default_scenario = scenario
    c = comfy()
    job = wait_for(c, submit(c).json()["id"])
    assert job["status"] == "failed"
    assert job["error"] == {"code": code, "message": text}
    assert job["result"] is None
    assert "/srv/ComfyUI" not in str(job) and "Traceback" not in str(job)   # no internals leak


def test_missing_model_is_reported(comfy, fake):
    fake.prompt_mode = "reject_model"
    c = comfy()
    job = wait_for(c, submit(c).json()["id"])
    assert job["error"] == {"code": "model_missing",
                            "message": "Model file wan2.2_vae.safetensors is not installed on the GPU server (expected in models/vae/)."}


@pytest.mark.parametrize("mode,code", [
    ("server_error", "comfyui_server_error"),
    ("malformed", "comfyui_bad_response"),
    ("no_id", "comfyui_bad_response"),
])
def test_bad_server_responses(comfy, fake, mode, code):
    fake.prompt_mode = mode
    c = comfy()
    job = wait_for(c, submit(c).json()["id"])
    assert job["status"] == "failed" and job["error"]["code"] == code


def test_request_timeout(comfy, fake):
    fake.prompt_mode = "hang"
    c = comfy(comfyui_timeout=0.5)
    job = wait_for(c, submit(c).json()["id"])
    assert job["error"] == {"code": "comfyui_timeout", "message": "The GPU server (ComfyUI) did not respond in time."}


def test_unreachable_server_fails_clearly_without_leaking_url(make_client):
    port = free_port()
    c = make_client(inference_provider="comfyui", comfyui_base_url=f"http://127.0.0.1:{port}", comfyui_timeout=2)
    job = wait_for(c, submit(c).json()["id"])
    assert job["error"] == {"code": "comfyui_unreachable", "message": "The GPU server (ComfyUI) is unreachable."}
    assert str(port) not in str(job)
    assert job["result"] is None   # never a mock video pretending to be real


def test_whole_job_timeout(comfy, fake):
    fake.default_scenario = "hang"
    c = comfy(inference_timeout_seconds=1)
    job = wait_for(c, submit(c).json()["id"])
    assert job["status"] == "failed" and job["error"]["code"] == "timeout"
    pid = c.app.state.manager.get(job["id"]).provider_job_id
    assert pid in fake.interrupts   # the GPU work was actually stopped


def test_job_lost_when_comfyui_forgets_it(comfy, fake):
    fake.paused = True
    c = comfy()
    job_id = submit(c).json()["id"]
    wait_for(c, job_id, statuses={"processing"})
    deadline = time.monotonic() + 5
    while not fake._pending and time.monotonic() < deadline:
        time.sleep(0.02)
    fake._pending.clear()        # simulate a ComfyUI restart: the prompt vanishes
    job = wait_for(c, job_id)
    assert job["error"]["code"] == "comfyui_job_lost"


# ====================================================================== state tracking
def test_queued_then_running_then_completed(comfy, fake):
    fake.paused = True
    c = comfy()
    job_id = submit(c).json()["id"]
    job = wait_for(c, job_id, statuses={"processing"})
    time.sleep(0.5)
    job = c.get(f"/api/jobs/{job_id}").json()
    assert job["status"] == "processing" and job["progress"] == 0.0   # waiting in ComfyUI's queue
    fake.resume()
    assert wait_for(c, job_id)["status"] == "completed"


def test_completes_from_polling_when_no_websocket_events(comfy, fake):
    fake.default_scenario = "silent_success"
    c = comfy()
    assert wait_for(c, submit(c).json()["id"])["status"] == "completed"


def test_survives_websocket_drop_mid_job(comfy, fake):
    fake.default_scenario = "ws_drop"
    fake.step_delay = 0.02
    c = comfy()
    assert wait_for(c, submit(c, steps=20).json()["id"])["status"] == "completed"


def test_follows_server_prompt_id_if_client_id_ignored(comfy, fake):
    fake.ignore_client_prompt_id = True
    c = comfy()
    job = wait_for(c, submit(c).json()["id"])
    assert job["status"] == "completed"
    pid = c.app.state.manager.get(job["id"]).provider_job_id
    assert pid in fake.prompts and pid != prompts_posted(fake)[0]["prompt_id"]


# ====================================================================== cancellation
def test_cancel_running_job_interrupts_only_that_prompt(comfy, fake):
    fake.default_scenario = "hang"
    c = comfy()
    job_id = submit(c).json()["id"]
    wait_for(c, job_id, statuses={"processing"})
    deadline = time.monotonic() + 5
    while fake._running is None and time.monotonic() < deadline:
        time.sleep(0.02)
    r = c.post(f"/api/jobs/{job_id}/cancel")
    assert r.status_code == 200 and r.json()["status"] == "cancelled"
    pid = c.app.state.manager.get(job_id).provider_job_id
    assert fake.interrupts == [pid]          # targeted interrupt, never a global one
    time.sleep(0.3)
    assert c.get(f"/api/jobs/{job_id}").json()["status"] == "cancelled"


def test_cancel_queued_job_deletes_it_from_comfyui_queue(comfy, fake):
    fake.paused = True
    c = comfy()
    job_id = submit(c).json()["id"]
    wait_for(c, job_id, statuses={"processing"})
    deadline = time.monotonic() + 5
    while not fake._pending and time.monotonic() < deadline:
        time.sleep(0.02)
    assert c.post(f"/api/jobs/{job_id}/cancel").json()["status"] == "cancelled"
    pid = c.app.state.manager.get(job_id).provider_job_id
    assert fake.deleted == [pid] and fake.interrupts == []
    fake.resume()
    time.sleep(0.3)
    assert pid not in fake.history   # it never ran


# ====================================================================== concurrency (condition 2)
def _request(job_id: str, steps: int) -> GenerationRequest:
    return GenerationRequest(job_id=job_id, mode=GenerationMode.t2v, prompt=f"prompt for {job_id}", negative_prompt="",
                             width=832, height=480, num_frames=17, fps=24, steps=steps, guidance_scale=5.0, seed=1)


def test_concurrent_jobs_only_see_their_own_progress(fake):
    """Two jobs share one WebSocket; foreign events (unknown prompt ids, no prompt id) are
    injected too. Each job must see only its own sampler steps."""
    fake.foreign_events = True
    fake.step_delay = 0.01

    async def scenario():
        provider = ComfyUIProvider(base_url=fake.url, api_token=TOKEN, poll_interval=0.2,
                                   workflows=WorkflowTemplates(REPO_ROOT / "workflows/comfyui"))
        try:
            a = await provider.submit(_request("job_aaaa", steps=10))
            b = await provider.submit(_request("job_bbbb", steps=7))

            async def collect(pid):
                return [u async for u in provider.progress(pid)]

            ua, ub = await asyncio.gather(collect(a), collect(b))
            ra, rb = await provider.fetch_result(a), await provider.fetch_result(b)
            sizes = (ra.path.stat().st_size, rb.path.stat().st_size)
            ra.cleanup(), rb.cleanup()
            return ua, ub, sizes
        finally:
            await provider.aclose()

    ua, ub, sizes = asyncio.run(scenario())
    steps_a = [u.message for u in ua if u.message.startswith("step ")]
    steps_b = [u.message for u in ub if u.message.startswith("step ")]
    assert steps_a and all(m.endswith("/10") for m in steps_a), steps_a
    assert steps_b and all(m.endswith("/7") for m in steps_b), steps_b
    assert ua[-1].state == ub[-1].state == "completed"
    assert all(s > 0 for s in sizes)
    for updates in (ua, ub):
        pcts = [u.percent for u in updates]
        assert pcts == sorted(pcts)


def test_dispatch_routes_strictly_by_prompt_id(fake):
    provider = ComfyUIProvider(base_url=fake.url, workflows=WorkflowTemplates(REPO_ROOT / "workflows/comfyui"))
    from app.providers.comfyui import _Run

    provider._runs = {"p1": _Run("p1", "job_1"), "p2": _Run("p2", "job_2")}
    for raw in ['{"type":"progress","data":{"value":1,"max":5,"node":"3","prompt_id":"p1"}}',
                '{"type":"progress","data":{"value":2,"max":9,"node":"3","prompt_id":"p2"}}',
                '{"type":"progress","data":{"value":3,"max":9,"node":"3","prompt_id":"other"}}',
                '{"type":"progress","data":{"value":4,"max":9,"node":"3"}}',
                '{"type":"status","data":{"status":{"exec_info":{"queue_remaining":1}}}}',
                'not json', '{"type":"executing","data":null}', '[1,2]']:
        provider._dispatch(raw)
    assert [provider._runs["p1"].events.get_nowait()[1]["value"]] == [1] and provider._runs["p1"].events.empty()
    assert [provider._runs["p2"].events.get_nowait()[1]["value"]] == [2] and provider._runs["p2"].events.empty()
    asyncio.run(provider.aclose())


# ====================================================================== secrets
def test_token_never_logged(comfy, fake, caplog):
    caplog.set_level(logging.DEBUG)
    fake.default_scenario = "error"
    c = comfy()
    wait_for(c, submit(c).json()["id"])
    c.get("/api/health")
    assert TOKEN not in caplog.text
    assert "Bearer" not in caplog.text


def test_provider_details_not_exposed_to_frontend(comfy, fake):
    c = comfy()
    job = wait_for(c, submit(c).json()["id"])
    text = str(job) + str(c.get("/api/health").json())
    assert fake.url not in text and str(fake.port) not in text and TOKEN not in text


# ====================================================================== honest model/inference health
def test_zero_byte_placeholders_are_never_reported_ready(comfy, fake):
    """The exact situation from Phase 4: correct filenames, 0-byte files. Uses the real
    /experiment/models payload recorded from ComfyUI 0.46.0."""
    import json as _json
    from comfyui_support import FIXTURES

    recorded = _json.loads((FIXTURES / "recorded_responses.json").read_text())["experiment_models_placeholders"]
    fake.sizes = {item["name"]: item["size"] for items in recorded.values() for item in items}
    assert set(fake.sizes.values()) == {0}
    h = comfy().get("/api/health").json()
    assert h["status"] == "degraded" and h["provider"]["ok"] is False
    assert checks_of(h) == {"comfyui_reachable": "ok", "workflow_nodes": "ok", "model_files_detected": "ok",
                            "model_file_sizes": "failed", "model_weights_integrity": "failed", "inference": "unverified"}
    detail = h["provider"]["detail"]
    assert detail.startswith("Model files have the wrong size on the GPU server")
    assert "diffusion_models/wan2.2_ti2v_5B_fp16.safetensors (0 bytes, expected 9,999,658,848)" in detail
    assert "ready" not in detail.lower()


def test_one_truncated_file_is_detected(comfy, fake):
    fake.sizes["wan2.2_vae.safetensors"] = 1_000_000          # interrupted download
    h = comfy().get("/api/health").json()
    assert h["provider"]["ok"] is False
    assert "vae/wan2.2_vae.safetensors (1,000,000 bytes, expected 1,409,400,960)" in h["provider"]["detail"]
    assert "diffusion_models" not in h["provider"]["detail"]  # only the bad file is named


def test_sizes_unavailable_means_unverified_not_ok(comfy, fake):
    fake.sizes_route = False                                 # e.g. a ComfyUI without the experimental route
    h = comfy().get("/api/health").json()
    assert h["provider"]["ok"] is True                        # it can accept work...
    assert checks_of(h)["model_file_sizes"] == "unverified"   # ...but nothing about the weights is claimed
    assert checks_of(h)["model_weights_integrity"] == "unverified" and checks_of(h)["inference"] == "unverified"
    assert "weights unverified" in h["provider"]["detail"] and "file sizes match" not in h["provider"]["detail"]
    assert "ready" not in h["provider"]["detail"].lower()


def test_inference_verified_only_after_a_real_wan_generation(comfy, fake):
    c = comfy()
    assert checks_of(c.get("/api/health").json())["inference"] == "unverified"
    fake.default_scenario = "error"
    wait_for(c, submit(c).json()["id"])                       # a failed generation proves nothing
    assert checks_of(c.get("/api/health").json())["inference"] == "unverified"
    fake.default_scenario = "success"
    assert wait_for(c, submit(c).json()["id"])["status"] == "completed"
    h = c.get("/api/health").json()
    assert checks_of(h)["inference"] == "ok" and checks_of(h)["model_weights_integrity"] == "ok"
    assert "inference verified by a successful generation at" in h["provider"]["detail"]


def test_non_wan_graph_success_does_not_verify_inference(fake):
    """A graph that never loads the Wan model files (e.g. the cheap live-test graph) must not
    count as proof that the weights work."""
    class NoModels(WorkflowTemplates):
        def build(self, mode, values):
            g = super().build(mode, values)
            for m in self.required_models:
                g.pop(m.node_id)
            return g

    async def scenario():
        provider = ComfyUIProvider(base_url=fake.url, api_token=TOKEN, poll_interval=0.2,
                                   workflows=NoModels(REPO_ROOT / "workflows/comfyui"))
        try:
            fake.prompt_mode = "ok"
            pid = await provider.submit(_request("job_cheap", steps=3))
            final = [u async for u in provider.progress(pid)][-1]
            return final.state, await provider.health()
        finally:
            await provider.aclose()

    fake.validate_prompts = False                            # the stripped graph isn't a valid Wan graph
    state, health = asyncio.run(scenario())
    assert state == "completed"
    assert {c.name: c.status for c in health.checks}["inference"] == "unverified"


# ====================================================================== Phase 6A: out-of-memory mapping
GPU_OOM = ("failed", "gpu_out_of_memory", "The GPU ran out of memory. Try a smaller resolution or fewer frames.")
RAM_OOM = ("failed", "system_out_of_memory",
           "The GPU server ran out of system memory (RAM). Try a smaller resolution or fewer frames.")


@pytest.mark.parametrize("etype,message,trace,expected", [
    ("torch.OutOfMemoryError", "CUDA out of memory. Tried to allocate 2.00 GiB", [], GPU_OOM),
    ("torch.cuda.OutOfMemoryError", "", [], GPU_OOM),                                   # type alone is enough
    ("RuntimeError", "CUDA error: out of memory", [], GPU_OOM),
    ("RuntimeError", "Allocation on device", [], GPU_OOM),                              # comfy.model_management
    ("RuntimeError", "CUBLAS_STATUS_ALLOC_FAILED when calling cublasCreate", [], GPU_OOM),
    ("RuntimeError", "boom", ["File x.py\n", "torch.OutOfMemoryError: CUDA out of memory.\n"], GPU_OOM),  # only in the traceback
    ("MemoryError", "", [], RAM_OOM),
    ("builtins.MemoryError", "x", [], RAM_OOM),
    ("RuntimeError", "[enforce fail at alloc_cpu.cpp:117] DefaultCPUAllocator: can't allocate memory", [], RAM_OOM),
    ("RuntimeError", "std::bad_alloc", [], RAM_OOM),
])
def test_execution_errors_that_are_out_of_memory_are_recognised(etype, message, trace, expected):
    p = ComfyUIProvider._execution_error({"exception_type": etype, "exception_message": message, "traceback": trace,
                                          "node_type": "KSampler"})
    assert (p.state, p.error_code, p.message) == expected


@pytest.mark.parametrize("etype,message", [
    ("safetensors.SafetensorError", "header too small"),
    ("ValueError", "bedroom memory is not a thing, nor is a hall of mirrors"),          # "room"/"memory" are not markers
    ("RuntimeError", "Expected all tensors to be on the same device"),
    ("KeyError", "'latent'"),
])
def test_other_execution_errors_are_not_reported_as_out_of_memory(etype, message):
    p = ComfyUIProvider._execution_error({"exception_type": etype, "exception_message": message, "node_type": "KSampler"})
    assert p.error_code == "comfyui_execution_error" and p.message.startswith("ComfyUI error in KSampler")


def test_system_memory_exhaustion_reaches_the_job(comfy, fake):
    fake.default_scenario = "oom_ram"
    c = comfy()
    job = wait_for(c, submit(c).json()["id"])
    assert job["status"] == "failed" and job["error"] == {"code": "system_out_of_memory", "message": RAM_OOM[2]}


def test_oom_is_recognised_when_only_the_history_has_it(comfy, fake):
    """No WebSocket events (e.g. the socket dropped): the failure is found by polling /history."""
    fake.default_scenario = "silent_oom"
    c = comfy()
    job = wait_for(c, submit(c).json()["id"])
    assert job["status"] == "failed" and job["error"] == {"code": "gpu_out_of_memory", "message": GPU_OOM[2]}


def test_a_job_after_an_oom_still_works(comfy, fake):
    fake.next_scenarios = ["oom"]
    c = comfy()
    assert wait_for(c, submit(c).json()["id"])["error"]["code"] == "gpu_out_of_memory"
    assert wait_for(c, submit(c).json()["id"])["status"] == "completed"


# ====================================================================== Phase 6A: connection interruptions
def start_running(c, fake):
    """Submit a job that ComfyUI holds in its queue (so it stays 'processing'), and wait until it is queued there."""
    fake.paused = True
    job_id = submit(c).json()["id"]
    wait_for(c, job_id, statuses={"processing"})
    deadline = time.monotonic() + 5
    while not fake._pending and time.monotonic() < deadline:
        time.sleep(0.02)
    assert fake._pending
    return job_id


def test_a_short_outage_does_not_fail_the_job(comfy, fake):
    c = comfy(comfyui_connection_grace_seconds=10)
    job_id = start_running(c, fake)
    fake.outage = True
    time.sleep(1.0)                                      # several failed polls
    assert c.get(f"/api/jobs/{job_id}").json()["status"] == "processing"
    fake.outage = False
    fake.resume()
    job = wait_for(c, job_id)
    assert job["status"] == "completed" and job["error"] is None


def test_a_long_outage_fails_with_a_clear_connection_lost_error(comfy, fake):
    c = comfy(comfyui_connection_grace_seconds=1)
    job_id = start_running(c, fake)
    fake.outage = True
    job = wait_for(c, job_id)
    assert job["status"] == "failed" and job["error"]["code"] == "comfyui_connection_lost"
    assert "lost for more than 1 s" in job["error"]["message"] and "Please try again" in job["error"]["message"]
    assert str(fake.port) not in str(job)                # the address is never shown to users


def test_a_dead_server_fails_as_connection_lost_not_a_hang(comfy, fake):
    c = comfy(comfyui_connection_grace_seconds=1)
    job_id = start_running(c, fake)
    fake.stop()                                          # connection refused from now on
    job = wait_for(c, job_id)
    assert job["status"] == "failed" and job["error"]["code"] == "comfyui_connection_lost"


def test_zero_grace_keeps_the_old_fail_fast_behaviour(comfy, fake):
    c = comfy(comfyui_connection_grace_seconds=0)
    job_id = start_running(c, fake)
    fake.outage = True
    assert wait_for(c, job_id)["error"]["code"] == "comfyui_server_error"


def test_only_connection_problems_are_retried(comfy, fake):
    """A rejected token during generation is an answer, not a blip: it fails at once, not after the grace period."""
    c = comfy(comfyui_connection_grace_seconds=600)
    job_id = start_running(c, fake)
    fake.token = "a-different-token"
    fake.drop_websockets()                               # no live events: the provider polls, and is refused
    t0 = time.monotonic()
    job = wait_for(c, job_id, timeout=15)
    assert job["error"]["code"] == "comfyui_auth_failed" and time.monotonic() - t0 < 10


def test_the_overall_job_timeout_still_applies_during_an_outage(comfy, fake):
    c = comfy(comfyui_connection_grace_seconds=600, inference_timeout_seconds=2)
    job_id = start_running(c, fake)
    fake.outage = True
    assert wait_for(c, job_id, timeout=15)["error"]["code"] == "timeout"


def test_cancelling_during_an_outage_ends_the_job(comfy, fake):
    c = comfy(comfyui_connection_grace_seconds=600)
    job_id = start_running(c, fake)
    fake.outage = True
    time.sleep(0.5)
    assert c.post(f"/api/jobs/{job_id}/cancel").json()["status"] == "cancelled"
    assert wait_for(c, job_id)["status"] == "cancelled"


def test_a_failed_download_is_retried_so_a_blip_does_not_lose_the_video(comfy, fake):
    fake.view_failures = 1                               # the first /view gets a 503 (e.g. a proxy hiccup)
    c = comfy()
    job = wait_for(c, submit(c).json()["id"])
    assert job["status"] == "completed" and job["result"] is not None
    assert len([r for r in fake.requests if r["path"] == "/view"]) == 2


def test_a_download_that_keeps_failing_gives_up_with_a_clear_error(comfy, fake):
    fake.view_failures = 99
    c = comfy()
    job = wait_for(c, submit(c).json()["id"])
    assert job["status"] == "failed"
    assert job["error"] == {"code": "output_download_failed", "message": "Downloading the video failed (HTTP 503)."}
    assert len([r for r in fake.requests if r["path"] == "/view"]) == 3     # bounded
    assert job["result"] is None


def test_a_missing_output_is_not_retried(comfy, fake):
    fake.view_override = (404, b"nope")
    c = comfy()
    job = wait_for(c, submit(c).json()["id"])
    assert job["error"]["code"] == "output_download_failed" and "404" in job["error"]["message"]
    assert len([r for r in fake.requests if r["path"] == "/view"]) == 1


# ====================================================================== Phase 6C Stage 2: read-only reconciliation of uncertain submissions
def posts_to(fake, path="/prompt"):
    return [r for r in fake.requests if r["path"] == path and r["method"] == "POST"]


@pytest.mark.parametrize("mode", ["queue_then_hang", "queue_then_500"])
def test_a_lost_submission_response_is_reconciled_without_resubmitting(comfy, fake, mode):
    """ComfyUI queued the prompt but the response was lost: the job must continue, and the prompt must be sent exactly once."""
    fake.prompt_mode = mode
    c = comfy(comfyui_timeout=0.5)
    job = wait_for(c, submit(c).json()["id"], timeout=40)
    assert job["status"] == "completed", job["error"]
    assert len(posts_to(fake)) == 1                                                     # never resubmitted


def test_reconciliation_uses_only_read_only_requests(comfy, fake):
    fake.prompt_mode = "queue_then_hang"
    c = comfy(comfyui_timeout=0.5)
    job_id = submit(c).json()["id"]
    wait_for(c, job_id, timeout=40)
    first_post = next(i for i, r in enumerate(fake.requests) if r["path"] == "/prompt")
    between = [r for r in fake.requests[first_post + 1:] if r["path"].startswith("/history/") or r["path"] == "/queue"]
    assert between and between[0]["method"] == "GET"                                    # the first thing after the lost response is a read
    assert [r["method"] for r in fake.requests if r["path"] in ("/prompt",)] == ["POST"]


def test_a_prompt_that_comfyui_does_not_know_is_reported_failed_and_not_resubmitted(comfy, fake):
    fake.prompt_mode = "hang"                                                           # never queued
    c = comfy(comfyui_timeout=0.5)
    job = wait_for(c, submit(c).json()["id"], timeout=40)
    assert job["status"] == "failed" and job["error"] == {"code": "comfyui_timeout", "message": "The GPU server (ComfyUI) did not respond in time."}
    assert len(posts_to(fake)) == 1 and len(fake.prompts) == 0


def test_an_unreachable_server_is_not_reconciled_it_fails_fast(make_client):
    port = free_port()
    c = make_client(inference_provider="comfyui", comfyui_base_url=f"http://127.0.0.1:{port}", comfyui_timeout=2)   # default 30 s grace
    t0 = time.monotonic()
    job = wait_for(c, submit(c).json()["id"], timeout=20)
    assert job["error"]["code"] == "comfyui_unreachable" and time.monotonic() - t0 < 8     # a refused connection cannot have queued anything


def test_zero_grace_disables_reconciliation(comfy, fake):
    fake.prompt_mode = "queue_then_hang"
    c = comfy(comfyui_timeout=0.5, comfyui_connection_grace_seconds=0)
    job = wait_for(c, submit(c).json()["id"], timeout=40)
    assert job["status"] == "failed" and job["error"]["code"] == "comfyui_timeout"
    assert len(posts_to(fake)) == 1
    first_post = next(i for i, r in enumerate(fake.requests) if r["path"] == "/prompt")
    assert [r for r in fake.requests[first_post + 1:] if r["path"].startswith("/history/") and r["method"] == "GET"] == []   # no lookups at all


def test_rejections_by_comfyui_are_not_reconciled(comfy, fake):
    fake.prompt_mode = "reject_model"
    c = comfy()
    job = wait_for(c, submit(c).json()["id"])
    assert job["error"]["code"] == "model_missing"
    assert [r for r in fake.requests if r["path"] == "/queue"] == []                    # no reconciliation lookups for a definite answer
