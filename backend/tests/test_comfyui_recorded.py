"""Keeps the fake ComfyUI and the provider's parsing honest: they are checked against
responses recorded from a REAL ComfyUI 0.46.0 (fixtures/comfyui/recorded_responses.json)."""

import json

from app.core.config import REPO_ROOT
from app.providers.comfyui import ComfyUIProvider, _Run
from app.workflows import WorkflowTemplates
from comfyui_support import FIXTURES

REC = json.loads((FIXTURES / "recorded_responses.json").read_text())
WF = WorkflowTemplates(REPO_ROOT / "workflows" / "comfyui")


def provider() -> ComfyUIProvider:
    return ComfyUIProvider(base_url="http://127.0.0.1:1", workflows=WF)


def test_real_success_history_yields_the_video():
    p = provider()
    run = _Run("pid", "job_x")
    update = p._from_history(run, REC["success_run"]["history_entry"])
    assert update.state == "completed"
    (video,) = run.outputs
    # Condition 1 (verified, not assumed): SaveVideo reports under "images" with animated=[true],
    # and the entry carries metadata.kind == "video".
    out = REC["success_run"]["history_entry"]["outputs"]["58"]
    assert out["animated"] == [True] and video in out["images"]
    assert video["filename"].endswith(".mp4") and video["type"] == "output" and video["metadata"]["kind"] == "video"


def test_real_events_all_carry_prompt_id():
    """Routing by prompt_id (condition 2) relies on every execution event carrying it."""
    pid = REC["success_run"]["prompt_id"]
    for e in REC["success_run"]["ws_events"]:
        assert e["data"].get("prompt_id") == pid, e["type"]


def test_real_execution_error_maps_to_clear_failure():
    p = provider()
    update = p._from_event(_Run("pid", "job_x"), "execution_error", REC["execution_error_event"]["data"])
    assert update.state == "failed" and update.error_code == "comfyui_execution_error"
    assert "header too small" in update.message
    assert p._from_history(_Run("pid", "job_x"), {"status": REC["execution_error_history_status"], "outputs": {}}).state == "failed"


def test_real_interrupt_maps_to_cancelled():
    p = provider()
    assert p._from_event(_Run("pid", "j"), "execution_interrupted", REC["interrupted_event"]["data"]).state == "cancelled"
    assert p._from_history(_Run("pid", "j"), {"status": REC["interrupted_history_status"], "outputs": {}}).state == "cancelled"


def test_real_rejections_map_to_clear_errors():
    p = provider()
    missing_model = p._rejection(REC["missing_model_rejection"]["resp"])
    assert REC["missing_model_rejection"]["status"] == 400
    assert missing_model.code == "model_missing" and "wan2.2_ti2v_5B_fp16.safetensors" in missing_model.message
    missing_image = p._rejection(REC["missing_image_rejection"]["resp"])
    assert missing_image.code == "comfyui_rejected" and "Invalid image file" in missing_image.message


def test_real_queue_and_upload_shapes_match_what_the_provider_reads():
    q = REC["queue_shape"]
    assert all(len(item) == 5 and isinstance(item[1], str) for item in q["queue_running"] + q["queue_pending"])
    up = REC["upload_image_response"]
    assert up["status"] == 200 and set(up["resp"]) == {"name", "subfolder", "type"}
    assert REC["history_of_deleted_pending_prompt"] == {}   # deleted pending prompts leave no trace
