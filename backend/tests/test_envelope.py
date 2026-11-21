"""The GPU generation envelope: unsupported requests are rejected up front with clear 422 messages."""

import pytest
from pydantic import ValidationError

from app.core.config import Settings
from app.jobs.envelope import PROFILES, GenerationEnvelope
from conftest import make_image, make_settings, submit, wait_for

# Exactly what Phase 5B proved on a Kaggle T4.
PROVEN = dict(resolution="832x480", num_frames=17, steps=20, fps=24, guidance_scale=5.0)


@pytest.fixture
def t4(make_client):
    return make_client(generation_profile="t4_verified")


def errors(r) -> dict[str, str]:
    assert r.status_code == 422, r.text
    return {e["loc"][-1]: e["msg"] for e in r.json()["detail"]}


# ------------------------------------------------------------------ defaults are unchanged
def test_default_is_unrestricted_and_mock_behaviour_is_unchanged(client):
    s = Settings(_env_file=None)
    assert s.generation_profile == "unrestricted" and s.generation_max_frames is None and s.inference_max_concurrent_jobs == 1
    assert client.app.state.manager.envelope.restricted is False
    assert submit(client, resolution="1280x704", num_frames=121, steps=60, mode="t2v").status_code == 202
    i2v = submit(client, mode="i2v", image=make_image())
    assert i2v.status_code == 202


# ------------------------------------------------------------------ the proven configuration works
def test_the_proven_kaggle_configuration_is_accepted_and_completes(t4):
    r = submit(t4, **PROVEN)
    assert r.status_code == 202, r.text
    assert wait_for(t4, r.json()["id"])["status"] == "completed"


@pytest.mark.parametrize("override", [dict(num_frames=17, steps=8), dict(steps=1), dict(fps=30), dict(guidance_scale=1.0),
                                      dict(guidance_scale=15.0), dict(fps=8)])
def test_values_inside_the_envelope_are_accepted(t4, override):
    assert submit(t4, **{**PROVEN, **override}).status_code == 202


# ------------------------------------------------------------------ rejections
@pytest.mark.parametrize("override,field,fragments", [
    (dict(resolution="1280x704"), "resolution", ["1280x704", "not enabled", "allowed: 832x480", "t4_verified"]),
    (dict(resolution="640x640"), "resolution", ["640x640", "allowed: 832x480"]),
    (dict(resolution="480x832"), "resolution", ["480x832", "allowed: 832x480"]),
    (dict(num_frames=21), "num_frames", ["21", "maximum of 17"]),
    (dict(num_frames=121), "num_frames", ["121", "maximum of 17"]),
    (dict(steps=21), "steps", ["21", "maximum of 20"]),
    (dict(steps=30), "steps", ["30", "maximum of 20"]),
])
def test_requests_beyond_the_envelope_are_rejected_clearly(t4, override, field, fragments):
    r = submit(t4, **{**PROVEN, **override})
    msg = errors(r)[field]
    for f in fragments:
        assert f in msg, msg
    assert t4.get("/api/jobs").json()["total"] == 0          # nothing was queued


def test_image_to_video_is_not_enabled_by_the_verified_preset_and_the_upload_is_not_stored(t4):
    r = submit(t4, mode="i2v", image=make_image(), **PROVEN)
    assert "image-to-video is not enabled" in errors(r)["mode"] and "allowed: t2v" in errors(r)["mode"]
    assert t4.get("/api/jobs").json()["total"] == 0
    root = t4.app.state.manager.settings.storage_local_root
    assert not list(root.rglob("*.png"))                      # the rejected upload was never stored


def test_every_violation_is_reported_at_once(t4):
    r = submit(t4, resolution="1280x704", num_frames=121, steps=60)
    assert set(errors(r)) == {"resolution", "num_frames", "steps"}


def test_the_api_defaults_exceed_the_envelope_and_say_what_to_send(t4):
    """Omitted fields use the API defaults (49 frames, 30 steps), which the T4 profile must refuse, clearly."""
    r = t4.post("/api/jobs", data={"mode": "t2v", "prompt": "a fox"})
    e = errors(r)
    assert "maximum of 17" in e["num_frames"] and "maximum of 20" in e["steps"] and "resolution" not in e


def test_the_error_shape_matches_what_the_frontend_already_parses(t4):
    detail = submit(t4, **{**PROVEN, "num_frames": 49}).json()["detail"]
    assert isinstance(detail, list) and detail[0]["loc"] == ["body", "num_frames"] and detail[0]["type"] == "value_error"


def test_ordinary_validation_still_comes_first(t4):
    # A malformed value is a normal 422 about the value, not an envelope message.
    assert "4k+1" in errors(submit(t4, **{**PROVEN, "num_frames": 18}))["num_frames"]


# ------------------------------------------------------------------ configuration
def test_individual_settings_override_the_preset(make_client):
    c = make_client(generation_profile="t4_verified", generation_allowed_modes="t2v,i2v",
                    generation_max_frames=33, generation_allowed_resolutions="832x480,480x832", generation_max_steps="30")
    assert submit(c, **{**PROVEN, "num_frames": 33, "steps": 30, "resolution": "480x832"}).status_code == 202
    assert submit(c, mode="i2v", image=make_image(), **PROVEN).status_code == 202
    assert "maximum of 33" in errors(submit(c, **{**PROVEN, "num_frames": 37}))["num_frames"]


def test_limits_work_without_a_profile(make_client):
    c = make_client(generation_max_frames=25)
    assert submit(c, num_frames=25, resolution="1280x704").status_code == 202
    assert "maximum of 25" in errors(submit(c, num_frames=29))["num_frames"]
    assert c.app.state.manager.envelope.summary() == "profile unrestricted: at most 25 frames"


def test_the_preset_is_the_proven_configuration():
    e = GenerationEnvelope.from_settings(Settings(_env_file=None, generation_profile="t4_verified"))
    assert PROFILES["t4_verified"] == {"modes": ("t2v",), "resolutions": ("832x480",), "max_frames": 17, "max_steps": 20}
    assert (e.modes, e.resolutions, e.max_frames, e.max_steps) == (("t2v",), ("832x480",), 17, 20)
    assert e.summary() == "profile t4_verified: modes t2v; resolutions 832x480; at most 17 frames; at most 20 steps"


def test_blank_limits_from_an_env_file_mean_unset():
    s = Settings(_env_file=None, generation_max_frames="", generation_max_steps=" ")
    assert s.generation_max_frames is None and s.generation_max_steps is None


@pytest.mark.parametrize("bad", [
    dict(generation_allowed_resolutions="1920x1080"),
    dict(generation_allowed_modes="t2v,v2v"),
    dict(generation_max_frames=200),
    dict(generation_max_frames=5),
    dict(generation_max_steps=0),
    dict(generation_profile="h100"),
])
def test_invalid_envelope_configuration_fails_at_startup(bad):
    with pytest.raises(ValidationError):
        Settings(_env_file=None, **bad)


def test_the_verified_preset_requires_one_concurrent_job(tmp_path):
    with pytest.raises(ValidationError, match="INFERENCE_MAX_CONCURRENT_JOBS=1"):
        make_settings(tmp_path, generation_profile="t4_verified", inference_max_concurrent_jobs=2)
    assert make_settings(tmp_path, generation_profile="t4_verified").inference_max_concurrent_jobs == 1
    assert make_settings(tmp_path, inference_max_concurrent_jobs=2).inference_max_concurrent_jobs == 2   # unrestricted: unchanged
