"""GET /api/limits: one authoritative description of what POST /api/jobs accepts."""

from __future__ import annotations

import copy
import json
import time

import pytest
from fastapi.testclient import TestClient

from app.core.config import REPO_ROOT, Settings
from app.jobs.envelope import GenerationEnvelope
from app.jobs.limits import FRAME_STEP, LimitsError, build_limits, verify_limits
from app.jobs.models import MAX_FRAMES, MIN_FRAMES, SUPPORTED_RESOLUTIONS
from app.main import create_app
from comfyui_support import FakeComfyUI
from conftest import make_image, make_settings, submit

REASON_T4 = "(verified GPU profile: t4_verified)."
IMAGE = {"mime_types": ["image/jpeg", "image/png", "image/webp"], "max_bytes": 10 * 1024 * 1024, "min_side": 256, "max_side": 4096}
RES = [("1280x704", 1280, 704), ("704x1280", 704, 1280), ("832x480", 832, 480), ("480x832", 480, 832), ("640x640", 640, 640)]

DEFAULT_LIMITS = {
    "profile": "unrestricted", "restricted": False, "max_concurrent_jobs": 1,
    "modes": [{"value": "t2v", "label": "Text to video", "enabled": True, "reason": None},
              {"value": "i2v", "label": "Image to video", "enabled": True, "reason": None}],
    "resolutions": [{"value": v, "width": w, "height": h, "enabled": True, "reason": None} for v, w, h in RES],
    "num_frames": {"min": 17, "max": 121, "step": 4, "allowed": list(range(17, 122, 4)), "limited": False, "reason": None},
    "steps": {"min": 1, "max": 60, "limited": False, "reason": None},
    "fps": {"min": 8, "max": 30}, "guidance_scale": {"min": 1.0, "max": 15.0}, "seed": {"min": 0, "max": 4294967295},
    "prompt": {"max_chars": 2000}, "image": IMAGE,
    "defaults": {"mode": "t2v", "resolution": "832x480", "num_frames": 49, "steps": 30, "fps": 24, "guidance_scale": 5.0},
}

T4_LIMITS = {
    "profile": "t4_verified", "restricted": True, "max_concurrent_jobs": 1,
    "modes": [{"value": "t2v", "label": "Text to video", "enabled": True, "reason": None},
              {"value": "i2v", "label": "Image to video", "enabled": False,
               "reason": "Image to video is not enabled on this server " + REASON_T4}],
    "resolutions": [{"value": v, "width": w, "height": h, "enabled": v == "832x480",
                     "reason": None if v == "832x480" else f"{w}x{h} is not enabled on this server " + REASON_T4} for v, w, h in RES],
    "num_frames": {"min": 17, "max": 17, "step": 4, "allowed": [17], "limited": True,
                   "reason": "This server allows at most 17 frames " + REASON_T4},
    "steps": {"min": 1, "max": 20, "limited": True, "reason": "This server allows at most 20 steps " + REASON_T4},
    "fps": {"min": 8, "max": 30}, "guidance_scale": {"min": 1.0, "max": 15.0}, "seed": {"min": 0, "max": 4294967295},
    "prompt": {"max_chars": 2000}, "image": IMAGE,
    "defaults": {"mode": "t2v", "resolution": "832x480", "num_frames": 17, "steps": 20, "fps": 24, "guidance_scale": 5.0},
}

PROFILES = {
    "default": {},
    "t4_verified": {"generation_profile": "t4_verified"},
    "t4_with_i2v": {"generation_profile": "t4_verified", "generation_allowed_modes": "t2v,i2v"},
    "custom_small": {"generation_max_frames": 33, "generation_max_steps": 8, "generation_allowed_resolutions": "832x480,640x640"},
    "i2v_only": {"generation_allowed_modes": "i2v", "generation_allowed_resolutions": "480x832"},
}


def limits_of(client) -> dict:
    r = client.get("/api/limits")
    assert r.status_code == 200, r.text
    return r.json()


# ====================================================================== the two main profiles
def test_the_default_profile_reports_the_models_full_capabilities(client):
    assert limits_of(client) == DEFAULT_LIMITS


def test_the_t4_verified_profile_reports_its_restrictive_text_to_video_capabilities(make_client):
    c = make_client(generation_profile="t4_verified")
    body = limits_of(c)
    assert body == T4_LIMITS
    assert [m["value"] for m in body["modes"] if m["enabled"]] == ["t2v"]
    assert [r["value"] for r in body["resolutions"] if r["enabled"]] == ["832x480"]
    assert body["num_frames"]["allowed"] == [17] and body["steps"]["max"] == 20 and body["max_concurrent_jobs"] == 1


def test_disabled_options_carry_a_human_reason_and_enabled_ones_do_not(make_client):
    body = limits_of(make_client(generation_profile="t4_verified"))
    for group in (body["modes"], body["resolutions"]):
        for item in group:
            assert (item["reason"] is None) == item["enabled"]
            assert item["reason"] is None or ("not enabled on this server" in item["reason"] and "t4_verified" in item["reason"])


def test_a_restriction_that_is_not_a_named_profile_says_it_is_configuration(make_client):
    body = limits_of(make_client(generation_max_frames=33))
    assert body["profile"] == "unrestricted" and body["restricted"] is True
    assert body["num_frames"]["max"] == 33 and "limited by its configuration" in body["num_frames"]["reason"]
    assert body["steps"]["limited"] is False and body["steps"]["reason"] is None


def test_overrides_are_reflected(make_client):
    body = limits_of(make_client(**PROFILES["custom_small"]))
    assert {r["value"] for r in body["resolutions"] if r["enabled"]} == {"832x480", "640x640"}
    assert body["num_frames"]["allowed"] == [17, 21, 25, 29, 33] and body["steps"]["max"] == 8
    assert body["defaults"] == {"mode": "t2v", "resolution": "832x480", "num_frames": 33, "steps": 8, "fps": 24, "guidance_scale": 5.0}


def test_when_text_to_video_is_off_the_default_mode_is_the_first_enabled_one(make_client):
    body = limits_of(make_client(**PROFILES["i2v_only"]))
    assert body["defaults"]["mode"] == "i2v" and body["defaults"]["resolution"] == "480x832"
    assert [m["enabled"] for m in body["modes"]] == [False, True]


# ====================================================================== consistency with POST /api/jobs validation
def base_fields(limits: dict, **over) -> dict:
    d = limits["defaults"]
    fields = {"mode": d["mode"], "prompt": "a red fox", "resolution": d["resolution"], "num_frames": d["num_frames"], "steps": d["steps"],
              "fps": d["fps"], "guidance_scale": d["guidance_scale"]}
    fields.update(over)
    return fields


def post(client, fields: dict):
    image = make_image() if fields["mode"] == "i2v" else None
    return client.post("/api/jobs", data={k: str(v) for k, v in fields.items()}, files={"image": ("in.png", image, "image/png")} if image else None)


def variants(limits: dict) -> list[tuple[str, dict, bool]]:
    """(dimension, field overrides, expected to be accepted) derived ONLY from the limits response."""
    out = []
    for m in limits["modes"]:
        out.append(("mode", {"mode": m["value"]}, m["enabled"]))
    for r in limits["resolutions"]:
        out.append(("resolution", {"resolution": r["value"]}, r["enabled"]))
    f = limits["num_frames"]
    for n in sorted({f["min"] - FRAME_STEP, f["min"], f["max"], f["max"] + FRAME_STEP, MAX_FRAMES, MAX_FRAMES + FRAME_STEP, f["min"] + 1}):
        out.append(("num_frames", {"num_frames": n}, n in f["allowed"]))
    s = limits["steps"]
    for n in sorted({0, s["min"], s["max"], s["max"] + 1, 60, 61}):
        out.append(("steps", {"steps": n}, s["min"] <= n <= s["max"]))
    p = limits["fps"]
    for n in (p["min"] - 1, p["min"], p["max"], p["max"] + 1):
        out.append(("fps", {"fps": n}, p["min"] <= n <= p["max"]))
    g = limits["guidance_scale"]
    for x in (round(g["min"] - 0.1, 1), g["min"], g["max"], round(g["max"] + 0.1, 1)):
        out.append(("guidance_scale", {"guidance_scale": x}, g["min"] <= x <= g["max"]))
    sd = limits["seed"]
    for x in (sd["min"] - 1, sd["min"], sd["max"], sd["max"] + 1):
        out.append(("seed", {"seed": x}, sd["min"] <= x <= sd["max"]))
    return out


@pytest.mark.parametrize("name", list(PROFILES))
def test_post_accepts_exactly_what_the_limits_advertise(make_client, name):
    c = make_client(**PROFILES[name])
    limits = limits_of(c)
    for dimension, over, expected in variants(limits):
        fields = base_fields(limits, **over)
        if dimension in ("resolution", "num_frames", "steps", "fps", "guidance_scale", "seed") and not any(
                m["enabled"] for m in limits["modes"]):
            continue
        r = post(c, fields)
        assert (r.status_code == 202) is expected, (name, dimension, over, r.status_code, r.text[:200])
        if not expected:
            assert r.status_code == 422 and dimension in [e["loc"][-1] for e in r.json()["detail"]], (name, dimension, r.text[:200])


@pytest.mark.parametrize("name", list(PROFILES))
def test_every_default_is_accepted_by_post(make_client, name):
    c = make_client(**PROFILES[name])
    limits = limits_of(c)
    assert post(c, base_fields(limits)).status_code == 202


@pytest.mark.parametrize("name", list(PROFILES))
def test_every_default_is_valid_under_its_own_active_limits(make_client, name):
    limits = limits_of(make_client(**PROFILES[name]))
    d = limits["defaults"]
    assert d["mode"] in [m["value"] for m in limits["modes"] if m["enabled"]]
    assert d["resolution"] in [r["value"] for r in limits["resolutions"] if r["enabled"]]
    assert d["num_frames"] in limits["num_frames"]["allowed"] and limits["steps"]["min"] <= d["steps"] <= limits["steps"]["max"]
    assert limits["fps"]["min"] <= d["fps"] <= limits["fps"]["max"]
    assert limits["guidance_scale"]["min"] <= d["guidance_scale"] <= limits["guidance_scale"]["max"]
    verify_limits(limits)


@pytest.mark.parametrize("name", list(PROFILES))
def test_boundary_values_of_every_numeric_limit_behave_exactly_as_advertised(make_client, name):
    c = make_client(**PROFILES[name])
    limits = limits_of(c)
    for key, field in (("steps", "steps"), ("fps", "fps"), ("guidance_scale", "guidance_scale")):
        lo, hi = limits[key]["min"], limits[key]["max"]
        assert post(c, base_fields(limits, **{field: lo})).status_code == 202 and post(c, base_fields(limits, **{field: hi})).status_code == 202
        step = 0.1 if key == "guidance_scale" else 1
        assert post(c, base_fields(limits, **{field: round(lo - step, 1)})).status_code == 422
        assert post(c, base_fields(limits, **{field: round(hi + step, 1)})).status_code == 422
    allowed = limits["num_frames"]["allowed"]
    assert allowed == list(range(limits["num_frames"]["min"], limits["num_frames"]["max"] + 1, FRAME_STEP))
    assert post(c, base_fields(limits, num_frames=allowed[0])).status_code == 202 and post(c, base_fields(limits, num_frames=allowed[-1])).status_code == 202


def test_every_supported_and_unsupported_mode_is_described_and_enforced(make_client):
    c = make_client(generation_profile="t4_verified")
    limits = limits_of(c)
    assert {m["value"] for m in limits["modes"]} == {"t2v", "i2v"}                       # every mode the API knows is listed
    assert post(c, base_fields(limits, mode="t2v")).status_code == 202
    r = post(c, base_fields(limits, mode="i2v"))
    assert r.status_code == 422 and "mode" in [e["loc"][-1] for e in r.json()["detail"]]
    r = c.post("/api/jobs", data={"mode": "v2v", "prompt": "x", **{k: str(v) for k, v in base_fields(limits).items() if k != "mode"}})
    assert r.status_code == 422                                                           # not listed, not accepted


def test_the_apis_own_field_defaults_may_exceed_a_restricted_servers_limits_so_clients_send_explicit_values(make_client):
    c = make_client(generation_profile="t4_verified")
    r = c.post("/api/jobs", data={"mode": "t2v", "prompt": "a fox"})                     # omitted fields take the API defaults (49, 30)
    assert r.status_code == 422 and {"num_frames", "steps"} <= {e["loc"][-1] for e in r.json()["detail"]}
    assert limits_of(c)["defaults"]["num_frames"] == 17                                   # the limits' defaults ARE valid


# ====================================================================== independent of the GPU
def test_the_endpoint_never_contacts_the_inference_provider():
    fake = FakeComfyUI(b"x" * 100).start()
    try:
        import tempfile
        from pathlib import Path
        settings = make_settings(Path(tempfile.mkdtemp()), inference_provider="comfyui", comfyui_base_url=fake.url)
        with TestClient(create_app(settings)) as c:
            before = len(fake.requests)
            for _ in range(5):
                assert c.get("/api/limits").status_code == 200
            assert len(fake.requests) == before                                           # no ComfyUI request at all
    finally:
        fake.stop()


def test_the_endpoint_works_when_the_gpu_host_is_down_and_matches_the_mock_answer(make_client):
    from test_inference_proxy import free_port
    down = make_client(inference_provider="comfyui", comfyui_base_url=f"http://127.0.0.1:{free_port()}", comfyui_timeout=2)
    t0 = time.monotonic()
    body = limits_of(down)
    assert time.monotonic() - t0 < 1.0 and body == DEFAULT_LIMITS
    assert down.get("/api/health").json()["status"] == "degraded"                         # health notices; limits do not care


# ====================================================================== nothing sensitive
SECRETS = {"token": "T0kenValue" + "9" * 30, "s3": "S3SecretKey" + "7" * 20, "host": "gpu-secret-host.example.com"}


def test_no_credential_url_path_or_infrastructure_detail_can_appear(make_client, tmp_path):
    c = make_client(inference_provider="comfyui", comfyui_base_url=f"https://{SECRETS['host']}", comfyui_api_token=SECRETS["token"],
                    s3_access_key_id=SECRETS["s3"], s3_secret_access_key=SECRETS["s3"] + "x", s3_endpoint_url="https://s3.internal.example.com",
                    database_url=f"sqlite:///{tmp_path}/secret-dir/vidsgen.db", storage_local_root=tmp_path / "secret-storage",
                    generation_profile="t4_verified")
    r = c.get("/api/limits")
    text = r.text
    for needle in (*SECRETS.values(), "secret-dir", "secret-storage", str(tmp_path), "internal.example.com", "https://", "http://", "gpu-secret",
                   "comfyui", "token", "password", "Bearer"):
        assert needle.lower() not in text.lower(), needle
    strings = []

    def walk(v):
        if isinstance(v, dict):
            for k, x in v.items():
                strings.append(k)
                walk(x)
        elif isinstance(v, list):
            for x in v:
                walk(x)
        elif isinstance(v, str):
            strings.append(v)
    walk(r.json())
    assert not [s for s in strings if "://" in s or s.startswith(("/", "~", ".")) or "\\" in s], [s for s in strings if "://" in s or s.startswith("/")]
    assert [s for s in strings if "/" in s] == ["image/jpeg", "image/png", "image/webp"]    # the only slashes are the three MIME types


def test_the_set_of_fields_is_exactly_the_documented_one(client):
    body = limits_of(client)
    assert set(body) == {"profile", "restricted", "max_concurrent_jobs", "modes", "resolutions", "num_frames", "steps", "fps", "guidance_scale",
                         "seed", "prompt", "image", "defaults"}
    assert set(body["modes"][0]) == {"value", "label", "enabled", "reason"} and set(body["resolutions"][0]) == {"value", "width", "height", "enabled", "reason"}
    assert set(body["num_frames"]) == {"min", "max", "step", "allowed", "limited", "reason"} and set(body["steps"]) == {"min", "max", "limited", "reason"}
    assert set(body["defaults"]) == {"mode", "resolution", "num_frames", "steps", "fps", "guidance_scale"} and set(body["image"]) == set(IMAGE)


def test_the_endpoint_is_read_only_and_ignores_query_parameters(client):
    for method in ("post", "put", "patch", "delete"):
        assert getattr(client, method)("/api/limits").status_code == 405
    assert client.get("/api/limits?profile=t4_verified&max_frames=17").json() == DEFAULT_LIMITS


def test_the_response_is_deterministic_json(client):
    a, b = client.get("/api/limits"), client.get("/api/limits")
    assert a.content == b.content and a.headers["content-type"] == "application/json"


# ====================================================================== invalid or unsupported configuration
BAD_ENVELOPES = {
    "unknown mode": GenerationEnvelope(modes=("v2v",)),
    "no mode enabled": GenerationEnvelope(modes=()),
    "unknown resolution": GenerationEnvelope(resolutions=("1x1",)),
    "no resolution enabled": GenerationEnvelope(resolutions=()),
    "frames below the minimum": GenerationEnvelope(max_frames=5),
    "zero frames": GenerationEnvelope(max_frames=0),
    "zero steps": GenerationEnvelope(max_steps=0),
    "negative steps": GenerationEnvelope(max_steps=-3),
}


@pytest.mark.parametrize("name", list(BAD_ENVELOPES))
def test_inconsistent_limits_are_an_error_never_permissive_numbers(name):
    with pytest.raises(LimitsError):
        build_limits(BAD_ENVELOPES[name], Settings(_env_file=None))


def test_the_endpoint_reports_a_misconfiguration_instead_of_guessing(client):
    client.app.state.manager.envelope = GenerationEnvelope(max_frames=5)
    r = client.get("/api/limits")
    assert r.status_code == 500 and r.json() == {"detail": {"code": "limits_unavailable", "message": "The server's generation limits are misconfigured."}}
    assert "num_frames" not in r.text                                                     # no numbers at all


def test_a_server_with_inconsistent_limits_refuses_to_start(make_client, monkeypatch):
    import app.jobs.manager as manager_module
    monkeypatch.setattr(manager_module.GenerationEnvelope, "from_settings", classmethod(lambda cls, s: GenerationEnvelope(modes=())))
    with pytest.raises(LimitsError):
        make_client()


@pytest.mark.parametrize("bad", [dict(generation_allowed_resolutions="1920x1080"), dict(generation_allowed_modes="t2v,v2v"),
                                 dict(generation_max_frames=5), dict(generation_max_steps=0), dict(generation_profile="h100")])
def test_unsupported_configuration_is_refused_when_the_settings_are_loaded(bad):
    with pytest.raises(Exception):
        Settings(_env_file=None, **bad)


@pytest.mark.parametrize("tamper", [
    lambda L: L["defaults"].update(num_frames=49),                        # not allowed under t4
    lambda L: L["defaults"].update(resolution="1280x704"),                # disabled
    lambda L: L["defaults"].update(mode="i2v"),                           # disabled
    lambda L: L["defaults"].update(steps=30),                             # above the maximum
    lambda L: L["defaults"].update(fps=60),
    lambda L: L["defaults"].update(guidance_scale=20.0),
    lambda L: L["num_frames"].update(allowed=[]),
    lambda L: L["num_frames"].update(allowed=[18]),
    lambda L: L["modes"].__setitem__(0, {**L["modes"][0], "enabled": False}),
    lambda L: [r.update(enabled=False) for r in L["resolutions"]],
])
def test_verify_limits_rejects_a_default_its_own_limits_would_refuse(tamper):
    limits = copy.deepcopy(T4_LIMITS)
    verify_limits(limits)
    tamper(limits)
    with pytest.raises(LimitsError):
        verify_limits(limits)


# ====================================================================== contract synchronisation
def test_the_static_frontend_constraints_agree_with_the_default_limits(client):
    static = json.loads((REPO_ROOT / "frontend" / "src" / "api" / "constraints.json").read_text())
    live = limits_of(client)
    assert [r["value"] for r in static["resolutions"]] == [r["value"] for r in live["resolutions"]]
    assert [(r["width"], r["height"]) for r in static["resolutions"]] == [(r["width"], r["height"]) for r in live["resolutions"]]
    assert static["defaultResolution"] == live["defaults"]["resolution"]
    assert static["numFrames"] == {"min": live["num_frames"]["min"], "max": live["num_frames"]["max"], "step": live["num_frames"]["step"],
                                   "default": live["defaults"]["num_frames"]}
    assert static["steps"] == {"default": live["defaults"]["steps"], "min": live["steps"]["min"], "max": live["steps"]["max"]}
    assert static["fps"] == {"default": live["defaults"]["fps"], "min": live["fps"]["min"], "max": live["fps"]["max"]}
    assert static["guidanceScale"] == {"default": live["defaults"]["guidance_scale"], "min": live["guidance_scale"]["min"], "max": live["guidance_scale"]["max"]}
    assert static["seed"] == live["seed"] and static["prompt"]["maxChars"] == live["prompt"]["max_chars"]
    assert static["image"] == {"mimeTypes": live["image"]["mime_types"], "maxBytes": live["image"]["max_bytes"],
                               "minSide": live["image"]["min_side"], "maxSide": live["image"]["max_side"]}


def test_the_openapi_spec_documents_the_endpoint_and_its_schema(client):
    spec = client.get("/api/openapi.json").json()
    assert list(spec["paths"]["/api/limits"]) == ["get"]
    op = spec["paths"]["/api/limits"]["get"]
    assert op["responses"]["200"]["content"]["application/json"]["schema"]["$ref"].endswith("/LimitsOut")
    assert "500" in op["responses"]
    props = spec["components"]["schemas"]["LimitsOut"]["properties"]
    assert set(props) == {"profile", "restricted", "max_concurrent_jobs", "modes", "resolutions", "num_frames", "steps", "fps", "guidance_scale",
                          "seed", "prompt", "image", "defaults"}
    assert set(spec["components"]["schemas"]["LimitsOut"]["required"]) == set(props)


def test_the_committed_contract_files_include_the_endpoint():
    openapi = json.loads((REPO_ROOT / "backend" / "openapi.json").read_text())
    assert "/api/limits" in openapi["paths"] and "LimitsOut" in openapi["components"]["schemas"]
    ts = (REPO_ROOT / "frontend" / "src" / "api" / "schema.d.ts").read_text()
    assert '"/api/limits"' in ts and "LimitsOut: {" in ts and "num_frames: components" in ts and "defaults: components" in ts


def test_the_resolutions_listed_are_exactly_the_ones_the_api_knows(client):
    assert [r["value"] for r in limits_of(client)["resolutions"]] == list(SUPPORTED_RESOLUTIONS)
    assert limits_of(client)["num_frames"]["min"] == MIN_FRAMES


def test_the_builders_own_output_has_exactly_the_documented_keys_before_any_response_filtering():
    s = Settings(_env_file=None, generation_profile="t4_verified")
    built = build_limits(GenerationEnvelope.from_settings(s), s)
    assert set(built) == {"profile", "restricted", "max_concurrent_jobs", "modes", "resolutions", "num_frames", "steps", "fps", "guidance_scale",
                          "seed", "prompt", "image", "defaults"}
    assert built == T4_LIMITS
