import pytest

from conftest import submit


@pytest.mark.parametrize("fields,bad_field", [
    ({"prompt": ""}, "prompt"),
    ({"prompt": "    "}, "prompt"),
    ({"prompt": "bad\x00prompt"}, "prompt"),
    ({"prompt": "x" * 2001}, "prompt"),
    ({"negative_prompt": "y" * 2001}, "negative_prompt"),
    ({"mode": "v2v"}, "mode"),
    ({"resolution": "1920x1080"}, "resolution"),
    ({"resolution": "abc"}, "resolution"),
    ({"num_frames": 50}, "num_frames"),       # not 4k+1
    ({"num_frames": 13}, "num_frames"),       # below minimum
    ({"num_frames": 125}, "num_frames"),      # above maximum
    ({"num_frames": "ten"}, "num_frames"),
    ({"steps": 0}, "steps"),
    ({"steps": 61}, "steps"),
    ({"guidance_scale": 0.5}, "guidance_scale"),
    ({"fps": 60}, "fps"),
    ({"seed": -1}, "seed"),
    ({"seed": 2**32}, "seed"),
])
def test_invalid_parameters_rejected(client, fields, bad_field):
    r = submit(client, **fields)
    assert r.status_code == 422, r.text
    locs = [e["loc"][-1] for e in r.json()["detail"]]
    assert bad_field in locs
    assert client.get("/api/jobs").json()["total"] == 0


def test_missing_required_fields(client):
    r = client.post("/api/jobs", data={"mode": "t2v"})
    assert r.status_code == 422
    assert ["body", "prompt"] in [e["loc"] for e in r.json()["detail"]]


def test_prompt_limit_is_configurable(make_client):
    client = make_client(max_prompt_chars=20)
    assert submit(client, prompt="x" * 21).status_code == 422
    assert submit(client, prompt="x" * 20).status_code == 202


def test_valid_edge_values_accepted(client):
    for frames in (17, 121):
        assert submit(client, num_frames=frames).status_code == 202
    for res in ("1280x704", "704x1280", "832x480", "480x832", "640x640", "1280X704"):
        assert submit(client, resolution=res).status_code == 202


def test_oversized_request_rejected_before_reading_body(make_client):
    client = make_client(max_upload_mb=0.01)
    body = b"x" * (2 * 1024 * 1024)
    r = client.post("/api/jobs", content=body, headers={"Content-Type": "multipart/form-data; boundary=x"})
    assert r.status_code == 413
    assert r.json()["detail"]["code"] == "request_too_large"
