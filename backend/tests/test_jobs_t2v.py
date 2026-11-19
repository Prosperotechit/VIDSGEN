import time

from conftest import probe_video, submit, wait_for


def test_submit_t2v_returns_202_and_queued_job(client):
    r = submit(client, prompt="  A drone shot over a coral reef  ", seed=1234, resolution="1280x704",
               num_frames=49, steps=20, guidance_scale=6.5, negative_prompt="blurry")
    assert r.status_code == 202
    job = r.json()
    assert job["id"].startswith("job_")
    assert job["mode"] == "t2v"
    assert job["status"] in ("queued", "processing")
    assert job["provider"] == "mock"
    assert job["created_at"] and job["completed_at"] is None
    assert job["result"] is None and job["error"] is None and job["input_image"] is None
    p = job["params"]
    assert p["prompt"] == "A drone shot over a coral reef"          # trimmed
    assert (p["width"], p["height"], p["num_frames"], p["fps"]) == (1280, 704, 49, 24)
    assert p["duration_seconds"] == round(49 / 24, 2)
    assert (p["steps"], p["guidance_scale"], p["seed"], p["negative_prompt"]) == (20, 6.5, 1234, "blurry")


def test_seed_is_generated_and_recorded_when_omitted(client):
    job = submit(client).json()
    assert isinstance(job["params"]["seed"], int)
    assert client.get(f"/api/jobs/{job['id']}").json()["params"]["seed"] == job["params"]["seed"]


def test_unique_job_ids(client):
    ids = {submit(client).json()["id"] for _ in range(5)}
    assert len(ids) == 5


def test_t2v_completes_with_timestamps(client):
    job_id = submit(client).json()["id"]
    job = wait_for(client, job_id)
    assert job["status"] == "completed", job["error"]
    assert job["progress"] == 100.0
    assert job["created_at"] <= job["started_at"] <= job["completed_at"]
    assert job["result"]["download_url"] == f"/api/jobs/{job_id}/download"
    assert job["result"]["preview_url"] == f"/api/jobs/{job_id}/download?inline=true"
    assert job["result"]["size_bytes"] > 0


def test_progress_advances_monotonically(make_client):
    client = make_client(mock_step_delay_seconds=0.03)
    job_id = submit(client, steps=30).json()["id"]
    seen, statuses = [], []
    deadline = time.monotonic() + 20
    while time.monotonic() < deadline:
        j = client.get(f"/api/jobs/{job_id}").json()
        seen.append(j["progress"])
        statuses.append(j["status"])
        if j["status"] in ("completed", "failed", "cancelled"):
            break
        time.sleep(0.01)
    assert statuses[-1] == "completed"
    assert "processing" in statuses
    assert seen == sorted(seen), "progress must never go backwards"
    assert any(0 < p < 100 for p in seen), "intermediate progress should be visible"
    assert max(p for p, s in zip(seen, statuses) if s != "completed") < 100  # 100 only once stored


def test_download_returns_valid_mp4_matching_request(client, tmp_path):
    job_id = submit(client, resolution="832x480", num_frames=21, fps=24).json()["id"]
    assert wait_for(client, job_id)["status"] == "completed"
    r = client.get(f"/api/jobs/{job_id}/download")
    assert r.status_code == 200
    assert r.headers["content-type"] == "video/mp4"
    assert r.headers["content-disposition"] == f'attachment; filename="vidsgen-{job_id}.mp4"'
    assert r.content[4:8] == b"ftyp"
    info = probe_video(r.content, tmp_path)
    assert (info["width"], info["height"], info["frames"]) == (832, 480, 21)
    assert info["fps"] == 24


def test_preview_inline_and_range_requests(client):
    job_id = submit(client).json()["id"]
    wait_for(client, job_id)
    r = client.get(f"/api/jobs/{job_id}/download", params={"inline": "true"})
    assert r.headers["content-disposition"].startswith("inline")
    full = r.content
    part = client.get(f"/api/jobs/{job_id}/download", headers={"Range": "bytes=0-99"})
    assert part.status_code == 206
    assert part.content == full[:100]
    assert part.headers["content-range"] == f"bytes 0-99/{len(full)}"


def test_output_stored_under_storage_key_not_in_api(client):
    job_id = submit(client).json()["id"]
    body = wait_for(client, job_id)
    manager = client.app.state.manager
    job = manager.get(job_id)
    assert job.output_key == f"outputs/{job_id}/video.mp4"
    path = manager.storage.local_path(job.output_key)
    assert path.is_file() and manager.storage.root in path.parents
    assert "output_key" not in str(body) and str(manager.storage.root) not in str(body)
