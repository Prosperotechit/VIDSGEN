import time

from conftest import submit, wait_for


# ---------------------------------------------------------------- failures
def test_provider_failure_marks_job_failed(client):
    job_id = submit(client, prompt="explode please [mock:fail]", steps=10).json()["id"]
    job = wait_for(client, job_id)
    assert job["status"] == "failed"
    assert job["error"] == {"code": "mock_failure", "message": "Simulated provider failure"}
    assert job["progress"] < 100
    assert job["completed_at"] is not None
    assert job["result"] is None
    r = client.get(f"/api/jobs/{job_id}/download")
    assert r.status_code == 409
    assert r.json()["detail"]["code"] == "job_not_completed"


def test_timeout_marks_job_failed(make_client):
    client = make_client(mock_step_delay_seconds=0.2, inference_timeout_seconds=0.3)
    job_id = submit(client, steps=20).json()["id"]
    job = wait_for(client, job_id)
    assert job["status"] == "failed"
    assert job["error"]["code"] == "timeout"


def test_worker_continues_after_failure(client):
    bad = submit(client, prompt="x [mock:fail]").json()["id"]
    good = submit(client).json()["id"]
    assert wait_for(client, bad)["status"] == "failed"
    assert wait_for(client, good)["status"] == "completed"


def test_unexpected_provider_crash_is_reported_safely(client):
    async def boom(request):
        raise RuntimeError("secret internal detail")

    client.app.state.manager.provider.submit = boom
    job = wait_for(client, submit(client).json()["id"])
    assert job["status"] == "failed"
    assert job["error"] == {"code": "internal_error", "message": "Unexpected error during generation."}


# ---------------------------------------------------------------- not found / not ready
def test_unknown_job_returns_404(client):
    for method, path in [("get", "/api/jobs/job_nope"), ("get", "/api/jobs/job_nope/download"),
                         ("post", "/api/jobs/job_nope/cancel")]:
        r = getattr(client, method)(path)
        assert r.status_code == 404
        assert r.json()["detail"]["code"] == "job_not_found"


def test_download_before_completion_returns_409(make_client):
    client = make_client(mock_step_delay_seconds=0.05)
    job_id = submit(client, steps=40).json()["id"]
    r = client.get(f"/api/jobs/{job_id}/download")
    assert r.status_code == 409
    client.post(f"/api/jobs/{job_id}/cancel")


def test_missing_output_file_returns_410(client):
    job_id = submit(client).json()["id"]
    wait_for(client, job_id)
    manager = client.app.state.manager
    manager.storage.local_path(manager.get(job_id).output_key).unlink()
    r = client.get(f"/api/jobs/{job_id}/download")
    assert r.status_code == 410
    assert r.json()["detail"]["code"] == "result_missing"


# ---------------------------------------------------------------- cancellation
def test_cancel_queued_job(make_client):
    client = make_client(mock_step_delay_seconds=0.05, inference_max_concurrent_jobs=1)
    running = submit(client, steps=40).json()["id"]          # occupies the only worker
    queued = submit(client).json()["id"]
    assert client.get(f"/api/jobs/{queued}").json()["status"] == "queued"

    r = client.post(f"/api/jobs/{queued}/cancel")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "cancelled" and body["completed_at"] is not None
    assert body["started_at"] is None                          # never started

    client.post(f"/api/jobs/{running}/cancel")
    time.sleep(0.3)
    assert client.get(f"/api/jobs/{queued}").json()["status"] == "cancelled"  # worker skipped it


def test_cancel_processing_job_stops_generation(make_client):
    client = make_client(mock_step_delay_seconds=0.05)
    job_id = submit(client, steps=60).json()["id"]
    job = wait_for(client, job_id, statuses={"processing"})
    while job["progress"] == 0:
        job = client.get(f"/api/jobs/{job_id}").json()

    r = client.post(f"/api/jobs/{job_id}/cancel")
    assert r.status_code == 200 and r.json()["status"] == "cancelled"
    progress_at_cancel = r.json()["progress"]

    time.sleep(0.4)  # the worker must not overwrite the cancellation
    job = client.get(f"/api/jobs/{job_id}").json()
    assert job["status"] == "cancelled"
    assert job["progress"] == progress_at_cancel
    assert job["result"] is None
    manager = client.app.state.manager
    assert manager.get(job_id).output_key is None
    assert not (manager.storage.root / "outputs" / job_id).exists()
    assert manager.provider._runs == {}  # provider resources released


def test_cancel_finished_job_returns_409(client):
    done = submit(client).json()["id"]
    wait_for(client, done)
    failed = submit(client, prompt="x [mock:fail]").json()["id"]
    wait_for(client, failed)
    for job_id, status in [(done, "completed"), (failed, "failed")]:
        r = client.post(f"/api/jobs/{job_id}/cancel")
        assert r.status_code == 409
        assert r.json()["detail"]["code"] == "job_not_cancellable"
        assert status in r.json()["detail"]["message"]


def test_cancel_twice_returns_409(make_client):
    client = make_client(mock_step_delay_seconds=0.05)
    job_id = submit(client, steps=40).json()["id"]
    assert client.post(f"/api/jobs/{job_id}/cancel").status_code == 200
    assert client.post(f"/api/jobs/{job_id}/cancel").status_code == 409
