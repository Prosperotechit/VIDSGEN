"""Lower-level tests of the job repository and manager (restart recovery, races)."""

from datetime import timedelta

from app.jobs.models import GenerationMode, GenerationParams, Job, JobStatus, new_job_id, utcnow
from app.jobs.repository import JobRepository
from conftest import make_settings, wait_for


def _job(status: JobStatus, minutes_ago: int = 0) -> Job:
    t = utcnow() - timedelta(minutes=minutes_ago)
    return Job(id=new_job_id(), mode=GenerationMode.t2v, status=status, provider="mock",
               params=GenerationParams(mode="t2v", prompt="recovered job", num_frames=17, steps=4),
               created_at=t, updated_at=t)


def test_transition_is_conditional(tmp_path):
    repo = JobRepository(tmp_path / "db.sqlite")
    job = _job(JobStatus.queued)
    repo.insert(job)
    assert repo.transition(job.id, [JobStatus.queued], status=JobStatus.cancelled)
    # A worker trying to start the cancelled job must lose the race.
    assert not repo.transition(job.id, [JobStatus.queued], status=JobStatus.processing)
    assert repo.get(job.id).status == JobStatus.cancelled
    repo.close()


def test_restart_recovery(tmp_path, make_client):
    settings = make_settings(tmp_path)
    repo = JobRepository(settings.database_path)
    interrupted, waiting = _job(JobStatus.processing, 2), _job(JobStatus.queued, 1)
    repo.insert(interrupted)
    repo.insert(waiting)
    repo.close()

    client = make_client()  # same DB: startup runs recovery
    j = client.get(f"/api/jobs/{interrupted.id}").json()
    assert j["status"] == "failed" and j["error"]["code"] == "interrupted"
    assert wait_for(client, waiting.id)["status"] == "completed"   # re-queued and processed


def test_concurrent_workers(make_client):
    client = make_client(inference_max_concurrent_jobs=3, mock_step_delay_seconds=0.02)
    from conftest import submit

    ids = [submit(client, steps=10).json()["id"] for _ in range(6)]
    assert all(wait_for(client, i)["status"] == "completed" for i in ids)
    assert client.get("/api/health").json()["jobs"]["completed"] == 6
