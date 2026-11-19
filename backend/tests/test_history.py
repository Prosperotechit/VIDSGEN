from conftest import submit, wait_for


def test_empty_history(client):
    assert client.get("/api/jobs").json() == {"items": [], "total": 0, "limit": 20, "offset": 0}


def test_history_newest_first_with_pagination(client):
    ids = [submit(client, prompt=f"prompt number {i}").json()["id"] for i in range(5)]
    for i in ids:
        wait_for(client, i)

    page1 = client.get("/api/jobs", params={"limit": 2}).json()
    assert page1["total"] == 5
    assert [j["id"] for j in page1["items"]] == ids[::-1][:2]
    page3 = client.get("/api/jobs", params={"limit": 2, "offset": 4}).json()
    assert [j["id"] for j in page3["items"]] == [ids[0]]
    assert page3["items"][0]["params"]["prompt"] == "prompt number 0"


def test_history_filter_by_status(client):
    ok = submit(client).json()["id"]
    bad = submit(client, prompt="z [mock:fail]").json()["id"]
    wait_for(client, ok), wait_for(client, bad)
    failed = client.get("/api/jobs", params={"status": "failed"}).json()
    assert failed["total"] == 1 and failed["items"][0]["id"] == bad
    completed = client.get("/api/jobs", params={"status": "completed"}).json()
    assert [j["id"] for j in completed["items"]] == [ok]


def test_history_invalid_query(client):
    assert client.get("/api/jobs", params={"status": "exploded"}).status_code == 422
    assert client.get("/api/jobs", params={"limit": 0}).status_code == 422
    assert client.get("/api/jobs", params={"limit": 101}).status_code == 422
    assert client.get("/api/jobs", params={"offset": -1}).status_code == 422


def test_history_persists_across_restart(make_client):
    c1 = make_client()
    job_id = submit(c1).json()["id"]
    wait_for(c1, job_id)
    c1.__exit__(None, None, None)

    c2 = make_client()  # same tmp storage + DB
    job = c2.get(f"/api/jobs/{job_id}").json()
    assert job["status"] == "completed"
    assert c2.get(f"/api/jobs/{job_id}/download").status_code == 200
