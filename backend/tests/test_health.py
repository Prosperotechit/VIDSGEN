def test_health_ok(client):
    r = client.get("/api/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["environment"] == "test"
    assert body["provider"] == {"name": "mock", "ok": True, "detail": "mock provider ready (no GPU)"}
    assert body["storage"]["name"] == "local" and body["storage"]["ok"] is True
    assert set(body["jobs"]) == {"queued", "processing", "completed", "failed", "cancelled"}


def test_health_reports_degraded_provider(client):
    from app.providers.base import ProviderHealth

    async def offline():
        return ProviderHealth(False, "GPU host unreachable")

    client.app.state.manager.provider.health = offline
    body = client.get("/api/health").json()
    assert body["status"] == "degraded"
    assert body["provider"]["ok"] is False


def test_swagger_and_openapi_available(client):
    assert client.get("/api/docs").status_code == 200
    spec = client.get("/api/openapi.json").json()
    for path in ["/api/health", "/api/jobs", "/api/jobs/{job_id}", "/api/jobs/{job_id}/download",
                 "/api/jobs/{job_id}/cancel"]:
        assert path in spec["paths"]
    r = client.get("/", follow_redirects=False)
    assert r.status_code == 307 and r.headers["location"] == "/api/docs"


def test_cors_allowed_origin(client):
    r = client.options("/api/jobs", headers={
        "Origin": "http://localhost:5173", "Access-Control-Request-Method": "POST"})
    assert r.status_code == 200
    assert r.headers["access-control-allow-origin"] == "http://localhost:5173"


def test_cors_disallowed_origin(client):
    r = client.options("/api/jobs", headers={
        "Origin": "https://evil.example", "Access-Control-Request-Method": "POST"})
    assert "access-control-allow-origin" not in r.headers
    r = client.get("/api/health", headers={"Origin": "https://evil.example"})
    assert "access-control-allow-origin" not in r.headers


def test_cors_origins_from_settings(make_client):
    c = make_client(cors_origins="https://app.example.com, https://staging.example.com")
    r = c.get("/api/health", headers={"Origin": "https://staging.example.com"})
    assert r.headers["access-control-allow-origin"] == "https://staging.example.com"
