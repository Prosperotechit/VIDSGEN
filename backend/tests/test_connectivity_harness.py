"""The Stage 4A connectivity harness (scripts/connectivity_harness.py): authentication, WebSocket, streaming, Range, interruption,
timeouts, cleanup and the 30-minute lifetime cap, all on loopback with a dummy upstream. No GPU, no Kaggle, no tunnel, no external network."""

from __future__ import annotations

import asyncio
import importlib.util
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

from app.core.config import REPO_ROOT

SCRIPT = REPO_ROOT / "scripts" / "connectivity_harness.py"
sys.path.insert(0, str(REPO_ROOT / "scripts"))
_spec = importlib.util.spec_from_file_location("connectivity_harness", SCRIPT)
ch = importlib.util.module_from_spec(_spec)
sys.modules["connectivity_harness"] = ch
_spec.loader.exec_module(ch)
import probe_upstream as up  # noqa: E402

WORKFLOWS = REPO_ROOT / "workflows" / "comfyui"
SMALL = ["--download-mib", "4", "--ws-idle-seconds", "1", "--burst", "12", "--timeout", "30"]


def run_harness(tmp, *extra, timeout=240, lifetime="120"):
    return subprocess.run([sys.executable, str(SCRIPT), "run-local", "--run-dir", str(tmp / "run"), "--report", str(tmp / "report.json"),
                           "--lifetime-seconds", lifetime, *SMALL, *extra], capture_output=True, text=True, timeout=timeout)


def wait_until(pred, timeout=15.0) -> bool:
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if pred():
            return True
        time.sleep(0.1)
    return False


@pytest.fixture(scope="module")
def full_run(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("harness")
    r = run_harness(tmp)
    report = json.loads((tmp / "report.json").read_text())
    return tmp, r, report


# ====================================================================== a complete local run
def test_a_full_local_run_passes_every_check(full_run):
    tmp, r, report = full_run
    assert r.returncode == 0, r.stdout + r.stderr
    assert report["verdict"] == "PASS" and report["failed_checks"] == []
    names = [c["name"] for c in report["checks"]]
    assert names == ["unauthenticated_rejected", "authenticated_ok", "route_allowlist", "submit_and_progress_over_websocket", "idle_websocket_survives",
                     "streamed_download_integrity", "range_requests", "concurrency_burst", "duplicate_submission_replay", "headers_hygiene",
                     "invalid_token_throttle", "timeout_handling", "proxy_interruption_recovery", "duplicate_after_restart_not_rerun", "clean_teardown"]
    assert all(c["ok"] for c in report["checks"]), [c for c in report["checks"] if not c["ok"]]
    assert "CONNECTIVITY VERDICT: PASS" in r.stdout


def test_the_run_covers_the_required_scenarios(full_run):
    _, _, report = full_run
    by = {c["name"]: c for c in report["checks"]}
    assert "identical bodies: True" in by["unauthenticated_rejected"]["detail"]                      # authentication
    assert "20 progress steps" in by["submit_and_progress_over_websocket"]["detail"]                 # WebSocket
    assert "sha256 matches" in by["streamed_download_integrity"]["detail"]                           # streaming, verified bit for bit
    assert by["range_requests"]["detail"] == "206/206/416/416"                                       # Range
    assert "websocket closed: True" in by["proxy_interruption_recovery"]["detail"]                   # interruption and recovery
    assert "recovered after restart: True" in by["proxy_interruption_recovery"]["detail"]
    assert "client timed out: True" in by["timeout_handling"]["detail"]                              # timeouts
    assert "existing-unverified" in by["duplicate_after_restart_not_rerun"]["detail"]                # duplicate protection after a restart
    assert report["config"]["bind"] == "127.0.0.1 only" and report["config"]["lifetime_seconds"] == 120
    assert report["tear_down"] == {"proxy_port_closed": True, "upstream_port_closed": True}


def test_the_harness_leaves_nothing_running_and_no_pid_file(full_run):
    tmp, _, _ = full_run
    assert not (tmp / "run" / "pids.json").exists()
    assert not subprocess.run(["pgrep", "-f", f"{tmp}/run"], capture_output=True).stdout.strip()


def test_the_report_logs_and_output_never_contain_the_token(full_run):
    tmp, r, report = full_run
    blob = r.stdout + r.stderr + json.dumps(report) + "".join(p.read_text(errors="replace") for p in (tmp / "run").glob("*.log"))
    assert "Bearer " not in blob and "PROXY_API_TOKEN" not in blob and "Authorization" not in blob


# ====================================================================== process hygiene, teardown and the lifetime cap
@pytest.mark.parametrize("value", ["0", "-5", "1801", "99999", "abc"])
def test_a_lifetime_beyond_thirty_minutes_is_refused(tmp_path, value):
    r = subprocess.run([sys.executable, str(SCRIPT), "run-local", "--run-dir", str(tmp_path / "r"), "--report", str(tmp_path / "x.json"),
                        "--lifetime-seconds", value], capture_output=True, text=True, timeout=60)
    assert r.returncode == 2 and not (tmp_path / "r" / "pids.json").exists() and not (tmp_path / "x.json").exists()


def test_the_default_lifetime_is_the_thirty_minute_maximum():
    assert ch.HARD_MAX_LIFETIME == 1800.0
    p = subprocess.run([sys.executable, str(SCRIPT), "run-local", "--help"], capture_output=True, text=True)
    assert "maximum 1800" in p.stdout


def test_when_the_lifetime_expires_everything_is_stopped_and_the_run_fails(tmp_path):
    started = time.monotonic()
    r = run_harness(tmp_path, "--ws-idle-seconds", "60", lifetime="4")
    assert r.returncode == 3 and "LIFETIME EXCEEDED" in r.stderr and time.monotonic() - started < 60
    report = json.loads((tmp_path / "report.json").read_text())
    assert report["verdict"] == "FAIL" and report["failed_checks"] == ["lifetime exceeded"]
    assert not (tmp_path / "run" / "pids.json").exists()
    assert wait_until(lambda: not subprocess.run(["pgrep", "-f", f"{tmp_path}/run"], capture_output=True).stdout.strip(), 10)


def start_long_run(tmp):
    proc = subprocess.Popen([sys.executable, str(SCRIPT), "run-local", "--run-dir", str(tmp / "run"), "--report", str(tmp / "report.json"),
                             "--lifetime-seconds", "300", "--download-mib", "1", "--ws-idle-seconds", "120", "--burst", "5"],
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env={**os.environ, "AWS_ACCESS_KEY_ID": "AKIA" + "Z" * 16,
                                                                                    "GITHUB_TOKEN": "ghp_" + "y" * 30})
    pids = tmp / "run" / "pids.json"
    assert wait_until(pids.exists, 30)
    assert wait_until(lambda: {"proxy", "upstream"} <= set(json.loads(pids.read_text())), 30)
    data = json.loads(pids.read_text())
    assert wait_until(lambda: ch.port_open(data["ports"]["proxy"]) and ch.port_open(data["ports"]["upstream"]), 30)
    return proc, data


def test_children_get_a_minimal_environment_and_the_token_is_not_on_any_command_line(tmp_path):
    proc, data = start_long_run(tmp_path)
    try:
        proxy_env = Path(f"/proc/{data['proxy']}/environ").read_bytes().decode(errors="replace")
        upstream_env = Path(f"/proc/{data['upstream']}/environ").read_bytes().decode(errors="replace")
        assert "AKIA" not in proxy_env + upstream_env and "ghp_" not in proxy_env + upstream_env      # no ambient credential reached a child
        assert "PROXY_API_TOKEN=" in proxy_env and "PROXY_API_TOKEN" not in upstream_env               # only the proxy gets its token
        token = proxy_env.split("PROXY_API_TOKEN=", 1)[1].split("\0", 1)[0]
        for pid in (data["proxy"], data["upstream"], proc.pid):
            assert token not in ch._cmdline(pid)
    finally:
        proc.send_signal(signal.SIGTERM)
        out = proc.communicate(timeout=30)[0].decode(errors="replace")
    assert token not in out


def test_every_service_listens_on_loopback_only(tmp_path):
    from test_wan22_gpu_test import g  # noqa: F401  (loads scripts/wan22_gpu_test.py for its /proc parser)
    gt = sys.modules["wan22_gpu_test"]
    proc, data = start_long_run(tmp_path)
    try:
        for port in data["ports"].values():
            addrs = gt.listening_addresses(port)
            assert addrs and all(gt.is_loopback(a) for a in addrs), (port, addrs)
    finally:
        proc.send_signal(signal.SIGTERM)
        proc.communicate(timeout=30)


def test_sigterm_tears_everything_down(tmp_path):
    proc, data = start_long_run(tmp_path)
    proc.send_signal(signal.SIGTERM)
    proc.communicate(timeout=30)
    assert proc.returncode == 128 + signal.SIGTERM
    assert wait_until(lambda: not ch.port_open(data["ports"]["proxy"]) and not ch.port_open(data["ports"]["upstream"]), 15)
    assert not ch.is_ours(data["proxy"]) and not ch.is_ours(data["upstream"]) and not (tmp_path / "run" / "pids.json").exists()


def test_children_die_with_a_killed_harness_and_teardown_cleans_up_the_rest(tmp_path):
    proc, data = start_long_run(tmp_path)
    proc.kill()                                                                    # the worst case: no handler runs
    proc.communicate(timeout=30)
    wait_until(lambda: not ch.is_ours(data["proxy"]) and not ch.is_ours(data["upstream"]), 20)   # (PDEATHSIG is best effort; teardown is the guarantee)
    r = subprocess.run([sys.executable, str(SCRIPT), "teardown", "--run-dir", str(tmp_path / "run")], capture_output=True, text=True, timeout=60)
    assert r.returncode == 0 and "teardown complete" in r.stdout
    assert not ch.is_ours(data["proxy"]) and not ch.is_ours(data["upstream"])
    assert wait_until(lambda: not ch.port_open(data["ports"]["proxy"]) and not ch.port_open(data["ports"]["upstream"]), 10)
    assert not (tmp_path / "run" / "pids.json").exists()


def test_teardown_with_nothing_to_do_is_harmless_and_never_kills_a_foreign_process(tmp_path):
    r = subprocess.run([sys.executable, str(SCRIPT), "teardown", "--run-dir", str(tmp_path / "none")], capture_output=True, text=True)
    assert r.returncode == 0 and "nothing to tear down" in r.stdout
    victim = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])             # NOT one of our scripts
    try:
        (tmp_path / "run").mkdir()
        (tmp_path / "run" / "pids.json").write_text(json.dumps({"proxy": victim.pid, "upstream": victim.pid}))
        r = subprocess.run([sys.executable, str(SCRIPT), "teardown", "--run-dir", str(tmp_path / "run")], capture_output=True, text=True)
        assert r.returncode == 0 and victim.poll() is None                         # a reused pid is left alone
    finally:
        victim.kill()
        victim.wait()


def test_a_service_that_fails_to_start_still_ends_with_a_clean_teardown(tmp_path):
    r = subprocess.run([sys.executable, str(SCRIPT), "run-local", "--run-dir", str(tmp_path / "run"), "--report", str(tmp_path / "report.json"),
                        "--workflows-dir", str(tmp_path / "no-such-dir"), *SMALL, "--lifetime-seconds", "60"], capture_output=True, text=True, timeout=120)
    report = json.loads((tmp_path / "report.json").read_text())
    assert r.returncode == 1 and report["verdict"] == "FAIL" and not (tmp_path / "run" / "pids.json").exists()
    assert report["tear_down"] == {"proxy_port_closed": True, "upstream_port_closed": True}


# ====================================================================== the probe command and its remote guard
def test_probe_refuses_a_remote_url_without_the_explicit_flag(monkeypatch, capsys):
    monkeypatch.setenv("PROXY_API_TOKEN", "t" * 40)
    for url in ("https://abc.trycloudflare.com", "http://192.168.1.5:8189", "https://example.com"):
        assert ch.main(["probe", "--url", url, "--expect-mib", "1"]) == 2
    assert "refusing a non-loopback URL" in capsys.readouterr().err


def test_probe_needs_a_token_in_the_environment_never_on_the_command_line(monkeypatch, capsys):
    monkeypatch.delenv("PROXY_API_TOKEN", raising=False)
    assert ch.main(["probe", "--url", "http://127.0.0.1:1", "--expect-mib", "1"]) == 2
    monkeypatch.setenv("PROXY_API_TOKEN", "short")
    assert ch.main(["probe", "--url", "http://127.0.0.1:1", "--expect-mib", "1"]) == 2
    capsys.readouterr()
    with pytest.raises(SystemExit):                                                # there is no --token option, and no abbreviation of --token-env
        ch.main(["probe", "--url", "http://127.0.0.1:1", "--expect-mib", "1", "--token", "x" * 40])
    assert "x" * 40 not in capsys.readouterr().err
    secret = "S" + "e" * 39
    assert ch.main(["probe", "--url", "http://127.0.0.1:1", "--expect-mib", "1", "--token-env", secret]) == 2     # a token pasted as the name
    assert secret not in capsys.readouterr().err


def test_probe_against_a_running_loopback_proxy_passes(tmp_path, monkeypatch):
    token = "p" * 43
    svc = ch.Services(tmp_path / "run", token, WORKFLOWS, download_mib=2, seed=9, lifetime=120)
    try:
        svc.start_upstream()
        svc.start_proxy()
        svc.wait_ready()
        monkeypatch.setenv("PROXY_API_TOKEN", token)
        rc = ch.main(["probe", "--url", svc.proxy_url, "--expect-mib", "2", "--expect-seed", "9", "--ws-idle-seconds", "0", "--burst", "6",
                      "--report", str(tmp_path / "probe.json")])
        report = json.loads((tmp_path / "probe.json").read_text())
        assert rc == 0 and report["verdict"] == "PASS" and report["config"]["target_host_class"] == "loopback"
        assert token not in json.dumps(report)
    finally:
        svc.stop_all()
    assert svc.all_stopped()


def test_a_wrong_expected_download_is_reported_as_a_failure(tmp_path, monkeypatch):
    token = "q" * 43
    svc = ch.Services(tmp_path / "run", token, WORKFLOWS, download_mib=2, seed=9, lifetime=120)
    try:
        svc.start_upstream(); svc.start_proxy(); svc.wait_ready()
        monkeypatch.setenv("PROXY_API_TOKEN", token)
        rc = ch.main(["probe", "--url", svc.proxy_url, "--expect-mib", "2", "--expect-seed", "10", "--ws-idle-seconds", "0", "--burst", "4",
                      "--report", str(tmp_path / "p.json")])                    # wrong seed: the bytes will not match
        report = json.loads((tmp_path / "p.json").read_text())
    finally:
        svc.stop_all()
    assert rc == 1 and "streamed_download_integrity" in report["failed_checks"] and "range_requests" in report["failed_checks"]


def test_checks_against_an_unreachable_target_fail_cleanly_without_raising_or_hanging():
    probe = ch.Probe("http://127.0.0.1:9", "z" * 43, WORKFLOWS, expect_bytes=1024, expect_seed=1, ws_idle=0, burst=3, timeout=2)
    t0 = time.monotonic()
    results = asyncio.run(probe.run_all())
    assert time.monotonic() - t0 < 60 and results and not any(r["ok"] for r in results)
    assert all("z" * 43 not in json.dumps(r) for r in results)                      # details carry exception types only


def test_a_report_that_somehow_contains_a_token_is_replaced_by_a_failure():
    report = ch.finish([{"name": "x", "ok": True, "detail": "Authorization: Bearer abcdefgh", "seconds": 0}], {}, time.monotonic())
    assert report["verdict"] == "FAIL" and report["failed_checks"] == ["report contained a token-like string"]


# ====================================================================== the probe workflow is the real, valid one
def test_the_probe_graph_is_accepted_by_the_proxys_strict_validator():
    import inference_proxy as px_mod  # noqa: F401
    from test_inference_proxy import px
    graph = ch.build_probe_graph(WORKFLOWS, ch.new_job_id())
    px.GraphValidator(WORKFLOWS, px.Policy()).validate(graph)
    assert px.PREFIX_RE.match(graph["58"]["inputs"]["filename_prefix"])


def test_the_harness_source_has_no_tunnel_and_no_public_binding():
    src = SCRIPT.read_text()
    for forbidden in ("cloudflared", "trycloudflare", "ngrok", "0.0.0.0"):
        assert forbidden not in src.lower(), forbidden
