#!/usr/bin/env python3
"""VIDSGEN connectivity harness (Phase 6C Stage 4A): CPU-only, loopback-only tests of the authenticated inference proxy against a
DUMMY ComfyUI (scripts/probe_upstream.py). No GPU, no model, no Kaggle, no tunnel, no external network.

  run-local   Start the dummy upstream and the real proxy (as separate processes, loopback only), run every probe check, simulate a
              proxy interruption and restart, write a JSON report, and ALWAYS tear everything down. The whole run is capped at
              --lifetime-seconds (maximum 1800 = 30 minutes): a watchdog then stops everything and the run fails.
  probe       Run the same checks against an already running proxy URL. Only loopback URLs are accepted unless --allow-remote is
              given; that flag exists for a FUTURE, separately approved tunnel test and must not be used before then.
  teardown    Stop whatever a previous run left behind (from <run-dir>/pids.json), verifying each pid is still one of our processes.

The bearer token is generated per run (run-local) or read from an environment variable (probe). It is never printed, logged, written
to the report or passed on a command line. Children get a minimal, allow-listed environment.
"""

from __future__ import annotations

import argparse
import asyncio
import atexit
import ctypes
import hashlib
import json
import os
import re
import resource
import secrets
import signal
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path
from urllib.parse import urlsplit

sys.path.insert(0, str(Path(__file__).resolve().parent))
import probe_upstream as up  # noqa: E402  (deterministic download bytes: the client verifies them independently)

REPO = Path(__file__).resolve().parents[1]
SCRIPTS = Path(__file__).resolve().parent
HARD_MAX_LIFETIME = 1800.0                       # 30 minutes: the most any run may live
LOOPBACK = ("127.0.0.1", "localhost", "::1")
OUR_SCRIPTS = ("inference_proxy.py", "probe_upstream.py")


# ===================================================================================== process management
def _die_with_parent() -> None:
    """Linux: ask the kernel to SIGTERM this child if the harness dies, even by SIGKILL (best effort; --max-lifetime-seconds is the backstop)."""
    try:
        ctypes.CDLL("libc.so.6", use_errno=True).prctl(1, signal.SIGTERM)       # PR_SET_PDEATHSIG
    except (OSError, AttributeError):
        pass


def free_port() -> int:
    import socket
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def port_open(port: int) -> bool:
    import socket
    with socket.socket() as s:
        s.settimeout(0.5)
        return s.connect_ex(("127.0.0.1", port)) == 0


def _cmdline(pid: int) -> str:
    try:
        return Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\0", b" ").decode(errors="replace")
    except OSError:
        return ""


def is_ours(pid: int) -> bool:
    """True only if `pid` is still a process running one of OUR scripts (guards against pid reuse)."""
    cmd = _cmdline(pid)
    return any(name in cmd for name in OUR_SCRIPTS)


def terminate(pid: int, wait: float = 5.0) -> bool:
    """SIGTERM the process group, then SIGKILL. Returns True once the process is gone."""
    if not is_ours(pid):
        return True
    try:
        group = os.getpgid(pid)
        os.killpg(group, signal.SIGTERM)
    except OSError:
        return True
    end = time.monotonic() + wait
    while time.monotonic() < end:
        if not is_ours(pid) or _zombie(pid):
            return True
        time.sleep(0.05)
    try:
        os.killpg(group, signal.SIGKILL)
    except OSError:
        pass
    end = time.monotonic() + 3
    while time.monotonic() < end:
        if not is_ours(pid) or _zombie(pid):
            return True
        time.sleep(0.05)
    return not is_ours(pid)


def _zombie(pid: int) -> bool:
    try:
        return Path(f"/proc/{pid}/stat").read_text().split(") ", 1)[1].split()[0] == "Z"
    except (OSError, IndexError):
        return True


class Services:
    """The dummy upstream and the proxy, as child processes of this harness. Loopback only."""

    def __init__(self, run_dir: Path, token: str, workflows_dir: Path, *, download_mib: float, seed: int, lifetime: float):
        self.run_dir, self.token, self.workflows_dir = Path(run_dir), token, Path(workflows_dir)
        self.download_mib, self.seed, self.lifetime = download_mib, seed, lifetime
        self.upstream_port, self.proxy_port = free_port(), free_port()
        self.procs: dict[str, subprocess.Popen] = {}
        self.run_dir.mkdir(parents=True, exist_ok=True)

    @property
    def proxy_url(self) -> str:
        return f"http://127.0.0.1:{self.proxy_port}"

    @property
    def upstream_url(self) -> str:
        return f"http://127.0.0.1:{self.upstream_port}"

    def _spawn(self, name: str, args: list[str], env: dict[str, str]) -> None:
        log = open(self.run_dir / f"{name}.log", "ab")
        try:
            self.procs[name] = subprocess.Popen([sys.executable, *args], env=env, stdout=log, stderr=subprocess.STDOUT,
                                                start_new_session=True, preexec_fn=_die_with_parent)
        finally:
            log.close()                                           # the child keeps its own copy
        self._write_pids()

    def _env(self, **extra: str) -> dict[str, str]:
        base = {k: os.environ[k] for k in ("PATH", "HOME", "LANG", "LC_ALL") if k in os.environ}      # an allow-list: no credential can pass
        return {**base, "PYTHONDONTWRITEBYTECODE": "1", **extra}

    def start_upstream(self) -> None:
        self._spawn("upstream", [str(SCRIPTS / "probe_upstream.py"), "--port", str(self.upstream_port), "--download-mib", str(self.download_mib),
                                 "--seed", str(self.seed), "--max-lifetime-seconds", str(self.lifetime + 30)], self._env())

    def start_proxy(self) -> None:
        self._spawn("proxy", [str(SCRIPTS / "inference_proxy.py"), "--upstream", self.upstream_url, "--port", str(self.proxy_port),
                              "--workflows-dir", str(self.workflows_dir), "--max-lifetime-seconds", str(self.lifetime + 30)],
                    self._env(PROXY_API_TOKEN=self.token))

    def _write_pids(self) -> None:
        (self.run_dir / "pids.json").write_text(json.dumps({"harness": os.getpid(), **{n: p.pid for n, p in self.procs.items()},
                                                           "ports": {"upstream": self.upstream_port, "proxy": self.proxy_port}}))

    def wait_ready(self, timeout: float = 60.0) -> None:
        import httpx
        deadline = time.monotonic() + timeout
        for name, url, headers in (("upstream", f"{self.upstream_url}/__probe_stats", {}),
                                   ("proxy", f"{self.proxy_url}/system_stats", {"Authorization": f"Bearer {self.token}"})):
            while True:
                if self.procs[name].poll() is not None:
                    raise RuntimeError(f"the {name} exited with code {self.procs[name].returncode}: " + self.log_tail(name))
                try:
                    if httpx.get(url, headers=headers, timeout=2).status_code == 200:
                        break
                except httpx.HTTPError:
                    pass
                if time.monotonic() > deadline:
                    raise RuntimeError(f"the {name} did not become ready in {timeout:.0f}s: " + self.log_tail(name))
                time.sleep(0.2)

    def log_tail(self, name: str, n: int = 15) -> str:
        try:
            text = (self.run_dir / f"{name}.log").read_text(errors="replace")
        except OSError:
            return ""
        return text.replace(self.token, "[token]")[-1500:] if text else ""

    def kill_proxy(self) -> None:
        proc = self.procs.pop("proxy", None)
        if proc is not None:
            terminate_proc(proc, hard=True)

    def stop_all(self) -> None:
        for name in ("proxy", "upstream"):                         # the gate first, then what it protects
            proc = self.procs.pop(name, None)
            if proc is not None:
                terminate_proc(proc)
        (self.run_dir / "pids.json").unlink(missing_ok=True)

    def all_stopped(self) -> bool:
        return not port_open(self.proxy_port) and not port_open(self.upstream_port)


def terminate_proc(proc: subprocess.Popen, hard: bool = False) -> None:
    if proc.poll() is not None:
        return
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGKILL if hard else signal.SIGTERM)
    except OSError:
        pass
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except OSError:
            pass
        proc.wait(timeout=5)


# ===================================================================================== the probe client
def build_probe_graph(workflows_dir: Path, job_id: str, seed: int = 1) -> dict:
    """The verified T2V workflow with VIDSGEN's own inputs filled in (so the proxy's strict validation accepts it)."""
    d = Path(workflows_dir)
    mapping = json.loads((d / "wan22_ti2v_5b.mapping.json").read_text(encoding="utf-8"))
    graph = json.loads((d / mapping["templates"]["t2v"]).read_text(encoding="utf-8"))
    values = {"prompt": "a paper boat on a rainy street", "negative_prompt": "", "width": 832, "height": 480, "num_frames": 17, "fps": 24,
              "steps": 20, "guidance_scale": 5.0, "seed": seed, "filename_prefix": f"vidsgen/{job_id}"}
    for param, target in mapping["inputs"].items():
        if param in values:
            graph[mapping["nodes"][target["node"]]["id"]]["inputs"][target["input"]] = values[param]
    return graph


def new_job_id() -> str:
    return "job_" + secrets.token_hex(12)


class Probe:
    """The connectivity checks. Every check returns a dict; none can raise, hang or print the token."""

    def __init__(self, base_url: str, token: str, workflows_dir: Path, *, expect_bytes: int, expect_seed: int, ws_idle: float, burst: int,
                 timeout: float):
        self.base, self.token, self.workflows_dir = base_url.rstrip("/"), token, Path(workflows_dir)
        self.expect_bytes, self.expect_seed, self.ws_idle, self.burst, self.timeout = expect_bytes, expect_seed, ws_idle, burst, timeout
        self.auth = {"Authorization": f"Bearer {token}"}
        self.ws_base = ("wss" if self.base.startswith("https") else "ws") + self.base[self.base.index("://"):]
        self.ctx: dict = {}

    def ws_connect(self, cid: str):
        from websockets.asyncio.client import connect
        return connect(f"{self.ws_base}/ws?clientId={cid}", additional_headers=self.auth, proxy=None, open_timeout=self.timeout, ping_interval=20,
                       ping_timeout=20)

    async def collect(self, ws, until: str = "execution_success") -> list[dict]:
        seen = []
        async with asyncio.timeout(self.timeout):
            async for raw in ws:
                msg = json.loads(raw)
                seen.append(msg)
                if msg["type"] == until:
                    break
        return seen

    def submit_body(self) -> dict:
        return {"prompt": build_probe_graph(self.workflows_dir, new_job_id()), "client_id": "probe-" + secrets.token_hex(6), "prompt_id": str(uuid.uuid4())}

    async def submit(self, client, pid: str | None = None, cid: str | None = None, job_id: str | None = None):
        job_id = job_id or new_job_id()
        pid = pid or str(uuid.uuid4())
        body = {"prompt": build_probe_graph(self.workflows_dir, job_id), "client_id": cid or "probe-" + secrets.token_hex(6), "prompt_id": pid}
        return await client.post(f"{self.base}/prompt", json=body, headers=self.auth), body, job_id

    # ---------------------------------------------------------------- checks
    async def c_unauthenticated_rejected(self, c) -> dict:
        a, b, wrong = (await c.get(f"{self.base}/system_stats"), await c.get(f"{self.base}/no/such/route"),
                       await c.get(f"{self.base}/system_stats", headers={"Authorization": "Bearer " + secrets.token_hex(24)}))
        same = (a.status_code, a.content) == (b.status_code, b.content) == (wrong.status_code, wrong.content)
        return {"ok": a.status_code == 401 and same and a.json() == {"error": "unauthorized"}, "detail": f"status {a.status_code}, identical bodies: {same}"}

    async def c_authenticated_ok(self, c) -> dict:
        r = await c.get(f"{self.base}/system_stats", headers=self.auth)
        leaked = [w for w in ("argv", "/probe", "python_version", "embedded_python") if w in r.text]
        return {"ok": r.status_code == 200 and set(r.json()) == {"system", "devices"} and not leaked, "detail": f"status {r.status_code}, leaked: {leaked}"}

    async def c_route_allowlist(self, c) -> dict:
        codes = {f"{m} {p}": (await c.request(m, f"{self.base}{p}", headers=self.auth)).status_code
                 for m, p in (("POST", "/free"), ("GET", "/history"), ("POST", "/upload/image"), ("GET", "/queue?x=1"), ("GET", "/extensions"))}
        return {"ok": all(v == 404 for v in codes.values()), "detail": json.dumps(codes)}

    async def c_submit_and_progress_over_websocket(self, c) -> dict:
        cid = "probe-" + secrets.token_hex(6)
        async with self.ws_connect(cid) as ws:
            r, body, job_id = await self.submit(c, cid=cid)
            events = await self.collect(ws)
        self.ctx.update(first_body=body, job_id=job_id, pid=body["prompt_id"])
        kinds = [e["type"] for e in events]
        values = [e["data"]["value"] for e in events if e["type"] == "progress"]
        text = json.dumps(events)
        ok = (r.status_code == 200 and kinds[0] == "execution_start" and kinds[-1] == "execution_success" and values == sorted(values)
              and len(values) >= 5 and all(e["data"]["prompt_id"] == body["prompt_id"] for e in events) and "/probe" not in text)
        return {"ok": ok, "detail": f"{len(events)} events, {len(values)} progress steps, status {r.status_code}"}

    async def c_idle_websocket_survives(self, c) -> dict:
        cid = "probe-" + secrets.token_hex(6)
        async with self.ws_connect(cid) as ws:
            await asyncio.sleep(self.ws_idle)
            r, _, _ = await self.submit(c, cid=cid)
            events = await self.collect(ws, until="execution_start")
        return {"ok": r.status_code == 200 and events[-1]["type"] == "execution_start", "detail": f"idle {self.ws_idle:g}s, then received events"}

    async def c_streamed_download_integrity(self, c) -> dict:
        job_id = self.ctx.get("job_id") or new_job_id()
        h, n, t0 = hashlib.sha256(), 0, time.monotonic()
        rss0 = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        async with c.stream("GET", f"{self.base}/view", params={"filename": f"{job_id}_00001_.mp4", "subfolder": "vidsgen", "type": "output"},
                            headers=self.auth) as r:
            status, length = r.status_code, r.headers.get("content-length")
            async for chunk in r.aiter_bytes(1024 * 1024):
                h.update(chunk)
                n += len(chunk)
        secs = max(time.monotonic() - t0, 1e-6)
        want = up.expected_sha256(self.expect_bytes, self.expect_seed)
        ok = status == 200 and n == self.expect_bytes and h.hexdigest() == want and length == str(self.expect_bytes)
        return {"ok": ok, "detail": f"{n} bytes, {n / 2**20 / secs:.1f} MiB/s, sha256 {'matches' if h.hexdigest() == want else 'DIFFERS'}",
                "mib_per_s": round(n / 2**20 / secs, 1), "client_rss_growth_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss - rss0}

    async def c_range_requests(self, c) -> dict:
        job_id = self.ctx.get("job_id") or new_job_id()
        params = {"filename": f"{job_id}_00001_.mp4", "subfolder": "vidsgen", "type": "output"}
        size, seed = self.expect_bytes, self.expect_seed
        r1 = await c.get(f"{self.base}/view", params=params, headers={**self.auth, "Range": "bytes=0-1023"})
        r2 = await c.get(f"{self.base}/view", params=params, headers={**self.auth, "Range": "bytes=-100"})
        r3 = await c.get(f"{self.base}/view", params=params, headers={**self.auth, "Range": f"bytes={size + 10}-"})
        r4 = await c.get(f"{self.base}/view", params=params, headers={**self.auth, "Range": "bytes=0-1,5-9"})
        want1, want2 = b"".join(up.iter_stream(size, seed, 0, 1023)), b"".join(up.iter_stream(size, seed, size - 100, size - 1))
        ok = (r1.status_code == 206 and r1.content == want1 and r1.headers.get("content-range") == f"bytes 0-1023/{size}"
              and r2.status_code == 206 and r2.content == want2 and r3.status_code == 416 and r4.status_code in (400, 416))
        return {"ok": ok, "detail": f"{r1.status_code}/{r2.status_code}/{r3.status_code}/{r4.status_code}"}

    async def c_concurrency_burst(self, c) -> dict:
        rs = await asyncio.gather(*[c.get(f"{self.base}/system_stats", headers=self.auth) for _ in range(self.burst)], return_exceptions=True)
        codes = [r.status_code if isinstance(r, __import__("httpx").Response) else type(r).__name__ for r in rs]
        counts = {str(k): codes.count(k) for k in set(codes)}
        return {"ok": all(k in (200, 429) for k in codes) and 200 in codes, "detail": json.dumps(counts)}

    async def c_duplicate_submission_replay(self, c) -> dict:
        body = self.ctx.get("first_body")
        if body is None:
            return {"ok": False, "detail": "no earlier submission to replay"}
        r = await c.post(f"{self.base}/prompt", json=body, headers=self.auth)
        return {"ok": r.status_code == 200 and r.headers.get("x-idempotent-replay") in ("1", "existing-unverified"),
                "detail": f"status {r.status_code}, replay header {r.headers.get('x-idempotent-replay')}"}

    async def c_headers_hygiene(self, c) -> dict:
        problems = []
        for r in (await c.get(f"{self.base}/system_stats", headers=self.auth), await c.get(f"{self.base}/system_stats")):
            for h in ("server", "x-powered-by", "date"):
                if h in r.headers and self.base.startswith("http://127"):         # (a tunnel's edge adds its own: only judged locally)
                    problems.append(h)
            if r.headers.get("cache-control") != "no-store" or r.headers.get("x-content-type-options") != "nosniff":
                problems.append("missing no-store/nosniff")
        return {"ok": not problems, "detail": ", ".join(sorted(set(problems))) or "clean"}

    async def c_invalid_token_throttle(self, c) -> dict:
        codes = [(await c.get(f"{self.base}/system_stats", headers={"Authorization": "Bearer " + secrets.token_hex(24)})).status_code
                 for _ in range(14)]
        good = await c.get(f"{self.base}/system_stats", headers=self.auth)
        return {"ok": 429 in codes and good.status_code == 200, "detail": f"{codes.count(401)}x401, {codes.count(429)}x429; valid token afterwards: {good.status_code}"}

    ORDER = ["unauthenticated_rejected", "authenticated_ok", "route_allowlist", "submit_and_progress_over_websocket", "idle_websocket_survives",
             "streamed_download_integrity", "range_requests", "concurrency_burst", "duplicate_submission_replay", "headers_hygiene",
             "invalid_token_throttle"]                       # (the throttle is last: it makes later bad-token requests 429 for a minute)

    async def run_check(self, client, name: str) -> dict:
        t0 = time.monotonic()
        try:
            async with asyncio.timeout(self.timeout + self.ws_idle + 30):
                out = await getattr(self, f"c_{name}")(client)
        except TimeoutError:
            out = {"ok": False, "detail": "timed out"}
        except Exception as e:                                # noqa: BLE001  (reported by type only: messages may carry URLs)
            out = {"ok": False, "detail": f"{type(e).__name__}"}
        return {"name": name, **out, "seconds": round(time.monotonic() - t0, 2)}

    async def run_all(self, names: list[str] | None = None) -> list[dict]:
        import httpx
        results = []
        async with httpx.AsyncClient(timeout=self.timeout, follow_redirects=False) as client:
            for name in names or self.ORDER:
                results.append(await self.run_check(client, name))
        return results


class SafeParser(argparse.ArgumentParser):
    """argparse echoes unrecognised arguments in its error text; if someone pastes a token as an option, it must not be printed back."""

    def __init__(self, *args, **kwargs):
        kwargs.setdefault("allow_abbrev", False)            # "--token" must not silently become "--token-env"
        super().__init__(*args, **kwargs)

    def error(self, message):
        import re as _re
        super().error(_re.sub(r"[A-Za-z0-9_\-]{24,}", "[redacted]", message))


# ===================================================================================== reports
def finish(results: list[dict], config: dict, started: float, extra: dict | None = None) -> dict:
    failed = [r["name"] for r in results if not r["ok"]]
    report = {"verdict": "PASS" if results and not failed else "FAIL", "failed_checks": failed, "checks": results, "config": config,
              "seconds": round(time.monotonic() - started, 1), **(extra or {})}
    text = json.dumps(report, indent=1, default=str)
    return report if "Bearer " not in text else {"verdict": "FAIL", "failed_checks": ["report contained a token-like string"], "checks": []}


def print_summary(report: dict) -> None:
    for r in report["checks"]:
        print(f"  {'PASS' if r['ok'] else 'FAIL'}  {r['name']:<38} {r['detail']}  ({r['seconds']}s)")
    print(f"\nCONNECTIVITY VERDICT: {report['verdict']} ({report['seconds']}s)")


# ===================================================================================== run-local
def run_local(a) -> int:
    lifetime = a.lifetime_seconds
    if not 0 < lifetime <= HARD_MAX_LIFETIME:
        print(f"FAILED: --lifetime-seconds must be between 1 and {HARD_MAX_LIFETIME:g} (30 minutes).", file=sys.stderr)
        return 2
    started = time.monotonic()
    run_dir = Path(a.run_dir)
    token = secrets.token_urlsafe(32)
    svc = Services(run_dir, token, Path(a.workflows_dir), download_mib=a.download_mib, seed=a.seed, lifetime=lifetime)
    done = threading.Event()
    expired = {"flag": False}

    def teardown() -> None:
        svc.stop_all()

    def on_signal(signum, frame):
        teardown()
        os._exit(128 + signum)

    def on_expire() -> None:
        if done.is_set():
            return
        expired["flag"] = True
        teardown()
        report = finish([], {"lifetime_seconds": lifetime}, started)
        report.update(verdict="FAIL", failed_checks=["lifetime exceeded"], reasons=[f"the run exceeded its {lifetime:g}s lifetime and was stopped"])
        Path(a.report).parent.mkdir(parents=True, exist_ok=True)
        Path(a.report).write_text(json.dumps(report, indent=1))
        print(f"LIFETIME EXCEEDED: stopped everything after {lifetime:g}s", file=sys.stderr)
        os._exit(3)

    atexit.register(teardown)
    signal.signal(signal.SIGINT, on_signal)
    signal.signal(signal.SIGTERM, on_signal)
    watchdog = threading.Timer(lifetime, on_expire)
    watchdog.daemon = True
    watchdog.start()
    results: list[dict] = []
    extra: dict = {}
    try:
        svc.start_upstream()
        svc.start_proxy()
        svc.wait_ready()
        size = int(a.download_mib * 1024 * 1024)
        probe = Probe(svc.proxy_url, token, Path(a.workflows_dir), expect_bytes=size, expect_seed=a.seed, ws_idle=a.ws_idle_seconds,
                      burst=a.burst, timeout=a.timeout)
        results = asyncio.run(probe.run_all())
        results += asyncio.run(local_interruption_checks(svc, probe, token))
        extra["tear_down"] = {}
    except Exception as e:                                     # noqa: BLE001
        results.append({"name": "run", "ok": False, "detail": f"{type(e).__name__}: {str(e).replace(token, '[token]')[:300]}", "seconds": 0})
    finally:
        done.set()
        watchdog.cancel()
        teardown()
    extra["tear_down"] = {"proxy_port_closed": not port_open(svc.proxy_port), "upstream_port_closed": not port_open(svc.upstream_port)}
    clean = all(extra["tear_down"].values())
    results.append({"name": "clean_teardown", "ok": clean, "detail": json.dumps(extra["tear_down"]), "seconds": 0})
    report = finish(results, {"lifetime_seconds": lifetime, "download_mib": a.download_mib, "ws_idle_seconds": a.ws_idle_seconds, "burst": a.burst,
                              "bind": "127.0.0.1 only", "upstream": "dummy (CPU only)"}, started, extra)
    Path(a.report).parent.mkdir(parents=True, exist_ok=True)
    Path(a.report).write_text(json.dumps(report, indent=1, default=str))
    print_summary(report)
    return 0 if report["verdict"] == "PASS" else 1


async def local_interruption_checks(svc: Services, probe: Probe, token: str) -> list[dict]:
    """Local only: hang handling, a killed and restarted proxy, and 'no second run after the restart'."""
    import httpx
    out: list[dict] = []
    auth = probe.auth
    async with httpx.AsyncClient(timeout=10) as c:
        # 1. A hung upstream must not hang the client: its own timeout fires, and the system stays usable.
        t0 = time.monotonic()
        await c.post(f"{svc.upstream_url}/__probe_control", json={"hang_prompt": True})
        try:
            await httpx.AsyncClient(timeout=2).post(f"{probe.base}/prompt", json=probe.submit_body(), headers=auth)
            hung = False
        except httpx.TimeoutException:
            hung = True
        await c.post(f"{svc.upstream_url}/__probe_control", json={"hang_prompt": False})
        alive = (await c.get(f"{probe.base}/system_stats", headers=auth)).status_code == 200
        out.append({"name": "timeout_handling", "ok": hung and alive and time.monotonic() - t0 < 8, "detail": f"client timed out: {hung}; proxy still serving: {alive}",
                    "seconds": round(time.monotonic() - t0, 2)})
        # 2. Kill the proxy mid-use, then restart it.
        before = (await c.get(f"{svc.upstream_url}/__probe_stats")).json()["prompts_received"]
        t0 = time.monotonic()
        cid = "probe-" + secrets.token_hex(6)
        ws = await probe.ws_connect(cid).__aenter__()
        svc.kill_proxy()
        refused = False
        try:
            await c.get(f"{probe.base}/system_stats", headers=auth)
        except httpx.HTTPError:
            refused = True
        try:
            async with asyncio.timeout(10):
                await ws.recv()
            ws_closed = False
        except Exception:                                      # noqa: BLE001  (a closed socket is the expected outcome)
            ws_closed = True
        svc.start_proxy()
        svc.wait_ready(30)
        recovered = (await c.get(f"{probe.base}/system_stats", headers=auth)).status_code == 200
        out.append({"name": "proxy_interruption_recovery", "ok": refused and ws_closed and recovered, "seconds": round(time.monotonic() - t0, 2),
                    "detail": f"requests failed while down: {refused}; websocket closed: {ws_closed}; recovered after restart: {recovered}"})
        # 3. After the restart the proxy has forgotten everything: a repeated submission must be recognised from ComfyUI and not run again.
        body = probe.ctx.get("first_body")
        if body is not None:
            r = await c.post(f"{probe.base}/prompt", json=body, headers=auth)
            after = (await c.get(f"{svc.upstream_url}/__probe_stats")).json()["prompts_received"]
            out.append({"name": "duplicate_after_restart_not_rerun", "seconds": 0,
                        "ok": r.status_code == 200 and r.headers.get("x-idempotent-replay") == "existing-unverified" and after == before,
                        "detail": f"status {r.status_code}, replay header {r.headers.get('x-idempotent-replay')}, upstream prompts {before} -> {after}"})
    return out


# ===================================================================================== probe / teardown commands
def run_probe(a) -> int:
    host = urlsplit(a.url).hostname or ""
    if host not in LOOPBACK and not a.allow_remote:
        print("FAILED: refusing a non-loopback URL. (--allow-remote exists only for a future, separately approved tunnel test.)", file=sys.stderr)
        return 2
    if not re.fullmatch(r"[A-Z_][A-Z0-9_]{0,63}", a.token_env):         # a token pasted here by mistake must not be echoed back
        print("FAILED: --token-env must be the NAME of an environment variable (upper case letters, digits, underscore).", file=sys.stderr)
        return 2
    token = os.environ.get(a.token_env, "")
    if len(token) < 32:
        print(f"FAILED: the token must be provided in the environment variable {a.token_env} (at least 32 characters).", file=sys.stderr)
        return 2
    started = time.monotonic()
    probe = Probe(a.url, token, Path(a.workflows_dir), expect_bytes=int(a.expect_mib * 1024 * 1024), expect_seed=a.expect_seed,
                  ws_idle=a.ws_idle_seconds, burst=a.burst, timeout=a.timeout)
    results = asyncio.run(probe.run_all())
    report = finish(results, {"target_host_class": "loopback" if host in LOOPBACK else "remote", "ws_idle_seconds": a.ws_idle_seconds, "burst": a.burst}, started)
    if a.report:
        Path(a.report).parent.mkdir(parents=True, exist_ok=True)
        Path(a.report).write_text(json.dumps(report, indent=1, default=str))
    print_summary(report)
    return 0 if report["verdict"] == "PASS" else 1


def run_teardown(a) -> int:
    pids = Path(a.run_dir) / "pids.json"
    if not pids.exists():
        print("nothing to tear down")
        return 0
    data = json.loads(pids.read_text())
    ok = True
    for name in ("proxy", "upstream"):
        if name in data:
            ok &= terminate(int(data[name]))
    pids.unlink(missing_ok=True)
    print("teardown complete" if ok else "FAILED: a process could not be stopped")
    return 0 if ok else 1


def main(argv: list[str] | None = None) -> int:
    ap = SafeParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True, parser_class=SafeParser)
    p = sub.add_parser("run-local")
    p.add_argument("--run-dir", required=True); p.add_argument("--report", required=True)
    p.add_argument("--workflows-dir", default=str(REPO / "workflows" / "comfyui"))
    p.add_argument("--lifetime-seconds", type=float, default=HARD_MAX_LIFETIME, help="maximum 1800 (30 minutes)")
    p.add_argument("--download-mib", type=float, default=100.0); p.add_argument("--seed", type=int, default=7)
    p.add_argument("--ws-idle-seconds", type=float, default=150.0); p.add_argument("--burst", type=int, default=40)
    p.add_argument("--timeout", type=float, default=60.0)
    p = sub.add_parser("probe")
    p.add_argument("--url", required=True); p.add_argument("--token-env", default="PROXY_API_TOKEN")
    p.add_argument("--workflows-dir", default=str(REPO / "workflows" / "comfyui")); p.add_argument("--report")
    p.add_argument("--expect-mib", type=float, required=True); p.add_argument("--expect-seed", type=int, default=7)
    p.add_argument("--ws-idle-seconds", type=float, default=150.0); p.add_argument("--burst", type=int, default=40)
    p.add_argument("--timeout", type=float, default=60.0)
    p.add_argument("--allow-remote", action="store_true", help="NOT for use until a tunnel test has been separately approved")
    p = sub.add_parser("teardown"); p.add_argument("--run-dir", required=True)
    a = ap.parse_args(argv)
    return {"run-local": run_local, "probe": run_probe, "teardown": run_teardown}[a.cmd](a)


if __name__ == "__main__":
    raise SystemExit(main())
