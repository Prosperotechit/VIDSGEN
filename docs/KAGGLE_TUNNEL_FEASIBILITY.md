# Kaggle tunnel feasibility: status, gates and the prepared (NOT executed) procedure

## Status: not approved
- Kaggle's Terms restrict use to internal, personal, non-commercial purposes, and its Acceptable Use Policy lists server farming and
  activity unrelated to ML/data science among prohibited resource abuse. **Permission for a public tunnel is not established.**
- Nothing in this repository starts a tunnel, uses Kaggle, or contacts an external server. All Stage 4A tooling is loopback-only and CPU-only.
- Next step before any tunnel work: send `docs/KAGGLE_SUPPORT_QUESTION.md` to Kaggle Support and **wait for a written yes**.
  A "no" or silence means: no tunnel on Kaggle. Use a permitted dedicated GPU environment instead (see below).

## What exists now (Stage 4A, local only)
| Tool | Purpose |
|---|---|
| `scripts/probe_upstream.py` | Dummy ComfyUI (CPU only, loopback only): the routes the proxy forwards, WebSocket progress events, a deterministic large download with Range support |
| `scripts/connectivity_harness.py run-local` | Starts the dummy upstream and the real proxy as loopback processes, runs every probe check, simulates a proxy kill/restart, and always tears down. **Hard maximum lifetime 30 minutes** (a watchdog stops everything and fails the run) |
| `scripts/connectivity_harness.py probe` | The same checks against a proxy URL. Refuses non-loopback URLs unless `--allow-remote`, which must not be used before a tunnel test is separately approved |
| `scripts/connectivity_harness.py teardown` | Stops leftovers from a crashed run (verifies each pid is still one of our processes first) |
| `scripts/cloudflared_pin.py` + `scripts/cloudflared_pins.json` | Version pin + SHA-256 verification for a future `cloudflared`. **Empty on purpose**: until a reviewed pin exists it refuses to download or approve any binary |

Run it locally (about 3 minutes with the defaults, or seconds with smaller values):

```bash
python scripts/connectivity_harness.py run-local --run-dir /tmp/vidsgen-conn --report /tmp/vidsgen-conn/report.json
```

## Safeguards added in Stage 4A
- **Pre-forward existence check.** Before forwarding a `prompt_id` the proxy has not seen, it asks ComfyUI (GET `/history/<id>`, GET `/queue`)
  whether it already knows that id. If so it answers "already accepted" and does **not** forward again. If ComfyUI cannot be asked, it does
  not forward (502).
- **This does NOT guarantee exactly-once execution.** Known gaps: (1) a prompt ComfyUI has accepted but not yet made visible in the queue or
  history (a visibility race) looks unknown; (2) a first request still in flight through another or an older proxy instance cannot be seen;
  (3) an id remembered only by ComfyUI is judged by id alone, so its payload cannot be compared; (4) the proxy's own memory is in RAM and is
  lost on restart. **Uncertain submissions are never resubmitted automatically**: the backend reconciles with read-only lookups and otherwise
  fails the job; a person decides what happens next.
- **Dead-man switch stays OFF by default.** When enabled (minimum 120 s) it acts only if the backend has been completely silent: no
  authenticated request and no open WebSocket for the whole period, re-checked before every action. A job's age alone can never trigger it.
- **`--max-lifetime-seconds`** on the proxy and the dummy upstream, and a kernel "die with parent" signal for harness children.

## Prepared tunnel procedure (DO NOT EXECUTE until every gate is met)
Gates, all required: (1) Kaggle Support's written permission; (2) your explicit approval of Stage 4B; (3) `cloudflared_pins.json` filled in
and reviewed, `cloudflared_pin.py verify` passing; (4) a CPU-only session (no GPU quota) for the first run.
1. In a new private CPU notebook (Internet on, no accelerator), keep the token in a Kaggle Secret and never print it.
2. Fetch the pinned binary with `cloudflared_pin.py download`; refuse to continue unless it verifies.
3. Start `probe_upstream.py` and `inference_proxy.py` (loopback, `--max-lifetime-seconds 1800`, dead-man optional) as in `run-local`.
4. Start `cloudflared` pointing only at the proxy port, wrapped in `timeout 1800`; print the tunnel URL once (it is not a secret, the token is).
5. From your machine run `connectivity_harness.py probe --url <tunnel> --allow-remote ...` with the token in an environment variable.
6. **Abort immediately** on any 2xx for an unauthenticated request, any token or path in a response, an unexpected route answering, or a
   Kaggle warning. Stop the session; discard the token.
7. Always: stop the tunnel and proxy, end the session, rotate the token. Review the report before any GPU test.

## Alternative if Kaggle does not approve: a permitted dedicated GPU environment
The `InferenceProvider` design makes the GPU host replaceable. A rented GPU VM or a serverless GPU endpoint run under terms that allow
hosting your own inference service can run the same pinned ComfyUI + this proxy, reachable over HTTPS with the bearer token (the backend
already requires https and a 32+ character token for any non-loopback endpoint). Advantages: a stable address, persistent model storage
(no 16.9 GiB re-download), no session limits. Check the provider's terms and pricing at the time; nothing here selects a provider.
