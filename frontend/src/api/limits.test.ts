import { afterEach, describe, expect, it, vi } from "vitest";
import { ApiError } from "./client";
import { canGenerate, createLimitsStore, getDefaultLimitsStore, retryDelay, validateLimits, type LimitsState, type LimitsStore } from "./limits";
import { getLimits } from "./jobs";
import type { Limits } from "./types";
import limitsSource from "./limits.ts?raw";
import errorsSource from "./errors.ts?raw";
import hookSource from "../hooks/useLimits.ts?raw";
import { DEFAULT_LIMITS, T4_LIMITS } from "./limits.fixtures";

const clone = <T,>(v: T): T => JSON.parse(JSON.stringify(v));

describe("validateLimits", () => {
  it("accepts the default and the t4_verified responses", () => {
    expect(validateLimits(DEFAULT_LIMITS)).toEqual([]);
    expect(validateLimits(T4_LIMITS)).toEqual([]);
  });

  it.each<[string, (l: any) => void]>([
    ["no enabled mode", (l) => l.modes.forEach((m: any) => (m.enabled = false))],
    ["no modes at all", (l) => (l.modes = [])],
    ["no enabled resolution", (l) => l.resolutions.forEach((r: any) => (r.enabled = false))],
    ["default mode disabled", (l) => (l.defaults.mode = "i2v")],
    ["default mode unknown", (l) => (l.defaults.mode = "v2v")],
    ["default resolution disabled", (l) => (l.defaults.resolution = "1280x704")],
    ["default frames not allowed", (l) => (l.defaults.num_frames = 49)],
    ["no allowed frames", (l) => (l.num_frames.allowed = [])],
    ["frames off the step grid", (l) => (l.num_frames.allowed = [18])],
    ["frames outside their range", (l) => (l.num_frames.allowed = [13, 17])],
    ["default steps above the maximum", (l) => (l.defaults.steps = 30)],
    ["steps range inverted", (l) => (l.steps = { min: 20, max: 1 })],
    ["default fps out of range", (l) => (l.defaults.fps = 60)],
    ["default guidance out of range", (l) => (l.defaults.guidance_scale = 99)],
    ["fractional fps range", (l) => (l.fps = { min: 8.5, max: 30 })],
    ["malformed prompt limit", (l) => (l.prompt = { max_chars: 0 })],
    ["malformed image limits", (l) => (l.image = { mime_types: "image/png" })],
    ["missing profile info", (l) => delete l.profile],
    ["missing defaults", (l) => delete l.defaults],
    ["frame step zero", (l) => (l.num_frames.step = 0)],
    ["a non-number where a number belongs", (l) => (l.steps.max = "20")],
  ])("rejects %s", (_name, tamper) => {
    const l = clone(T4_LIMITS);
    tamper(l);
    expect(validateLimits(l).length).toBeGreaterThan(0);
  });

  it.each([null, undefined, 42, "limits", [], true])("rejects a non-object: %j", (x) => {
    expect(validateLimits(x)).toEqual(["the response is not an object"]);
  });
});

// ------------------------------------------------------------------ a store with a manual clock, scheduler and fetcher
function harness(opts: { ttlMs?: number; maxAutoRetries?: number } = {}) {
  let t = 1_000_000;
  const timers = new Map<number, { at: number; fn: () => void }>();
  let nextId = 1;
  const calls: { resolve: (l: Limits) => void; reject: (e: unknown) => void; signal: AbortSignal }[] = [];
  const fetcher = vi.fn((signal: AbortSignal) => new Promise<Limits>((resolve, reject) => { calls.push({ resolve, reject, signal }); }));
  const store = createLimitsStore({
    fetcher, now: () => t, ttlMs: opts.ttlMs, maxAutoRetries: opts.maxAutoRetries,
    setTimer: (fn, ms) => { const id = nextId++; timers.set(id, { at: t + ms, fn }); return id; },
    clearTimer: (h) => { timers.delete(h as number); },
  });
  const flush = async () => { for (let i = 0; i < 5; i++) await Promise.resolve(); };
  const advance = async (ms: number) => {
    t += ms;
    for (const [id, tm] of [...timers]) if (tm.at <= t) { timers.delete(id); tm.fn(); }
    await flush();
  };
  const last = () => calls[calls.length - 1]!;
  const states: LimitsState[] = [];
  const watch = (s: LimitsStore) => s.subscribe(() => states.push(s.getState()));
  return { store, fetcher, calls, timers, flush, advance, last, states, watch, setNow: (v: number) => { t = v; }, now: () => t };
}

const net = () => new ApiError("Can't reach the VIDSGEN server.", { status: 0, code: "network_error" });

describe("limits store: loading and success", () => {
  it("starts in loading, fetches once when first watched, and becomes ready with the active limits", async () => {
    const h = harness();
    expect(h.store.getState()).toEqual({ kind: "loading", attempt: 1 });
    expect(h.fetcher).not.toHaveBeenCalled();                                  // nothing happens until someone is watching
    h.watch(h.store);
    expect(h.fetcher).toHaveBeenCalledTimes(1);
    expect(h.store.getState().kind).toBe("loading");
    h.last().resolve(T4_LIMITS);
    await h.flush();
    const s = h.store.getState();
    expect(s.kind).toBe("ready");
    expect(s.kind === "ready" && s.limits).toEqual(T4_LIMITS);
    expect(canGenerate(s)).toBe(true);
  });

  it("generation is not allowed while loading", () => {
    const h = harness();
    h.watch(h.store);
    expect(canGenerate(h.store.getState())).toBe(false);
  });

  it("notifies subscribers on every change and returns a stable state object between changes", async () => {
    const h = harness();
    h.watch(h.store);
    const a = h.store.getState();
    expect(h.store.getState()).toBe(a);
    h.last().resolve(DEFAULT_LIMITS);
    await h.flush();
    expect(h.states.map((s) => s.kind)).toEqual(["loading", "ready"]);
    expect(h.store.getState()).toBe(h.store.getState());
  });
});

describe("limits store: caching and sharing", () => {
  it("shares ONE request between simultaneous watchers", async () => {
    const h = harness();
    h.watch(h.store); h.watch(h.store); h.watch(h.store);
    expect(h.fetcher).toHaveBeenCalledTimes(1);
    h.last().resolve(DEFAULT_LIMITS);
    await h.flush();
    expect(h.states.length).toBeGreaterThanOrEqual(2);
  });

  it("serves a fresh cached answer to later watchers without another request", async () => {
    const h = harness({ ttlMs: 60_000 });
    const off = h.watch(h.store);
    h.last().resolve(DEFAULT_LIMITS);
    await h.flush();
    off();
    await h.advance(30_000);
    h.watch(h.store);
    expect(h.fetcher).toHaveBeenCalledTimes(1);
    expect(h.store.getState().kind).toBe("ready");
  });

  it("re-checks a stale answer in the background and stays ready while doing so", async () => {
    const h = harness({ ttlMs: 60_000 });
    const off = h.watch(h.store);
    h.last().resolve(DEFAULT_LIMITS);
    await h.flush();
    off();
    await h.advance(61_000);
    h.watch(h.store);
    expect(h.fetcher).toHaveBeenCalledTimes(2);
    expect(h.store.getState().kind).toBe("ready");                              // not flickering to loading
    h.last().resolve(T4_LIMITS);                                                // the server now reports a stricter profile
    await h.flush();
    const s = h.store.getState();
    expect(s.kind === "ready" && s.limits.profile).toBe("t4_verified");
  });

  it("refresh() is a no-op while the cache is fresh and re-checks when forced or stale", async () => {
    const h = harness({ ttlMs: 60_000 });
    h.watch(h.store);
    h.last().resolve(DEFAULT_LIMITS);
    await h.flush();
    h.store.refresh();
    expect(h.fetcher).toHaveBeenCalledTimes(1);
    h.store.refresh(true);
    expect(h.fetcher).toHaveBeenCalledTimes(2);
    h.last().resolve(DEFAULT_LIMITS);
    await h.flush();
    await h.advance(61_000);
    h.store.refresh();
    expect(h.fetcher).toHaveBeenCalledTimes(3);
  });

  it("a failed background re-check turns generation OFF instead of trusting stale limits", async () => {
    const h = harness({ ttlMs: 1_000 });
    h.watch(h.store);
    h.last().resolve(DEFAULT_LIMITS);
    await h.flush();
    await h.advance(2_000);
    h.store.refresh();
    h.last().reject(net());
    await h.flush();
    const s = h.store.getState();
    expect(s.kind).toBe("unavailable");
    expect(canGenerate(s)).toBe(false);
  });

  it("stops all work when the last watcher leaves, and starts again for the next one", async () => {
    const h = harness();
    const off = h.watch(h.store);
    const first = h.last();
    off();
    expect(first.signal.aborted).toBe(true);
    expect(h.store.getState()).toEqual({ kind: "loading", attempt: 1 });
    first.resolve(DEFAULT_LIMITS);                                             // a late answer from the abandoned request is ignored
    await h.flush();
    expect(h.store.getState().kind).toBe("loading");
    h.watch(h.store);
    expect(h.fetcher).toHaveBeenCalledTimes(2);
  });

  it("keeps the cache but cancels pending retries when everyone leaves", async () => {
    const h = harness();
    const off = h.watch(h.store);
    h.last().reject(net());
    await h.flush();
    expect(h.timers.size).toBe(1);
    off();
    expect(h.timers.size).toBe(0);
    h.watch(h.store);
    expect(h.fetcher).toHaveBeenCalledTimes(2);                                // a new viewer gets a fresh attempt
  });
});

describe("limits store: failure and retry", () => {
  it("reports unavailable with a friendly description and schedules an automatic retry with exponential backoff", async () => {
    const h = harness();
    h.watch(h.store);
    h.last().reject(net());
    await h.flush();
    let s = h.store.getState();
    expect(s.kind === "unavailable" && s.error.kind).toBe("connection");
    expect(s.kind === "unavailable" && s.retryInMs).toBe(1000);
    for (const delay of [1000, 2000, 4000, 8000, 16000]) {
      expect(h.fetcher).toHaveBeenCalledTimes([1000, 2000, 4000, 8000, 16000].indexOf(delay) + 1);
      await h.advance(delay);
      expect(h.store.getState().kind).toBe("loading");
      h.last().reject(net());
      await h.flush();
    }
    s = h.store.getState();
    expect(s.kind === "unavailable" && s.retryInMs).toBeNull();               // the retries are used up: manual retry only
    expect(h.timers.size).toBe(0);
    expect(h.fetcher).toHaveBeenCalledTimes(6);
  });

  it("the retry delay doubles and is capped", () => {
    expect([1, 2, 3, 4, 5, 6, 7, 8].map(retryDelay)).toEqual([1000, 2000, 4000, 8000, 16000, 30000, 30000, 30000]);
  });

  it("recovers by itself when a retry succeeds", async () => {
    const h = harness();
    h.watch(h.store);
    h.last().reject(net());
    await h.flush();
    await h.advance(1000);
    h.last().resolve(T4_LIMITS);
    await h.flush();
    expect(h.store.getState().kind).toBe("ready");
    expect(h.timers.size).toBe(0);
  });

  it("manual retry fetches immediately, cancels the schedule and resets the attempt counter", async () => {
    const h = harness();
    h.watch(h.store);
    h.last().reject(net());
    await h.flush();
    await h.advance(1000);
    h.last().reject(net());
    await h.flush();
    expect(h.timers.size).toBe(1);
    h.store.retry();
    expect(h.timers.size).toBe(0);
    expect(h.store.getState()).toEqual({ kind: "loading", attempt: 1 });
    expect(h.fetcher).toHaveBeenCalledTimes(3);
    h.last().resolve(DEFAULT_LIMITS);
    await h.flush();
    expect(h.store.getState().kind).toBe("ready");
  });

  it("manual retry works after the automatic retries are exhausted", async () => {
    const h = harness({ maxAutoRetries: 1 });
    h.watch(h.store);
    h.last().reject(net());
    await h.flush();
    await h.advance(1000);
    h.last().reject(net());
    await h.flush();
    const s = h.store.getState();
    expect(s.kind === "unavailable" && s.retryInMs).toBeNull();
    h.store.retry();
    h.last().resolve(DEFAULT_LIMITS);
    await h.flush();
    expect(h.store.getState().kind).toBe("ready");
  });

  it("an older server without the endpoint is reported clearly and not retried automatically", async () => {
    const h = harness();
    h.watch(h.store);
    h.last().reject(new ApiError("Not Found", { status: 404, code: "http_404" }));
    await h.flush();
    const s = h.store.getState();
    expect(s.kind === "unavailable" && s.error.code).toBe("limits_not_supported");
    expect(s.kind === "unavailable" && s.error.retryable).toBe(false);
    expect(s.kind === "unavailable" && s.retryInMs).toBeNull();
    expect(h.timers.size).toBe(0);
  });

  it.each<[string, ApiError, boolean]>([
    ["server misconfigured (500 limits_unavailable)", new ApiError("x", { status: 500, code: "limits_unavailable" }), true],
    ["busy (429)", new ApiError("x", { status: 429, code: "http_429" }), true],
    ["bad gateway (502)", new ApiError("x", { status: 502, code: "http_502" }), true],
    ["timeout", new ApiError("x", { status: 0, code: "timeout" }), true],
    ["forbidden (403)", new ApiError("x", { status: 403, code: "http_403" }), false],
  ])("%s -> unavailable (retry scheduled: %s)", async (_n, err, retries) => {
    const h = harness();
    h.watch(h.store);
    h.last().reject(err);
    await h.flush();
    const s = h.store.getState();
    expect(s.kind).toBe("unavailable");
    expect(s.kind === "unavailable" && s.retryInMs !== null).toBe(retries);
    expect(canGenerate(s)).toBe(false);
  });

  it("an unexpected exception is unavailable too, with generic wording", async () => {
    const h = harness();
    h.watch(h.store);
    h.last().reject(new Error("TypeError: cannot read properties of undefined at /srv/app/x.js"));
    await h.flush();
    const s = h.store.getState();
    expect(s.kind === "unavailable" && s.error.message).toBe("An unexpected error occurred. Please try again.");
    expect(JSON.stringify(s)).not.toContain("/srv");
  });
});

describe("limits store: never falls back to permissive numbers", () => {
  it("rejects a response that fails verification and stays unavailable", async () => {
    const h = harness();
    h.watch(h.store);
    const bad = clone(T4_LIMITS) as any;
    bad.defaults.num_frames = 49;                                                // a default its own limits refuse
    h.last().resolve(bad);
    await h.flush();
    const s = h.store.getState();
    expect(s.kind).toBe("unavailable");
    expect(s.kind === "unavailable" && s.error.code).toBe("limits_invalid");
    expect(canGenerate(s)).toBe(false);
  });

  it("a malformed body (not even an object) never becomes ready", async () => {
    const h = harness();
    h.watch(h.store);
    h.last().resolve(null as unknown as Limits);
    await h.flush();
    expect(h.store.getState().kind).toBe("unavailable");
  });

  it("is never ready until a real response has been verified, whatever else happens", async () => {
    const h = harness({ maxAutoRetries: 2 });
    h.watch(h.store);
    for (let i = 0; i < 6; i++) {
      expect(canGenerate(h.store.getState())).toBe(false);
      h.last().reject(net());
      await h.flush();
      await h.advance(30_000);
    }
    expect(canGenerate(h.store.getState())).toBe(false);
  });

  it("the module does not import or reference the static constraints", () => {
    for (const source of [limitsSource, hookSource, errorsSource]) {
      expect(source.length).toBeGreaterThan(100);
      expect(source).not.toMatch(/from\s+["'][^"']*constraints/);
      expect(source).not.toMatch(/frameOptions|durationSeconds|isNative/);
    }
  });
});

// ------------------------------------------------------------------ the typed client and the shared default store
describe("getLimits", () => {
  afterEach(() => vi.unstubAllGlobals());

  it("calls GET /api/limits and returns the typed body", async () => {
    const fetchMock = vi.fn().mockResolvedValue(new Response(JSON.stringify(T4_LIMITS), { status: 200 }));
    vi.stubGlobal("fetch", fetchMock);
    const out = await getLimits();
    expect(out.profile).toBe("t4_verified");
    expect(String(fetchMock.mock.calls[0]![0])).toMatch(/\/api\/limits$/);
    expect(fetchMock.mock.calls[0]![1]).toMatchObject({ headers: { Accept: "application/json" } });
    expect(fetchMock.mock.calls[0]![1].method).toBeUndefined();                 // a plain GET
  });

  it("turns a server error into a described ApiError", async () => {
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(new Response(
      JSON.stringify({ detail: { code: "limits_unavailable", message: "The server's generation limits are misconfigured." } }), { status: 500 })));
    const err = (await getLimits().catch((e: unknown) => e)) as ApiError;
    expect(err).toBeInstanceOf(ApiError);
    expect(err.code).toBe("limits_unavailable");
  });

  it("the default store fetches through getLimits and is shared", async () => {
    const fetchMock = vi.fn().mockImplementation(() => Promise.resolve(new Response(JSON.stringify(DEFAULT_LIMITS), { status: 200 })));
    vi.stubGlobal("fetch", fetchMock);
    const store = getDefaultLimitsStore();
    expect(getDefaultLimitsStore()).toBe(store);
    const off = store.subscribe(() => {});
    for (let i = 0; i < 20 && store.getState().kind !== "ready"; i++) await new Promise((r) => setTimeout(r, 0));
    expect(store.getState().kind).toBe("ready");
    expect(fetchMock).toHaveBeenCalledTimes(1);
    off();
  });
});
