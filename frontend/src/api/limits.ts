// Runtime generation limits: what THIS server will accept (GET /api/limits).
//
// Safety rule: generation is allowed only while safe, active limits are known. If they cannot be fetched or fail the checks below, the
// state is "unavailable" and stays that way until a fetch succeeds. The static constraints bundled with the app describe the model's
// full capabilities, not this server's, so they are NEVER used as a stand-in (this module does not even import them).
import { describeError, type DescribedError } from "./errors";
import { ApiError } from "./client";
import { getLimits } from "./jobs";
import type { Limits } from "./types";

export type LimitsState =
  | { kind: "loading"; attempt: number }
  | { kind: "ready"; limits: Limits; fetchedAt: number }
  | { kind: "unavailable"; error: DescribedError; attempt: number; retryInMs: number | null };

export const canGenerate = (s: LimitsState): s is Extract<LimitsState, { kind: "ready" }> => s.kind === "ready";

const isObj = (v: unknown): v is Record<string, unknown> => typeof v === "object" && v !== null && !Array.isArray(v);
const isInt = (v: unknown): v is number => typeof v === "number" && Number.isInteger(v);
const isNum = (v: unknown): v is number => typeof v === "number" && Number.isFinite(v);

/** Everything wrong with a limits response (empty = safe to rely on). Checked on the client because the data comes over the network. */
export function validateLimits(x: unknown): string[] {
  const p: string[] = [];
  if (!isObj(x)) return ["the response is not an object"];
  const { modes, resolutions, num_frames: nf, steps, fps, guidance_scale: gs, seed, prompt, image, defaults: d } = x as Record<string, unknown>;

  const enabled = (list: unknown, what: string): string[] => {
    if (!Array.isArray(list) || list.length === 0) { p.push(`${what} is missing`); return []; }
    const names: string[] = [];
    for (const item of list) {
      if (!isObj(item) || typeof item.value !== "string" || typeof item.enabled !== "boolean") p.push(`a ${what} entry is malformed`);
      else if (item.enabled) names.push(item.value);
    }
    if (names.length === 0) p.push(`no ${what} is enabled`);
    return names;
  };
  const enabledModes = enabled(modes, "mode");
  const enabledRes = enabled(resolutions, "resolution");

  let allowed: number[] = [];
  if (!isObj(nf) || !Array.isArray(nf.allowed) || !nf.allowed.every(isInt) || !isInt(nf.min) || !isInt(nf.max) || !isInt(nf.step) || nf.step <= 0) {
    p.push("num_frames is malformed");
  } else {
    allowed = nf.allowed as number[];
    if (allowed.length === 0) p.push("no frame count is allowed");
    if (allowed.some((n) => n < (nf.min as number) || n > (nf.max as number) || (n - (nf.min as number)) % (nf.step as number) !== 0)) p.push("a frame count is outside its own range");
  }
  const range = (r: unknown, what: string, integer: boolean): [number, number] | null => {
    const ok = isObj(r) && (integer ? isInt(r.min) && isInt(r.max) : isNum(r.min) && isNum(r.max)) && (r.min as number) <= (r.max as number);
    if (!ok) { p.push(`${what} is malformed`); return null; }
    return [(r as { min: number }).min, (r as { max: number }).max];
  };
  const stepsR = range(steps, "steps", true), fpsR = range(fps, "fps", true), gsR = range(gs, "guidance_scale", false);
  range(seed, "seed", true);
  if (!isObj(prompt) || !isInt(prompt.max_chars) || prompt.max_chars <= 0) p.push("prompt limits are malformed");
  if (!isObj(image) || !Array.isArray(image.mime_types) || !isInt(image.max_bytes) || image.max_bytes <= 0 || !isInt(image.min_side) || !isInt(image.max_side)) p.push("image limits are malformed");
  if (typeof x.profile !== "string" || typeof x.restricted !== "boolean" || !isInt(x.max_concurrent_jobs)) p.push("profile information is missing");

  if (!isObj(d)) p.push("defaults are missing");
  else {
    if (typeof d.mode !== "string" || !enabledModes.includes(d.mode)) p.push("the default mode is not enabled");
    if (typeof d.resolution !== "string" || !enabledRes.includes(d.resolution)) p.push("the default resolution is not enabled");
    if (!isInt(d.num_frames) || !allowed.includes(d.num_frames)) p.push("the default frame count is not allowed");
    if (!isInt(d.steps) || (stepsR && (d.steps < stepsR[0] || d.steps > stepsR[1]))) p.push("the default step count is out of range");
    if (!isInt(d.fps) || (fpsR && (d.fps < fpsR[0] || d.fps > fpsR[1]))) p.push("the default fps is out of range");
    if (!isNum(d.guidance_scale) || (gsR && (d.guidance_scale < gsR[0] || d.guidance_scale > gsR[1]))) p.push("the default guidance scale is out of range");
  }
  return p;
}

// ------------------------------------------------------------------ the store
export interface LimitsStoreOptions {
  fetcher: (signal: AbortSignal) => Promise<Limits>;
  now?: () => number;
  setTimer?: (fn: () => void, ms: number) => unknown;
  clearTimer?: (handle: unknown) => void;
  /** How long a fetched answer is trusted before it is re-checked (default 5 minutes). */
  ttlMs?: number;
  /** Automatic retries after a retryable failure, with exponential backoff (default 5). Manual retry() is always available. */
  maxAutoRetries?: number;
}

export interface LimitsStore {
  getState(): LimitsState;
  subscribe(listener: () => void): () => void;
  /** Start over now: clears any pending automatic retry and fetches immediately. */
  retry(): void;
  /** Re-check the limits if the cached answer is older than the TTL (or always, with `force`). A ready state stays ready while it runs. */
  refresh(force?: boolean): void;
}

export const RETRY_BASE_MS = 1_000;
export const RETRY_MAX_MS = 30_000;
export const retryDelay = (attempt: number) => Math.min(RETRY_BASE_MS * 2 ** (attempt - 1), RETRY_MAX_MS);

export function createLimitsStore(opts: LimitsStoreOptions): LimitsStore {
  const now = opts.now ?? (() => Date.now());
  const setTimer = opts.setTimer ?? ((fn, ms) => setTimeout(fn, ms));
  const clearTimer = opts.clearTimer ?? ((h) => clearTimeout(h as ReturnType<typeof setTimeout>));
  const ttl = opts.ttlMs ?? 5 * 60_000;
  const maxAuto = opts.maxAutoRetries ?? 5;

  let state: LimitsState = { kind: "loading", attempt: 1 };
  let started = false;
  let subscribers = 0;
  let controller: AbortController | null = null;
  let timer: unknown = null;
  let generation = 0;                                   // invalidates the results of superseded fetches
  const listeners = new Set<() => void>();

  const set = (next: LimitsState) => { state = next; listeners.forEach((l) => l()); };
  const cancelTimer = () => { if (timer !== null) { clearTimer(timer); timer = null; } };

  function fail(err: unknown, attempt: number) {
    const error = err instanceof ApiError && err.status === 404 ? describeError(new ApiError("not found", { status: 404, code: "limits_not_supported" })) : describeError(err);
    const retryInMs = error.retryable && attempt <= maxAuto ? retryDelay(attempt) : null;
    set({ kind: "unavailable", error, attempt, retryInMs });
    if (retryInMs !== null && subscribers > 0) {
      timer = setTimer(() => { timer = null; run(attempt + 1, true); }, retryInMs);
    }
  }

  // Callers never start a second request while one is in flight (subscribe only starts the first, refresh() checks, retry() aborts first).
  /** `visible`: show the loading state while fetching (false for a silent background refresh that keeps a ready state). */
  function run(attempt: number, visible: boolean) {
    cancelTimer();
    started = true;
    const mine = ++generation;
    const c = (controller = new AbortController());
    if (visible || state.kind !== "ready") set({ kind: "loading", attempt });
    opts.fetcher(c.signal).then(
      (limits) => {
        if (mine !== generation) return;
        controller = null;
        const problems = validateLimits(limits);
        if (problems.length > 0) {
          fail(new ApiError("The server's limits could not be verified.", { status: 200, code: "limits_invalid" }), attempt);
        } else {
          set({ kind: "ready", limits, fetchedAt: now() });
        }
      },
      (err: unknown) => {
        if (mine !== generation) return;
        controller = null;
        fail(err, attempt);
      },
    );
  }

  function abortAll() {
    generation++;
    cancelTimer();
    if (controller) { controller.abort(); controller = null; }
  }

  return {
    getState: () => state,
    subscribe(listener) {
      listeners.add(listener);
      subscribers++;
      if (!started) run(1, true);
      else if (state.kind === "ready" && now() - state.fetchedAt >= ttl) run(1, false);
      else if (state.kind === "unavailable" && timer === null && !controller) run(1, true);        // a new viewer gets a fresh attempt
      return () => {
        listeners.delete(listener);
        subscribers--;
        if (subscribers === 0) {                          // nobody is looking: stop all work; the state is kept for the next subscriber
          abortAll();
          if (state.kind === "loading") state = { kind: "loading", attempt: 1 };
          started = state.kind !== "loading";
        }
      };
    },
    retry() {
      abortAll();
      run(1, true);
    },
    refresh(force = false) {
      if (!force && state.kind === "ready" && now() - state.fetchedAt < ttl) return;
      if (controller) return;
      abortAll();
      run(1, false);
    },
  };
}

let defaultStore: LimitsStore | null = null;
/** The shared store the app uses (created on first use). Every component sees one fetch and one cache. */
export function getDefaultLimitsStore(): LimitsStore {
  return (defaultStore ??= createLimitsStore({ fetcher: (signal) => getLimits(signal) }));
}
