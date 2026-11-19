import { useCallback, useEffect, useRef, useState } from "react";
import { ApiError } from "../api/client";
import { getJob } from "../api/jobs";
import { isTerminal, type Job } from "../api/types";

/** Polling intervals (ms). Exported so tests can reason about them. */
export const POLL = {
  processing: 1500,     // while progress is moving
  processingMax: 6000,  // when progress hasn't changed for a while
  queued: 3000,
  hidden: 15000,        // tab in background
  errorMax: 20000,      // network trouble: exponential backoff up to this
} as const;

/** Next delay given the latest job, the previous one, and the current delay. Pure, for testing. */
export function nextDelay(job: Job, prev: Job | null, current: number, hidden: boolean): number {
  if (hidden) return POLL.hidden;
  if (job.status === "queued") return POLL.queued;
  const moved = !prev || prev.progress !== job.progress || prev.status !== job.status;
  return moved ? POLL.processing : Math.min(Math.round(current * 1.5), POLL.processingMax);
}

interface State {
  job: Job | null;
  error: ApiError | null;     // last error (cleared on the next success)
  notFound: boolean;
}

/**
 * Keeps a job up to date by polling GET /api/jobs/{id} until it reaches a terminal state.
 * Pass `initial` (e.g. the POST response) to render immediately without a round trip.
 */
export function useJobPolling(jobId: string | null, initial?: Job | null) {
  const [state, setState] = useState<State>({ job: initial ?? null, error: null, notFound: false });
  const timer = useRef<number | undefined>(undefined);
  const delay = useRef<number>(POLL.processing);
  const errDelay = useRef<number>(2000);
  const kick = useRef<() => void>(() => {});

  useEffect(() => {
    if (!jobId) {
      setState({ job: null, error: null, notFound: false });
      return;
    }
    const controller = new AbortController();
    let prev: Job | null = initial && initial.id === jobId ? initial : null;
    setState({ job: prev, error: null, notFound: false });
    delay.current = POLL.processing;
    errDelay.current = 2000;

    const schedule = (ms: number) => {
      window.clearTimeout(timer.current);
      timer.current = window.setTimeout(tick, ms);
    };

    async function tick() {
      try {
        const job = await getJob(jobId!, controller.signal);
        errDelay.current = 2000;
        setState({ job, error: null, notFound: false });
        if (isTerminal(job.status)) return; // done: stop polling
        delay.current = nextDelay(job, prev, delay.current, document.hidden);
        prev = job;
        schedule(delay.current);
      } catch (e) {
        if (controller.signal.aborted) return;
        const err = e instanceof ApiError ? e : new ApiError(String(e), { status: 0, code: "unknown" });
        if (err.status === 404) {
          setState((s) => ({ ...s, error: err, notFound: true }));
          return;
        }
        setState((s) => ({ ...s, error: err }));
        errDelay.current = Math.min(errDelay.current * 2, POLL.errorMax);
        schedule(errDelay.current);
      }
    }

    kick.current = () => schedule(0);
    // Skip the first fetch if we already have a terminal job (e.g. opened from history).
    if (!(prev && isTerminal(prev.status))) schedule(prev ? POLL.processing : 0);

    // When the tab becomes visible again, refresh at once instead of waiting.
    const onVisible = () => {
      if (!document.hidden) kick.current();
    };
    document.addEventListener("visibilitychange", onVisible);
    return () => {
      controller.abort();
      window.clearTimeout(timer.current);
      document.removeEventListener("visibilitychange", onVisible);
    };
    // `initial` only seeds the first render for a given id.
  }, [jobId]);

  /** Apply a job returned by another call (e.g. cancel) and resume polling if needed. */
  const update = useCallback((job: Job) => {
    setState({ job, error: null, notFound: false });
    if (!isTerminal(job.status)) kick.current();
  }, []);

  return { ...state, update, refresh: () => kick.current() };
}
