import { useEffect, useSyncExternalStore } from "react";
import { canGenerate, getDefaultLimitsStore, type LimitsState, type LimitsStore } from "../api/limits";
import type { Limits } from "../api/types";

export interface UseLimits {
  state: LimitsState;
  /** The active limits, only while they are known and verified. */
  limits: Limits | null;
  /** False until safe active limits are available (loading, or unavailable): the generate action must stay disabled. */
  canGenerate: boolean;
  /** Fetch again right now (resets the automatic retry schedule). */
  retry: () => void;
}

/** Fetches, caches and shares the server's active generation limits. Never falls back to static constraints. */
export function useLimits(store: LimitsStore = getDefaultLimitsStore()): UseLimits {
  const state = useSyncExternalStore(store.subscribe, store.getState, store.getState);
  useEffect(() => { store.refresh(); }, [store]);          // re-check a stale cached answer when a component mounts
  return { state, limits: state.kind === "ready" ? state.limits : null, canGenerate: canGenerate(state), retry: store.retry };
}
