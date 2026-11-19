import { useEffect, useState } from "react";
import { getHealth } from "../api/jobs";
import type { Health } from "../api/types";

export type HealthState =
  | { kind: "loading" }
  | { kind: "ok" | "degraded"; health: Health }
  | { kind: "offline"; message: string };

/** Checks GET /api/health on mount and every `intervalMs` (default 30 s). */
export function useHealth(intervalMs = 30_000): HealthState {
  const [state, setState] = useState<HealthState>({ kind: "loading" });

  useEffect(() => {
    const controller = new AbortController();
    let timer: number | undefined;
    const check = async () => {
      try {
        const health = await getHealth(controller.signal);
        setState({ kind: health.status, health });
      } catch (e) {
        if (controller.signal.aborted) return;
        setState({ kind: "offline", message: e instanceof Error ? e.message : "API unreachable" });
      }
      timer = window.setTimeout(check, intervalMs);
    };
    check();
    return () => {
      controller.abort();
      window.clearTimeout(timer);
    };
  }, [intervalMs]);

  return state;
}
