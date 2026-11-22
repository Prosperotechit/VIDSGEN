// Low-level HTTP client for the VIDSGEN API. This is the only module that calls fetch().
// The frontend talks ONLY to the VIDSGEN FastAPI backend: never to ComfyUI, Hugging Face
// or any GPU provider directly.

import { redactSensitive } from "./redact";

const RAW_BASE = (import.meta.env.VITE_API_BASE_URL ?? "").trim();
/** API origin without trailing slash. Empty string = same origin (dev proxy / same domain). */
export const API_BASE_URL = RAW_BASE.replace(/\/+$/, "");

/** Turns an API path (e.g. a job's download_url) into a URL the browser can load. */
export function apiUrl(path: string): string {
  if (/^https?:\/\//i.test(path)) return path;
  return `${API_BASE_URL}${path.startsWith("/") ? path : `/${path}`}`;
}

export interface FieldError {
  field: string;
  message: string;
}

/** Normalised error for every failure mode: HTTP errors, validation, network, timeouts. */
export class ApiError extends Error {
  readonly status: number;            // 0 = network error / no response
  readonly code: string;
  readonly fieldErrors: FieldError[];

  constructor(message: string, opts: { status: number; code: string; fieldErrors?: FieldError[] }) {
    super(redactSensitive(message));      // nothing that reaches the UI may carry a credential, URL, address or file path
    this.name = "ApiError";
    this.status = opts.status;
    this.code = opts.code;
    this.fieldErrors = opts.fieldErrors ?? [];
  }

  get isNetworkError() {
    return this.status === 0;
  }
}

interface ValidationItem {
  loc?: (string | number)[];
  msg?: string;
}

// Pydantic prefixes custom validator messages with "Value error, ". Strip it for display.
const cleanMsg = (m: string) => redactSensitive(m.replace(/^Value error,\s*/i, ""));

/** Parses the backend's two error formats into an ApiError. Exported for tests. */
export function parseErrorBody(status: number, body: unknown): ApiError {
  const detail = (body as { detail?: unknown } | null)?.detail;

  // FastAPI validation errors: {"detail": [{"loc": ["body", "prompt"], "msg": "..."}]}
  if (Array.isArray(detail)) {
    const fieldErrors = (detail as ValidationItem[]).map((d) => ({
      field: String(d.loc?.[d.loc.length - 1] ?? "request"),
      message: cleanMsg(d.msg ?? "Invalid value"),
    }));
    const first = fieldErrors[0];
    return new ApiError(first ? `${first.field}: ${first.message}` : "Invalid request", {
      status, code: "validation_error", fieldErrors,
    });
  }
  // VIDSGEN errors: {"detail": {"code": "...", "message": "..."}}
  if (detail && typeof detail === "object" && "message" in detail) {
    const d = detail as { code?: string; message: string };
    return new ApiError(d.message, { status, code: d.code ?? `http_${status}` });
  }
  if (typeof detail === "string") return new ApiError(detail, { status, code: `http_${status}` });
  return new ApiError(`Request failed (HTTP ${status})`, { status, code: `http_${status}` });
}

export async function request<T>(path: string, init: RequestInit & { timeoutMs?: number } = {}): Promise<T> {
  const { timeoutMs = 30_000, signal, ...rest } = init;
  const timeout = AbortSignal.timeout(timeoutMs);
  const combined = signal ? AbortSignal.any([signal, timeout]) : timeout;

  let res: Response;
  try {
    res = await fetch(apiUrl(path), { ...rest, signal: combined, headers: { Accept: "application/json", ...rest.headers } });
  } catch (e) {
    if (signal?.aborted) throw e;  // caller cancelled: propagate AbortError untouched
    const timedOut = timeout.aborted;
    throw new ApiError(
      timedOut ? "The server took too long to respond." : "Can't reach the VIDSGEN server. Check that the API is running.",
      { status: 0, code: timedOut ? "timeout" : "network_error" },
    );
  }

  if (!res.ok) {
    let body: unknown = null;
    try {
      body = await res.json();
    } catch {
      /* non-JSON error body */
    }
    throw parseErrorBody(res.status, body);
  }
  return (await res.json()) as T;
}
