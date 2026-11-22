// Turns API failures into messages a person can act on, without ever exposing credentials or internal infrastructure details.
//
// Rule: the server's raw text is shown only when it is a short, clean validation message (see safeMessage). Provider and GPU-host
// failures are described in our own words, by error code first and by HTTP status second.
import { ApiError, type FieldError } from "./client";
import { safeMessage } from "./redact";

export type ErrorKind =
  | "capacity" | "busy" | "connection" | "timeout" | "auth" | "validation" | "unavailable"
  | "rejected" | "limits" | "not_found" | "conflict" | "unknown";

export const PARAM_FIELDS = ["mode", "resolution", "num_frames", "steps", "fps", "guidance_scale", "seed", "prompt", "negative_prompt", "image"] as const;
export type ParamField = (typeof PARAM_FIELDS)[number];
export type FieldMessages = Partial<Record<ParamField, string>>;

export interface DescribedError {
  kind: ErrorKind;
  title: string;
  message: string;
  /** True if trying again can help (a busy or unreachable server), false if the request itself must change. */
  retryable: boolean;
  /** The machine-readable code (letters, digits and underscores only), or "unknown". */
  code: string;
  status: number;
  fieldErrors: FieldMessages;
  /** Validation problems that are not about a known parameter. */
  formError: string | null;
}

interface Entry { kind: ErrorKind; title: string; message: string; retryable: boolean }
const e = (kind: ErrorKind, title: string, message: string, retryable: boolean): Entry => ({ kind, title, message, retryable });

const GPU_MEMORY = e("capacity", "Not enough GPU memory", "The GPU ran out of memory for this request. Try a smaller resolution or fewer frames.", false);
const SYSTEM_MEMORY = e("capacity", "Not enough server memory", "The generation server ran out of memory for this request. Try a smaller resolution or fewer frames.", false);
const BUSY = e("busy", "The server is busy", "Too many requests are being processed right now. Wait a moment and try again.", true);
const LOST = e("connection", "Connection to the generation server was lost", "The connection to the generation server was interrupted. Check that it is running, then try again.", true);
const NO_REPLY = e("timeout", "The generation server did not respond", "The generation server took too long to respond. Try again in a moment.", true);
const AUTH = e("auth", "Generation service not authorised", "This server isn't authorised to use the generation service. Ask the administrator to check its configuration.", false);
const DOWN = e("unavailable", "Generation isn't available right now", "The generation service can't take requests at the moment. Try again later.", true);
const NOT_SET_UP = e("unavailable", "Generation isn't set up", "The generation service isn't fully set up on the server yet. Ask the administrator to check it.", false);
const REJECTED = e("rejected", "The request was rejected", "The generation service rejected this request. Try different settings.", false);
const BAD_OUTPUT = e("unavailable", "The video couldn't be produced", "The generation service didn't return a usable video. Try again.", true);
const LIMITS_DOWN = e("limits", "Server limits unavailable", "Couldn't determine what this server supports, so generation is turned off. Try again in a moment.", true);
const UNEXPECTED = e("unknown", "Something went wrong", "An unexpected error occurred. Please try again.", true);

const BY_CODE: Readonly<Record<string, Entry>> = {
  // GPU / provider failures (job errors and API errors)
  gpu_out_of_memory: GPU_MEMORY,
  system_out_of_memory: SYSTEM_MEMORY,
  comfyui_busy: BUSY,
  busy: BUSY,
  comfyui_connection_lost: LOST,
  comfyui_unreachable: LOST,
  comfyui_job_lost: e("connection", "The generation was lost", "The generation server no longer knows this job (it may have restarted). Please submit it again.", true),
  comfyui_timeout: NO_REPLY,
  timeout: e("timeout", "The generation took too long", "The generation didn't finish in time and was stopped. Try fewer frames or steps.", true),
  comfyui_server_error: DOWN,
  comfyui_bad_response: DOWN,
  provider_error: DOWN,
  unknown_provider_job: DOWN,
  result_not_ready: DOWN,
  comfyui_auth_failed: AUTH,
  comfyui_rejected: REJECTED,
  workflow_error: NOT_SET_UP,
  model_missing: NOT_SET_UP,
  missing_image: e("validation", "An image is required", "Image-to-video needs a start image.", false),
  no_video_output: BAD_OUTPUT,
  unsupported_output: BAD_OUTPUT,
  output_download_failed: BAD_OUTPUT,
  interrupted: e("unavailable", "The server restarted", "The server restarted while this job was running. Please submit it again.", true),
  internal_error: UNEXPECTED,
  // limits endpoint
  limits_unavailable: LIMITS_DOWN,
  limits_invalid: LIMITS_DOWN,
  limits_not_supported: e("limits", "Server limits unavailable", "This server doesn't report its generation limits (it may be an older version), so generation is turned off.", false),
  // requests and jobs
  job_not_found: e("not_found", "Not found", "That generation no longer exists.", false),
  job_not_cancellable: e("conflict", "Already finished", "That generation has already finished and can't be cancelled.", false),
  job_not_completed: e("conflict", "Not ready yet", "The video isn't ready yet.", false),
  result_missing: e("not_found", "Video unavailable", "The video file is no longer available.", false),
  input_missing: e("not_found", "Image unavailable", "The uploaded image is no longer available. Please upload it again.", false),
  request_too_large: e("validation", "Upload too large", "The upload is too large.", false),
  invalid_image: e("validation", "Invalid image", "That image can't be used. Try a different file.", false),
  network_error: e("connection", "Can't reach the VIDSGEN server", "Can't reach the VIDSGEN server. Check that the API is running.", true),
};

const BY_STATUS: Readonly<Record<number, Entry>> = {
  0: e("connection", "Can't reach the VIDSGEN server", "Can't reach the VIDSGEN server. Check that the API is running.", true),
  401: AUTH, 403: AUTH,
  404: e("not_found", "Not found", "That wasn't found on the server.", false),
  409: e("conflict", "Conflict", "That can't be done in the job's current state.", false),
  413: e("validation", "Request too large", "The request is too large.", false),
  422: e("validation", "Check your settings", "Some settings aren't valid for this server.", false),
  429: BUSY,
  500: UNEXPECTED,
  502: DOWN, 503: DOWN, 504: NO_REPLY,
};

const CODE_SHAPE = /^[a-z0-9_]{1,64}$/;

const FIELD_FALLBACKS: Readonly<Record<ParamField, string>> = {
  mode: "That generation mode isn't available on this server.",
  resolution: "That resolution isn't available on this server. Pick one that is offered.",
  num_frames: "That number of frames isn't allowed. Pick one of the offered lengths.",
  steps: "That number of steps isn't allowed on this server.",
  fps: "That frame rate isn't allowed.",
  guidance_scale: "That guidance scale isn't allowed.",
  seed: "That seed isn't valid.",
  prompt: "Check the prompt.",
  negative_prompt: "Check the negative prompt.",
  image: "Check the image.",
};

const isParamField = (name: string): name is ParamField => (PARAM_FIELDS as readonly string[]).includes(name);

/** Per-parameter messages for a validation error, plus a form-level message for anything unrecognised. */
export function mapFieldErrors(errors: readonly FieldError[]): { fields: FieldMessages; form: string | null } {
  const fields: FieldMessages = {};
  let form: string | null = null;
  for (const fe of errors) {
    if (isParamField(fe.field)) {
      if (!fields[fe.field]) fields[fe.field] = safeMessage(fe.message, FIELD_FALLBACKS[fe.field]);   // the first message per field wins
    } else if (form === null) {
      form = "Some settings aren't valid for this server.";                                            // never echo an unknown field's text
    }
  }
  return { fields, form };
}

function build(entry: Entry, code: string, status: number, fields: FieldMessages = {}, form: string | null = null): DescribedError {
  return { ...entry, code, status, fieldErrors: fields, formError: form };
}

/** Describes any error thrown while talking to the API (an ApiError, or something unexpected). */
export function describeError(err: unknown): DescribedError {
  if (!(err instanceof ApiError)) return build(UNEXPECTED, "unknown", 0);
  const code = CODE_SHAPE.test(err.code) ? err.code : "unknown";
  if (err.code === "validation_error" || (err.status === 422 && err.fieldErrors.length > 0)) {
    const { fields, form } = mapFieldErrors(err.fieldErrors);
    return build(BY_STATUS[422]!, "validation_error", err.status, fields, form);
  }
  const entry = BY_CODE[code] ?? BY_STATUS[err.status] ?? (err.status >= 500 ? DOWN : UNEXPECTED);
  return build(entry, code, err.status);
}

/** Describes the `error` object stored on a failed job: {code, message}. The message is deliberately not shown. */
export function describeJobError(error: { code?: string | null; message?: string | null } | null | undefined): DescribedError {
  const code = error?.code && CODE_SHAPE.test(error.code) ? error.code : "unknown";
  return build(BY_CODE[code] ?? UNEXPECTED, code, 0);
}
