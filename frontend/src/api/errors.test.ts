import { describe, expect, it } from "vitest";
import { ApiError, parseErrorBody } from "./client";
import { describeError, describeJobError, mapFieldErrors, PARAM_FIELDS, type ErrorKind } from "./errors";

const api = (status: number, code: string, message = "server text") => new ApiError(message, { status, code });

describe("friendly messages for every kind of failure", () => {
  it.each<[string, string, number, ErrorKind, boolean]>([
    ["GPU out of memory", "gpu_out_of_memory", 200, "capacity", false],
    ["system out of memory", "system_out_of_memory", 200, "capacity", false],
    ["server busy (code)", "comfyui_busy", 200, "busy", true],
    ["server busy (HTTP 429)", "http_429", 429, "busy", true],
    ["connection lost", "comfyui_connection_lost", 200, "connection", true],
    ["server unreachable", "comfyui_unreachable", 200, "connection", true],
    ["job lost", "comfyui_job_lost", 200, "connection", true],
    ["network failure", "network_error", 0, "connection", true],
    ["no reply from the generation server", "comfyui_timeout", 200, "timeout", true],
    ["client timeout", "timeout", 0, "timeout", true],
    ["authentication failure (code)", "comfyui_auth_failed", 200, "auth", false],
    ["HTTP 401", "http_401", 401, "auth", false],
    ["HTTP 403", "http_403", 403, "auth", false],
    ["inference unavailable (5xx from the GPU host)", "comfyui_server_error", 200, "unavailable", true],
    ["inference unavailable (bad response)", "comfyui_bad_response", 200, "unavailable", true],
    ["HTTP 503", "http_503", 503, "unavailable", true],
    ["model not installed", "model_missing", 200, "unavailable", false],
    ["workflow problem", "workflow_error", 200, "unavailable", false],
    ["no video produced", "no_video_output", 200, "unavailable", true],
    ["rejected by the GPU host", "comfyui_rejected", 200, "rejected", false],
    ["limits unavailable", "limits_unavailable", 500, "limits", true],
    ["limits not supported by an older server", "limits_not_supported", 404, "limits", false],
    ["limits failed verification", "limits_invalid", 200, "limits", true],
    ["job not found", "job_not_found", 404, "not_found", false],
    ["cannot cancel a finished job", "job_not_cancellable", 409, "conflict", false],
    ["upload too large", "request_too_large", 413, "validation", false],
    ["invalid image", "invalid_image", 415, "validation", false],
    ["internal error", "internal_error", 500, "unknown", true],
    ["an unknown code on HTTP 500", "something_new", 500, "unknown", true],
    ["an unknown code on HTTP 502", "something_new", 502, "unavailable", true],
    ["an unknown code on HTTP 418", "something_new", 418, "unknown", true],
  ])("%s", (_name, code, status, kind, retryable) => {
    const d = describeError(api(status, code));
    expect(d.kind).toBe(kind);
    expect(d.retryable).toBe(retryable);
    expect(d.title.length).toBeGreaterThan(3);
    expect(d.message.length).toBeGreaterThan(10);
    expect(d.status).toBe(status);
  });

  it("describes GPU memory failures with advice the person can act on", () => {
    expect(describeError(api(200, "gpu_out_of_memory")).message).toMatch(/smaller resolution or fewer frames/);
  });
  it("describes a busy server as something to retry shortly", () => {
    expect(describeError(api(429, "http_429")).message).toMatch(/try again/i);
  });
  it("tells a person with an auth failure to ask the administrator, without mentioning tokens", () => {
    const d = describeError(api(200, "comfyui_auth_failed", "The GPU server rejected the API token (check COMFYUI_API_TOKEN)."));
    expect(d.message).toMatch(/administrator/i);
    expect(`${d.title} ${d.message}`.toLowerCase()).not.toMatch(/token|comfyui|api_key|secret/);     // (the short code is an identifier, not shown text)
  });
  it("treats anything that is not an ApiError as unexpected", () => {
    for (const thing of [new Error("boom at /srv/app/x.py"), "text", null, undefined, { code: "gpu_out_of_memory" }]) {
      const d = describeError(thing);
      expect(d.kind).toBe("unknown");
      expect(d.message).toBe("An unexpected error occurred. Please try again.");
    }
  });
});

describe("no credentials or infrastructure details in any description", () => {
  const nasty = [
    "connect ECONNREFUSED 127.0.0.1:8188", "https://abc-def.trycloudflare.com/prompt failed", "Bearer sk-abcdefghijklmnopqrstu",
    "File \"/kaggle/tmp/vidsgen-gpu/ComfyUI/execution.py\", line 5", "COMFYUI_API_TOKEN=supersecretvalue123", "gpu-box.lan:8188 timed out",
    "hf_abcdefghijklmnopqrstuvwxyz", "C:\\Users\\me\\.env",
  ];
  const codes = ["gpu_out_of_memory", "comfyui_rejected", "comfyui_server_error", "comfyui_auth_failed", "comfyui_unreachable", "model_missing", "something_new",
    "http_500", "http_502", "limits_unavailable", "internal_error"];
  const FORBIDDEN = /127\.0\.0\.1|trycloudflare|https?:|Bearer|sk-abc|\/kaggle|\/srv|COMFYUI|supersecret|gpu-box|hf_abc|C:\\|ECONNREFUSED|\.env/i;

  it.each(nasty)("a hostile server message is never repeated: %j", (message) => {
    for (const code of codes) {
      for (const status of [0, 200, 401, 404, 409, 422, 429, 500, 502, 503]) {
        const d = describeError(new ApiError(message, { status, code }));
        expect(`${d.title} ${d.message} ${d.formError ?? ""} ${Object.values(d.fieldErrors).join(" ")}`).not.toMatch(FORBIDDEN);
      }
    }
  });

  it("ApiError messages themselves are redacted at the source", () => {
    expect(new ApiError("failed at https://gpu.example.com with Bearer abcdefghijkl", { status: 500, code: "x" }).message).not.toMatch(/gpu\.example|abcdefghijkl/);
    const parsed = parseErrorBody(500, { detail: { code: "provider_error", message: "cannot reach http://10.1.2.3:8188/ws" } });
    expect(parsed.message).not.toMatch(/10\.1\.2\.3/);
    const fieldParsed = parseErrorBody(422, { detail: [{ loc: ["body", "steps"], msg: "Value error, bad at /srv/secret/path/file.py" }] });
    expect(fieldParsed.fieldErrors[0]?.message).not.toMatch(/\/srv/);
  });

  it("codes are limited to a safe shape", () => {
    const d = describeError(new ApiError("x", { status: 500, code: "Bearer abcdef <script>" }));
    expect(d.code).toBe("unknown");
    expect(d.kind).toBe("unknown");
  });

  it("job errors are described by code only, never by the stored message", () => {
    const d = describeJobError({ code: "comfyui_unreachable", message: "cannot reach https://gpu.example.com:8443" });
    expect(d.kind).toBe("connection");
    expect(`${d.title} ${d.message}`).not.toMatch(/gpu\.example|8443|https/);
    expect(describeJobError({ code: "gpu_out_of_memory", message: "x" }).kind).toBe("capacity");
    expect(describeJobError({ code: "timeout", message: "x" }).kind).toBe("timeout");
    expect(describeJobError({ code: "interrupted", message: "x" }).retryable).toBe(true);
    expect(describeJobError(null).kind).toBe("unknown");
    expect(describeJobError({ code: "evil code!", message: "x" }).code).toBe("unknown");
  });
});

describe("validation errors for every generation parameter", () => {
  const body = (field: string, msg: string) => parseErrorBody(422, { detail: [{ loc: ["body", field], msg: `Value error, ${msg}` }] });

  it.each([
    ["mode", "image-to-video is not enabled on this server; allowed: t2v. This GPU server only accepts requests it has been verified for."],
    ["resolution", "resolution 1280x704 is not enabled on this server; allowed: 832x480."],
    ["num_frames", "num_frames 49 exceeds this server's maximum of 17."],
    ["steps", "steps 30 exceeds this server's maximum of 20."],
    ["fps", "fps must be between 8 and 30"],
    ["guidance_scale", "guidance_scale must be between 1 and 15"],
    ["seed", "seed must be between 0 and 4294967295"],
    ["prompt", "prompt must not be empty"],
    ["negative_prompt", "negative_prompt must be at most 2000 characters"],
    ["image", "an image is required for image-to-video (mode=i2v)"],
  ])("maps %s to its own field with the server's explanation", (field, message) => {
    const d = describeError(body(field, message));
    expect(d.kind).toBe("validation");
    expect(d.code).toBe("validation_error");
    expect(d.fieldErrors[field as keyof typeof d.fieldErrors]).toBe(message);
    expect(d.formError).toBeNull();
  });

  it("covers every parameter the API accepts", () => {
    expect([...PARAM_FIELDS].sort()).toEqual(["fps", "guidance_scale", "image", "mode", "negative_prompt", "num_frames", "prompt", "resolution", "seed", "steps"]);
  });

  it("uses a clear fallback when the server's text is empty, too long or sensitive", () => {
    for (const field of PARAM_FIELDS) {
      for (const msg of ["", "x ".repeat(300), "failed at https://gpu.example.com/x", "Bearer abcdefghijkl"]) {
        const { fields } = mapFieldErrors([{ field, message: msg }]);
        expect(fields[field]).toBeTruthy();
        expect(fields[field]).not.toMatch(/gpu\.example|Bearer|xx xx/);
      }
    }
  });

  it("collects several fields at once, keeping the first message per field", () => {
    const { fields } = mapFieldErrors([
      { field: "steps", message: "steps 30 exceeds this server's maximum of 20." },
      { field: "num_frames", message: "num_frames 49 exceeds this server's maximum of 17." },
      { field: "steps", message: "second message" },
    ]);
    expect(fields).toEqual({ steps: "steps 30 exceeds this server's maximum of 20.", num_frames: "num_frames 49 exceeds this server's maximum of 17." });
  });

  it("puts unknown fields in a generic form message without echoing them", () => {
    const { fields, form } = mapFieldErrors([{ field: "secret_internal_field", message: "boom at /srv/x/y" }]);
    expect(fields).toEqual({});
    expect(form).toBe("Some settings aren't valid for this server.");
  });

  it("handles a validation error that carries no fields", () => {
    const d = describeError(new ApiError("Invalid request", { status: 422, code: "validation_error" }));
    expect(d.kind).toBe("validation");
    expect(d.fieldErrors).toEqual({});
  });
});
