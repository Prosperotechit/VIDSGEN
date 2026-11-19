import { afterEach, describe, expect, it, vi } from "vitest";
import { nextDelay, POLL } from "../hooks/useJobPolling";
import { apiUrl, ApiError, parseErrorBody, request } from "./client";
import { constraints, durationSeconds, frameOptions } from "./constraints";
import { toFormData } from "./jobs";
import type { Job } from "./types";

describe("parseErrorBody", () => {
  it("maps FastAPI validation errors to field errors and strips the 'Value error' prefix", () => {
    const e = parseErrorBody(422, {
      detail: [{ loc: ["body", "num_frames"], msg: "Value error, num_frames must be 4k+1 (e.g. 17, 49, 81, 121)" }],
    });
    expect(e.status).toBe(422);
    expect(e.code).toBe("validation_error");
    expect(e.fieldErrors).toEqual([{ field: "num_frames", message: "num_frames must be 4k+1 (e.g. 17, 49, 81, 121)" }]);
  });

  it("maps VIDSGEN {code, message} errors", () => {
    const e = parseErrorBody(409, { detail: { code: "job_not_cancellable", message: "Job is already completed" } });
    expect(e.code).toBe("job_not_cancellable");
    expect(e.message).toBe("Job is already completed");
  });

  it("falls back for unknown bodies", () => {
    expect(parseErrorBody(502, null).message).toBe("Request failed (HTTP 502)");
    expect(parseErrorBody(500, { detail: "boom" }).message).toBe("boom");
  });
});

describe("request", () => {
  afterEach(() => vi.unstubAllGlobals());

  it("turns network failures into a friendly ApiError with status 0", async () => {
    vi.stubGlobal("fetch", vi.fn().mockRejectedValue(new TypeError("Failed to fetch")));
    const err = await request<never>("/api/health").catch((e: ApiError) => e);
    expect(err).toBeInstanceOf(ApiError);
    expect(err.isNetworkError).toBe(true);
    expect(err.code).toBe("network_error");
  });

  it("parses JSON error responses", async () => {
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(new Response(
      JSON.stringify({ detail: { code: "job_not_found", message: "No job with id x" } }), { status: 404 })));
    const err = await request<never>("/api/jobs/x").catch((e: ApiError) => e);
    expect(err.status).toBe(404);
    expect(err.code).toBe("job_not_found");
  });
});

describe("apiUrl", () => {
  it("keeps absolute URLs and prefixes relative API paths", () => {
    expect(apiUrl("https://cdn.example.com/v.mp4")).toBe("https://cdn.example.com/v.mp4");
    expect(apiUrl("/api/jobs/1/download")).toBe("/api/jobs/1/download"); // same-origin default
  });
});

describe("toFormData", () => {
  const base = { prompt: "a fox", resolution: "832x480", num_frames: 49, fps: 24, steps: 30, guidance_scale: 5 } as const;

  it("builds a t2v body without image or seed when random", () => {
    const fd = toFormData({ ...base, mode: "t2v", seed: null, image: new File(["x"], "ignored.png", { type: "image/png" }) });
    expect(fd.get("mode")).toBe("t2v");
    expect(fd.get("num_frames")).toBe("49");
    expect(fd.get("guidance_scale")).toBe("5");
    expect(fd.has("seed")).toBe(false);
    expect(fd.has("image")).toBe(false); // never send an image for t2v
  });

  it("includes the image and an explicit seed for i2v", () => {
    const img = new File(["img"], "start.png", { type: "image/png" });
    const fd = toFormData({ ...base, mode: "i2v", seed: 0, image: img, negative_prompt: "blurry" });
    expect(fd.get("seed")).toBe("0");
    expect((fd.get("image") as File).name).toBe("start.png");
    expect(fd.get("negative_prompt")).toBe("blurry");
  });
});

describe("constraints (generated from the backend)", () => {
  it("offers only valid 4k+1 frame counts", () => {
    expect(frameOptions[0]).toBe(17);
    expect(frameOptions.at(-1)).toBe(121);
    expect(frameOptions.every((n) => (n - 1) % 4 === 0)).toBe(true);
    expect(frameOptions).toContain(constraints.numFrames.default);
  });

  it("contains the backend's resolutions and default", () => {
    expect(constraints.resolutions.map((r) => r.value)).toContain(constraints.defaultResolution);
    expect(durationSeconds(121, 24)).toBe(5.04);
  });
});

describe("polling backoff", () => {
  const job = (status: Job["status"], progress: number) => ({ status, progress }) as Job;

  it("polls quickly while progress moves and backs off when it stalls", () => {
    expect(nextDelay(job("processing", 10), job("processing", 5), POLL.processing, false)).toBe(POLL.processing);
    let d: number = POLL.processing;
    for (let i = 0; i < 10; i++) d = nextDelay(job("processing", 50), job("processing", 50), d, false);
    expect(d).toBe(POLL.processingMax);
  });

  it("uses slower intervals for queued jobs and hidden tabs", () => {
    expect(nextDelay(job("queued", 0), null, POLL.processing, false)).toBe(POLL.queued);
    expect(nextDelay(job("processing", 10), job("processing", 5), POLL.processing, true)).toBe(POLL.hidden);
  });
});
