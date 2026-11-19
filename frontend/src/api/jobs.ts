// Typed functions for every VIDSGEN API endpoint the UI uses.
import { apiUrl, request } from "./client";
import type { CreateJobRequest, Health, Job, JobList, JobStatus } from "./types";

export function getHealth(signal?: AbortSignal) {
  return request<Health>("/api/health", { signal, timeoutMs: 10_000 });
}

/** Builds the multipart body exactly as the backend's Form/File parameters expect. */
export function toFormData(req: CreateJobRequest): FormData {
  const fd = new FormData();
  fd.set("mode", req.mode);
  fd.set("prompt", req.prompt);
  if (req.negative_prompt) fd.set("negative_prompt", req.negative_prompt);
  fd.set("resolution", req.resolution);
  fd.set("num_frames", String(req.num_frames));
  fd.set("fps", String(req.fps));
  fd.set("steps", String(req.steps));
  fd.set("guidance_scale", String(req.guidance_scale));
  if (req.seed !== undefined && req.seed !== null) fd.set("seed", String(req.seed));
  if (req.mode === "i2v" && req.image) fd.set("image", req.image, req.image.name);
  return fd;
}

export function createJob(req: CreateJobRequest, signal?: AbortSignal) {
  // Don't set Content-Type: the browser adds the multipart boundary itself.
  return request<Job>("/api/jobs", { method: "POST", body: toFormData(req), signal, timeoutMs: 120_000 });
}

export function getJob(id: string, signal?: AbortSignal) {
  return request<Job>(`/api/jobs/${encodeURIComponent(id)}`, { signal });
}

export function listJobs(opts: { limit?: number; offset?: number; status?: JobStatus } = {}, signal?: AbortSignal) {
  const q = new URLSearchParams();
  q.set("limit", String(opts.limit ?? 20));
  q.set("offset", String(opts.offset ?? 0));
  if (opts.status) q.set("status", opts.status);
  return request<JobList>(`/api/jobs?${q}`, { signal });
}

export function cancelJob(id: string) {
  return request<Job>(`/api/jobs/${encodeURIComponent(id)}/cancel`, { method: "POST" });
}

/** Browser-loadable URLs for a completed job's video. */
export function videoUrls(job: Job) {
  if (!job.result) return null;
  return { preview: apiUrl(job.result.preview_url), download: apiUrl(job.result.download_url) };
}
