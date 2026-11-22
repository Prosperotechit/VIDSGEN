// Friendly aliases over the types generated from the backend's OpenAPI spec
// (src/api/schema.d.ts, produced by `npm run gen:api`). Never hand-write API shapes.
import type { components } from "./schema";

type S = components["schemas"];

export type Job = S["JobOut"];
export type JobList = S["JobListOut"];
export type JobStatus = S["JobStatus"];
export type GenerationMode = S["GenerationMode"];
export type JobParams = S["JobParamsOut"];
export type JobResult = S["ResultOut"];
export type JobError = S["JobError"];
export type Health = S["HealthOut"];
export type Limits = S["LimitsOut"];

/** Fields accepted by POST /api/jobs (multipart/form-data). */
export interface CreateJobRequest {
  mode: GenerationMode;
  prompt: string;
  negative_prompt?: string;
  resolution: string;
  num_frames: number;
  fps: number;
  steps: number;
  guidance_scale: number;
  /** Omit for a random seed chosen (and recorded) by the server. */
  seed?: number | null;
  /** Required for i2v, must be omitted for t2v. */
  image?: File | null;
}

export const TERMINAL_STATUSES: readonly JobStatus[] = ["completed", "failed", "cancelled"];
export const isTerminal = (s: JobStatus) => TERMINAL_STATUSES.includes(s);
