// Golden /api/limits responses for tests: the same values the backend tests assert for each generation profile.
import type { Limits } from "./types";

const RES = [["1280x704", 1280, 704], ["704x1280", 704, 1280], ["832x480", 832, 480], ["480x832", 480, 832], ["640x640", 640, 640]] as const;
const IMAGE = { mime_types: ["image/jpeg", "image/png", "image/webp"], max_bytes: 10485760, min_side: 256, max_side: 4096 };

/** The two profiles' real responses (the same golden values the backend tests assert). */
export const DEFAULT_LIMITS = {
  profile: "unrestricted", restricted: false, max_concurrent_jobs: 1,
  modes: [{ value: "t2v", label: "Text to video", enabled: true, reason: null }, { value: "i2v", label: "Image to video", enabled: true, reason: null }],
  resolutions: RES.map(([value, width, height]) => ({ value, width, height, enabled: true, reason: null })),
  num_frames: { min: 17, max: 121, step: 4, allowed: Array.from({ length: 27 }, (_, i) => 17 + 4 * i), limited: false, reason: null },
  steps: { min: 1, max: 60, limited: false, reason: null }, fps: { min: 8, max: 30 }, guidance_scale: { min: 1, max: 15 },
  seed: { min: 0, max: 4294967295 }, prompt: { max_chars: 2000 }, image: IMAGE,
  defaults: { mode: "t2v", resolution: "832x480", num_frames: 49, steps: 30, fps: 24, guidance_scale: 5 },
} as unknown as Limits;

export const T4_LIMITS = {
  ...DEFAULT_LIMITS, profile: "t4_verified", restricted: true,
  modes: [{ value: "t2v", label: "Text to video", enabled: true, reason: null },
          { value: "i2v", label: "Image to video", enabled: false, reason: "Image to video is not enabled on this server (verified GPU profile: t4_verified)." }],
  resolutions: RES.map(([value, width, height]) => ({ value, width, height, enabled: value === "832x480", reason: value === "832x480" ? null : `${width}x${height} is not enabled on this server (verified GPU profile: t4_verified).` })),
  num_frames: { min: 17, max: 17, step: 4, allowed: [17], limited: true, reason: "This server allows at most 17 frames (verified GPU profile: t4_verified)." },
  steps: { min: 1, max: 20, limited: true, reason: "This server allows at most 20 steps (verified GPU profile: t4_verified)." },
  defaults: { mode: "t2v", resolution: "832x480", num_frames: 17, steps: 20, fps: 24, guidance_scale: 5 },
} as unknown as Limits;
