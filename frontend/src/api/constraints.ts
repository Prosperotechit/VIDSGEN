// Typed access to the backend's validation rules. The values come from
// constraints.json, which is GENERATED from backend code (backend/scripts/export_contract.py)
// and checked by a backend test, so the UI only ever offers values the API accepts.
import raw from "./constraints.json";

export interface Resolution {
  value: string;
  width: number;
  height: number;
}

export const constraints = raw as {
  resolutions: Resolution[];
  defaultResolution: string;
  numFrames: { min: number; max: number; step: number; default: number };
  fps: { min: number; max: number; default: number };
  steps: { min: number; max: number; default: number };
  guidanceScale: { min: number; max: number; default: number };
  seed: { min: number; max: number };
  prompt: { maxChars: number };
  image: { mimeTypes: string[]; maxBytes: number; minSide: number; maxSide: number };
};

/** All valid frame counts (4k+1 between min and max). */
export const frameOptions: number[] = (() => {
  const { min, max, step } = constraints.numFrames;
  const out: number[] = [];
  for (let n = min; n <= max; n += step) out.push(n);
  return out;
})();

export const durationSeconds = (frames: number, fps: number) => Math.round((frames / fps) * 100) / 100;

export function aspectLabel({ width, height }: Resolution): string {
  if (width === height) return "1:1";
  return width > height ? "16:9" : "9:16";
}

/** The model's native (720p) sizes. */
export const isNative = ({ width, height }: Resolution) => Math.max(width, height) === 1280;

export function randomSeed(): number {
  const buf = new Uint32Array(1);
  crypto.getRandomValues(buf);
  return buf[0]!;
}

export function formatBytes(n: number): string {
  if (n < 1024) return `${n} B`;
  if (n < 1024 * 1024) return `${(n / 1024).toFixed(0)} KB`;
  return `${(n / 1024 / 1024).toFixed(1)} MB`;
}

/** Client-side mirror of the backend's image checks, for instant feedback.
 *  The backend validates again; this only avoids pointless uploads. */
export async function checkImageFile(file: File): Promise<string | null> {
  const { mimeTypes, maxBytes, minSide, maxSide } = constraints.image;
  if (!mimeTypes.includes(file.type)) return "Use a JPEG, PNG or WEBP image.";
  if (file.size > maxBytes) return `Image is ${formatBytes(file.size)}; the limit is ${formatBytes(maxBytes)}.`;
  try {
    const bmp = await createImageBitmap(file);
    const { width, height } = bmp;
    bmp.close();
    if (Math.min(width, height) < minSide) return `Image is ${width}×${height}; the shortest side must be at least ${minSide}px.`;
    if (Math.max(width, height) > maxSide) return `Image is ${width}×${height}; the longest side must be at most ${maxSide}px.`;
  } catch {
    return "This file couldn't be read as an image.";
  }
  return null;
}
