import { describe, expect, it } from "vitest";
import { redactSensitive, safeMessage, wasRedacted } from "./redact";

// Built at run time so that no token-like literal sits in the repository (the hygiene check would rightly flag it).
const GITHUB_STYLE = ["gh", "p_", "abcdefghijklmnopqrstuvwxyz0123"].join("");

describe("redactSensitive", () => {
  it.each([
    ["Bearer abcdef0123456789abcdef", "abcdef0123456789abcdef"],
    ["failed with Authorization: Bearer sk-abcdefghijklmnop1234", "sk-abcdefghijklmnop1234"],
    ["token=abc123XYZ789 was rejected", "abc123XYZ789"],
    ["COMFYUI_API_TOKEN is wrong", "COMFYUI_API_TOKEN"],
    ["see https://gpu.internal.example.com:8443/prompt for details", "gpu.internal.example.com"],
    ["connect ECONNREFUSED 127.0.0.1:8188", "127.0.0.1"],
    ["connect to 10.0.3.7 failed", "10.0.3.7"],
    ["could not reach localhost:8189", "localhost"],
    ["host gpu-box.lan:8188 timed out", "gpu-box.lan"],
    ["File \"/kaggle/tmp/vidsgen-gpu/ComfyUI/execution.py\", line 5", "/kaggle/tmp"],
    ["failed at /home/user/.cache/x", "/home/user"],
    ["C:\\Users\\me\\secrets\\key.pem not found", "C:\\Users"],
    ["key hf_abcdefghijklmnopqrstu leaked", "hf_abcdefghijklmnopqrstu"],
    [`id ${GITHUB_STYLE} found`, GITHUB_STYLE],
    ["opaque a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4e5f6 string", "a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4e5f6"],
    ["postgres://user:pw@db.internal/app", "user:pw"],
    ["ws://127.0.0.1:8188/ws?clientId=abc", "8188"],
  ])("hides sensitive content in %j", (text, secret) => {
    const out = redactSensitive(text);
    expect(out).not.toContain(secret);
    expect(wasRedacted(text)).toBe(true);
  });

  it.each([
    "num_frames must be 4k+1 (e.g. 17, 49, 81, 121)",
    "steps must be an integer between 1 and 20 (this GPU's verified limit)",
    "resolution 1280x704 is not enabled on this server (verified GPU profile: t4_verified).",
    "Image-to-video needs a start image.",
    "image/png, image/jpeg and image/webp are accepted",
    "prompt must be at most 2000 characters",
    "The GPU ran out of memory. Try a smaller resolution or fewer frames.",
    "Job is already completed and cannot be cancelled",
  ])("leaves ordinary messages alone: %j", (text) => {
    expect(redactSensitive(text)).toBe(text);
    expect(wasRedacted(text)).toBe(false);
  });
});

describe("safeMessage", () => {
  it("returns the server's wording when short and clean", () => {
    expect(safeMessage("  steps must be between 1 and 20 ", "fallback")).toBe("steps must be between 1 and 20");
  });
  it("uses the fallback for empty, non-string, very long and sensitive text", () => {
    expect(safeMessage("", "fb")).toBe("fb");
    expect(safeMessage(undefined, "fb")).toBe("fb");
    expect(safeMessage(42, "fb")).toBe("fb");
    expect(safeMessage("x ".repeat(200), "fb")).toBe("fb");
    expect(safeMessage("cannot reach https://gpu.example.com", "fb")).toBe("fb");
    expect(safeMessage("Bearer abcdefghijkl", "fb")).toBe("fb");
  });
});
