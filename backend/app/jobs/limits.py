"""The generation limits a client may rely on (GET /api/limits).

There is ONE source of truth. The numbers here come from the same objects that validate POST /api/jobs:
  * the model-level bounds in `GenerationParams` (frames, steps, fps, guidance, seed) and `SUPPORTED_RESOLUTIONS`,
  * the server's `GenerationEnvelope` (profile such as `t4_verified`, plus any GENERATION_* overrides),
  * the upload and prompt limits in `Settings`.
Nothing is read from a provider, so the result never depends on a GPU host being reachable, and nothing infrastructure-related
(URLs, tokens, paths, hosts) can appear in it.

`build_limits` verifies its own answer before returning it. If the configuration is inconsistent (no usable mode, resolution or frame
count, or a default that its own limits would reject) it raises `LimitsError`: the server never advertises permissive limits it
cannot honour. The JobManager runs the same check at startup, so such a server refuses to start.
"""

from __future__ import annotations

from typing import Any

from ..core.config import Settings
from .envelope import GenerationEnvelope
from .images import ALLOWED_FORMATS
from .models import MAX_FRAMES, MIN_FRAMES, SUPPORTED_RESOLUTIONS, GenerationMode, GenerationParams

FRAME_STEP = 4
MODE_LABELS = {"t2v": "Text to video", "i2v": "Image to video"}


class LimitsError(Exception):
    """The configured limits are inconsistent or unsupported."""


def _bounds(field: str) -> dict[str, Any]:
    info = GenerationParams.model_fields[field]
    out: dict[str, Any] = {"default": info.default}
    for m in info.metadata:
        for attr, key in (("ge", "min"), ("le", "max")):
            if hasattr(m, attr):
                out[key] = getattr(m, attr)
    return out


def _why(envelope: GenerationEnvelope, what: str) -> str:
    if envelope.profile != "unrestricted":
        return f"{what} is not enabled on this server (verified GPU profile: {envelope.profile})."
    return f"{what} is not enabled on this server (limited by its configuration)."


def _limited_reason(envelope: GenerationEnvelope, what: str, limit: int) -> str:
    source = f"verified GPU profile: {envelope.profile}" if envelope.profile != "unrestricted" else "limited by its configuration"
    return f"This server allows at most {limit} {what} ({source})."


def build_limits(envelope: GenerationEnvelope, settings: Settings) -> dict[str, Any]:
    """The limits as a plain dict (the API wraps it in `LimitsOut`). Raises LimitsError if they are not self-consistent."""
    known_modes = [m.value for m in GenerationMode]
    for m in envelope.modes or ():
        if m not in known_modes:
            raise LimitsError(f"unknown generation mode {m!r} in the configured limits")
    for r in envelope.resolutions or ():
        if r not in SUPPORTED_RESOLUTIONS:
            raise LimitsError(f"unsupported resolution {r!r} in the configured limits")

    modes = [{"value": m, "label": MODE_LABELS[m], "enabled": envelope.modes is None or m in envelope.modes,
              "reason": None if envelope.modes is None or m in envelope.modes else _why(envelope, MODE_LABELS[m])} for m in known_modes]
    resolutions = [{"value": k, "width": w, "height": h, "enabled": envelope.resolutions is None or k in envelope.resolutions,
                    "reason": None if envelope.resolutions is None or k in envelope.resolutions else _why(envelope, f"{w}x{h}")}
                   for k, (w, h) in SUPPORTED_RESOLUTIONS.items()]

    frames_max = MAX_FRAMES if envelope.max_frames is None else min(envelope.max_frames, MAX_FRAMES)
    allowed_frames = list(range(MIN_FRAMES, frames_max + 1, FRAME_STEP))
    frames = {"min": MIN_FRAMES, "max": allowed_frames[-1] if allowed_frames else MIN_FRAMES, "step": FRAME_STEP, "allowed": allowed_frames,
              "limited": envelope.max_frames is not None and envelope.max_frames < MAX_FRAMES,
              "reason": _limited_reason(envelope, "frames", envelope.max_frames) if envelope.max_frames is not None and envelope.max_frames < MAX_FRAMES else None}

    base_steps, base_fps, base_cfg = _bounds("steps"), _bounds("fps"), _bounds("guidance_scale")
    steps_max = base_steps["max"] if envelope.max_steps is None else min(envelope.max_steps, base_steps["max"])
    steps = {"min": base_steps["min"], "max": steps_max,
             "limited": envelope.max_steps is not None and envelope.max_steps < base_steps["max"],
             "reason": _limited_reason(envelope, "steps", envelope.max_steps) if envelope.max_steps is not None and envelope.max_steps < base_steps["max"] else None}

    enabled_modes = [m["value"] for m in modes if m["enabled"]]
    enabled_res = [r["value"] for r in resolutions if r["enabled"]]
    base_frames_default = GenerationParams.model_fields["num_frames"].default
    base_res_default = GenerationParams.model_fields["resolution"].default
    defaults = {
        "mode": "t2v" if "t2v" in enabled_modes else (enabled_modes[0] if enabled_modes else None),
        "resolution": base_res_default if base_res_default in enabled_res else (enabled_res[0] if enabled_res else None),
        "num_frames": max((n for n in allowed_frames if n <= base_frames_default), default=allowed_frames[0] if allowed_frames else None),
        "steps": min(base_steps["default"], steps_max),
        "fps": base_fps["default"],
        "guidance_scale": base_cfg["default"],
    }

    limits = {
        "profile": envelope.profile,
        "restricted": envelope.restricted,
        "max_concurrent_jobs": settings.inference_max_concurrent_jobs,
        "modes": modes,
        "resolutions": resolutions,
        "num_frames": frames,
        "steps": steps,
        "fps": {"min": base_fps["min"], "max": base_fps["max"]},
        "guidance_scale": {"min": base_cfg["min"], "max": base_cfg["max"]},
        "seed": {"min": 0, "max": 2**32 - 1},
        "prompt": {"max_chars": settings.max_prompt_chars},
        "image": {"mime_types": sorted(ALLOWED_FORMATS.values()), "max_bytes": settings.max_upload_bytes,
                  "min_side": settings.min_image_side, "max_side": settings.max_image_side},
        "defaults": defaults,
    }
    verify_limits(limits)
    return limits


def verify_limits(limits: dict[str, Any]) -> None:
    """Raises LimitsError unless the limits are usable and every default is valid under them (the server must never advertise a default
    that POST /api/jobs would reject)."""
    d = limits["defaults"]
    if not any(m["enabled"] for m in limits["modes"]):
        raise LimitsError("no generation mode is enabled")
    if not any(r["enabled"] for r in limits["resolutions"]):
        raise LimitsError("no resolution is enabled")
    frames = limits["num_frames"]
    if not frames["allowed"] or any(n < frames["min"] or n > frames["max"] or (n - 1) % FRAME_STEP for n in frames["allowed"]):
        raise LimitsError("no valid frame count is available")
    if limits["steps"]["max"] < limits["steps"]["min"]:
        raise LimitsError("no valid step count is available")
    if d["mode"] not in [m["value"] for m in limits["modes"] if m["enabled"]]:
        raise LimitsError("the default mode is not enabled")
    if d["resolution"] not in [r["value"] for r in limits["resolutions"] if r["enabled"]]:
        raise LimitsError("the default resolution is not enabled")
    if d["num_frames"] not in frames["allowed"]:
        raise LimitsError("the default frame count is not allowed")
    if not limits["steps"]["min"] <= d["steps"] <= limits["steps"]["max"]:
        raise LimitsError("the default step count is outside the allowed range")
    if not limits["fps"]["min"] <= d["fps"] <= limits["fps"]["max"]:
        raise LimitsError("the default fps is outside the allowed range")
    if not limits["guidance_scale"]["min"] <= d["guidance_scale"] <= limits["guidance_scale"]["max"]:
        raise LimitsError("the default guidance scale is outside the allowed range")
