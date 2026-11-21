"""The GPU generation envelope: which requests this server's GPU host is allowed to run.

A GPU host has been *verified* for particular settings only. A Kaggle T4 completed 832x480, 17 frames,
20 steps using 14,885 of 15,360 MiB of VRAM, so anything much larger is likely to run out of memory after
minutes of GPU time. The envelope rejects such requests up front, with a message that says what is allowed.

Where it is enforced: at the API boundary only (`POST /api/jobs` in app/api/jobs.py). `JobManager.create_job` does
not re-check it, so any new code path that creates jobs (a CLI, a batch importer, another API) must call
`manager.envelope.violations(params)` itself. Likewise, ComfyUIProvider never retries a submission whose outcome is
uncertain (for example a timeout on POST /prompt): a retry could run the same generation twice on a paid GPU.

Configuration (all optional; the default is "unrestricted", which is what the mock provider uses):

    GENERATION_PROFILE=t4_verified          a named preset (see PROFILES)
    GENERATION_ALLOWED_MODES=t2v,i2v        each of these overrides the preset's value when set
    GENERATION_ALLOWED_RESOLUTIONS=832x480
    GENERATION_MAX_FRAMES=17
    GENERATION_MAX_STEPS=20
"""

from __future__ import annotations

from dataclasses import dataclass

from ..core.config import Settings
from .models import MAX_FRAMES, MIN_FRAMES, SUPPORTED_RESOLUTIONS, GenerationMode, GenerationParams

# name -> limits. `t4_verified` is exactly what was proven on a Kaggle T4 (Phase 5B). Image-to-video has not
# been run on a GPU yet, so the preset does not allow it; set GENERATION_ALLOWED_MODES=t2v,i2v to test it.
PROFILES: dict[str, dict] = {
    "unrestricted": {},
    "t4_verified": {"modes": ("t2v",), "resolutions": ("832x480",), "max_frames": 17, "max_steps": 20},
}


def _csv(value: str) -> tuple[str, ...]:
    return tuple(dict.fromkeys(p.strip().lower() for p in value.split(",") if p.strip()))


@dataclass(frozen=True)
class GenerationEnvelope:
    profile: str = "unrestricted"
    modes: tuple[str, ...] | None = None          # None = no limit
    resolutions: tuple[str, ...] | None = None
    max_frames: int | None = None
    max_steps: int | None = None

    @classmethod
    def from_settings(cls, s: Settings) -> GenerationEnvelope:
        """Preset values, overridden field by field by the explicit settings. Raises ValueError on bad values."""
        base = PROFILES[s.generation_profile]
        modes = _csv(s.generation_allowed_modes) or base.get("modes")
        resolutions = _csv(s.generation_allowed_resolutions) or base.get("resolutions")
        max_frames = s.generation_max_frames if s.generation_max_frames is not None else base.get("max_frames")
        max_steps = s.generation_max_steps if s.generation_max_steps is not None else base.get("max_steps")

        for m in modes or ():
            if m not in {x.value for x in GenerationMode}:
                raise ValueError(f"GENERATION_ALLOWED_MODES: unknown mode {m!r} (use t2v and/or i2v)")
        for r in resolutions or ():
            if r not in SUPPORTED_RESOLUTIONS:
                raise ValueError(f"GENERATION_ALLOWED_RESOLUTIONS: unsupported resolution {r!r}; "
                                 f"choose from {sorted(SUPPORTED_RESOLUTIONS)}")
        if max_frames is not None and not MIN_FRAMES <= max_frames <= MAX_FRAMES:
            raise ValueError(f"GENERATION_MAX_FRAMES must be between {MIN_FRAMES} and {MAX_FRAMES}")
        return cls(profile=s.generation_profile, modes=tuple(modes) if modes else None,
                   resolutions=tuple(resolutions) if resolutions else None, max_frames=max_frames, max_steps=max_steps)

    @property
    def restricted(self) -> bool:
        return any(v is not None for v in (self.modes, self.resolutions, self.max_frames, self.max_steps))

    def summary(self) -> str:
        if not self.restricted:
            return "unrestricted"
        parts = []
        if self.modes is not None:
            parts.append("modes " + "/".join(self.modes))
        if self.resolutions is not None:
            parts.append("resolutions " + ", ".join(self.resolutions))
        if self.max_frames is not None:
            parts.append(f"at most {self.max_frames} frames")
        if self.max_steps is not None:
            parts.append(f"at most {self.max_steps} steps")
        return f"profile {self.profile}: " + "; ".join(parts)

    def violations(self, p: GenerationParams) -> list[tuple[str, str]]:
        """(field, message) for every limit the request exceeds. Empty = allowed."""
        out: list[tuple[str, str]] = []
        why = f"This GPU server only accepts requests it has been verified for ({self.summary()})."
        if self.modes is not None and p.mode.value not in self.modes:
            names = {"t2v": "text-to-video", "i2v": "image-to-video"}
            out.append(("mode", f"{names[p.mode.value]} is not enabled on this server; allowed: {', '.join(self.modes)}. {why}"))
        if self.resolutions is not None and p.resolution not in self.resolutions:
            out.append(("resolution", f"resolution {p.resolution} is not enabled on this server; "
                                      f"allowed: {', '.join(self.resolutions)}. {why}"))
        if self.max_frames is not None and p.num_frames > self.max_frames:
            out.append(("num_frames", f"num_frames {p.num_frames} exceeds this server's maximum of {self.max_frames}. {why}"))
        if self.max_steps is not None and p.steps > self.max_steps:
            out.append(("steps", f"steps {p.steps} exceeds this server's maximum of {self.max_steps}. {why}"))
        return out
