from __future__ import annotations

from ..core.config import Settings
from .base import (
    GenerationRequest, InferenceProvider, ProviderError, ProviderHealth, ProviderProgress, ProviderResult,
)
from .mock import MockProvider

__all__ = [
    "GenerationRequest", "InferenceProvider", "MockProvider", "ProviderError", "ProviderHealth",
    "ProviderProgress", "ProviderResult", "build_provider",
]


def build_provider(settings: Settings) -> InferenceProvider:
    """Select the inference provider from INFERENCE_PROVIDER. Add new providers here."""
    if settings.inference_provider == "mock":
        return MockProvider(step_delay_seconds=settings.mock_step_delay_seconds)
    if settings.inference_provider == "comfyui":
        # Phase 4: ComfyUIProvider(url=settings.comfyui_url, token=settings.comfyui_api_token, ...)
        raise NotImplementedError(
            "INFERENCE_PROVIDER=comfyui is planned for Phase 4 and not implemented yet; use 'mock'."
        )
    raise ValueError(f"Unknown inference provider {settings.inference_provider!r}")
