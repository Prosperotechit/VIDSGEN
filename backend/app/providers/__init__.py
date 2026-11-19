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
        # Imported lazily so the mock setup never needs the ComfyUI client code paths.
        from ..workflows import WorkflowTemplates
        from .comfyui import ComfyUIProvider

        if not settings.comfyui_base_url:
            raise ValueError("INFERENCE_PROVIDER=comfyui requires COMFYUI_BASE_URL to be set.")
        return ComfyUIProvider(
            base_url=settings.comfyui_base_url,
            api_token=settings.comfyui_api_token.get_secret_value(),
            timeout=settings.comfyui_timeout,
            client_id=settings.comfyui_client_id,
            poll_interval=settings.comfyui_poll_interval,
            workflows=WorkflowTemplates(settings.comfyui_workflows_dir),  # validated here: fails fast
        )
    raise ValueError(f"Unknown inference provider {settings.inference_provider!r}")
