"""Backward-compatibility shim for legacy imports.

This module keeps the old class/file name alive while the codebase transitions
from Ollama-specific naming to provider-agnostic naming.
"""

from llm_provider_mixin import LlmProviderMixin


class OllamaMixin(LlmProviderMixin):
    """Legacy alias for LlmProviderMixin."""

    pass
