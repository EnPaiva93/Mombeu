"""
Global state and supported model catalog for mombeu.
"""

from typing import Optional

# ---------------------------------------------------------------------------
# Supported model catalog
# Maps user-facing slug → HuggingFace repo ID
# Add / remove entries here as new models are published or deprecated.
# ---------------------------------------------------------------------------
SUPPORTED_MODELS: dict[str, str] = {
    "mombeu-v1":   "mombeu-org/mombeu-model-v1",
    "mombeu-v2":   "mombeu-org/mombeu-model-v2",
    "mombeu-fast": "mombeu-org/mombeu-fast-model",
}

# Deprecated models: still recognized but emit a warning.
DEPRECATED_MODELS: dict[str, str] = {
    # "mombeu-v0": "Model 'mombeu-v0' is deprecated. Please upgrade to 'mombeu-v1'.",
}

# ---------------------------------------------------------------------------
# Runtime state (module-level singletons)
# ---------------------------------------------------------------------------
_pipeline = None          # transformers Pipeline instance
_current_model: Optional[str] = None
_initialized: bool = False


# ---------------------------------------------------------------------------
# Accessors
# ---------------------------------------------------------------------------

def get_pipeline():
    return _pipeline


def get_current_model() -> Optional[str]:
    return _current_model


def is_initialized() -> bool:
    return _initialized


def set_pipeline(pipe, model_slug: str) -> None:
    global _pipeline, _current_model, _initialized
    _pipeline = pipe
    _current_model = model_slug
    _initialized = True


def reset() -> None:
    """Clear all runtime state (useful for testing or switching models)."""
    global _pipeline, _current_model, _initialized
    _pipeline = None
    _current_model = None
    _initialized = False
