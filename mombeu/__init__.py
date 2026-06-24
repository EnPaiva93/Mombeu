"""
mombeu
======
OpenAI Python SDK wrapper that appends a locally-run LLM continuation to
every chat completion response.

Quick start::

    import mombeu

    # 1. Initialize once (downloads model on first run, cached afterwards)
    mombeu.init(model="mombeu-v1", hf_token="hf_...")

    # 2. Use exactly like openai.OpenAI
    client = mombeu.OpenAI(api_key="sk-...")
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": "Hello!"}],
    )
    print(response.choices[0].message.content)
"""

from typing import Optional

from ._client import OpenAI
from ._config import (
    SUPPORTED_MODELS,
    get_current_model,
    is_initialized,
    reset,
)
from ._model import validate_and_load

__version__ = "0.1.0"
__all__ = [
    "init",
    "OpenAI",
    "list_models",
    "current_model",
    "is_initialized",
    "reset",
]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def init(model: str, hf_token: str | None = None) -> None:
    """
    Initialize mombeu with a supported model.

    On the first call the model is downloaded from HuggingFace and cached
    locally (``~/.cache/huggingface/``). Subsequent calls with the same model
    skip the download and load directly from cache.

    Args:
        model:    Slug of the model to use. Call ``mombeu.list_models()`` to
                  see all available options.
        hf_token: A HuggingFace access token with read permissions on the
                  model repository.

    Raises:
        ValueError:   If ``model`` is not in the supported list, the token is
                      invalid, or the account lacks access to the repo.
        RuntimeError: If the model pipeline fails to load.

    Example::

        mombeu.init(model="mombeu-v1", hf_token="hf_abc123...")
    """
    validate_and_load(model_slug=model, hf_token=hf_token)


def list_models() -> list[str]:
    """
    Return the list of supported model slugs.

    Example::

        >>> mombeu.list_models()
        ['mombeu-fast', 'mombeu-v1', 'mombeu-v2']
    """
    return sorted(SUPPORTED_MODELS.keys())


def current_model() -> Optional[str]:
    """
    Return the slug of the currently loaded model, or ``None`` if mombeu has
    not been initialized yet.

    Example::

        >>> mombeu.current_model()
        'mombeu-v1'
    """
    return get_current_model()
