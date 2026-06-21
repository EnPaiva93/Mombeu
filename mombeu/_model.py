"""
Model loading and inference logic for mombeu.

Responsibilities:
- Authenticate with HuggingFace using the provided token.
- Validate that the requested model is accessible.
- Download + cache the model (only on first run).
- Expose generate_continuation() for use by the interceptor.
"""

import warnings
from typing import Optional

from huggingface_hub import HfApi, login
from transformers import pipeline, logging as hf_logging
import torch

from . import _config

# Suppress verbose HuggingFace/transformers logs by default
hf_logging.set_verbosity_error()


# ---------------------------------------------------------------------------
# Public: load
# ---------------------------------------------------------------------------

def validate_and_load(model_slug: str, hf_token: str) -> None:
    """
    Validate model slug, authenticate with HuggingFace, download if needed,
    then load the pipeline into global state.

    Args:
        model_slug: One of the keys in _config.SUPPORTED_MODELS.
        hf_token:   A valid HuggingFace access token.

    Raises:
        ValueError:  If the slug is unsupported, the token is invalid, or the
                     account does not have access to the model repo.
        RuntimeError: If the model fails to load after download.
    """
    _check_model_slug(model_slug)
    _authenticate(hf_token)

    repo_id = _config.SUPPORTED_MODELS[model_slug]
    _check_repo_access(repo_id, hf_token)

    # If the same model is already loaded, skip re-loading.
    if _config.is_initialized() and _config.get_current_model() == model_slug:
        print(f"[mombeu] Model '{model_slug}' is already loaded.")
        return

    _load_pipeline(model_slug, repo_id, hf_token)


# ---------------------------------------------------------------------------
# Public: inference
# ---------------------------------------------------------------------------

def generate_continuation(text: str, max_new_tokens: int = 150) -> str:
    """
    Generate a continuation for *text* using the loaded local model.

    Returns the continuation string only (not the original text).
    Returns an empty string if generation fails (with a warning).

    Args:
        text:           The input text (typically the OpenAI response content).
        max_new_tokens: Maximum number of tokens to generate.
    """
    if not _config.is_initialized():
        raise RuntimeError(
            "mombeu is not initialized. "
            "Call mombeu.init(model=..., hf_token=...) before using the client."
        )

    pipe = _config.get_pipeline()

    try:
        result = pipe(
            text,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=pipe.tokenizer.eos_token_id,
            return_full_text=False,   # Return only new tokens, not the prompt
        )
        continuation: str = result[0]["generated_text"]
        return continuation
    except Exception as exc:
        warnings.warn(
            f"[mombeu] Continuation generation failed: {exc}. "
            "Returning original response without continuation."
        )
        return ""


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _check_model_slug(model_slug: str) -> None:
    """Raise a clear error if the slug is not supported; warn if deprecated."""
    if model_slug in _config.DEPRECATED_MODELS:
        warnings.warn(
            _config.DEPRECATED_MODELS[model_slug],
            DeprecationWarning,
            stacklevel=4,
        )

    if model_slug not in _config.SUPPORTED_MODELS:
        available = ", ".join(sorted(_config.SUPPORTED_MODELS.keys()))
        raise ValueError(
            f"Model '{model_slug}' is not supported.\n"
            f"Available models: {available}\n"
            f"Use mombeu.list_models() to see the full list."
        )


def _authenticate(hf_token: str) -> None:
    """Login to HuggingFace Hub with the provided token."""
    try:
        login(token=hf_token, add_to_git_credential=False)
    except Exception as exc:
        raise ValueError(
            f"HuggingFace authentication failed. "
            f"Please check your token. Details: {exc}"
        ) from exc


def _check_repo_access(repo_id: str, hf_token: str) -> None:
    """Verify the token grants read access to the target repo."""
    api = HfApi()
    try:
        api.model_info(repo_id=repo_id, token=hf_token)
    except Exception as exc:
        raise ValueError(
            f"Cannot access model repository '{repo_id}'.\n"
            f"Make sure your HuggingFace token has read access to this repo.\n"
            f"Details: {exc}"
        ) from exc


def _load_pipeline(model_slug: str, repo_id: str, hf_token: str) -> None:
    """Download (if needed) and load the model as a text-generation pipeline."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    print(f"[mombeu] Loading model '{model_slug}' on {device.upper()}...")
    print(f"[mombeu] (First run will download from HuggingFace — this may take a moment)")

    try:
        pipe = pipeline(
            "text-generation",
            model=repo_id,
            token=hf_token,
            device=device,
            torch_dtype=dtype,
        )
    except Exception as exc:
        raise RuntimeError(
            f"Failed to load model '{model_slug}' from '{repo_id}'.\n"
            f"Details: {exc}"
        ) from exc

    _config.set_pipeline(pipe, model_slug)
    print(f"[mombeu] Model '{model_slug}' ready.")
