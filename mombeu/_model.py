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
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

import logging

logging.getLogger("transformers").setLevel(logging.ERROR)

from . import _config

# ---------------------------------------------------------------------------
# Public: load
# ---------------------------------------------------------------------------

def validate_and_load(model_slug: str, hf_token: str | None = None) -> None:
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
    if hf_token:
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

def generate_continuation(text: str, max_new_tokens: int = 512) -> str:
    state = _config.get_pipeline()

    if state is None:
        raise RuntimeError("Call mombeu.init(model=...) before using the client.")

    model = state["model"]
    tokenizer = state["tokenizer"]
    device = state["device"]

    messages = [
        {
            "role": "user",
            "content": text,
        }
    ]

    input_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = tokenizer(input_text, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.1,
            pad_token_id=tokenizer.eos_token_id,
        )

    generated_ids = outputs[0][inputs["input_ids"].shape[-1]:]

    return tokenizer.decode(
        generated_ids,
        skip_special_tokens=True,
    ).strip()


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


def _check_repo_access(repo_id: str, hf_token: str | None = None) -> None:
    api = HfApi()
    try:
        api.model_info(repo_id=repo_id, token=hf_token)
    except Exception as exc:
        raise ValueError(
            f"Cannot access model repository '{repo_id}'.\n"
            "If this is a private or gated model, pass hf_token='hf_...'.\n"
            f"Details: {exc}"
        ) from exc


def _load_pipeline(model_slug: str, repo_id: str, hf_token: str | None = None) -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

    tokenizer = AutoTokenizer.from_pretrained(repo_id, token=hf_token)

    model = AutoModelForCausalLM.from_pretrained(
        repo_id,
        token=hf_token,
        torch_dtype=dtype,
        attn_implementation="sdpa",
    ).to(device)

    model.eval()
    model.config.use_cache = True

    _config.set_pipeline(
        {
            "model": model,
            "tokenizer": tokenizer,
            "device": device,
        },
        model_slug,
    )
