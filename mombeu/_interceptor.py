"""
Interceptor layer for mombeu.

Handles two cases:
  1. Non-streaming: receives a ChatCompletion, appends continuation, returns it.
  2. Streaming:     yields all original chunks unchanged, then yields one final
                    chunk containing the locally-generated continuation.
"""

import copy
import warnings
from typing import Iterator

from . import _model


# ---------------------------------------------------------------------------
# Non-streaming
# ---------------------------------------------------------------------------

def process_response(response):
    """
    Append a local-model continuation to every choice in a ChatCompletion.

    The continuation is appended directly to `choice.message.content` so the
    caller receives a single, seamless string (Opcion A).

    Args:
        response: openai.types.chat.ChatCompletion

    Returns:
        The same response object with modified content fields.
    """
    for choice in response.choices:
        if not (choice.message and choice.message.content):
            continue

        original = choice.message.content
        continuation = _model.generate_continuation(original)

        if not continuation:
            continue

        new_content = original + continuation
        _set_content(choice.message, new_content)

    return response


# ---------------------------------------------------------------------------
# Streaming
# ---------------------------------------------------------------------------

def process_stream_and_continue(stream) -> Iterator:
    """
    Pass-through generator for an OpenAI streaming response.

    Behaviour:
    - Every chunk from OpenAI is yielded immediately (real-time streaming).
    - After the stream ends, the accumulated text is fed to the local model.
    - The continuation is yielded as a single final chunk.
    - A closing stop-chunk is then yielded to properly terminate the stream.

    Args:
        stream: The raw stream returned by openai client.chat.completions.create
                with stream=True.

    Yields:
        openai.types.chat.ChatCompletionChunk
    """
    accumulated_content: str = ""
    last_chunk = None

    # ── Phase 1: stream OpenAI response through unchanged ──────────────────
    for chunk in stream:
        last_chunk = chunk

        delta_content = (
            chunk.choices[0].delta.content
            if chunk.choices and chunk.choices[0].delta
            else None
        )
        if delta_content:
            accumulated_content += delta_content

        yield chunk

    # ── Phase 2: generate & yield continuation ─────────────────────────────
    if not accumulated_content or last_chunk is None:
        return

    continuation = _model.generate_continuation(accumulated_content)

    if not continuation:
        return

    try:
        # Continuation chunk — reuses metadata from the last real chunk
        cont_chunk = copy.deepcopy(last_chunk)
        _set_delta_content(cont_chunk, continuation)
        _set_finish_reason(cont_chunk, None)
        yield cont_chunk

        # Final stop chunk — signals end of stream to callers
        stop_chunk = copy.deepcopy(last_chunk)
        _set_delta_content(stop_chunk, None)
        _set_finish_reason(stop_chunk, "stop")
        yield stop_chunk

    except Exception as exc:
        warnings.warn(
            f"[mombeu] Could not yield continuation chunk: {exc}. "
            "Stream ended without continuation."
        )


# ---------------------------------------------------------------------------
# Pydantic-safe attribute setters
# ---------------------------------------------------------------------------

def _set_content(message, value: str) -> None:
    """Set message.content, bypassing Pydantic frozen validation if needed."""
    try:
        message.content = value
    except Exception:
        object.__setattr__(message, "content", value)


def _set_delta_content(chunk, value) -> None:
    """Set chunk.choices[0].delta.content safely."""
    try:
        chunk.choices[0].delta.content = value
    except Exception:
        object.__setattr__(chunk.choices[0].delta, "content", value)


def _set_finish_reason(chunk, value) -> None:
    """Set chunk.choices[0].finish_reason safely."""
    try:
        chunk.choices[0].finish_reason = value
    except Exception:
        object.__setattr__(chunk.choices[0], "finish_reason", value)
