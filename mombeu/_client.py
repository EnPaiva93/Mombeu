"""
OpenAI client wrapper for mombeu.

Wraps only the endpoints where a text continuation makes sense:
  - chat.completions.create  ✓  (streaming and non-streaming)

All other OpenAI API surface (embeddings, images, audio, etc.) is delegated
directly to the underlying openai.OpenAI instance without modification.
"""

from typing import Any

from openai import OpenAI as _OriginalOpenAI

from . import _config, _interceptor


# ---------------------------------------------------------------------------
# Completions wrapper
# ---------------------------------------------------------------------------

class _Completions:
    """Wraps openai.resources.chat.Completions."""

    def __init__(self, original_completions):
        self._original = original_completions

    def create(self, *args, **kwargs):
        """
        Call OpenAI chat.completions.create, then append a local continuation.

        stream=False (default):
            Waits for the full OpenAI response, generates a continuation,
            appends it to response.choices[0].message.content, and returns.

        stream=True:
            Yields all original chunks in real-time, then yields one extra
            chunk containing the local continuation, followed by a stop chunk.
        """
        _assert_initialized()

        stream: bool = kwargs.get("stream", False)
        response = self._original.create(*args, **kwargs)

        if stream:
            return _interceptor.process_stream_and_continue(response)
        else:
            return _interceptor.process_response(response)


# ---------------------------------------------------------------------------
# Chat wrapper
# ---------------------------------------------------------------------------

class _Chat:
    """Wraps openai.resources.Chat."""

    def __init__(self, original_chat):
        self._original = original_chat
        self.completions = _Completions(original_chat.completions)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._original, name)


# ---------------------------------------------------------------------------
# Public OpenAI wrapper
# ---------------------------------------------------------------------------

class OpenAI:
    """
    Drop-in replacement for ``openai.OpenAI`` that appends a locally-run LLM
    continuation to every chat completion response.

    Usage::

        import mombeu

        mombeu.init(model="mombeu-v1", hf_token="hf_...")

        client = mombeu.OpenAI(api_key="sk-...")

        # Non-streaming
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": "Tell me about space."}],
        )
        print(response.choices[0].message.content)

        # Streaming
        for chunk in client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": "Tell me about space."}],
            stream=True,
        ):
            delta = chunk.choices[0].delta.content or ""
            print(delta, end="", flush=True)

    All other ``openai.OpenAI`` attributes and methods (``embeddings``,
    ``images``, ``audio``, etc.) are forwarded transparently to the underlying
    client.
    """

    def __init__(self, **kwargs):
        _assert_initialized()
        self._client = _OriginalOpenAI(**kwargs)
        self.chat = _Chat(self._client.chat)

    # Delegate everything else to the real client
    def __getattr__(self, name: str) -> Any:
        return getattr(self._client, name)


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _assert_initialized() -> None:
    if not _config.is_initialized():
        raise RuntimeError(
            "mombeu has not been initialized.\n"
            "Call mombeu.init(model='mombeu-v1', hf_token='hf_...') "
            "before creating a client."
        )
