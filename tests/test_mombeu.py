"""
Unit tests for mombeu.

Run with:
    pytest tests/ -v
"""

import copy
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Helpers to build fake OpenAI response objects
# ---------------------------------------------------------------------------

def _make_chat_completion(content: str):
    """Build a minimal fake ChatCompletion object."""
    message = MagicMock()
    message.content = content

    choice = MagicMock()
    choice.message = message

    response = MagicMock()
    response.choices = [choice]
    return response


def _make_stream_chunks(texts: list[str], finish_reason: str = "stop"):
    """Yield fake ChatCompletionChunk objects."""
    for i, text in enumerate(texts):
        delta = MagicMock()
        delta.content = text

        choice = MagicMock()
        choice.delta = delta
        choice.finish_reason = finish_reason if i == len(texts) - 1 else None

        chunk = MagicMock()
        chunk.choices = [choice]
        yield chunk


# ---------------------------------------------------------------------------
# _config
# ---------------------------------------------------------------------------

class TestConfig:
    def setup_method(self):
        from mombeu import _config
        _config.reset()

    def test_not_initialized_by_default(self):
        from mombeu import _config
        assert not _config.is_initialized()
        assert _config.get_current_model() is None
        assert _config.get_pipeline() is None

    def test_set_pipeline_marks_initialized(self):
        from mombeu import _config
        fake_pipe = MagicMock()
        _config.set_pipeline(fake_pipe, "mombeu-v1")
        assert _config.is_initialized()
        assert _config.get_current_model() == "mombeu-v1"
        assert _config.get_pipeline() is fake_pipe

    def test_reset_clears_state(self):
        from mombeu import _config
        _config.set_pipeline(MagicMock(), "mombeu-v1")
        _config.reset()
        assert not _config.is_initialized()
        assert _config.get_current_model() is None


# ---------------------------------------------------------------------------
# list_models / current_model
# ---------------------------------------------------------------------------

class TestPublicHelpers:
    def setup_method(self):
        from mombeu import _config
        _config.reset()

    def test_list_models_returns_sorted_slugs(self):
        import mombeu
        models = mombeu.list_models()
        assert isinstance(models, list)
        assert len(models) > 0
        assert models == sorted(models)

    def test_current_model_none_before_init(self):
        import mombeu
        assert mombeu.current_model() is None

    def test_current_model_after_init(self):
        from mombeu import _config
        _config.set_pipeline(MagicMock(), "mombeu-v2")
        import mombeu
        assert mombeu.current_model() == "mombeu-v2"


# ---------------------------------------------------------------------------
# init() validation
# ---------------------------------------------------------------------------

class TestInit:
    def setup_method(self):
        from mombeu import _config
        _config.reset()

    def test_unsupported_model_raises_value_error(self):
        import mombeu
        with pytest.raises(ValueError, match="not supported"):
            mombeu.init(model="nonexistent-model", hf_token="hf_fake")

    def test_error_message_lists_available_models(self):
        import mombeu
        with pytest.raises(ValueError) as exc_info:
            mombeu.init(model="bad-model", hf_token="hf_fake")
        assert "mombeu-v1" in str(exc_info.value)

    @patch("mombeu._model._authenticate")
    @patch("mombeu._model._check_repo_access")
    @patch("mombeu._model._load_pipeline")
    def test_valid_model_calls_load(self, mock_load, mock_access, mock_auth):
        import mombeu
        mombeu.init(model="mombeu-v1", hf_token="hf_fake")
        mock_auth.assert_called_once_with("hf_fake")
        mock_access.assert_called_once()
        mock_load.assert_called_once()


# ---------------------------------------------------------------------------
# _interceptor — non-streaming
# ---------------------------------------------------------------------------

class TestProcessResponse:
    def setup_method(self):
        from mombeu import _config
        _config.set_pipeline(MagicMock(), "mombeu-v1")

    def test_appends_continuation_to_content(self):
        from mombeu import _interceptor

        response = _make_chat_completion("Hello from OpenAI.")
        with patch("mombeu._model.generate_continuation", return_value=" And here is more."):
            result = _interceptor.process_response(response)

        assert result.choices[0].message.content == "Hello from OpenAI. And here is more."

    def test_empty_continuation_leaves_content_unchanged(self):
        from mombeu import _interceptor

        response = _make_chat_completion("Original.")
        with patch("mombeu._model.generate_continuation", return_value=""):
            result = _interceptor.process_response(response)

        assert result.choices[0].message.content == "Original."

    def test_none_content_is_skipped(self):
        from mombeu import _interceptor

        response = _make_chat_completion(None)
        with patch("mombeu._model.generate_continuation") as mock_gen:
            _interceptor.process_response(response)
            mock_gen.assert_not_called()


# ---------------------------------------------------------------------------
# _interceptor — streaming
# ---------------------------------------------------------------------------

class TestProcessStreamAndContinue:
    def setup_method(self):
        from mombeu import _config
        _config.set_pipeline(MagicMock(), "mombeu-v1")

    def test_yields_all_original_chunks(self):
        from mombeu import _interceptor

        original_chunks = list(_make_stream_chunks(["Hello", " world"]))

        with patch("mombeu._model.generate_continuation", return_value=" Extra."):
            output = list(_interceptor.process_stream_and_continue(iter(original_chunks)))

        # At minimum the original chunks must be present (plus continuation chunks)
        assert output[0] is original_chunks[0]
        assert output[1] is original_chunks[1]

    def test_yields_continuation_chunk_after_stream(self):
        from mombeu import _interceptor

        original_chunks = list(_make_stream_chunks(["Hello", " world"]))

        with patch("mombeu._model.generate_continuation", return_value=" Extra."):
            output = list(_interceptor.process_stream_and_continue(iter(original_chunks)))

        # Should have: 2 original + 1 continuation + 1 stop = 4 chunks
        assert len(output) == 4

    def test_no_extra_chunks_when_continuation_empty(self):
        from mombeu import _interceptor

        original_chunks = list(_make_stream_chunks(["Hello"]))

        with patch("mombeu._model.generate_continuation", return_value=""):
            output = list(_interceptor.process_stream_and_continue(iter(original_chunks)))

        assert len(output) == 1  # Only the original chunk


# ---------------------------------------------------------------------------
# OpenAI client wrapper
# ---------------------------------------------------------------------------

class TestOpenAIWrapper:
    def setup_method(self):
        from mombeu import _config
        _config.set_pipeline(MagicMock(), "mombeu-v1")

    def test_raises_if_not_initialized(self):
        from mombeu import _config, _client
        _config.reset()
        with pytest.raises(RuntimeError, match="not been initialized"):
            _client.OpenAI(api_key="sk-fake")

    @patch("mombeu._client._OriginalOpenAI")
    def test_chat_completions_create_non_stream(self, MockOriginal):
        from mombeu._client import OpenAI
        from mombeu import _interceptor

        fake_response = _make_chat_completion("OpenAI says hi.")
        MockOriginal.return_value.chat.completions.create.return_value = fake_response

        with patch.object(_interceptor, "process_response", return_value=fake_response) as mock_proc:
            client = OpenAI(api_key="sk-fake")
            result = client.chat.completions.create(model="gpt-4o", messages=[])
            mock_proc.assert_called_once_with(fake_response)
            assert result is fake_response

    @patch("mombeu._client._OriginalOpenAI")
    def test_chat_completions_create_stream(self, MockOriginal):
        from mombeu._client import OpenAI
        from mombeu import _interceptor

        fake_stream = iter([MagicMock()])
        MockOriginal.return_value.chat.completions.create.return_value = fake_stream

        with patch.object(_interceptor, "process_stream_and_continue", return_value=iter([])) as mock_s:
            client = OpenAI(api_key="sk-fake")
            client.chat.completions.create(model="gpt-4o", messages=[], stream=True)
            mock_s.assert_called_once_with(fake_stream)

    @patch("mombeu._client._OriginalOpenAI")
    def test_non_chat_attributes_delegated(self, MockOriginal):
        from mombeu._client import OpenAI

        fake_embeddings = MagicMock()
        MockOriginal.return_value.embeddings = fake_embeddings

        client = OpenAI(api_key="sk-fake")
        assert client.embeddings is fake_embeddings
