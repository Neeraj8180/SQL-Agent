"""Phase 2.3 tests — HuggingFace provider.

Split into two tiers:

    1. Fast tests (always run):
         * JSON extraction helper
         * Message conversion helper
         * Structured-output parsing + retry logic (no real model needed)

    2. Torch-gated integration tests:
         * Real model load (SmolLM2-135M-Instruct — ~270 MB, CPU-viable)
         * invoke() returns non-empty content

Torch-gated tests are automatically skipped if torch / transformers are
not installed, so pure-OpenAI setups remain green.
"""

from __future__ import annotations

import pytest
from pydantic import BaseModel, Field

from sql_agent.llm_serving.base import ProviderUnavailableError
from sql_agent.llm_serving.hf_provider import (
    _HFStructuredInvoker,
    _extract_json,
    _to_chat_messages,
)


# ---------------------------------------------------------------------------
# 1. Fast tests
# ---------------------------------------------------------------------------


def test_extract_json_fenced_block():
    text = "Here it is:\n```json\n{\"x\": 1, \"y\": \"hi\"}\n```\nThanks"
    assert _extract_json(text) == '{"x": 1, "y": "hi"}'


def test_extract_json_greedy_braces():
    text = 'Prefix text {"a": 1, "b": [1, 2, 3]} suffix'
    assert _extract_json(text) == '{"a": 1, "b": [1, 2, 3]}'


def test_extract_json_none_returned_when_no_object():
    assert _extract_json("no json anywhere here") is None


def test_extract_json_prefers_fence_over_inline():
    # Fence should win even if both are present.
    text = "first {inline: bad} then ```json\n{\"good\": true}\n```"
    assert _extract_json(text) == '{"good": true}'


def test_to_chat_messages_from_dicts():
    msgs = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "hello"},
    ]
    assert _to_chat_messages(msgs) == msgs


def test_to_chat_messages_from_langchain_objects():
    from langchain_core.messages import HumanMessage, SystemMessage

    out = _to_chat_messages([SystemMessage("sys"), HumanMessage("hi")])
    assert out == [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "hi"},
    ]


def test_to_chat_messages_normalizes_unknown_role():
    # Unknown role -> defaults to "user", doesn't explode.
    out = _to_chat_messages([{"role": "weird", "content": "x"}])
    assert out == [{"role": "user", "content": "x"}]


# ---------------------------------------------------------------------------
# Structured-output invoker (uses a fake generator — no real model needed)
# ---------------------------------------------------------------------------


class _SimpleOutput(BaseModel):
    x: int
    y: str = Field(default="")


class _FakeGenerator:
    """Object with the minimal surface the structured invoker needs."""

    def __init__(self, scripted_responses):
        self._responses = list(scripted_responses)
        self.calls = 0

    def _generate(self, _chat_msgs):
        self.calls += 1
        if not self._responses:
            raise AssertionError("FakeGenerator exhausted unexpectedly")
        return self._responses.pop(0)


def test_structured_output_happy_path_parses_valid_json():
    fake = _FakeGenerator(['{"x": 42, "y": "hello"}'])
    inv = _HFStructuredInvoker(fake, _SimpleOutput)
    result = inv.invoke([])
    assert isinstance(result, _SimpleOutput)
    assert result.x == 42
    assert result.y == "hello"
    assert fake.calls == 1  # no retry needed


def test_structured_output_retries_on_invalid_json():
    fake = _FakeGenerator(
        [
            "This is not JSON at all, just prose.",
            '{"x": 7, "y": "retry-ok"}',
        ]
    )
    inv = _HFStructuredInvoker(fake, _SimpleOutput)
    result = inv.invoke([])
    assert result.x == 7
    assert result.y == "retry-ok"
    assert fake.calls == 2


def test_structured_output_retries_on_schema_mismatch():
    fake = _FakeGenerator(
        [
            '{"wrong_field": 99}',  # valid JSON but not matching schema
            '{"x": 1}',
        ]
    )
    inv = _HFStructuredInvoker(fake, _SimpleOutput)
    result = inv.invoke([])
    assert result.x == 1
    assert fake.calls == 2


def test_structured_output_raises_after_second_failure():
    fake = _FakeGenerator(
        ["still not json", "also not json"]
    )
    inv = _HFStructuredInvoker(fake, _SimpleOutput)
    with pytest.raises(ProviderUnavailableError, match="failed to produce valid"):
        inv.invoke([])
    assert fake.calls == 2


def test_structured_output_extracts_from_noisy_response():
    # Real-world-ish: model prepends a thought then emits JSON.
    fake = _FakeGenerator(
        ['Sure, here is the answer:\n{"x": 5, "y": "great"}\nLet me know if you need more.']
    )
    inv = _HFStructuredInvoker(fake, _SimpleOutput)
    result = inv.invoke([])
    assert result.x == 5


# ---------------------------------------------------------------------------
# 2. Torch-gated integration tests
# ---------------------------------------------------------------------------


def _torch_available() -> bool:
    try:
        import torch  # noqa: F401
        import transformers  # noqa: F401

        return True
    except ImportError:
        return False


requires_torch = pytest.mark.skipif(
    not _torch_available(),
    reason="torch/transformers not installed; pip install -r requirements-llm-local.txt",
)


@pytest.fixture(scope="module")
def _smol_provider():
    """Load SmolLM2-135M once per test module.

    Model is ~270 MB on first run (cached under ~/.cache/huggingface/hub);
    subsequent runs load from disk in seconds. We share the instance across
    the two integration tests below so we don't pay the load cost twice.
    """
    if not _torch_available():
        pytest.skip("torch/transformers not installed")
    from sql_agent.llm_serving.hf_provider import HuggingFaceProvider

    return HuggingFaceProvider(
        model_id="HuggingFaceTB/SmolLM2-135M-Instruct",
        device="cpu",
        max_new_tokens=24,
        quantization="none",
    )


@requires_torch
@pytest.mark.timeout(600)
def test_hf_provider_loads_tiny_model_and_generates(_smol_provider):
    """Real model: SmolLM2-135M-Instruct. Proves the full load + generate
    pipeline works end-to-end on CPU."""
    provider = _smol_provider
    assert provider.name == "hf"
    assert provider.device == "cpu"
    assert provider.model_id == "HuggingFaceTB/SmolLM2-135M-Instruct"

    chat = provider.chat_model(temperature=0.0)
    resp = chat.invoke(
        [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Say the single word 'hello'."},
        ]
    )
    assert isinstance(resp.content, str)
    assert len(resp.content) > 0


@requires_torch
def test_hf_provider_raises_when_transformers_missing(monkeypatch):
    """If transformers is mocked to be unimportable, construction raises a
    clean ProviderUnavailableError rather than a bare ImportError."""
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "transformers":
            raise ImportError("mocked: no transformers")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    from sql_agent.llm_serving.hf_provider import HuggingFaceProvider

    with pytest.raises(ProviderUnavailableError, match="torch \\+ transformers"):
        HuggingFaceProvider(model_id="some/model", device="cpu")


@requires_torch
@pytest.mark.timeout(600)
def test_hf_provider_accessible_through_registry(_smol_provider):
    """Registry should return the cached HuggingFaceProvider when primed.

    We prime with the shared module-scope provider (no second model load).
    """
    from sql_agent.llm_serving import registry
    from sql_agent.llm_serving.hf_provider import HuggingFaceProvider

    registry.reset_caches()
    with registry._lock:  # type: ignore[attr-defined]
        registry._llm_cache["hf"] = _smol_provider  # type: ignore[attr-defined]

    prov = registry.get_llm_provider("hf")
    assert isinstance(prov, HuggingFaceProvider)
    assert prov is _smol_provider
    registry.reset_caches()
