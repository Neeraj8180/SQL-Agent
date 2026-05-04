"""Phase 2.1 tests for the new llm_serving package.

Scope: the package itself. Integration with services/llm.py comes in 2.2.
"""

from __future__ import annotations

import pytest

from sql_agent.llm_serving import base, hardware, registry
from sql_agent.llm_serving.mock_embedder import MockEmbedder
from sql_agent.llm_serving.mock_provider import MockProvider
from sql_agent.llm_serving.openai_embedder import OpenAIEmbedder
from sql_agent.llm_serving.openai_provider import OpenAIProvider


# ---------------------------------------------------------------------------
# Hardware detection
# ---------------------------------------------------------------------------


def test_detect_device_forced_cpu():
    assert hardware.detect_device(force_cpu=True, force_gpu=False) == "cpu"


def test_detect_device_rejects_both_forced(monkeypatch):
    with pytest.raises(RuntimeError, match="cannot both be true"):
        hardware.detect_device(force_cpu=True, force_gpu=True)


def test_detect_device_force_gpu_without_cuda_raises(monkeypatch):
    # Force _torch_module to report no CUDA.
    monkeypatch.setattr(hardware, "_torch_module", lambda: None)
    with pytest.raises(RuntimeError, match="no CUDA device"):
        hardware.detect_device(force_cpu=False, force_gpu=True)


def test_detect_device_default_cpu_when_no_torch(monkeypatch):
    monkeypatch.setattr(hardware, "_torch_module", lambda: None)
    assert hardware.detect_device(force_cpu=False, force_gpu=False) == "cpu"


def test_log_execution_mode_does_not_raise(caplog):
    # Just make sure it doesn't blow up for any device spelling.
    for dev in ("cuda", "mps", "cpu", "unknown"):
        hardware.log_execution_mode(dev, "some/model")


# ---------------------------------------------------------------------------
# Protocols (runtime_checkable)
# ---------------------------------------------------------------------------


def test_mock_provider_conforms_to_llm_provider_protocol():
    p = MockProvider()
    assert isinstance(p, base.LLMProvider)
    assert p.name == "mock"
    assert p.device == "cpu"


def test_openai_provider_conforms_to_llm_provider_protocol():
    # No API key required to *construct* — only to call chat_model.
    p = OpenAIProvider(api_key="")
    assert isinstance(p, base.LLMProvider)
    assert p.name == "openai"
    assert p.device == "remote"


def test_mock_embedder_conforms_to_embedding_provider_protocol():
    e = MockEmbedder()
    assert isinstance(e, base.EmbeddingProvider)
    # Default is small (64) now that phase 2.4 dim-namespaces FAISS files.
    # Callers that want OpenAI-compatible dims can pass dim=1536 explicitly.
    assert e.dimension == 64
    assert len(e.embed("hello")) == 64
    # Custom dim still works.
    assert len(MockEmbedder(dim=128).embed("x")) == 128


def test_openai_embedder_conforms_to_embedding_provider_protocol():
    e = OpenAIEmbedder(api_key="")
    assert isinstance(e, base.EmbeddingProvider)
    assert e.dimension == 1536  # default text-embedding-3-small


# ---------------------------------------------------------------------------
# MockProvider behavior
# ---------------------------------------------------------------------------


def test_mock_provider_chat_model_structured_output_for_intent():
    from sql_agent.models import Intent

    chat = MockProvider().chat_model()
    result = chat.with_structured_output(Intent).invoke([])
    assert isinstance(result, Intent)
    assert result.output_type.value == "count"


def test_mock_provider_chat_model_structured_output_for_param_plan():
    # ParamPlan is defined inside param_builder_agent; import its class.
    from sql_agent.agents.param_builder_agent import ParamPlan

    chat = MockProvider().chat_model()
    result = chat.with_structured_output(ParamPlan).invoke([])
    assert isinstance(result, ParamPlan)
    assert result.params.table_names == ["orders"]


def test_mock_provider_unknown_schema_raises_descriptive_error():
    from pydantic import BaseModel

    class Unknown(BaseModel):
        x: int = 0

    chat = MockProvider().chat_model()
    with pytest.raises(NotImplementedError, match="Unknown"):
        chat.with_structured_output(Unknown).invoke([])


def test_mock_provider_custom_registration_overrides_default():
    from sql_agent.models import Intent

    p = MockProvider()
    p.register("Intent", lambda cls: cls(output_type="list_unique"))
    result = p.chat_model().with_structured_output(Intent).invoke([])
    assert result.output_type.value == "list_unique"


def test_mock_provider_plain_invoke_has_content():
    resp = MockProvider().chat_model().invoke([])
    assert hasattr(resp, "content")
    assert isinstance(resp.content, str)


# ---------------------------------------------------------------------------
# MockEmbedder determinism
# ---------------------------------------------------------------------------


def test_mock_embedder_is_deterministic():
    e = MockEmbedder(dim=32)
    v1 = e.embed("the quick brown fox")
    v2 = e.embed("the quick brown fox")
    assert v1 == v2
    assert len(v1) == 32


def test_mock_embedder_batch_matches_single():
    e = MockEmbedder(dim=8)
    batch = e.embed_batch(["a", "b", "c"])
    assert batch == [e.embed("a"), e.embed("b"), e.embed("c")]


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


def test_registry_builds_mock_llm(monkeypatch):
    registry.reset_caches()
    prov = registry.get_llm_provider("mock")
    assert isinstance(prov, MockProvider)


def test_registry_caches_llm_providers(monkeypatch):
    registry.reset_caches()
    p1 = registry.get_llm_provider("mock")
    p2 = registry.get_llm_provider("mock")
    assert p1 is p2


def test_registry_builds_openai_llm_lazily():
    registry.reset_caches()
    prov = registry.get_llm_provider("openai")
    assert isinstance(prov, OpenAIProvider)


def test_registry_rejects_unknown_provider():
    registry.reset_caches()
    with pytest.raises(base.ProviderUnavailableError, match="Unknown"):
        registry.get_llm_provider("wizardlm")


def test_registry_embedder_auto_follows_llm_provider():
    """When LLM_PROVIDER=mock and EMBEDDING_PROVIDER=auto -> MockEmbedder."""
    from sql_agent.config import settings as s

    # pydantic BaseSettings doesn't play well with monkeypatch.setattr;
    # save/restore manually.
    orig_llm = s.llm_provider
    orig_emb = s.embedding_provider
    try:
        s.llm_provider = "mock"
        s.embedding_provider = "auto"
        registry.reset_caches()
        e = registry.get_embedding_provider()
        assert isinstance(e, MockEmbedder)
    finally:
        s.llm_provider = orig_llm
        s.embedding_provider = orig_emb
        registry.reset_caches()


# NOTE: test_registry_hf_provider_not_available_yet (phase 2.1) was removed
# in phase 2.3 because the HF provider is now real. The torch-missing path is
# covered by tests/test_hf_provider.py::test_hf_provider_raises_when_transformers_missing.
