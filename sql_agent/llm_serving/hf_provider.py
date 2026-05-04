"""HuggingFace local-model provider.

Loads an instruction-tuned causal LM via ``transformers``, honors hardware
detection (CUDA > MPS > CPU, subject to FORCE_* overrides), and exposes the
same ``ChatModel`` surface the agents already use.

Structured output is implemented by prompting the model with the target
JSON schema, parsing the response with pydantic, and retrying once on
failure. No extra libraries (outlines / instructor) are required — this
keeps the dependency footprint small and works with any chat-template-
capable model.

Torch and transformers are imported lazily (inside ``__init__`` and inside
the generation helpers) so that importing this module on a system without
torch does NOT raise. The provider registry propagates a clean
``ProviderUnavailableError`` instead.
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional, Type

from pydantic import BaseModel, ValidationError

from sql_agent.config import get_logger, settings

from .base import ChatModel, ProviderUnavailableError
from .hardware import detect_device, log_execution_mode


_log = get_logger("llm_serving.hf")


# ---------------------------------------------------------------------------
# Message conversion
# ---------------------------------------------------------------------------


_ROLE_MAP = {
    "human": "user",
    "system": "system",
    "ai": "assistant",
    "assistant": "assistant",
    "user": "user",
    "tool": "tool",
}


def _to_chat_messages(messages: List[Any]) -> List[Dict[str, str]]:
    """Normalize LangChain message objects or plain dicts into HF chat dicts.

    Roles are always run through the same role map so the downstream chat
    template receives a known role name. Unknown roles default to ``"user"``.
    """
    out: List[Dict[str, str]] = []
    for m in messages:
        if isinstance(m, dict):
            raw_role = str(m.get("role", "user")).lower()
            content = str(m.get("content", ""))
        else:
            raw_role = str(
                getattr(m, "type", None) or getattr(m, "role", None) or "user"
            ).lower()
            content = str(getattr(m, "content", ""))
        out.append({"role": _ROLE_MAP.get(raw_role, "user"), "content": content})
    return out


# ---------------------------------------------------------------------------
# JSON extraction / structured-output parsing
# ---------------------------------------------------------------------------


_FENCE_RE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL)


def _extract_json(text: str) -> Optional[str]:
    """Best-effort JSON extraction from LLM prose.

    Preference order:
        1. first fenced ```json ... ``` block
        2. first '{' to the LAST '}' in the text (greedy)

    Returns the raw JSON string, or None if no candidate is found.
    """
    m = _FENCE_RE.search(text)
    if m:
        return m.group(1)
    first = text.find("{")
    last = text.rfind("}")
    if first != -1 and last != -1 and last > first:
        return text[first : last + 1]
    return None


_STRUCTURED_SYS_PROMPT = (
    "You must respond with VALID JSON matching the schema below.\n"
    "Output a SINGLE JSON object — no prose, no markdown, no code fences.\n\n"
    "JSON Schema:\n{schema}"
)


class _HFStructuredInvoker:
    """Wraps an _HFChatModel to enforce pydantic-typed output."""

    def __init__(self, chat: "_HFChatModel", model_cls: Type[BaseModel]) -> None:
        self._chat = chat
        self._model_cls = model_cls
        try:
            schema = model_cls.model_json_schema()
        except Exception:  # pragma: no cover
            schema = {}
        self._schema_str = json.dumps(schema, indent=2)

    def invoke(self, messages: List[Any]) -> BaseModel:
        sys_prefix = {
            "role": "system",
            "content": _STRUCTURED_SYS_PROMPT.format(schema=self._schema_str),
        }
        chat_msgs = [sys_prefix] + _to_chat_messages(messages)

        text = self._chat._generate(chat_msgs)
        parsed = self._try_parse(text)
        if parsed is not None:
            return parsed

        retry_nudge = {
            "role": "system",
            "content": (
                "Your previous response was not valid JSON for the required "
                "schema. Respond with ONE JSON object matching the schema and "
                "nothing else."
            ),
        }
        text2 = self._chat._generate(chat_msgs + [retry_nudge])
        parsed = self._try_parse(text2)
        if parsed is not None:
            return parsed

        raise ProviderUnavailableError(
            f"HFProvider: failed to produce valid {self._model_cls.__name__} "
            f"JSON after 2 attempts. Last response (first 400 chars): "
            f"{text2[:400]!r}"
        )

    def _try_parse(self, text: str) -> Optional[BaseModel]:
        raw = _extract_json(text) or text
        try:
            data = json.loads(raw)
        except json.JSONDecodeError as e:
            _log.debug("JSON decode failed: %s", e)
            return None
        try:
            return self._model_cls(**data)
        except (ValidationError, TypeError) as e:
            _log.debug("Pydantic validation failed: %s", e)
            return None


# ---------------------------------------------------------------------------
# HF-backed ChatModel
# ---------------------------------------------------------------------------


class _HFResp:
    """Minimal response object matching the ``HasContent`` protocol."""

    def __init__(self, content: str) -> None:
        self.content = content


class _HFChatModel:
    """Concrete ChatModel implementation calling transformers.generate()."""

    def __init__(
        self,
        *,
        model,
        tokenizer,
        device: str,
        temperature: float,
        max_new_tokens: int,
    ) -> None:
        self._model = model
        self._tokenizer = tokenizer
        self._device = device
        self.temperature = float(temperature)
        self._max_new_tokens = int(max_new_tokens)

    def with_structured_output(self, model_cls: Type[BaseModel]) -> _HFStructuredInvoker:
        return _HFStructuredInvoker(self, model_cls)

    def invoke(self, messages: List[Any]) -> _HFResp:
        chat_msgs = _to_chat_messages(messages)
        text = self._generate(chat_msgs)
        return _HFResp(text)

    def _generate(self, chat_msgs: List[Dict[str, str]]) -> str:
        import time as _time

        import torch  # lazy

        from sql_agent.request_context import record_token_usage

        tokenizer = self._tokenizer
        prompt = tokenizer.apply_chat_template(
            chat_msgs,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = tokenizer(prompt, return_tensors="pt")
        if self._device != "cpu":
            inputs = {k: v.to(self._device) for k, v in inputs.items()}

        do_sample = self.temperature > 0
        gen_kwargs: Dict[str, Any] = {
            "max_new_tokens": self._max_new_tokens,
            "do_sample": do_sample,
            "pad_token_id": tokenizer.pad_token_id or tokenizer.eos_token_id,
        }
        if do_sample:
            # HF rejects temperature=0 when do_sample=True.
            gen_kwargs["temperature"] = max(self.temperature, 0.01)

        t0 = _time.perf_counter()
        with torch.no_grad():
            output_ids = self._model.generate(**inputs, **gen_kwargs)
        latency_ms = (_time.perf_counter() - t0) * 1000.0

        prompt_len = inputs["input_ids"].shape[1]
        new_tokens = output_ids[0, prompt_len:]
        output_len = int(new_tokens.shape[0])

        model_id = getattr(self._model, "name_or_path", "hf-unknown") or "hf-unknown"
        # Phase 8.3: per-request token accumulator.
        record_token_usage(
            provider="hf", model=model_id,
            input_tokens=int(prompt_len), output_tokens=output_len,
        )
        # Phase 8.6: Prometheus metrics (per-provider latency + tokens).
        try:
            from sql_agent.observability.metrics import get_metrics

            get_metrics().record_llm_call(
                "hf", model_id, latency_ms,
                input_tokens=int(prompt_len), output_tokens=output_len,
            )
        except Exception:  # pragma: no cover
            pass

        return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


# ---------------------------------------------------------------------------
# Provider
# ---------------------------------------------------------------------------


class HuggingFaceProvider:
    """LLMProvider backed by a local HuggingFace causal LM.

    Construction is expensive (downloads + loads weights). Use the registry
    cache so the model is loaded at most once per process.
    """

    name: str = "hf"

    def __init__(
        self,
        *,
        model_id: Optional[str] = None,
        device: Optional[str] = None,
        max_new_tokens: Optional[int] = None,
        quantization: Optional[str] = None,
        cache_dir: Optional[str] = None,
    ) -> None:
        try:
            import torch  # noqa: F401
            import transformers  # noqa: F401
        except ImportError as exc:
            raise ProviderUnavailableError(
                "HuggingFace provider requires torch + transformers. "
                "Install with: pip install -r requirements-llm-local.txt "
                f"(underlying error: {exc})"
            ) from exc

        self.model_id = model_id or settings.hf_chat_model
        self.device = device or detect_device()
        self._max_new_tokens = max_new_tokens or settings.hf_max_new_tokens
        self._quantization = (quantization or settings.hf_quantization or "none").lower()
        self._cache_dir = cache_dir or (settings.hf_cache_dir or None)
        self._cache: Dict[Any, _HFChatModel] = {}

        log_execution_mode(self.device, self.model_id)
        self._tokenizer, self._model = self._load()

    # ------------------------------------------------------------------
    def _load(self):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        common_kwargs: Dict[str, Any] = {}
        if self._cache_dir:
            common_kwargs["cache_dir"] = self._cache_dir

        _log.info(
            "Loading HF tokenizer: model_id=%s cache_dir=%s",
            self.model_id,
            self._cache_dir or "<default>",
        )
        tokenizer = AutoTokenizer.from_pretrained(self.model_id, **common_kwargs)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model_kwargs: Dict[str, Any] = dict(common_kwargs)
        use_int8_dynamic = False

        if self._quantization in ("4bit", "8bit"):
            try:
                from transformers import BitsAndBytesConfig

                model_kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=(self._quantization == "4bit"),
                    load_in_8bit=(self._quantization == "8bit"),
                )
                model_kwargs["device_map"] = "auto"
                _log.info("bitsandbytes quantization enabled: %s", self._quantization)
            except Exception as exc:
                _log.warning(
                    "bitsandbytes (%s) unavailable (%s); falling back to full precision. "
                    "Tip: set HF_QUANTIZATION=int8_dynamic for a CPU-portable alternative.",
                    self._quantization,
                    exc,
                )
                self._quantization = "none"

        elif self._quantization == "int8_dynamic":
            # Phase 8.4: torch.quantization.quantize_dynamic on CPU. No
            # external dependencies, works on Windows/macOS/Linux.
            # Applied *after* model load (see below).
            if self.device != "cpu":
                _log.warning(
                    "int8_dynamic quantization is CPU-only; device=%s — "
                    "falling back to full precision on this device.",
                    self.device,
                )
                self._quantization = "none"
            else:
                use_int8_dynamic = True

        if self._quantization == "none" or use_int8_dynamic:
            model_kwargs["torch_dtype"] = (
                torch.float16 if self.device == "cuda" else torch.float32
            )

        _log.info(
            "Loading HF model: model_id=%s device=%s quantization=%s dtype=%s",
            self.model_id,
            self.device,
            self._quantization,
            model_kwargs.get("torch_dtype", "auto"),
        )
        model = AutoModelForCausalLM.from_pretrained(self.model_id, **model_kwargs)

        # bitsandbytes places the model via device_map; otherwise move explicitly.
        if "device_map" not in model_kwargs:
            model = model.to(self.device)

        # Phase 8.4: dynamic int8 on CPU — applied AFTER placement so the
        # model is materialized on the right device first.
        if use_int8_dynamic:
            _log.info("Applying torch.quantization.quantize_dynamic (int8, Linear layers)")
            model = torch.quantization.quantize_dynamic(
                model, {torch.nn.Linear}, dtype=torch.qint8
            )
            _log.info("int8_dynamic quantization applied")

        model.eval()
        return tokenizer, model

    # ------------------------------------------------------------------
    def chat_model(self, temperature: float = 0.0) -> ChatModel:
        key = round(float(temperature), 3)
        if key in self._cache:
            return self._cache[key]
        chat = _HFChatModel(
            model=self._model,
            tokenizer=self._tokenizer,
            device=self.device,
            temperature=temperature,
            max_new_tokens=self._max_new_tokens,
        )
        self._cache[key] = chat
        return chat
