"""Centralized settings loaded from environment / .env file."""

from __future__ import annotations

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


PROJECT_ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    """Application settings driven by environment variables."""

    model_config = SettingsConfigDict(
        env_file=str(PROJECT_ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    openai_api_key: str = Field(default="", alias="OPENAI_API_KEY")
    openai_chat_model: str = Field(default="gpt-4o-mini", alias="OPENAI_CHAT_MODEL")
    openai_embedding_model: str = Field(
        default="text-embedding-3-small", alias="OPENAI_EMBEDDING_MODEL"
    )

    database_url: str = Field(
        default="sqlite:///sqlite_db/demo.db", alias="DATABASE_URL"
    )

    faiss_index_dir: str = Field(
        default="sqlite_db/faiss_index", alias="FAISS_INDEX_DIR"
    )
    chat_history_dir: str = Field(
        default="sqlite_db/chat_histories", alias="CHAT_HISTORY_DIR"
    )

    schema_cache_ttl_seconds: int = Field(
        default=300, alias="SCHEMA_CACHE_TTL_SECONDS"
    )
    data_fetch_default_limit: int = Field(
        default=100, alias="DATA_FETCH_DEFAULT_LIMIT"
    )
    data_fetch_max_limit: int = Field(default=10000, alias="DATA_FETCH_MAX_LIMIT")

    log_level: str = Field(default="INFO", alias="LOG_LEVEL")

    # Phase 1 (gRPC gateway). Additive; ignored by the CLI / Streamlit UI.
    grpc_port: int = Field(default=50051, alias="GRPC_PORT")

    # Phase 8.5 — gRPC TLS + bearer auth. Opt-in for production deployments
    # behind a service mesh (Istio / Linkerd); service meshes typically
    # terminate TLS + do mTLS themselves, so this is deliberately simple.
    grpc_tls_cert_file: str = Field(default="", alias="GRPC_TLS_CERT_FILE")
    grpc_tls_key_file: str = Field(default="", alias="GRPC_TLS_KEY_FILE")
    # Bearer-token auth. When non-empty, every RPC must carry
    # `authorization: Bearer <token>` metadata matching this value.
    grpc_auth_token: str = Field(default="", alias="GRPC_AUTH_TOKEN")

    # -----------------------------------------------------------------
    # Phase 2 (LLM serving abstraction). All additive; defaults preserve
    # the original OpenAI-backed behavior exactly.
    # -----------------------------------------------------------------
    llm_provider: str = Field(default="openai", alias="LLM_PROVIDER")
    embedding_provider: str = Field(default="auto", alias="EMBEDDING_PROVIDER")

    # Hardware controls (mutually exclusive; both false => auto-detect).
    force_cpu: bool = Field(default=False, alias="FORCE_CPU")
    force_gpu: bool = Field(default=False, alias="FORCE_GPU")

    # HuggingFace provider defaults (only used when LLM_PROVIDER=hf).
    hf_chat_model: str = Field(
        default="Qwen/Qwen2.5-1.5B-Instruct", alias="HF_CHAT_MODEL"
    )
    # Qwen2.5-1.5B-Instruct: ~3 GB, CPU-viable, follows JSON reasonably well.
    # Override via HF_CHAT_MODEL for other hardware (e.g. Qwen2.5-3B-Instruct
    # on a 16 GB machine or Qwen2.5-7B-Instruct on a GPU).
    hf_embedding_model: str = Field(
        default="sentence-transformers/all-MiniLM-L6-v2",
        alias="HF_EMBEDDING_MODEL",
    )
    hf_quantization: str = Field(
        default="none", alias="HF_QUANTIZATION"
    )  # "none" | "4bit" | "8bit" (bitsandbytes; Linux/WSL) | "int8_dynamic"
    # (torch.quantization.quantize_dynamic; CPU-portable, no bitsandbytes)
    hf_max_new_tokens: int = Field(default=512, alias="HF_MAX_NEW_TOKENS")
    hf_cache_dir: str = Field(default="", alias="HF_CACHE_DIR")

    # -----------------------------------------------------------------
    # Phase 3 (routing / A/B testing). All opt-in; disabled by default,
    # which makes the codebase behaviorally identical to post-phase-2.
    # -----------------------------------------------------------------
    llm_routing_enabled: bool = Field(default=False, alias="LLM_ROUTING_ENABLED")
    llm_routing_weights: str = Field(
        default="openai:100", alias="LLM_ROUTING_WEIGHTS"
    )  # format: "openai:70,hf:30"
    llm_routing_decision_log: str = Field(
        default="logs/routing/decisions.jsonl", alias="LLM_ROUTING_DECISION_LOG"
    )  # path, resolved relative to project root. Lives under logs/ so it is
    # clearly distinct from the sql_agent/routing/ source package — keeping
    # them separate avoids any confusion where a log-cleanup command could
    # accidentally delete source code.

    # Phase 8.7 — routing strategy + circuit breaker.
    llm_routing_strategy: str = Field(
        default="weighted", alias="LLM_ROUTING_STRATEGY"
    )  # "weighted" | "hash_by_id"
    llm_routing_circuit_breaker: bool = Field(
        default=False, alias="LLM_ROUTING_CIRCUIT_BREAKER"
    )
    llm_routing_breaker_threshold: int = Field(
        default=3, alias="LLM_ROUTING_BREAKER_THRESHOLD"
    )
    llm_routing_breaker_cooldown_seconds: float = Field(
        default=30.0, alias="LLM_ROUTING_BREAKER_COOLDOWN_SECONDS"
    )

    # -----------------------------------------------------------------
    # Phase 4 (ML lifecycle tracking). Opt-in; disabled by default.
    # -----------------------------------------------------------------
    tracking_enabled: bool = Field(default=False, alias="TRACKING_ENABLED")
    tracking_backend: str = Field(default="auto", alias="TRACKING_BACKEND")
    # "auto" tries mlflow, falls back to file; "mlflow"/"file"/"noop" force.
    tracking_file_log: str = Field(
        default="logs/tracking/turns.jsonl", alias="TRACKING_FILE_LOG"
    )
    mlflow_tracking_uri: str = Field(default="", alias="MLFLOW_TRACKING_URI")
    # Empty => file:<project_root>/logs/tracking/mlruns
    mlflow_experiment_name: str = Field(
        default="sql_agent", alias="MLFLOW_EXPERIMENT_NAME"
    )
    # Cap user_query length stored in tracking params (MLflow limit is 6000,
    # but shorter values keep the UI usable and reduce disk pressure).
    tracking_query_max_chars: int = Field(
        default=500, alias="TRACKING_QUERY_MAX_CHARS"
    )

    # -----------------------------------------------------------------
    # Phase 5 (observability). All opt-in; defaults preserve today's
    # human-readable logs and don't open any metrics port.
    # -----------------------------------------------------------------
    log_json: bool = Field(default=False, alias="LOG_JSON")
    metrics_enabled: bool = Field(default=False, alias="METRICS_ENABLED")
    metrics_port: int = Field(default=9090, alias="METRICS_PORT")
    metrics_addr: str = Field(default="", alias="METRICS_ADDR")
    # Empty addr => bind all interfaces (0.0.0.0). Set to "127.0.0.1" to
    # restrict to loopback.

    # Phase 8.6: append-only log rotation. Both DecisionLogWriter (routing
    # decisions) and FileTracker (per-turn tracking) rotate files when they
    # exceed ``log_rotation_max_bytes`` and keep at most
    # ``log_rotation_backup_count`` backups. 0 disables rotation.
    log_rotation_max_bytes: int = Field(
        default=50 * 1024 * 1024, alias="LOG_ROTATION_MAX_BYTES"
    )  # 50 MB
    log_rotation_backup_count: int = Field(
        default=5, alias="LOG_ROTATION_BACKUP_COUNT"
    )
    # OpenTelemetry tracing (phase 8.6). Off by default; emits spans around
    # run_turn and each agent node when enabled. OTLP exporter target
    # uses standard OTEL_* env vars (OTEL_EXPORTER_OTLP_ENDPOINT, etc.)
    # which OpenTelemetry's SDK picks up natively.
    otel_enabled: bool = Field(default=False, alias="OTEL_ENABLED")
    otel_service_name: str = Field(
        default="sql-agent", alias="OTEL_SERVICE_NAME"
    )

    # Phase 6 (containerization): optional demo-DB seeding at gRPC boot.
    # Off by default (production deployments should pre-seed or use a real
    # DATABASE_URL). The Dockerfile default leaves this off; the compose
    # "default" profile flips it on so a bare `docker compose up` gives a
    # working demo out of the box.
    seed_demo_on_boot: bool = Field(default=False, alias="SEED_DEMO_ON_BOOT")

    # Phase 7 (Kubernetes readiness): read-only mode for memory writes.
    # When true, record_reward / record_penalty no-op on this pod.
    # Historically (phases 7-8) required for multi-replica Deployments
    # with FAISS to avoid divergent memories. Phase 9 made this optional:
    # with MEMORY_STORE_BACKEND=qdrant all replicas can safely write to
    # the shared DB, but this flag remains useful for read-heavy HPA
    # pods where you want to reduce write contention.
    read_only_memory: bool = Field(default=False, alias="READ_ONLY_MEMORY")

    # -----------------------------------------------------------------
    # Phase 9 (vector-DB memory backends). Pluggable reward/penalty
    # memory: FAISS (default, pod-local) or Qdrant (shared, multi-
    # replica-safe). Fully additive — existing deployments stay on FAISS
    # unless they opt into Qdrant.
    # -----------------------------------------------------------------
    memory_store_backend: str = Field(
        default="faiss", alias="MEMORY_STORE_BACKEND"
    )  # "faiss" | "qdrant" | "auto"  (auto => qdrant iff QDRANT_URL set)

    # Qdrant configuration (only used when backend resolves to "qdrant").
    # Empty QDRANT_URL => embedded / on-disk mode at <faiss_dir>/qdrant.
    # Set to http://qdrant:6333 (or similar) for a shared service.
    qdrant_url: str = Field(default="", alias="QDRANT_URL")
    qdrant_api_key: str = Field(default="", alias="QDRANT_API_KEY")
    qdrant_collection_prefix: str = Field(
        default="sql_agent_", alias="QDRANT_COLLECTION_PREFIX"
    )

    @property
    def project_root(self) -> Path:
        return PROJECT_ROOT

    def resolved_path(self, value: str) -> Path:
        """Resolve a relative path against the project root."""
        p = Path(value)
        return p if p.is_absolute() else (PROJECT_ROOT / p)

    @property
    def faiss_dir(self) -> Path:
        return self.resolved_path(self.faiss_index_dir)

    @property
    def chat_dir(self) -> Path:
        return self.resolved_path(self.chat_history_dir)


settings = Settings()
