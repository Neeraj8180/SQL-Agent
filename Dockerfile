# syntax=docker/dockerfile:1.7
# ============================================================================
# SQL Agent — default image (CPU-minimal, OpenAI/Mock providers only).
#
# Target size: ~300 MB. Does NOT bundle torch / transformers / mlflow —
# use Dockerfile.local-llm for those. Enough to run:
#   * gRPC gateway with OpenAI or Mock LLM_PROVIDER
#   * Routing, tracking (file-backed), observability (prometheus)
#
# Build:
#   docker build -t sql-agent:latest .
#
# Run:
#   docker run --rm -p 50051:50051 -p 9090:9090 \
#       -e LLM_PROVIDER=mock -e EMBEDDING_PROVIDER=mock \
#       -e METRICS_ENABLED=true \
#       sql-agent:latest
# ============================================================================

ARG PYTHON_IMAGE=python:3.11.9-slim-bookworm

# ---------------------------------------------------------------------------
# Builder stage — installs wheels into a venv, then this stage is discarded.
# Keeping build tools out of the final image saves ~150 MB.
# ---------------------------------------------------------------------------
FROM ${PYTHON_IMAGE} AS builder

# NOTE: we deliberately skip apt-get install. All our pip dependencies
# (grpcio, faiss-cpu, pydantic-core, psycopg2-binary, etc.) publish
# prebuilt manylinux wheels, so no compiler is required. This keeps the
# builder stage lean AND avoids dependency on the system apt mirror in
# build environments where HTTP egress to deb.debian.org is filtered.

WORKDIR /build

# Create a venv in /opt/venv so the final stage can just COPY it.
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install dependencies first (cache-friendly). We need a minimal stub
# package for `pip install -e .` to succeed without the full source.
COPY pyproject.toml /build/pyproject.toml
COPY sql_agent/__init__.py /build/sql_agent/__init__.py

RUN pip install --no-cache-dir --upgrade pip setuptools wheel \
 && pip install --no-cache-dir ".[grpc,observability,tracking]" \
 && pip freeze > /opt/venv/requirements.lock

# Copy full source last so code edits don't invalidate the dependency cache.
COPY sql_agent /build/sql_agent
COPY scripts /build/scripts

# Install our package in no-deps mode (everything is already satisfied).
RUN pip install --no-cache-dir --no-deps -e "."

# ---------------------------------------------------------------------------
# Final stage — slim runtime.
# ---------------------------------------------------------------------------
FROM ${PYTHON_IMAGE} AS final

# ca-certificates is already present in python:*-slim base images, so we
# don't need apt at all in the final stage. This bypasses any apt mirror
# filtering in the build environment and keeps the image minimal.

# grpc_health_probe v0.4.38 — small static binary for HEALTHCHECK.
ARG GRPC_HEALTH_PROBE_VERSION=v0.4.38
ADD --chmod=0755 https://github.com/grpc-ecosystem/grpc-health-probe/releases/download/${GRPC_HEALTH_PROBE_VERSION}/grpc_health_probe-linux-amd64 \
    /usr/local/bin/grpc_health_probe

# Non-root user; UID 10001 is well above reserved ranges.
RUN groupadd --system --gid 10001 sql_agent \
 && useradd --system --uid 10001 --gid 10001 --home-dir /app --shell /sbin/nologin sql_agent

WORKDIR /app

# Venv + code from builder, owned by the non-root runtime user directly.
# Using --chown here instead of a subsequent RUN chown avoids an extra
# ~800 MB duplicate layer (chown touches every file and thus rewrites it).
COPY --from=builder --chown=sql_agent:sql_agent /opt/venv /opt/venv
COPY --from=builder --chown=sql_agent:sql_agent /build/sql_agent /app/sql_agent
COPY --from=builder --chown=sql_agent:sql_agent /build/scripts /app/scripts

# Writeable runtime dirs (SQLite, FAISS indices, logs, tracking).
RUN mkdir -p /app/sql_agent/sqlite_db \
             /app/sql_agent/logs \
             /app/logs \
 && chown sql_agent:sql_agent /app/sql_agent/sqlite_db \
                              /app/sql_agent/logs \
                              /app/logs

ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    GRPC_PORT=50051 \
    METRICS_PORT=9090

USER sql_agent

EXPOSE 50051 9090

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD grpc_health_probe -addr=127.0.0.1:${GRPC_PORT} || exit 1

ENTRYPOINT ["python", "scripts/run_grpc_server.py"]
CMD []
