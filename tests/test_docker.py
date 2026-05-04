"""Phase 6 tests — Docker build and container smoke.

These tests are SKIPPED when the Docker CLI or daemon is unavailable, so
they don't block CI on machines without Docker. When Docker is present
they:

    * Build the minimal Dockerfile (sql-agent:test)
    * Assert image size is within budget
    * Run the container with LLM_PROVIDER=mock + a gRPC port
    * Run grpc_health_probe against it via `docker exec`
    * Tear down

Build time is ~60 s cold and ~5 s warm thanks to layer caching. The
local-LLM and GPU Dockerfiles are NOT built in CI by default (they're
multi-GB) — smoke-build is a `-m slow` opt-in.
"""

from __future__ import annotations

import shutil
import subprocess
import time
import uuid

import pytest


IMAGE_TAG = "sql-agent:pytest"
MINIMAL_IMAGE_SIZE_MB_BUDGET = 500  # Ceiling; observed ~300 MB.


def _docker_available() -> bool:
    if shutil.which("docker") is None:
        return False
    try:
        r = subprocess.run(
            ["docker", "info", "--format", "{{.ServerVersion}}"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        return r.returncode == 0 and bool(r.stdout.strip())
    except Exception:
        return False


requires_docker = pytest.mark.skipif(
    not _docker_available(),
    reason="Docker CLI or daemon unavailable",
)


def _run(cmd, **kwargs) -> subprocess.CompletedProcess:
    kwargs.setdefault("capture_output", True)
    kwargs.setdefault("text", True)
    kwargs.setdefault("timeout", 600)
    return subprocess.run(cmd, **kwargs)


# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------


@requires_docker
@pytest.mark.timeout(900)
def test_minimal_image_builds_and_fits_size_budget(tmp_path):
    """docker build the default Dockerfile and check the final size."""
    import os

    # Build from repo root (so .dockerignore and Dockerfile are picked up).
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    r = _run(
        [
            "docker",
            "build",
            "-t",
            IMAGE_TAG,
            "-f",
            "Dockerfile",
            repo_root,
        ],
        cwd=repo_root,
        timeout=900,
    )
    assert r.returncode == 0, f"build failed:\nSTDOUT:\n{r.stdout}\nSTDERR:\n{r.stderr}"

    # Image size check.
    r2 = _run(
        ["docker", "inspect", "-f", "{{.Size}}", IMAGE_TAG],
        cwd=repo_root,
    )
    assert r2.returncode == 0, r2.stderr
    size_mb = int(r2.stdout.strip()) / (1024 * 1024)
    assert size_mb < MINIMAL_IMAGE_SIZE_MB_BUDGET, (
        f"image size {size_mb:.0f} MB exceeds budget {MINIMAL_IMAGE_SIZE_MB_BUDGET} MB"
    )


# ---------------------------------------------------------------------------
# Run + HEALTHCHECK
# ---------------------------------------------------------------------------


@requires_docker
@pytest.mark.timeout(180)
def test_container_healthcheck_reports_serving():
    """Boot the image and probe the standard grpc.health.v1.Health service."""
    # Prerequisite: the build test must have run first. If it didn't, build now.
    r = _run(["docker", "image", "inspect", IMAGE_TAG])
    if r.returncode != 0:
        pytest.skip(
            f"{IMAGE_TAG} not present; run "
            "'docker build -t sql-agent:pytest .' or the build test first"
        )

    container_name = f"sql-agent-test-{uuid.uuid4().hex[:8]}"
    r = _run(
        [
            "docker",
            "run",
            "-d",
            "--rm",
            "--name",
            container_name,
            "-e",
            "LLM_PROVIDER=mock",
            "-e",
            "EMBEDDING_PROVIDER=mock",
            "-p",
            "0:50051",   # auto-assign host port so parallel runs don't collide
            IMAGE_TAG,
        ],
    )
    assert r.returncode == 0, f"docker run failed: {r.stderr}"

    try:
        # Wait up to 30s for the container to be in a running state and
        # grpc_health_probe to report SERVING.
        deadline = time.time() + 30
        last_err = ""
        while time.time() < deadline:
            r = _run(
                [
                    "docker",
                    "exec",
                    container_name,
                    "grpc_health_probe",
                    "-addr=127.0.0.1:50051",
                ],
            )
            if r.returncode == 0:
                # grpc_health_probe writes "status: SERVING" to stderr;
                # the returncode is the real success signal.
                assert "status: SERVING" in (r.stdout + r.stderr)
                return
            last_err = (r.stdout + r.stderr).strip()
            time.sleep(1)
        pytest.fail(f"health probe never succeeded within 30s. Last: {last_err!r}")
    finally:
        _run(["docker", "stop", "-t", "2", container_name])


# ---------------------------------------------------------------------------
# Smoke: container imports succeed without running the server
# ---------------------------------------------------------------------------


@requires_docker
@pytest.mark.timeout(60)
def test_container_can_import_servicer():
    """A one-shot python invocation inside the image should import cleanly."""
    r = _run(["docker", "image", "inspect", IMAGE_TAG])
    if r.returncode != 0:
        pytest.skip(f"{IMAGE_TAG} not present")

    r = _run(
        [
            "docker",
            "run",
            "--rm",
            "--entrypoint",
            "python",
            IMAGE_TAG,
            "-c",
            (
                "from sql_agent.grpc_server import SqlAgentServicer; "
                "from sql_agent.llm_serving.mock_provider import MockProvider; "
                "print('IMPORTS_OK')"
            ),
        ]
    )
    assert r.returncode == 0, f"import test failed: {r.stderr}"
    assert "IMPORTS_OK" in r.stdout


# ---------------------------------------------------------------------------
# Compose syntax smoke
# ---------------------------------------------------------------------------


@requires_docker
@pytest.mark.timeout(30)
def test_docker_compose_config_is_valid():
    """`docker compose config` validates the compose file's schema."""
    r = _run(["docker", "compose", "-f", "docker-compose.yml", "config"])
    assert r.returncode == 0, f"compose config invalid: {r.stderr}"
    # Basic sanity: the 'sql-agent' service is present.
    assert "sql-agent" in r.stdout
