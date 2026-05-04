"""Phase 7 tests — Kubernetes manifests + READ_ONLY_MEMORY behavior.

Two clusters:
    1. Static YAML validation (always runs):
         * Every manifest is parseable YAML.
         * Required kinds / names / invariants hold.
         * HPA sizing is sensible.
         * PVC documentation matches the multi-replica caveat.
    2. kubectl-backed schema validation (skipped if kubectl absent):
         * `kubectl apply --dry-run=client -f deploy/k8s/` passes.
    3. READ_ONLY_MEMORY contract (always runs):
         * record_reward / record_penalty no-op when flag is true.
         * recall still works (reads remain active).
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import Any, Dict, List

import pytest
import yaml

from sql_agent.config import settings


_REPO = Path(__file__).resolve().parent.parent
_K8S_DIR = _REPO / "deploy" / "k8s"
_MANIFESTS = [
    "namespace.yaml",
    "configmap.yaml",
    "secret.example.yaml",
    "deployment.yaml",
    "service.yaml",
    "hpa.yaml",
    "pvc.yaml",            # still present as an opt-in for single-replica users
    "servicemonitor.yaml",
    "kustomization.yaml",
    # Phase 8.9 additions
    "networkpolicy.yaml",
    "poddisruptionbudget.yaml",
    "statefulset-writer.yaml",
]


def _load_docs(path: Path) -> List[Dict[str, Any]]:
    raw = path.read_text(encoding="utf-8")
    return [d for d in yaml.safe_load_all(raw) if d is not None]


# ---------------------------------------------------------------------------
# 1. Static YAML validation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("filename", _MANIFESTS)
def test_manifest_parses_as_yaml(filename):
    path = _K8S_DIR / filename
    assert path.exists(), f"missing manifest: {path}"
    docs = _load_docs(path)
    assert len(docs) >= 1
    for d in docs:
        assert isinstance(d, dict), f"{filename}: top-level must be a mapping"
        assert "kind" in d or "apiVersion" in d or d.get("kind") == "Kustomization"


def test_deployment_invariants():
    docs = _load_docs(_K8S_DIR / "deployment.yaml")
    assert len(docs) == 1
    d = docs[0]
    assert d["kind"] == "Deployment"
    assert d["metadata"]["name"] == "sql-agent"
    spec = d["spec"]
    assert spec["replicas"] >= 1
    container = spec["template"]["spec"]["containers"][0]
    assert container["name"] == "sql-agent"
    # Image name is set; actual tag will be replaced in overlays.
    assert "sql-agent" in container["image"]

    port_names = [p["name"] for p in container["ports"]]
    assert "grpc" in port_names
    assert "metrics" in port_names

    # Resources set.
    assert "requests" in container["resources"]
    assert "limits" in container["resources"]

    # Non-root security.
    podsec = spec["template"]["spec"]["securityContext"]
    assert podsec["runAsNonRoot"] is True
    assert podsec["runAsUser"] == 10001

    # Probes use gRPC (K8s 1.24+).
    assert "grpc" in container["livenessProbe"]
    assert "grpc" in container["readinessProbe"]

    # Phase 8.9: horizontally-scalable Deployment uses emptyDir (NOT a PVC).
    volumes = spec["template"]["spec"]["volumes"]
    for v in volumes:
        assert "emptyDir" in v, (
            f"Deployment volume {v.get('name')!r} is not emptyDir — a PVC "
            "on a multi-replica Deployment is an anti-pattern; move it to "
            "statefulset-writer.yaml or use an RWX PVC."
        )


def test_service_has_both_ports():
    docs = _load_docs(_K8S_DIR / "service.yaml")
    # Should contain ClusterIP + headless variants.
    assert len(docs) >= 1
    svc = next(d for d in docs if d.get("metadata", {}).get("name") == "sql-agent")
    port_names = {p["name"] for p in svc["spec"]["ports"]}
    assert port_names == {"grpc", "metrics"}


def test_configmap_has_phase_defaults():
    docs = _load_docs(_K8S_DIR / "configmap.yaml")
    cm = docs[0]
    assert cm["kind"] == "ConfigMap"
    data = cm["data"]

    # Phase 2 defaults
    assert data["LLM_PROVIDER"] == "mock"
    assert data["EMBEDDING_PROVIDER"] == "mock"
    # Phase 4 tracking
    assert data["TRACKING_ENABLED"] == "true"
    assert data["TRACKING_BACKEND"] == "file"
    # Phase 5 observability
    assert data["LOG_JSON"] == "true"
    assert data["METRICS_ENABLED"] == "true"
    # Phase 7: multi-replica safety
    assert data["READ_ONLY_MEMORY"] == "true"


def test_hpa_sizing_is_sensible():
    docs = _load_docs(_K8S_DIR / "hpa.yaml")
    hpa = docs[0]
    assert hpa["kind"] == "HorizontalPodAutoscaler"
    s = hpa["spec"]
    assert s["minReplicas"] >= 1
    assert s["maxReplicas"] >= s["minReplicas"]
    # Target CPU utilization is an integer in (0, 100].
    cpu_metric = next(m for m in s["metrics"] if m["resource"]["name"] == "cpu")
    target = cpu_metric["resource"]["target"]["averageUtilization"]
    assert 0 < target <= 100


def test_pvc_is_rwo_with_documented_warning():
    docs = _load_docs(_K8S_DIR / "pvc.yaml")
    pvc = docs[0]
    assert pvc["kind"] == "PersistentVolumeClaim"
    assert pvc["spec"]["accessModes"] == ["ReadWriteOnce"]
    # The manifest text must call out the RWO + multi-replica caveat so
    # operators don't silently mis-configure. This also makes the limitation
    # part of the contract: if someone deletes the warning, the test fails.
    raw = (_K8S_DIR / "pvc.yaml").read_text(encoding="utf-8")
    assert "ReadWriteOnce" in raw
    assert "multi-replica" in raw.lower() or "readwritemany" in raw.lower()


def test_kustomization_lists_all_expected_resources():
    docs = _load_docs(_K8S_DIR / "kustomization.yaml")
    k = docs[0]
    assert k["kind"] == "Kustomization"
    assert k["namespace"] == "sql-agent"
    # Phase 8.9: pvc.yaml removed from base (Deployment uses emptyDir now).
    required = {
        "namespace.yaml", "configmap.yaml", "deployment.yaml",
        "service.yaml", "hpa.yaml",
        "poddisruptionbudget.yaml", "networkpolicy.yaml",
    }
    listed = set(k["resources"])
    assert required.issubset(listed), f"missing from kustomization: {required - listed}"


# ---------------------------------------------------------------------------
# 2. kubectl dry-run schema validation
# ---------------------------------------------------------------------------


def _kubectl_available() -> bool:
    if shutil.which("kubectl") is None:
        return False
    try:
        r = subprocess.run(
            ["kubectl", "version", "--client=true", "--output=json"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        return r.returncode == 0
    except Exception:
        return False


requires_kubectl = pytest.mark.skipif(
    not _kubectl_available(),
    reason="kubectl CLI unavailable",
)


@requires_kubectl
@pytest.mark.timeout(60)
def test_kubectl_kustomize_builds_offline():
    """`kubectl kustomize <dir>` renders the base without contacting a cluster.

    `--dry-run=client --validate=true` requires a live cluster to fetch the
    OpenAPI schema. Since Phase 7 explicitly validates manifests "without a
    cluster", we use `kubectl kustomize` instead — purely a template
    operation — then parse every rendered doc with PyYAML to prove the
    output is valid YAML AND contains the expected kinds.
    """
    r = subprocess.run(
        ["kubectl", "kustomize", str(_K8S_DIR)],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert r.returncode == 0, (
        f"kubectl kustomize failed:\nSTDOUT:\n{r.stdout}\nSTDERR:\n{r.stderr}"
    )
    # Parse every document in the rendered output.
    docs = [d for d in yaml.safe_load_all(r.stdout) if d is not None]
    assert len(docs) >= 7  # namespace + configmap + 2 services + deployment + hpa + pdb + networkpolicies

    kinds = [d.get("kind") for d in docs]
    assert "Namespace" in kinds
    assert "ConfigMap" in kinds
    assert "Deployment" in kinds
    assert "Service" in kinds
    assert "HorizontalPodAutoscaler" in kinds
    # Phase 8.9 additions:
    assert "PodDisruptionBudget" in kinds
    assert "NetworkPolicy" in kinds

    # All objects landed in the sql-agent namespace (except cluster-scoped
    # ones like Namespace).
    for d in docs:
        if d.get("kind") in {"Namespace"}:
            continue
        assert d.get("metadata", {}).get("namespace") == "sql-agent", (
            f"{d.get('kind')} {d.get('metadata', {}).get('name')} is missing namespace"
        )


@requires_kubectl
@pytest.mark.timeout(60)
def test_kubectl_kustomize_has_no_deprecation_warnings():
    """New `labels:` syntax should not trigger kustomize deprecation warnings."""
    r = subprocess.run(
        ["kubectl", "kustomize", str(_K8S_DIR)],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert r.returncode == 0
    # Only well-known, non-actionable warnings are allowed through; the
    # `commonLabels` deprecation is one we explicitly fixed.
    assert "commonLabels" not in (r.stderr or "")
    assert "deprecated" not in (r.stderr or "").lower()


# ---------------------------------------------------------------------------
# Phase 8.9 — real kind-cluster deployment test (slow, opt-in)
# ---------------------------------------------------------------------------


def _kind_available() -> bool:
    return shutil.which("kind") is not None


requires_kind_and_docker = pytest.mark.skipif(
    not (_kind_available() and shutil.which("docker")),
    reason="kind + docker required for cluster-runtime test",
)


@requires_kind_and_docker
@pytest.mark.slow
@pytest.mark.timeout(600)
def test_kind_deploy_reaches_ready():
    """End-to-end: spin up a single-node kind cluster, load the sql-agent
    image, apply the kustomize base, assert both HPA-managed pods become
    Ready. ~2 min.

    Requires the `sql-agent:latest` image to exist locally (build via
    `docker build -t sql-agent:latest .`).
    """
    # Skip if the image doesn't exist yet.
    check_img = subprocess.run(
        ["docker", "image", "inspect", "sql-agent:latest"],
        capture_output=True, text=True, timeout=10,
    )
    if check_img.returncode != 0:
        pytest.skip("sql-agent:latest image not built locally")

    cluster = "sql-agent-pytest-kind"
    try:
        create = subprocess.run(
            ["kind", "create", "cluster", "--name", cluster, "--wait", "120s"],
            capture_output=True, text=True, timeout=180,
        )
        assert create.returncode == 0, (
            f"kind create failed:\n{create.stdout}\n{create.stderr}"
        )

        load = subprocess.run(
            ["kind", "load", "docker-image", "sql-agent:latest", "--name", cluster],
            capture_output=True, text=True, timeout=120,
        )
        assert load.returncode == 0, f"kind load failed: {load.stderr}"

        apply = subprocess.run(
            [
                "kubectl",
                "--context", f"kind-{cluster}",
                "apply", "-k", str(_K8S_DIR),
            ],
            capture_output=True, text=True, timeout=60,
        )
        assert apply.returncode == 0, f"kubectl apply failed: {apply.stderr}"

        wait = subprocess.run(
            [
                "kubectl",
                "--context", f"kind-{cluster}",
                "-n", "sql-agent",
                "wait", "--for=condition=Ready", "pod",
                "-l", "app.kubernetes.io/name=sql-agent",
                "--timeout=180s",
            ],
            capture_output=True, text=True, timeout=240,
        )
        assert wait.returncode == 0, (
            f"pods didn't become ready:\n{wait.stdout}\n{wait.stderr}"
        )
        # Should have >=2 pods due to HPA minReplicas.
        assert wait.stdout.count("condition met") >= 2, (
            f"expected >=2 ready pods, saw:\n{wait.stdout}"
        )
    finally:
        subprocess.run(
            ["kind", "delete", "cluster", "--name", cluster],
            capture_output=True, text=True, timeout=120,
        )


# ---------------------------------------------------------------------------
# 3. READ_ONLY_MEMORY behavior
# ---------------------------------------------------------------------------


@pytest.fixture
def read_only_mode(tmp_path, monkeypatch):
    """Switch memory_manager to read-only + mock embedder for one test."""
    from sql_agent.llm_serving import registry as llm_reg
    from sql_agent.services.memory_manager import reset_memory_manager

    orig_llm = settings.llm_provider
    orig_emb = settings.embedding_provider
    orig_ro = settings.read_only_memory
    orig_faiss = settings.faiss_index_dir
    try:
        settings.llm_provider = "mock"
        settings.embedding_provider = "mock"
        settings.read_only_memory = True
        settings.faiss_index_dir = str(tmp_path / "faiss")
        llm_reg.reset_caches()
        reset_memory_manager()
        yield
    finally:
        settings.llm_provider = orig_llm
        settings.embedding_provider = orig_emb
        settings.read_only_memory = orig_ro
        settings.faiss_index_dir = orig_faiss
        llm_reg.reset_caches()
        reset_memory_manager()


def test_read_only_memory_blocks_reward_and_penalty_writes(read_only_mode):
    from sql_agent.services.memory_manager import MemoryManager

    mm = MemoryManager(root_dir=Path(settings.faiss_dir))
    assert mm.reward_size == 0

    mm.record_reward(
        "How many orders?",
        parameters={"table_names": ["orders"]},
        tool_used="count",
    )
    mm.record_penalty("bad query", reason="syntax")

    # Writes silently skipped.
    assert mm.reward_size == 0
    assert mm.penalty_size == 0


def test_read_only_memory_preserves_recall_path(read_only_mode):
    """Reads remain functional; recall returns whatever is already present
    (here, nothing — which is still a valid, non-exceptional result)."""
    from sql_agent.services.memory_manager import MemoryManager

    mm = MemoryManager(root_dir=Path(settings.faiss_dir))
    rules = mm.recall("any query", k_reward=3, k_penalty=3)
    assert rules == []


def test_read_only_memory_default_is_false():
    """Safety: the default must keep writes enabled so single-replica and
    dev setups don't silently drop memory."""
    # Freshly-parsed default (not the possibly-mutated instance).
    from sql_agent.config.settings import Settings

    fresh = Settings()
    assert fresh.read_only_memory is False
