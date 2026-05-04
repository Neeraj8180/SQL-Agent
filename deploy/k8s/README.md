# SQL Agent — Kubernetes manifests

Base manifests for deploying the SQL Agent gRPC server to a Kubernetes
cluster. Tested with `kubectl apply --dry-run=client` on kubectl 1.34.

## Files

| File | Purpose |
|---|---|
| `namespace.yaml` | `sql-agent` namespace |
| `configmap.yaml` | Non-secret env (LLM_PROVIDER, LOG_JSON, metrics, …) |
| `secret.example.yaml` | Template for the Secret holding `OPENAI_API_KEY` |
| `deployment.yaml` | 2-replica Deployment, gRPC liveness/readiness, resources, non-root |
| `service.yaml` | ClusterIP (gRPC + /metrics) + headless variant |
| `hpa.yaml` | HorizontalPodAutoscaler min=2 max=10 target=70% CPU |
| `pvc.yaml` | PVC for demo SQLite + FAISS (RWO, dev-only) |
| `servicemonitor.yaml` | Prometheus Operator CRD (optional) |
| `kustomization.yaml` | Base for overlays |

## Quick start

```bash
# 1. Build and push the image to your registry
docker build -t your-registry/sql-agent:v0.1.0 .
docker push your-registry/sql-agent:v0.1.0

# 2. Patch the image in deployment.yaml (or via overlay):
kustomize edit set image sql-agent=your-registry/sql-agent:v0.1.0

# 3. Create the OpenAI secret (if using OpenAI provider)
kubectl create secret generic sql-agent-secrets \
  --namespace sql-agent \
  --from-literal=OPENAI_API_KEY='sk-your-real-key'

# 4. Apply
kubectl apply -k deploy/k8s/

# 5. Watch pods come up
kubectl -n sql-agent get pods -w

# 6. Port-forward to test
kubectl -n sql-agent port-forward svc/sql-agent 50051:50051 9090:9090
grpc_health_probe -addr=localhost:50051     # -> status: SERVING
curl http://localhost:9090/metrics | head   # Prometheus metrics
```

## Known limitation: FAISS memory divergence

The agent's reward/penalty memory is backed by FAISS files on the pod's
local filesystem. **In a multi-replica Deployment each pod has its own
divergent copy** — a reward written by pod-A is invisible to pod-B.

Phase 7's mitigation: the `configmap.yaml` sets `READ_ONLY_MEMORY=true`
for all Deployment replicas, so they CONSULT memory but don't WRITE to
it. That prevents divergence but means no new rewards are ever recorded
in this topology.

If you need write-through memory:

1. **Quickest fix** — Deploy a dedicated writer pod as a StatefulSet
   (`replicas: 1`, its own RWO PVC, `READ_ONLY_MEMORY=false`) and keep
   this Deployment for reads. Use the `sql-agent-headless` Service to
   route writes directly to the writer pod.

2. **Proper fix** — Replace the file-based FAISS with a real vector DB
   (Qdrant, Weaviate, pgvector, Pinecone). That's out of scope for
   Phase 7; deferred to a future phase.

## Known limitation: PVC access mode

The `pvc.yaml` requests `ReadWriteOnce` (RWO), which **cannot be mounted
by >1 pod simultaneously on most clusters**. If HPA scales >1 replica
with this PVC, new pods will be stuck Pending with a mount error.

Options:

- **Multi-replica with RWO PVC**: remove `volumeMounts.agent-data` from
  `deployment.yaml` and switch the volume to `emptyDir`. Pods are
  stateless (`READ_ONLY_MEMORY=true` ensures no state is lost). Use a
  real external `DATABASE_URL` for persistence.
- **Multi-replica with shared memory**: use a ReadWriteMany (RWX) PVC
  backed by NFS / EFS / Azure Files / GCE Filestore. Set
  `accessModes: [ReadWriteMany]` in `pvc.yaml`.
- **Single replica**: set `hpa.yaml` `maxReplicas: 1` and this PVC works
  as-is.

## Resource sizing

Defaults target the *local-LLM image* (torch + transformers loaded):

| Resource | Request | Limit |
|---|---|---|
| CPU | 500m | 2 |
| Memory | 1 Gi | 4 Gi |

For the **minimal image** (OpenAI/Mock only, no torch), these are
oversized. Tune down in your overlay:

```yaml
# overlays/prod/patch-resources.yaml
- op: replace
  path: /spec/template/spec/containers/0/resources/requests/cpu
  value: "100m"
- op: replace
  path: /spec/template/spec/containers/0/resources/requests/memory
  value: "256Mi"
```

## Health probes

`deployment.yaml` uses **gRPC-native probes** (K8s 1.24+ required). These
hit the standard `grpc.health.v1.Health/Check` service registered by the
container at startup. No sidecar binary needed.

If you're on K8s < 1.24, replace with an `exec:` probe calling
`grpc_health_probe` (already baked into the image):

```yaml
livenessProbe:
  exec:
    command: ["grpc_health_probe", "-addr=127.0.0.1:50051"]
  initialDelaySeconds: 20
  periodSeconds: 30
```

## Dry-run validation

```bash
kubectl apply --dry-run=client -f deploy/k8s/
# or, with kustomize:
kustomize build deploy/k8s/ | kubectl apply --dry-run=client -f -
```
