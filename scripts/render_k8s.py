"""Quick local inspection of the Kubernetes manifests.

Usage:
    python scripts/render_k8s.py           # prints all manifests concatenated
    python scripts/render_k8s.py --list    # just filenames
    python scripts/render_k8s.py --kubectl # pipe through kubectl --dry-run=client

Does not require kustomize; just concatenates the YAMLs in the declared
kustomization order and prints them to stdout. This is a debugging aid,
not a production deploy tool — use `kubectl apply -k deploy/k8s/` for real.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import yaml


_ROOT = Path(__file__).resolve().parent.parent
_K8S_DIR = _ROOT / "deploy" / "k8s"


def _ordered_manifests() -> list[Path]:
    kfile = _K8S_DIR / "kustomization.yaml"
    if not kfile.exists():
        # Fallback: glob all YAMLs.
        return sorted(_K8S_DIR.glob("*.yaml"))

    k = yaml.safe_load(kfile.read_text(encoding="utf-8")) or {}
    resources = k.get("resources") or []
    return [_K8S_DIR / r for r in resources if (_K8S_DIR / r).exists()]


def main() -> int:
    p = argparse.ArgumentParser(description="Render the K8s base manifests.")
    p.add_argument("--list", action="store_true", help="Print only manifest filenames.")
    p.add_argument(
        "--kubectl",
        action="store_true",
        help="Run `kubectl apply --dry-run=client` on the concatenated output.",
    )
    args = p.parse_args()

    files = _ordered_manifests()
    if args.list:
        for f in files:
            print(f.relative_to(_ROOT))
        return 0

    chunks: list[str] = []
    for f in files:
        chunks.append(f"# ---- {f.name} ----\n")
        chunks.append(f.read_text(encoding="utf-8"))
        chunks.append("\n---\n")
    rendered = "\n".join(chunks)

    if args.kubectl:
        proc = subprocess.run(
            ["kubectl", "apply", "--dry-run=client", "-f", "-"],
            input=rendered,
            text=True,
            capture_output=True,
        )
        if proc.returncode != 0:
            sys.stderr.write(proc.stderr)
            return proc.returncode
        print(proc.stdout)
        return 0

    print(rendered)
    return 0


if __name__ == "__main__":
    sys.exit(main())
