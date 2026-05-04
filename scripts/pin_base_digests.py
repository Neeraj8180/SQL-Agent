"""Phase 8.8 — resolve and stamp base-image digests into the Dockerfiles.

Docker best-practice for production builds is to pin FROM lines by
sha256 digest, not by mutable tag (the ``3.11.9-slim`` tag CAN be
overwritten upstream).

This script uses ``docker buildx imagetools inspect --format '{{json
.Manifest.Digest}}'`` to resolve the current digest for each base tag
declared in this project, then rewrites the Dockerfile's PYTHON_IMAGE
/ CUDA_IMAGE ARG defaults to include the digest.

Run this in CI (before build) — not on every dev machine. The committed
Dockerfile falls back to tag-based FROM when no digest is pinned, which
keeps local dev friction low.

Usage:
    python scripts/pin_base_digests.py           # stamps all three Dockerfiles
    python scripts/pin_base_digests.py --dry-run # print what it would change
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent


# (Dockerfile path, ARG name, tag to resolve)
_TARGETS = [
    (_ROOT / "Dockerfile", "PYTHON_IMAGE", "python:3.11.9-slim-bookworm"),
    (_ROOT / "Dockerfile.local-llm", "PYTHON_IMAGE", "python:3.11.9-slim-bookworm"),
    (_ROOT / "Dockerfile.gpu", "CUDA_IMAGE", "nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04"),
]


def _resolve_digest(tag: str) -> str:
    """Ask the local Docker daemon for the current digest of ``tag``."""
    try:
        r = subprocess.run(
            ["docker", "buildx", "imagetools", "inspect", tag,
             "--format", "{{json .Manifest.Digest}}"],
            capture_output=True, text=True, timeout=60, check=False,
        )
    except FileNotFoundError:
        print("ERROR: docker CLI not found", file=sys.stderr)
        sys.exit(1)
    if r.returncode != 0:
        raise RuntimeError(
            f"Failed to resolve {tag!r}:\nSTDOUT:\n{r.stdout}\nSTDERR:\n{r.stderr}"
        )
    # Output is a JSON string like `"sha256:be9fb..."`; strip quotes.
    return r.stdout.strip().strip('"')


def _stamp(dockerfile: Path, arg_name: str, tag: str, digest: str, dry_run: bool) -> bool:
    """Rewrite `ARG <name>=<tag>` → `ARG <name>=<tag>@<digest>`. Returns
    True if the file was changed (or would be, in dry-run)."""
    src = dockerfile.read_text(encoding="utf-8")
    # Preserve the tag AND append the digest so humans can still read the
    # version; Docker's FROM pinning semantics honor the digest.
    new_value = f"{tag}@{digest}"
    pat = re.compile(rf"^(ARG\s+{re.escape(arg_name)}=)(.*)$", re.MULTILINE)

    def _repl(m: re.Match) -> str:
        existing = m.group(2).strip()
        if existing == new_value:
            return m.group(0)  # no-op
        return f"{m.group(1)}{new_value}"

    out, n = pat.subn(_repl, src)
    if n == 0:
        raise RuntimeError(f"{dockerfile}: no `ARG {arg_name}=` line found")
    if out == src:
        return False
    if dry_run:
        print(f"[dry-run] {dockerfile.relative_to(_ROOT)}: {arg_name} -> {new_value}")
    else:
        dockerfile.write_text(out, encoding="utf-8")
        print(f"updated {dockerfile.relative_to(_ROOT)}: {arg_name} -> {new_value}")
    return True


def main() -> int:
    p = argparse.ArgumentParser(description="Pin base-image digests in Dockerfiles.")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    any_changes = False
    for dockerfile, arg_name, tag in _TARGETS:
        if not dockerfile.exists():
            print(f"skipping {dockerfile}: not found")
            continue
        try:
            digest = _resolve_digest(tag)
        except RuntimeError as exc:
            print(f"WARN: {exc}", file=sys.stderr)
            continue
        changed = _stamp(dockerfile, arg_name, tag, digest, args.dry_run)
        any_changes = any_changes or changed

    if not any_changes:
        print("All digests already current.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
