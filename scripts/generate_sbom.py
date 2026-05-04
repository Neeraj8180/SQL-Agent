"""Phase 8.8 — Generate a CycloneDX SBOM for a built sql-agent image.

Uses ``docker sbom`` (powered by syft). No external tools required beyond
Docker Desktop ≥ 4.7 or Docker Engine with the Buildx + SBOM plugins.

Usage:
    python scripts/generate_sbom.py                             # default image
    python scripts/generate_sbom.py --image sql-agent:v0.2.0   # specific tag
    python scripts/generate_sbom.py --format cyclonedx-json    # CycloneDX

Outputs the SBOM to ``artifacts/sbom/<image-tag>.<fmt>``. Commit the
artifacts directory per release, not per-dev-build.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent


def main() -> int:
    p = argparse.ArgumentParser(description="Emit an SBOM for a sql-agent image.")
    p.add_argument("--image", default="sql-agent:latest")
    p.add_argument(
        "--format",
        default="cyclonedx-json",
        choices=("cyclonedx-json", "spdx-json", "syft-json", "table"),
    )
    p.add_argument("--out-dir", default=str(_ROOT / "artifacts" / "sbom"))
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tag_safe = args.image.replace(":", "-").replace("/", "-")
    ext = {
        "cyclonedx-json": "cdx.json",
        "spdx-json": "spdx.json",
        "syft-json": "syft.json",
        "table": "txt",
    }[args.format]
    out_path = out_dir / f"{tag_safe}.{ext}"

    cmd = [
        "docker", "sbom",
        "--format", args.format,
        args.image,
    ]
    print(f"Running: {' '.join(cmd)}")
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stderr, file=sys.stderr)
        return r.returncode

    out_path.write_text(r.stdout, encoding="utf-8")
    print(f"SBOM written: {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
