"""Runtime device detection with explicit force-overrides.

Priority order:
    1. settings.force_cpu  -> "cpu"
    2. settings.force_gpu  -> "cuda" (errors if no CUDA device)
    3. auto:  "cuda" > "mps" > "cpu"

Torch is imported lazily so this module is importable on systems without
torch installed (they'll simply never see a non-"cpu" answer, which is the
correct behavior).
"""

from __future__ import annotations

from typing import Optional

from sql_agent.config import get_logger, settings


_log = get_logger("llm_serving.hardware")


# ---------------------------------------------------------------------------
# Lazy torch probes
# ---------------------------------------------------------------------------


def _torch_module():
    try:
        import torch  # noqa: F401

        return torch
    except Exception:
        return None


def _cuda_available(torch_mod) -> bool:
    if torch_mod is None:
        return False
    try:
        return bool(torch_mod.cuda.is_available())
    except Exception:
        return False


def _mps_available(torch_mod) -> bool:
    if torch_mod is None:
        return False
    try:
        backends = getattr(torch_mod, "backends", None)
        mps = getattr(backends, "mps", None) if backends is not None else None
        return bool(mps and mps.is_available())
    except Exception:
        return False


def _cuda_name(torch_mod) -> str:
    if torch_mod is None:
        return "unknown"
    try:
        return torch_mod.cuda.get_device_name(0)
    except Exception:
        return "unknown"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def detect_device(
    *,
    force_cpu: Optional[bool] = None,
    force_gpu: Optional[bool] = None,
) -> str:
    """Return ``"cuda"``, ``"mps"``, or ``"cpu"`` per the resolution rules.

    Parameters let callers override settings (useful for tests).
    """
    fc = settings.force_cpu if force_cpu is None else force_cpu
    fg = settings.force_gpu if force_gpu is None else force_gpu

    if fc and fg:
        raise RuntimeError(
            "FORCE_CPU and FORCE_GPU cannot both be true. Set at most one."
        )
    if fc:
        _log.info("device: 'cpu' (FORCE_CPU=true)")
        return "cpu"

    torch_mod = _torch_module()
    cuda_ok = _cuda_available(torch_mod)

    if fg:
        if not cuda_ok:
            raise RuntimeError(
                "FORCE_GPU=true but no CUDA device is available. "
                "Install CUDA-enabled torch or unset FORCE_GPU."
            )
        return "cuda"

    if cuda_ok:
        return "cuda"
    if _mps_available(torch_mod):
        return "mps"
    return "cpu"


def log_execution_mode(device: str, model_id: str) -> None:
    """One-line startup banner per the phase-2 spec."""
    torch_mod = _torch_module()
    if device == "cuda":
        _log.info(
            "Running on GPU (%s): model=%s", _cuda_name(torch_mod), model_id
        )
    elif device == "mps":
        _log.info("Running on Apple Silicon (MPS): model=%s", model_id)
    else:
        _log.info("Running on CPU fallback: model=%s", model_id)


def torch_available() -> bool:
    """Utility: is torch importable at all?"""
    return _torch_module() is not None
