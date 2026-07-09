"""Guarded PyTorch access for optional training builds.

Future training code should import PyTorch through this module. The lightweight
StarAI build intentionally excludes torch, so direct ``import torch`` statements
elsewhere can break startup.
"""

from __future__ import annotations


try:
    import torch as _torch
except (ImportError, OSError):
    _torch = None


TORCH_AVAILABLE = _torch is not None


class TorchUnavailableError(RuntimeError):
    """Raised when training functionality is used without PyTorch installed."""


def get_torch():
    """Return the imported torch module, or None in lightweight builds."""
    return _torch


def require_torch():
    """Return torch or raise a clear error for optional-training boundaries."""
    if _torch is None:
        raise TorchUnavailableError("PyTorch is required for AI training")
    return _torch


def preferred_device():
    """Return the best PyTorch device, preferring CUDA when available."""
    if _torch is None:
        return None
    return _torch.device("cuda" if _torch.cuda.is_available() else "cpu")
