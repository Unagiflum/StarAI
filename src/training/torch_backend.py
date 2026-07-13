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
DEVICE_AUTO = "auto"
DEVICE_CPU = "cpu"
DEVICE_GPU = "gpu"
DEVICE_CHOICES = (DEVICE_AUTO, DEVICE_CPU, DEVICE_GPU)


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


def cuda_available() -> bool:
    """Return whether the active PyTorch install can use CUDA."""
    return bool(_torch is not None and _torch.cuda.is_available())


def training_device(choice: str | None = DEVICE_AUTO):
    """Resolve a user-facing training device choice to a torch device."""
    if _torch is None:
        return None
    choice = (choice or DEVICE_AUTO).lower()
    if choice == DEVICE_AUTO:
        return preferred_device()
    if choice == DEVICE_CPU:
        return _torch.device("cpu")
    if choice == DEVICE_GPU:
        if not _torch.cuda.is_available():
            raise RuntimeError("GPU PyTorch is not available")
        return _torch.device("cuda")
    raise ValueError(f"Unsupported training device: {choice}")


def training_device_key(choice: str | None = DEVICE_AUTO) -> str:
    """Return a stable cache key for a resolved training device."""
    device = training_device(choice)
    if device is None:
        return DEVICE_CPU
    return DEVICE_GPU if str(device).startswith("cuda") else DEVICE_CPU


def training_device_selector_visible() -> bool:
    """Return whether CPU/GPU selection should be offered to the user."""
    return cuda_available()


def move_optimizer_state_to_device(optimizer, device) -> None:
    """Move loaded optimizer tensor state to the selected training device."""
    if _torch is None or optimizer is None or device is None:
        return
    for state in optimizer.state.values():
        for key, value in list(state.items()):
            if hasattr(value, "to"):
                state[key] = value.to(device)
