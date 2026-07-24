"""Windows performance policy for active training work.

Scheduling priority and power-throttling QoS are independent Windows policies.
Training keeps its existing below-normal scheduling priority where configured,
but explicitly disables execution-speed throttling so an occluded StarAI window
does not move active training to Low/Eco QoS.
"""

from __future__ import annotations

from contextlib import contextmanager
import ctypes
import sys
import threading


_PROCESS_POWER_THROTTLING = 4
_THREAD_POWER_THROTTLING = 3
_POWER_THROTTLING_CURRENT_VERSION = 1
_POWER_THROTTLING_EXECUTION_SPEED = 0x1


class _PowerThrottlingState(ctypes.Structure):
    _fields_ = (
        ("Version", ctypes.c_uint32),
        ("ControlMask", ctypes.c_uint32),
        ("StateMask", ctypes.c_uint32),
    )


_process_qos_lock = threading.Lock()
_process_qos_users = 0
_thread_qos_state = threading.local()


def _power_throttling_state(*, controlled: bool) -> _PowerThrottlingState:
    return _PowerThrottlingState(
        _POWER_THROTTLING_CURRENT_VERSION,
        _POWER_THROTTLING_EXECUTION_SPEED if controlled else 0,
        0,
    )


def request_current_process_high_qos() -> bool:
    """Disable execution-speed throttling for the current Windows process."""

    if sys.platform != "win32":
        return False
    try:
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        get_current_process = kernel32.GetCurrentProcess
        get_current_process.restype = ctypes.c_void_p
        set_process_information = kernel32.SetProcessInformation
        set_process_information.argtypes = (
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_void_p,
            ctypes.c_uint32,
        )
        set_process_information.restype = ctypes.c_int
        state = _power_throttling_state(controlled=True)
        return bool(
            set_process_information(
                get_current_process(),
                _PROCESS_POWER_THROTTLING,
                ctypes.byref(state),
                ctypes.sizeof(state),
            )
        )
    except (AttributeError, OSError):
        # Older or restricted Windows environments may not expose the policy.
        return False


def request_current_thread_high_qos() -> bool:
    """Disable execution-speed throttling for the current Windows thread."""

    if sys.platform != "win32":
        return False
    try:
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        get_current_thread = kernel32.GetCurrentThread
        get_current_thread.restype = ctypes.c_void_p
        set_thread_information = kernel32.SetThreadInformation
        set_thread_information.argtypes = (
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_void_p,
            ctypes.c_uint32,
        )
        set_thread_information.restype = ctypes.c_int
        state = _power_throttling_state(controlled=True)
        return bool(
            set_thread_information(
                get_current_thread(),
                _THREAD_POWER_THROTTLING,
                ctypes.byref(state),
                ctypes.sizeof(state),
            )
        )
    except (AttributeError, OSError):
        return False


def _restore_current_thread_qos() -> bool:
    """Return the current thread to Windows-managed power throttling."""

    if sys.platform != "win32":
        return False
    try:
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        get_current_thread = kernel32.GetCurrentThread
        get_current_thread.restype = ctypes.c_void_p
        set_thread_information = kernel32.SetThreadInformation
        set_thread_information.argtypes = (
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_void_p,
            ctypes.c_uint32,
        )
        set_thread_information.restype = ctypes.c_int
        state = _power_throttling_state(controlled=False)
        return bool(
            set_thread_information(
                get_current_thread(),
                _THREAD_POWER_THROTTLING,
                ctypes.byref(state),
                ctypes.sizeof(state),
            )
        )
    except (AttributeError, OSError):
        return False


def _restore_current_process_qos() -> bool:
    """Return the current process to Windows-managed power throttling."""

    if sys.platform != "win32":
        return False
    try:
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        get_current_process = kernel32.GetCurrentProcess
        get_current_process.restype = ctypes.c_void_p
        set_process_information = kernel32.SetProcessInformation
        set_process_information.argtypes = (
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_void_p,
            ctypes.c_uint32,
        )
        set_process_information.restype = ctypes.c_int
        state = _power_throttling_state(controlled=False)
        return bool(
            set_process_information(
                get_current_process(),
                _PROCESS_POWER_THROTTLING,
                ctypes.byref(state),
                ctypes.sizeof(state),
            )
        )
    except (AttributeError, OSError):
        return False


@contextmanager
def training_high_qos():
    """Keep the current process and training thread at High QoS for a run.

    Multiple in-process training sessions share one process-level request. The
    last session to finish restores Windows-managed process QoS. Nested callers
    on one thread likewise share a thread-level request and restore it on exit.
    """

    global _process_qos_users

    process_claimed = False
    with _process_qos_lock:
        if _process_qos_users > 0:
            _process_qos_users += 1
            process_claimed = True
        elif request_current_process_high_qos():
            _process_qos_users = 1
            process_claimed = True

    thread_claimed = False
    thread_qos_users = int(getattr(_thread_qos_state, "users", 0))
    if thread_qos_users > 0:
        _thread_qos_state.users = thread_qos_users + 1
        thread_claimed = True
    elif request_current_thread_high_qos():
        _thread_qos_state.users = 1
        thread_claimed = True
    try:
        yield
    finally:
        if thread_claimed:
            thread_qos_users = int(getattr(_thread_qos_state, "users", 1)) - 1
            _thread_qos_state.users = thread_qos_users
            if thread_qos_users == 0:
                _restore_current_thread_qos()
        if process_claimed:
            restore = False
            with _process_qos_lock:
                _process_qos_users -= 1
                if _process_qos_users == 0:
                    restore = True
            if restore:
                _restore_current_process_qos()
