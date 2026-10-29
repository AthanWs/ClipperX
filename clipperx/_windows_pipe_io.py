"""Lazy Win32 synchronous-I/O calls; returned handles belong to the caller.

The host must serialize handle use and retain uncertain resources. This module
owns no process, thread or pipe and supplies no completion/cleanup guarantee.
"""

from __future__ import annotations

import ctypes
import platform
import sys
from collections.abc import Callable
from typing import Literal, Protocol, cast

CancelDisposition = Literal["requested", "no_pending"]
_POINTER_MAX = (1 << (8 * ctypes.sizeof(ctypes.c_void_p))) - 1
_DWORD_MAX = 0xFFFFFFFF
_FFI_FAILURES = (
    OSError,
    ValueError,
    TypeError,
    OverflowError,
    RuntimeError,
    AttributeError,
    ctypes.ArgumentError,
)


class WindowsIoError(ValueError):
    def __init__(
        self,
        operation: str,
        *,
        winerror: int | None = None,
        handle: int | None = None,
        ownership_uncertain: bool = False,
    ) -> None:
        self.operation = operation
        self.winerror = winerror
        self.handle = handle
        self.ownership_uncertain = ownership_uncertain
        super().__init__("Windows synchronous I/O operation failed")


class _NativeFunction(Protocol):
    argtypes: list[object]
    restype: object

    def __call__(self, *args: object) -> object: ...


def _is_supported_host() -> bool:
    return sys.platform == "win32" and platform.python_implementation() == "CPython"


def _real_handle(value: object, operation: str = "validate_handle") -> int:
    if type(value) is not int or not 0 < value < _POINTER_MAX - 1:
        raise WindowsIoError(operation)
    return value


class WindowsSyncIo:
    def __init__(
        self, *, kernel32: object | None = None, last_error: Callable[[], int] | None = None
    ) -> None:
        if (kernel32 is None) != (last_error is None) or (
            last_error is not None and not callable(last_error)
        ):
            raise WindowsIoError("configure_api")
        self._kernel32 = kernel32
        self._last_error = last_error
        self._functions: dict[str, _NativeFunction] | None = None

    def _error_code(
        self, operation: str, handle: int | None = None, *, uncertain: bool = False
    ) -> int:
        getter = self._last_error
        if getter is None:
            raise WindowsIoError(operation, handle=handle, ownership_uncertain=uncertain)
        try:
            code = getter()
        except _FFI_FAILURES:
            raise WindowsIoError(operation, handle=handle, ownership_uncertain=uncertain) from None
        if type(code) is not int or not 0 <= code <= _DWORD_MAX:
            raise WindowsIoError(operation, handle=handle, ownership_uncertain=uncertain)
        return code

    def _call(
        self,
        function: _NativeFunction,
        operation: str,
        *args: object,
        handle: int | None = None,
        uncertain: bool = False,
    ) -> object:
        try:
            return function(*args)
        except _FFI_FAILURES:
            raise WindowsIoError(operation, handle=handle, ownership_uncertain=uncertain) from None

    def _api(self) -> dict[str, _NativeFunction]:
        if self._functions is not None:
            return self._functions
        kernel = self._kernel32
        if kernel is None:
            if not _is_supported_host():
                raise WindowsIoError("load_api")
            try:
                kernel = ctypes.WinDLL("kernel32", use_last_error=True)
                self._last_error = ctypes.get_last_error
            except _FFI_FAILURES:
                raise WindowsIoError("load_api") from None
        handle, dword, boolean = ctypes.c_void_p, ctypes.c_uint32, ctypes.c_int32
        bindings: dict[str, tuple[list[object], object]] = {
            "GetCurrentProcess": ([], handle),
            "GetCurrentThread": ([], handle),
            "DuplicateHandle": (
                [handle, handle, handle, ctypes.POINTER(handle), dword, boolean, dword],
                boolean,
            ),
            "CancelSynchronousIo": ([handle], boolean),
            "WaitForSingleObject": ([handle, dword], dword),
            "CloseHandle": ([handle], boolean),
        }
        functions: dict[str, _NativeFunction] = {}
        for name, (arguments, result) in bindings.items():
            try:
                function = cast(_NativeFunction, getattr(kernel, name))
                if not callable(function):
                    raise WindowsIoError("configure_api")
                function.argtypes = arguments
                function.restype = result
            except _FFI_FAILURES:
                raise WindowsIoError("configure_api") from None
            functions[name] = function
        self._functions = functions
        return functions

    def duplicate_current_thread(self) -> int:
        api = self._api()
        operation = "duplicate_current_thread"
        process = self._call(api["GetCurrentProcess"], operation)
        thread = self._call(api["GetCurrentThread"], operation)
        output = ctypes.c_void_p()
        result = self._call(
            api["DuplicateHandle"],
            operation,
            process,
            thread,
            process,
            ctypes.byref(output),
            0x00100001,
            0,
            0,
        )
        if not result:
            code = self._error_code(operation)
            raise WindowsIoError(operation, winerror=code)
        return _real_handle(output.value, operation)

    def request_cancel(self, thread_handle: int) -> CancelDisposition:
        handle = _real_handle(thread_handle)
        result = self._call(
            self._api()["CancelSynchronousIo"], "cancel_sync", handle, handle=handle
        )
        if result:
            return "requested"
        code = self._error_code("cancel_sync", handle)
        if code == 1168:
            return "no_pending"
        raise WindowsIoError("cancel_sync", winerror=code, handle=handle)

    def is_signaled(self, handle: int) -> bool:
        raw = _real_handle(handle)
        status = self._call(self._api()["WaitForSingleObject"], "wait_zero", raw, 0, handle=raw)
        if type(status) is int:
            if status == 0:
                return True
            if status == 258:
                return False
            if status == _DWORD_MAX:
                code = self._error_code("wait_zero", raw)
                raise WindowsIoError("wait_zero", winerror=code, handle=raw)
        raise WindowsIoError("wait_zero", handle=raw)

    def close_handle(self, handle: int) -> None:
        raw = _real_handle(handle)
        result = self._call(
            self._api()["CloseHandle"], "close_handle", raw, handle=raw, uncertain=True
        )
        if not result:
            code = self._error_code("close_handle", raw, uncertain=True)
            raise WindowsIoError(
                "close_handle", winerror=code, handle=raw, ownership_uncertain=True
            )

    def detach_reaped_process(self, process: object) -> int:
        try:
            returncode: object = getattr(process, "returncode")
            wrapper: object = getattr(process, "_handle")
            closed: object = getattr(wrapper, "closed")
            detach: object = getattr(wrapper, "Detach")
        except _FFI_FAILURES:
            raise WindowsIoError("detach_popen") from None
        if (
            type(returncode) is not int
            or not 0 <= returncode <= _DWORD_MAX
            or not isinstance(wrapper, int)
            or isinstance(wrapper, bool)
            or closed is not False
            or not callable(detach)
        ):
            raise WindowsIoError("detach_popen")
        raw = _real_handle(int.__int__(wrapper), "detach_popen")
        if not self.is_signaled(raw):
            raise WindowsIoError("detach_popen", handle=raw)
        try:
            transferred = cast(Callable[[], object], detach)()
            closed_after: object = getattr(wrapper, "closed")
        except _FFI_FAILURES:
            raise WindowsIoError("detach_popen", handle=raw, ownership_uncertain=True) from None
        if type(transferred) is not int or transferred != raw or closed_after is not True:
            raise WindowsIoError("detach_popen", handle=raw, ownership_uncertain=True)
        return transferred
