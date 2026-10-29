"""Pure in-memory tests: every native operation uses a scripted fake DLL."""

from __future__ import annotations

import ctypes
import importlib
from collections.abc import Callable
from typing import cast

import pytest


class FakeFunction:
    def __init__(self, action: Callable[..., object], before: Callable[[], None]) -> None:
        self.action = action
        self.before = before
        self.argtypes: object = None
        self.restype: object = None
        self.calls: list[tuple[object, ...]] = []

    def __call__(self, *args: object) -> object:
        self.calls.append(args)
        self.before()
        return self.action(*args)


class FakeKernel:
    def __init__(self) -> None:
        self.events: list[str] = []
        self.error: object = 5
        self.output: int | None = 0x1234
        self.GetCurrentProcess = self.function("process", lambda: -1)
        self.GetCurrentThread = self.function("thread", lambda: -2)
        self.DuplicateHandle = self.function("duplicate", self.duplicate)
        self.CancelSynchronousIo = self.function("cancel", lambda handle: 1)
        self.WaitForSingleObject = self.function("wait", lambda handle, timeout: 0)
        self.CloseHandle = self.function("close", lambda handle: 1)

    def function(self, name: str, action: Callable[..., object]) -> FakeFunction:
        def recorded() -> None:
            self.events.append(name)

        return FakeFunction(action, recorded)

    def duplicate(self, *args: object) -> int:
        output = args[3]
        ctypes.cast(cast(ctypes.c_void_p, output), ctypes.POINTER(ctypes.c_void_p))[0] = self.output
        return 1

    def last_error(self) -> int:
        self.events.append("last_error")
        # Deliberately permits malformed scripted values for boundary tests.
        return cast(int, self.error)


class IntSubclass(int):
    pass


def test_windows_adapter_lazy_import_construction_and_abi(monkeypatch: pytest.MonkeyPatch) -> None:
    kernel = FakeKernel()
    loads: list[tuple[str, bool]] = []

    def loader(name: str, *, use_last_error: bool) -> object:
        loads.append((name, use_last_error))
        return kernel

    monkeypatch.setattr(ctypes, "WinDLL", loader, raising=False)
    monkeypatch.setattr(ctypes, "get_last_error", kernel.last_error, raising=False)
    windows = importlib.import_module("clipperx._windows_pipe_io")
    windows = importlib.reload(windows)
    from clipperx._windows_pipe_io import WindowsIoError

    adapter = windows.WindowsSyncIo()
    assert not loads and not kernel.events
    monkeypatch.setattr(windows, "_is_supported_host", lambda: True)
    assert adapter.duplicate_current_thread() == 0x1234
    assert loads == [("kernel32", True)]
    handle, dword, boolean = ctypes.c_void_p, ctypes.c_uint32, ctypes.c_int32
    table = {
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
    assert ctypes.sizeof(dword) == ctypes.sizeof(boolean) == 4
    assert ctypes.sizeof(handle) == ctypes.sizeof(ctypes.c_size_t)
    for name, (arguments, result) in table.items():
        function = getattr(kernel, name)
        assert function.argtypes == arguments
        assert function.restype is result
    assert adapter.duplicate_current_thread() == 0x1234
    assert len(loads) == 1
    injected = FakeKernel()
    explicit = windows.WindowsSyncIo(kernel32=injected, last_error=injected.last_error)
    assert injected.DuplicateHandle.argtypes is None
    assert explicit.duplicate_current_thread() == 0x1234
    assert len(loads) == 1
    monkeypatch.setattr(windows, "_is_supported_host", lambda: False)
    with pytest.raises(windows.WindowsIoError) as unsupported:
        windows.WindowsSyncIo().duplicate_current_thread()
    assert unsupported.value.operation == "load_api"
    assert len(loads) == 1
    error = WindowsIoError("cancel_sync", winerror=5, handle=0x1234)
    assert isinstance(error, ValueError)
    assert (error.operation, error.winerror, error.handle, error.ownership_uncertain) == (
        "cancel_sync",
        5,
        0x1234,
        False,
    )


@pytest.mark.parametrize(
    "case",
    [
        "ok",
        "wide",
        "unsigned_pseudo",
        "false",
        "null",
        "pseudo1",
        "pseudo2",
        "getter",
        "process_raises",
        "thread_raises",
        "duplicate_raises",
    ],
)
def test_windows_adapter_duplicates_current_thread_without_id_cast(case: str) -> None:
    from clipperx._windows_pipe_io import WindowsIoError, WindowsSyncIo

    kernel = FakeKernel()
    pointer_max = (1 << (8 * ctypes.sizeof(ctypes.c_void_p))) - 1
    if case == "wide":
        kernel.output = pointer_max - 2
    elif case == "unsigned_pseudo":
        kernel.GetCurrentProcess.action = lambda: pointer_max
        kernel.GetCurrentThread.action = lambda: pointer_max - 1
    elif case == "null":
        kernel.output = None
    elif case == "pseudo1":
        kernel.output = pointer_max
    elif case == "pseudo2":
        kernel.output = pointer_max - 1
    elif case in {"false", "getter"}:
        kernel.DuplicateHandle.action = lambda *args: 0
    elif case.endswith("raises"):

        def fail(*args: object) -> object:
            raise OSError("private detail must not escape")

        getattr(
            kernel,
            {
                "process_raises": "GetCurrentProcess",
                "thread_raises": "GetCurrentThread",
                "duplicate_raises": "DuplicateHandle",
            }[case],
        ).action = fail
    getter = kernel.last_error
    if case == "getter":

        def getter() -> int:
            kernel.events.append("last_error")
            raise ValueError("private getter detail")

    adapter = WindowsSyncIo(kernel32=kernel, last_error=getter)
    if case in {"ok", "wide", "unsigned_pseudo"}:
        assert adapter.duplicate_current_thread() == kernel.output
        assert kernel.events == ["process", "thread", "duplicate"]
    else:
        with pytest.raises(WindowsIoError) as caught:
            adapter.duplicate_current_thread()
        assert caught.value.operation == "duplicate_current_thread"
        assert caught.value.handle is None
        assert "private" not in str(caught.value)
        if case in {"false", "getter"}:
            assert kernel.events == ["process", "thread", "duplicate", "last_error"]
            assert caught.value.winerror == (5 if case == "false" else None)
    if kernel.DuplicateHandle.calls:
        args = kernel.DuplicateHandle.calls[0]
        assert args[:3] == (
            (pointer_max, pointer_max - 1, pointer_max)
            if case == "unsigned_pseudo"
            else (-1, -2, -1)
        )
        assert args[4:] == (0x00100001, 0, 0)
        assert args[3] is not None
    assert not kernel.CloseHandle.calls


@pytest.mark.parametrize(
    "result,code,expected",
    [
        (1, 5, "requested"),
        (-1, 1168, "requested"),
        (0, 1168, "no_pending"),
        (0, 5, None),
        (0, 6, None),
        (0, 0, None),
        (0, True, None),
        (0, -1, None),
        (0, 0x100000000, None),
        (0, "1168", None),
        (0, None, None),
        ("raises", 5, None),
    ],
)
def test_windows_adapter_classifies_cancel_without_claiming_completion(
    result: object,
    code: object,
    expected: str | None,
) -> None:
    from clipperx._windows_pipe_io import WindowsIoError, WindowsSyncIo

    kernel = FakeKernel()
    kernel.error = code

    def cancel(handle: object) -> object:
        if result == "raises":
            raise OSError("hidden details")
        return result

    kernel.CancelSynchronousIo.action = cancel

    def extra_call() -> None:
        kernel.error = 9999  # Any intervening native operation clobbers the saved error.
        kernel.events.append("extra_native")

    for extra in (
        kernel.GetCurrentProcess,
        kernel.GetCurrentThread,
        kernel.DuplicateHandle,
        kernel.WaitForSingleObject,
        kernel.CloseHandle,
    ):
        extra.before = extra_call

    def getter() -> int:
        assert kernel.events == ["cancel"]
        return kernel.last_error()

    adapter = WindowsSyncIo(kernel32=kernel, last_error=getter)
    if expected is not None:
        assert adapter.request_cancel(0x1234) == expected
    else:
        with pytest.raises(WindowsIoError) as caught:
            adapter.request_cancel(0x1234)
        assert caught.value.operation == "cancel_sync"
        assert caught.value.handle == 0x1234
        assert not caught.value.ownership_uncertain
        assert caught.value.winerror == (
            code if type(code) is int and 0 <= code <= 0xFFFFFFFF and result != "raises" else None
        )
    assert kernel.CancelSynchronousIo.calls == [(0x1234,)]
    assert kernel.events == (["cancel", "last_error"] if result == 0 else ["cancel"])
    assert not kernel.WaitForSingleObject.calls and not kernel.CloseHandle.calls


@pytest.mark.parametrize(
    "operation,result,code,expected",
    [
        ("wait", 0, 5, True),
        ("wait", 258, 5, False),
        ("wait", 0xFFFFFFFF, 6, None),
        ("wait", 1, 5, None),
        ("wait", 128, 5, None),
        ("wait", -1, 5, None),
        ("wait", True, 5, None),
        ("wait", "raises", 5, None),
        ("wait", "argument_error", 5, None),
        ("wait", 0xFFFFFFFF, True, None),
        ("close", 1, 5, "ok"),
        ("close", 0, 6, None),
        ("close", "raises", 5, None),
        ("close", "argument_error", 5, None),
        ("close", 0, True, None),
        ("close", 0, "getter_raises", None),
        ("close", 0, "getter_argument_error", None),
    ],
)
def test_windows_adapter_wait_zero_and_checked_close(
    operation: str,
    result: object,
    code: object,
    expected: object,
) -> None:
    from clipperx._windows_pipe_io import WindowsIoError, WindowsSyncIo

    kernel = FakeKernel()
    kernel.error = code

    def action(*args: object) -> object:
        if result == "raises":
            raise OSError("hidden native detail")
        if result == "argument_error":
            raise ctypes.ArgumentError("hidden synthetic native detail")
        return result

    function = kernel.WaitForSingleObject if operation == "wait" else kernel.CloseHandle
    function.action = action

    def getter() -> int:
        assert kernel.events == [operation]
        if code == "getter_raises":
            raise ValueError("hidden error detail")
        if code == "getter_argument_error":
            raise ctypes.ArgumentError("hidden synthetic getter detail")
        return kernel.last_error()

    adapter = WindowsSyncIo(kernel32=kernel, last_error=getter)
    if expected is not None:
        if operation == "wait":
            assert adapter.is_signaled(0x1234) is expected
        else:
            adapter.close_handle(0x1234)
    else:
        with pytest.raises(WindowsIoError) as caught:
            if operation == "wait":
                adapter.is_signaled(0x1234)
            else:
                adapter.close_handle(0x1234)
        assert caught.value.operation == ("wait_zero" if operation == "wait" else "close_handle")
        assert caught.value.handle == 0x1234
        assert caught.value.ownership_uncertain is (operation == "close")
        if type(code) is int and result in (0, 0xFFFFFFFF):
            assert caught.value.winerror == code
        else:
            assert caught.value.winerror is None
        assert "hidden" not in str(caught.value)
        assert str(caught.value) == "Windows synchronous I/O operation failed"
        if result == "argument_error" or code == "getter_argument_error":
            assert caught.value.__suppress_context__ is True
    assert function.calls == ([(0x1234, 0)] if operation == "wait" else [(0x1234,)])
    assert not kernel.CancelSynchronousIo.calls
    assert not (
        kernel.CloseHandle.calls if operation == "wait" else kernel.WaitForSingleObject.calls
    )
    if result in (0, 0xFFFFFFFF) and expected is None:
        assert kernel.events[:1] == [operation]
        if code not in {"getter_raises", "getter_argument_error"}:
            assert kernel.events == [operation, "last_error"]
        else:
            assert kernel.events == [operation]
    else:
        assert kernel.events == [operation]


_POINTER_MAX = (1 << (8 * ctypes.sizeof(ctypes.c_void_p))) - 1


@pytest.mark.parametrize(
    "case,value",
    [
        ("handle", True),
        ("handle", False),
        ("handle", 0),
        ("handle", -1),
        ("handle", -2),
        ("handle", _POINTER_MAX),
        ("handle", _POINTER_MAX - 1),
        ("handle", _POINTER_MAX + 1),
        ("handle", 1 << 512),
        ("handle", 1.5),
        ("handle", "1234"),
        ("handle", None),
        ("kernel_only", None),
        ("handle", IntSubclass(0x1234)),
        ("handle", ctypes.c_void_p(0x1234)),
        ("getter_only", None),
        ("bad_getter", 5),
        ("bad_getter", False),
        ("missing_function", None),
        ("noncallable_function", None),
        ("load_raises", None),
        ("configuration_raises", None),
    ],
    ids=lambda value: (
        str(value) if type(value) in (str, int, float, bool, type(None)) else "object"
    ),
)
def test_windows_adapter_rejects_handles_before_native_work(
    monkeypatch: pytest.MonkeyPatch,
    case: str,
    value: object,
) -> None:
    from clipperx import _windows_pipe_io as windows

    kernel = FakeKernel()
    loads: list[str] = []

    def loader(name: str, *, use_last_error: bool) -> object:
        loads.append(name)
        if case == "load_raises":
            raise OSError("private loader failure")
        return kernel

    monkeypatch.setattr(ctypes, "WinDLL", loader, raising=False)
    monkeypatch.setattr(ctypes, "get_last_error", kernel.last_error, raising=False)
    monkeypatch.setattr(windows, "_is_supported_host", lambda: True)
    if case == "handle":
        # Both the lazy default and explicit fake must reject BEFORE loading/configuring.
        for adapter in (
            windows.WindowsSyncIo(),
            windows.WindowsSyncIo(kernel32=kernel, last_error=kernel.last_error),
        ):
            for method in (adapter.request_cancel, adapter.is_signaled, adapter.close_handle):
                with pytest.raises(windows.WindowsIoError) as caught:
                    method(cast(int, value))
                assert caught.value.operation == "validate_handle"
                assert caught.value.handle is None
        assert not loads and not kernel.events
        assert kernel.CloseHandle.argtypes is None
    elif case in {"kernel_only", "getter_only", "bad_getter"}:
        with pytest.raises(windows.WindowsIoError) as caught:
            if case == "kernel_only":
                windows.WindowsSyncIo(kernel32=kernel)
            elif case == "getter_only":
                windows.WindowsSyncIo(last_error=kernel.last_error)
            else:
                windows.WindowsSyncIo(kernel32=kernel, last_error=cast(Callable[[], int], value))
        assert caught.value.operation == "configure_api"
        assert not loads and not kernel.events
    else:
        candidate: object = kernel
        if case == "missing_function":
            candidate = object()
        elif case == "noncallable_function":
            kernel.CloseHandle = cast(FakeFunction, object())
        elif case == "configuration_raises":

            class FrozenFunction:
                __slots__ = ()

                def __call__(self, *args: object) -> int:
                    raise AssertionError("must never call malformed function")

            kernel.CloseHandle = cast(FakeFunction, FrozenFunction())
        adapter = (
            windows.WindowsSyncIo()
            if case == "load_raises"
            else windows.WindowsSyncIo(kernel32=candidate, last_error=kernel.last_error)
        )
        with pytest.raises(windows.WindowsIoError) as caught:
            adapter.close_handle(0x1234)
        assert caught.value.operation == ("load_api" if case == "load_raises" else "configure_api")
        assert not kernel.events


class FakeHandle(int):
    closed: object
    detach_calls: int
    mode: str

    def __new__(cls, value: int) -> FakeHandle:
        wrapper = super().__new__(cls, value)
        wrapper.closed = False
        wrapper.detach_calls = 0
        wrapper.mode = "ok"
        return wrapper

    def Detach(self) -> object:
        self.detach_calls += 1
        if self.mode == "raises":
            raise OSError("hidden transfer detail")
        if self.mode == "argument_error":
            raise ctypes.ArgumentError("hidden synthetic transfer detail")
        if self.mode != "not_marked":
            self.closed = True
        if self.mode == "mismatch":
            return int(self) + 1
        if self.mode == "bool_result":
            return True
        if self.mode == "subclass_result":
            return FakeHandle(int(self))
        return int(self)


class FakeProcess:
    def __init__(self, wrapper: FakeHandle) -> None:
        self.returncode: object = 0
        self._handle: object = wrapper

    def forbidden(self) -> None:
        raise AssertionError("adapter must not operate on a process")

    poll = wait = terminate = kill = forbidden


@pytest.mark.parametrize(
    "case",
    [
        "ok",
        "max_returncode",
        "unreaped",
        "bool_returncode",
        "negative_returncode",
        "overflow_returncode",
        "subclass_returncode",
        "missing_returncode",
        "missing_handle",
        "nonint_wrapper",
        "bool_wrapper",
        "plain_int_wrapper",
        "closed",
        "closed_zero",
        "missing_closed",
        "noncallable_detach",
        "missing_detach",
        "null",
        "pseudo",
        "overflow_pointer",
        "live",
        "wait_failed",
        "wait_raises",
        "raises",
        "argument_error",
        "mismatch",
        "bool_result",
        "subclass_result",
        "not_marked",
    ],
)
def test_windows_adapter_detaches_only_signaled_reaped_popen(case: str) -> None:
    from clipperx._windows_pipe_io import WindowsIoError, WindowsSyncIo

    kernel = FakeKernel()
    wrapper = FakeHandle(
        0
        if case == "null"
        else -1
        if case == "pseudo"
        else _POINTER_MAX + 1
        if case == "overflow_pointer"
        else 0x1234
    )
    process = FakeProcess(wrapper)
    malformed = {
        "unreaped": None,
        "bool_returncode": True,
        "negative_returncode": -1,
        "overflow_returncode": 0x100000000,
        "subclass_returncode": FakeHandle(0),
        "max_returncode": 0xFFFFFFFF,
    }
    if case in malformed:
        process.returncode = malformed[case]
    if case == "missing_returncode":
        del process.returncode
    elif case == "missing_handle":
        del process._handle
    elif case == "nonint_wrapper":
        process._handle = object()
    elif case == "bool_wrapper":
        process._handle = True
    elif case == "plain_int_wrapper":
        process._handle = 0x1234
    elif case == "closed":
        wrapper.closed = True
    elif case == "closed_zero":
        wrapper.closed = 0
    elif case == "missing_closed":
        del wrapper.closed
    elif case == "noncallable_detach":
        setattr(wrapper, "Detach", None)
    elif case == "missing_detach":

        class NoDetach(int):
            closed = False

        process._handle = NoDetach(0x1234)
    elif case in {"live", "wait_failed"}:
        kernel.WaitForSingleObject.action = lambda *args: 258 if case == "live" else 0xFFFFFFFF
    elif case == "wait_raises":

        def fail(*args: object) -> int:
            raise OSError("hidden wait failure")

        kernel.WaitForSingleObject.action = fail
    elif case in {
        "raises",
        "argument_error",
        "mismatch",
        "bool_result",
        "subclass_result",
        "not_marked",
    }:
        wrapper.mode = case
    adapter = WindowsSyncIo(kernel32=kernel, last_error=kernel.last_error)
    if case in {"ok", "max_returncode"}:
        transferred = adapter.detach_reaped_process(process)
        assert type(transferred) is int and transferred == 0x1234
        assert wrapper.closed is True and wrapper.detach_calls == 1
        assert kernel.events == ["wait"]
        assert kernel.WaitForSingleObject.calls == [(0x1234, 0)]
        assert not kernel.CloseHandle.calls
        with pytest.raises(WindowsIoError) as repeated:
            adapter.detach_reaped_process(process)
        assert repeated.value.operation == "detach_popen"
        assert wrapper.detach_calls == 1 and kernel.events == ["wait"]
        adapter.close_handle(transferred)  # Caller explicitly closes only AFTER transfer.
        assert kernel.CloseHandle.calls == [(0x1234,)]
    else:
        with pytest.raises(WindowsIoError) as caught:
            adapter.detach_reaped_process(process)
        transfer_attempted = case in {
            "raises",
            "argument_error",
            "mismatch",
            "bool_result",
            "subclass_result",
            "not_marked",
        }
        assert caught.value.operation == (
            "wait_zero" if case in {"wait_failed", "wait_raises"} else "detach_popen"
        )
        assert caught.value.ownership_uncertain is transfer_attempted
        if transfer_attempted:
            assert caught.value.handle == 0x1234
        assert wrapper.detach_calls == (1 if transfer_attempted else 0)
        assert not kernel.CloseHandle.calls and not kernel.CancelSynchronousIo.calls
        if case not in {"live", "wait_failed", "wait_raises"} and not transfer_attempted:
            assert not kernel.events and kernel.WaitForSingleObject.argtypes is None
        assert "hidden" not in str(caught.value)
        if case == "argument_error":
            assert str(caught.value) == "Windows synchronous I/O operation failed"
            assert caught.value.__suppress_context__ is True
