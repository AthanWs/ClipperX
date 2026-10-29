# Windows synchronous-I/O adapter — Task6b1 /41

Status: ADOPTED internal technical prerequisite after pure binding review40
Approved. Human-authorized automatic vertical-short/editor vision, original
fonts, authored ranges/effects and final acceptance requirements are unchanged.
This is a TWO-new-path native-call adapter, NOT the full supervisor/executor.
Its implementation gate uses injected fake native functions only. No Windows
API/process/thread/pipe/media qualification is authorized by this document.

## Choice and independent deliverable

Use the smallest synchronous anonymous-pipe candidate from the FULL-read
phase3-windows-pipe-io-proposal.md. Overlapped named pipes would replace process
plumbing; a helper process changes containment/budgets. Neither is a fallback.
First isolate the exact Win32 ABI/error/handle-transfer boundary. A later owner
will implement independent source/relay/log pumps, repeated sticky cancellation,
one failure grace, launch handoff and resource accounting using this adapter.

The adapter owns no worker, process, pipe or global resource registry. Returned
real handles are immediately the caller's owned resources. It cannot authenticate
a raw integer, enforce caller linearity, prove a join or report own0. The later
supervisor must serialize cancellation/wait/close against handle use and retain
uncertain resources. Thin native mapping is independently testable and may be
rejected without changing compiler or source observation contracts.

## Exact paths and API

Create only clipperx/_windows_pipe_io.py and
clipperx/tests/test_windows_pipe_io.py. No imports from compiler, decoder,
settings/providers/models, legacy editing or render-runtime qualification.
Python >=3.10, standard library only. Native default supports Windows CPython;
other platforms fail explicitly, with no alternate mechanism. Injected fakes
may run portably. No actual DLL load/call, child, thread, timer or I/O on import
or default construction. No destructor/finalizer performs native work.

```python
CancelDisposition = Literal['requested', 'no_pending']

class WindowsIoError(ValueError):
    def __init__(self, operation: str, *, winerror: int | None = None,
                 handle: int | None = None,
                 ownership_uncertain: bool = False) -> None: ...
    operation: str
    winerror: int | None
    handle: int | None
    ownership_uncertain: bool

class WindowsSyncIo:
    def __init__(self, *, kernel32: object | None = None,
                 last_error: Callable[[], int] | None = None) -> None: ...
    def duplicate_current_thread(self) -> int: ...
    def request_cancel(self, thread_handle: int) -> CancelDisposition: ...
    def is_signaled(self, handle: int) -> bool: ...
    def close_handle(self, handle: int) -> None: ...
    def detach_reaped_process(self, process: object) -> int: ...
```

kernel32/last_error must be supplied together or both omitted; the supplied error
getter must be callable. This is private dependency injection, not a runtime
fallback/selector. An injected DLL models EXACTLY the same function ABI/branches
as the lazy default. Configure it lazily on first use. No separate fake engine.
The adapter is not a shared-state concurrency owner: the HOST serializes use
per instance or gives registration workers separate instances. It must serialize
cancel/wait/close of each real handle regardless of which adapter issued it.

Errors carry bounded operation names, optional DWORD error, optional known raw
handle and explicit close/transfer uncertainty. Never include process args/env,
credentials or arbitrary exception text. Native false/failure and unsupported
shape/platform must not become a successful cancellation/closure/transfer.
Operation tokens are load_api,configure_api,validate_handle,
duplicate_current_thread,cancel_sync,wait_zero,close_handle,detach_popen.

## Native ABI and operations

Use ctypes.WinDLL('kernel32', use_last_error=True) only at first explicit native
operation. WinDLL uses the Windows convention; no PyDLL or raw C-default handles.
HANDLE=c_void_p, BOOL=c_int32, DWORD=c_uint32. Explicit argtypes/restype:

| Function | argtypes | restype |
| --- | --- | --- |
| GetCurrentProcess | [] | HANDLE |
| GetCurrentThread | [] | HANDLE |
| DuplicateHandle | [HANDLE,HANDLE,HANDLE,POINTER(HANDLE),DWORD,BOOL,DWORD] | BOOL |
| CancelSynchronousIo | [HANDLE] | BOOL |
| WaitForSingleObject | [HANDLE,DWORD] | DWORD |
| CloseHandle | [HANDLE] | BOOL |

Use non-null output storage for DuplicateHandle. Duplicate the current worker's
pseudo thread handle into THIS current process with desired access
THREAD_TERMINATE|SYNCHRONIZE = 0x00100001, inherit FALSE, options0. Never close
pseudo handles, request DUPLICATE_CLOSE_SOURCE/SAME_ACCESS, OpenThread by ID or
cast threading IDs to HANDLE. The caller invokes registration ON the I/O worker,
then transfers the returned real handle through its later admission handshake.

Public real-handle arguments/results require plain positive int fitting the
native pointer width, excluding null and all pseudo sentinel encodings -1/-2
or their unsigned pointer values. Reject bool, negative, oversized/malformed
values BEFORE any native call or ctypes narrowing. Pseudo values are permitted
ONLY as the internally obtained GetCurrentProcess/GetCurrentThread inputs.

request_cancel returns requested on nonzero native success, no_pending ONLY on
zero + ERROR_NOT_FOUND1168. Every other failure raises WindowsIoError. Read
last_error immediately after the failing call, before any other native call.
Requested and no_pending establish neither completion nor permission to free a
buffer/close a live handle. Completion may be normal, aborted or another error.

is_signaled calls WaitForSingleObject(handle,0) ONLY: WAIT_OBJECT_0=0 ->True,
WAIT_TIMEOUT258 ->False; WAIT_FAILED0xFFFFFFFF captures error immediately;
unexpected statuses fail. No positive wait, INFINITE, polling loop, sleep or
deadline resetting. A signaled thread handle is not itself a Python join.

close_handle performs ONE checked CloseHandle, captures failure immediately,
and never retries. A failure/exception has ownership_uncertain=True; retain its
known raw handle in the error. Caller must not blindly retry an integer which
could have been recycled. Caller closes ONLY its duplicated thread handle or
successfully detached reaped-process handle, never a FileIO-owned borrowed fd,
child inherited endpoint, another application's resource or a pseudo handle.

## Narrow CPython process-handle transfer

Public Popen has no explicit general handle-close method. The local configured
3.12.6 subprocess source inspected by root has Handle(int).Detach(): mark closed
then return its integer, or raise if already closed. Destructor aliases Close.
This is a CPython-private dependency, not a Python public portability promise.

detach_reaped_process is called ONLY after the HOST has observed wait/reap and
permanently stopped all Popen uses. Before transfer require a bounded plain
integer returncode (Windows DWORD0..0xFFFFFFFF), an int-derived non-bool _handle,
plain closed=False, callable Detach and valid real pointer value. Also require
is_signaled(raw_handle)==True BEFORE Detach. Missing/closed/malformed/live state
fails without transfer/native close. No process.poll/wait/terminate/kill call.

Invoke actual wrapper.Detach ONCE; require returned plain int equal to the
validated original raw handle. Return that transferred handle to caller, DO NOT
close it here. Wrapper closed-state prevents its GC destructor double-close.
If transfer raises or returns inconsistent data, return a typed uncertain error
with the known original handle; never automatically close/retry/drop the Popen.
The HOST retains original object plus disposition until resolution. Already
detached wrappers fail before reuse. No tombstone added to foreign objects and
no generic process registry. Caller owns idempotence and future handle closure.

No native API call has a documented universal hard-latency guarantee here. This
adapter does not implement the candidate five-second cleanup or guarantee it.
Actual host registration/cancellation/reap/close and CPython compatibility need
a separate bounded manifest after fake gate and supervisor ownership review.

## Verification and preserved scope

Pure fake DLL functions retain argtypes/restype and scripted results/errors;
DuplicateHandle writes into the real supplied ctypes output cell in memory.
Assert actual call order/flags, pointer width, immediate error capture and lack
of call on malformed input. No monkeypatch that replaces whole positive adapter
methods. Cover success/NoPending/error, exact wait outcomes, close uncertainty,
real fake-Popen int wrapper transfer and no transfer while live/already closed.
Lazy-default tests substitute DLL loader/error getter BEFORE first operation;
there is no native WinDLL or child import-isolation test. No real thread/timer.

Final TWO-path pytest/Ruff/mypy/py_compile and fresh independent spec+quality
review are required. No old198/268/225 suite, native fixture, runtime probe,
Windows census, media/font/model/API campaign or original proof is repeated.
No source41 consumer is integrated by this unit. Parent6b/Task7 output/A-V,
resource/phase/precision, fonts/visual QA, one final encode, five sidecars and
model/human acceptance remain open.

## Documentation used (not host qualification)

Root resolved Context7 Win32 + CPython, queried cancellation/ctypes. Win32 query
established pseudo-handle semantics but not cancellation/ABI in sufficient
detail; official fallback and local configured stdlib were inspected:

- https://learn.microsoft.com/en-us/windows/win32/api/ioapiset/nf-ioapiset-cancelsynchronousio
- https://learn.microsoft.com/en-us/windows/win32/api/handleapi/nf-handleapi-duplicatehandle
- https://learn.microsoft.com/en-us/windows/win32/api/synchapi/nf-synchapi-waitforsingleobject
- https://docs.python.org/3.12/library/ctypes.html
- local Python312/Lib/subprocess.py:214-230 and1538-1595, text only.

API source/docs establish intended mapping, not installed binary ABI execution,
actual cancellation/reap, latency, RSS or full-editor readiness.
