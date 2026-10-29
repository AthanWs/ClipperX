"""Finite local speech subprocess; retire its owned Windows venv process tree."""
from __future__ import annotations

import math
import os
from pathlib import Path
import signal
import subprocess
import sys


def _retire(process: subprocess.Popen[str]) -> None:
    # A Windows venv python.exe is a launcher whose actual interpreter is a
    # child. Killing only the launcher can leave inference and its GPU alive.
    # Retain the exact Popen handle while addressing its PID, never image names.
    detail = ""
    if sys.platform == "win32":
        result = subprocess.run(
            ["taskkill", "/PID", str(process.pid), "/T", "/F"],
            capture_output=True, timeout=10, check=False,
            creationflags=subprocess.CREATE_NO_WINDOW,
        )
        # A descendant's exit can make a waiting venv launcher exit before
        # taskkill reaches it. A nonzero code alone is not a live-child proof.
        detail = f" (taskkill={result.returncode}: {result.stderr.decode(errors='replace')[-800:]})"
    else:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    try:
        process.communicate(timeout=5)
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(f"El worker propio PID {process.pid} no cerró sus pipes{detail}") from exc
    if process.poll() is None:
        raise RuntimeError(f"No se confirmó el cierre del worker propio PID {process.pid}{detail}")


def run_worker(
    command: list[str], *, cwd: Path, env: dict[str, str], timeout: float,
) -> subprocess.CompletedProcess[str]:
    if not math.isfinite(timeout) or not 0 < timeout <= 86400:
        raise ValueError("El worker necesita un límite finito de hasta 24 horas")
    process = subprocess.Popen(
        command, cwd=cwd, env=env, stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        encoding="utf-8", errors="replace", start_new_session=os.name != "nt",
        creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
    )
    try:
        stdout, stderr = process.communicate(timeout=timeout)
    except BaseException:
        _retire(process)
        raise
    return subprocess.CompletedProcess(command, process.returncode, stdout, stderr)
