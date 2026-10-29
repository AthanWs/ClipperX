"""Speech worker lifetime, including the Windows venv launcher child."""
import json
import os
import subprocess
import sys
import time
from types import SimpleNamespace

import pytest


@pytest.mark.skipif(os.name!="nt",reason="Windows launcher exit race")
def test_naturally_retired_launcher_still_settles_its_pipes_after_tree_kill_race(monkeypatch):
    from clipperx import speech_process
    observed=[]
    class AlreadyExited:
        pid=123
        def communicate(self,timeout):
            observed.append(timeout)
            return "",""
        def poll(self):
            return 0
    monkeypatch.setattr(speech_process.subprocess,"run",lambda *a,**k:SimpleNamespace(returncode=128,stderr=b"launcher already exited"))
    speech_process._retire(AlreadyExited())
    assert observed==[5]


def test_speech_process_returns_captured_text_and_exit_code(tmp_path):
    from clipperx.speech_process import run_worker
    result=run_worker([sys.executable,"-c","print('native worker'); raise SystemExit(7)"],
                      cwd=tmp_path,env=os.environ.copy(),timeout=20)
    assert result.returncode==7 and result.stdout.strip()=="native worker"


@pytest.mark.skipif(os.name!="nt",reason="Windows process-tree lifetime")
def test_timeout_retires_owned_venv_launcher_and_descendant(tmp_path):
    from clipperx.speech_process import run_worker
    receipt=tmp_path/"child.json"
    interpreter_receipt=tmp_path/"actual-interpreter.txt"
    grandchild=("import os,sys,time; from pathlib import Path; "
                "Path(sys.argv[1]).write_text(str(os.getpid())); time.sleep(90)")
    code=("import os,subprocess,sys,time,json; from pathlib import Path; "
          "child=subprocess.Popen([sys.executable,'-c',sys.argv[2],sys.argv[3]]); "
          "Path(sys.argv[1]).write_text(json.dumps({'child_launcher':child.pid,"
          "'owner_interpreter':os.getpid()})); time.sleep(90)")
    started=time.monotonic()
    with pytest.raises(subprocess.TimeoutExpired):
        run_worker([sys.executable,"-c",code,str(receipt),grandchild,str(interpreter_receipt)],cwd=tmp_path,
                   env=os.environ.copy(),timeout=8)
    assert time.monotonic()-started < 25
    descendants=list(json.loads(receipt.read_text()).values())
    descendants.append(int(interpreter_receipt.read_text()))
    import ctypes
    kernel=ctypes.WinDLL("kernel32",use_last_error=True)
    kernel.OpenProcess.restype=ctypes.c_void_p
    for child in descendants:
        handle=kernel.OpenProcess(0x1000,False,child)
        if handle:
            try:
                status=ctypes.c_ulong()
                assert kernel.GetExitCodeProcess(ctypes.c_void_p(handle),ctypes.byref(status))
                assert status.value!=259, "owned descendant still running"
            finally:
                kernel.CloseHandle(ctypes.c_void_p(handle))
