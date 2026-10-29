"""Explicit routing stays independent of legacy initialization."""

import ast
import builtins
import importlib
from pathlib import Path
import sys
from typing import Any

import pytest


def test_explicit_routes_exist_without_eager_legacy_imports() -> None:
    tree = ast.parse(Path("clipperx/plan_render.py").read_text(encoding="utf-8"))
    names = {node.name for node in tree.body if isinstance(node, ast.FunctionDef)}
    assert {"render_visual_composition_plan", "render_legacy_composition_plan"} <= names
    forbidden = {"editing", "config", "face", "media"}
    assert not any(isinstance(n, ast.ImportFrom) and n.module in forbidden for n in tree.body)


def test_fresh_visual_import_denies_legacy_and_provider_initialization(monkeypatch: pytest.MonkeyPatch) -> None:
    saved = {k: v for k, v in sys.modules.items() if k.startswith("clipperx.") and ".tests" not in k}
    package = sys.modules["clipperx"]
    package_attributes = dict(vars(package))
    for key in saved:
        monkeypatch.delitem(sys.modules, key)
    original = builtins.__import__
    forbidden = {"editing", "config", "face", "media", "models", "providers", "structured_client"}
    def checked(name: str, globals: Any = None, locals: Any = None, fromlist: Any = (), level: int = 0) -> Any:
        if name.split(".")[-1] in forbidden:
            pytest.fail(f"fresh v2 imported {name}")
        return original(name, globals, locals, fromlist, level)
    monkeypatch.setattr(builtins, "__import__", checked)
    importlib.import_module("clipperx.plan_render")
    importlib.import_module("clipperx._visual_render_host")
    # Restore all original module identities before other tests use imported
    # immutable dataclasses and exact-type guards.
    for key in list(sys.modules):
        if key.startswith("clipperx.") and ".tests" not in key and key not in saved:
            del sys.modules[key]
    sys.modules.update(saved)
    for name in list(vars(package)):
        if name not in package_attributes:
            delattr(package, name)
    vars(package).update(package_attributes)
