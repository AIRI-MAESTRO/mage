"""Sandbox for synthesized tool code — plan item P1.17 Phase B.

Static analysis (`scan_source`) is pure and fast; the dynamic smoke run
(`run_module_function`) executes trivial, safe code in a real subprocess, so a few
tests spawn subprocesses (kept tiny + short-timeout).
"""

from __future__ import annotations

from mmar_mage.security.sandbox import (
    run_module_function,
    scan_source,
    synth_inputs,
    validate_tool_source,
)


def _kinds(violations) -> set[str]:
    return {v.kind for v in violations}


# --------------------------------------------------------------------------- #
# Static analysis
# --------------------------------------------------------------------------- #
def test_clean_source_has_no_violations():
    assert scan_source("import json\nimport re\ndef f(a):\n    return json.dumps(a)\n") == []


def test_banned_module_imports_flagged():
    assert _kinds(scan_source("import subprocess")) == {"banned_import"}
    assert _kinds(scan_source("import pickle")) == {"banned_import"}
    assert _kinds(scan_source("from ctypes import CDLL")) == {"banned_import"}
    assert _kinds(scan_source("import importlib")) == {"banned_import"}


def test_undeclared_network_import_flagged_but_declared_allowed():
    assert _kinds(scan_source("import requests")) == {"network_import"}
    assert scan_source("import requests", {"requests"}) == []  # declared dependency → allowed
    # submodule still maps to its root
    assert scan_source("import urllib.request", {"urllib"}) == []


def test_banned_builtins_flagged():
    assert _kinds(scan_source("x = eval('1+1')")) == {"banned_call"}
    assert _kinds(scan_source("exec('x=1')")) == {"banned_call"}
    assert _kinds(scan_source("compile('1', '<s>', 'eval')")) == {"banned_call"}
    assert _kinds(scan_source("__import__('os')")) == {"banned_call"}


def test_dangerous_os_and_subprocess_calls_flagged():
    assert _kinds(scan_source("import os\nos.system('ls')")) == {"banned_call"}
    assert _kinds(scan_source("import os\nos.remove('/x')")) == {"banned_call"}
    assert _kinds(scan_source("import subprocess\nsubprocess.run(['ls'])")) >= {"banned_call"}


def test_syntax_error_reported():
    assert _kinds(scan_source("def f(:\n  pass")) == {"syntax_error"}


def test_safe_os_attribute_not_flagged():
    # os.path.join is fine; only the destructive/process attrs are banned
    assert scan_source("import os\np = os.path.join('a', 'b')\n") == []


# --------------------------------------------------------------------------- #
# synth_inputs
# --------------------------------------------------------------------------- #
def test_synth_inputs_maps_types():
    out = synth_inputs({"name": "a str", "count": "int value", "ratio": "float", "items": "list of x"})
    assert out == {"name": "test", "count": 1, "ratio": 1.0, "items": []}


def test_synth_inputs_unknown_type_defaults_empty_string():
    assert synth_inputs({"x": "some custom object"}) == {"x": ""}


# --------------------------------------------------------------------------- #
# Dynamic smoke run (real subprocess)
# --------------------------------------------------------------------------- #
def test_run_safe_function_passes():
    res = run_module_function("def add(a, b):\n    return a + b\n", "add", {"a": 1, "b": 2})
    assert res.passed is True
    assert res.returned_repr == "3"


def test_run_missing_function_fails():
    res = run_module_function("def other():\n    return 1\n", "add", {})
    assert res.passed is False
    assert "not defined" in res.error


def test_run_import_error_fails():
    res = run_module_function("import this_module_does_not_exist_xyz\n", "f", {})
    assert res.passed is False


def test_run_call_exception_is_non_fatal():
    # imports + is callable → passes; the call raises, recorded but not fatal
    src = "def f(a):\n    raise ValueError('bad input')\n"
    res = run_module_function(src, "f", {"a": 1})
    assert res.passed is True
    assert "ValueError" in res.call_error


def test_run_timeout_fails():
    res = run_module_function("def loop():\n    while True:\n        pass\n", "loop", {}, time_limit=2)
    assert res.passed is False
    assert "timeout" in res.error


# --------------------------------------------------------------------------- #
# Combined validation
# --------------------------------------------------------------------------- #
def test_validate_static_failure_short_circuits_run():
    report = validate_tool_source("import os\nos.system('x')\ndef f():\n    return 1\n", "f")
    assert report.ok is False
    assert report.static_violations  # caught statically
    assert report.run is None  # never executed banned code


def test_validate_clean_source_runs_and_passes():
    report = validate_tool_source("def double(x):\n    return x * 2\n", "double", input_schema={"x": "int"})
    assert report.ok is True
    assert report.run is not None and report.run.passed


def test_report_feedback_describes_failure():
    report = validate_tool_source("import socket\ndef f():\n    return 1\n", "f")
    assert "network" in report.feedback().lower()
