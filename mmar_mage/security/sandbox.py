"""Sandbox for synthesized tool code (plan item P1.17 Phase B).

Two layers, defence-in-depth:

1. **Static analysis** (``scan_source``) — parse the AST and reject banned imports
   (process control, dynamic code, deserialization, and *undeclared* network) and
   banned calls (``eval`` / ``exec`` / ``compile`` / ``__import__`` /
   ``os.system`` / ``subprocess.*`` / ``os.exec*`` / file-destroying ``os.*``).
   This is the primary guardrail and runs before any code is executed.

2. **Dynamic smoke run** (``run_module_function``) — import the module and call
   the target function with synthetic inputs in a *separate subprocess* under
   resource limits (CPU + address space via :mod:`resource`) and a wall-clock
   timeout, with a minimal environment. Catches import errors, missing/undefined
   functions, infinite loops, and crashes.

Network isolation is enforced **statically** (network modules are banned unless
declared in the tool's ``dependencies``); OS-level isolation (seccomp / network
namespaces) is a follow-up — macOS has no portable primitive, so the static ban
is the contract today.
"""

from __future__ import annotations

import ast
import json
import logging
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)

# Modules that are never allowed in synthesized tools.
_BANNED_MODULES: frozenset[str] = frozenset(
    {
        "subprocess",
        "ctypes",
        "multiprocessing",
        "pickle",  # deserialization RCE
        "marshal",
        "shelve",
        "pty",
        "importlib",  # dynamic import bypasses this scan
        "builtins",  # re-reaching banned builtins
    }
)

# Network-capable modules — allowed ONLY if declared in the tool's dependencies.
_NETWORK_MODULES: frozenset[str] = frozenset(
    {
        "socket",
        "ssl",
        "urllib",
        "http",
        "ftplib",
        "smtplib",
        "telnetlib",
        "poplib",
        "imaplib",
        "requests",
        "httpx",
        "aiohttp",
        "urllib3",
        "websocket",
        "websockets",
    }
)

# Builtins that enable dynamic code execution / import.
_BANNED_BUILTINS: frozenset[str] = frozenset({"eval", "exec", "compile", "__import__"})

# Dangerous ``os`` attributes (process control + filesystem destruction).
_BANNED_OS_ATTRS: frozenset[str] = frozenset(
    {
        "system",
        "popen",
        "remove",
        "unlink",
        "rmdir",
        "removedirs",
        "kill",
        "killpg",
        "fork",
        "forkpty",
        "abort",
        "execv",
        "execve",
        "execvp",
        "execvpe",
        "execl",
        "execle",
        "execlp",
        "spawnv",
        "spawnve",
        "spawnl",
    }
)


@dataclass
class StaticViolation:
    """One static-analysis finding."""

    kind: str  # "banned_import" | "network_import" | "banned_call" | "syntax_error"
    detail: str
    lineno: int = 0

    def __str__(self) -> str:
        loc = f" (line {self.lineno})" if self.lineno else ""
        return f"[{self.kind}] {self.detail}{loc}"


def _root_module(name: str) -> str:
    return (name or "").split(".", 1)[0]


def scan_source(source: str, allowed_imports: set[str] | None = None) -> list[StaticViolation]:
    """Static-scan ``source`` for banned imports/calls.

    Parameters
    ----------
    source : str
        Python source of the synthesized tool.
    allowed_imports : set[str] | None
        Module roots explicitly permitted (from the tool's declared
        ``dependencies``) — these exempt *network* modules from the ban. Process /
        dynamic-code modules are banned regardless.

    Returns
    -------
    list[StaticViolation]
        Empty when the source is clean.
    """
    allowed = {_root_module(a) for a in (allowed_imports or set())}
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        return [StaticViolation("syntax_error", str(exc), getattr(exc, "lineno", 0) or 0)]

    violations: list[StaticViolation] = []

    for node in ast.walk(tree):
        # --- imports ---
        if isinstance(node, ast.Import):
            for alias in node.names:
                _check_module(_root_module(alias.name), allowed, node.lineno, violations)
        elif isinstance(node, ast.ImportFrom):
            # `from x import y` — x may be None for relative imports.
            _check_module(_root_module(node.module or ""), allowed, node.lineno, violations)

        # --- calls ---
        elif isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Name) and func.id in _BANNED_BUILTINS:
                violations.append(StaticViolation("banned_call", f"call to {func.id}()", node.lineno))
            elif isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name):
                base, attr = func.value.id, func.attr
                if base == "os" and attr in _BANNED_OS_ATTRS:
                    violations.append(StaticViolation("banned_call", f"os.{attr}()", node.lineno))
                elif base == "subprocess":
                    violations.append(StaticViolation("banned_call", f"subprocess.{attr}()", node.lineno))

    return violations


def _check_module(root: str, allowed: set[str], lineno: int, out: list[StaticViolation]) -> None:
    if not root:
        return
    if root in _BANNED_MODULES:
        out.append(StaticViolation("banned_import", f"import of '{root}'", lineno))
    elif root in _NETWORK_MODULES and root not in allowed:
        out.append(StaticViolation("network_import", f"undeclared network import '{root}'", lineno))


# --------------------------------------------------------------------------- #
# Dynamic smoke run
# --------------------------------------------------------------------------- #
@dataclass
class SandboxRunResult:
    """Outcome of a dynamic subprocess smoke run."""

    passed: bool
    error: str = ""
    call_error: str = ""  # exception from calling the function (non-fatal — synth inputs are approximate)
    returned_repr: str = ""
    example_failures: list[str] = field(default_factory=list)  # WP-8: failed behaviour cases


_TYPE_DEFAULTS: dict[str, object] = {
    "str": "test",
    "int": 1,
    "float": 1.0,
    "bool": True,
    "bytes": b"test",
    "list": [],
    "dict": {},
    "tuple": (),
    "set": set(),
}


def synth_inputs(input_schema: dict[str, object]) -> dict[str, object]:
    """Derive trivial synthetic kwargs from an ``input_schema`` (best-effort).

    Each value's type is guessed from a substring of its description (``"str"``,
    ``"int"``, …); unknown types default to an empty string. Approximate by design
    — a signature mismatch is reported as a non-fatal ``call_error``, not a failure.
    """
    out: dict[str, object] = {}
    for arg, desc in (input_schema or {}).items():
        text = str(desc).lower()
        chosen: object = ""
        for type_name, default in _TYPE_DEFAULTS.items():
            if type_name in text:
                chosen = default
                break
        out[str(arg)] = chosen
    return out


def _set_limits(cpu_seconds: int, mem_mb: int):  # pragma: no cover - runs in child preexec
    """preexec_fn: cap CPU time and address space in the child before exec."""
    try:
        import resource

        resource.setrlimit(resource.RLIMIT_CPU, (cpu_seconds, cpu_seconds))
        try:
            nbytes = mem_mb * 1024 * 1024
            resource.setrlimit(resource.RLIMIT_AS, (nbytes, nbytes))
        except (ValueError, OSError):
            pass  # RLIMIT_AS not enforceable on some platforms (e.g. macOS)
    except Exception:
        pass


# The in-child runner: import the module by path, verify the function, try a call.
_RUNNER_TEMPLATE = """\
import importlib.util, json, sys, traceback
verdict = {{"imported": False, "callable": False, "called": False, "call_error": "", "returned": "", "example_results": []}}
try:
    spec = importlib.util.spec_from_file_location("synth_tool", {module_path!r})
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    verdict["imported"] = True
    fn = getattr(mod, {func_name!r}, None)
    if callable(fn):
        verdict["callable"] = True
        try:
            result = fn(**{inputs!r})
            verdict["called"] = True
            verdict["returned"] = repr(result)[:200]
        except Exception as exc:
            verdict["call_error"] = f"{{type(exc).__name__}}: {{exc}}"
        for ex in json.loads({examples_json!r}):
            rec = {{"input": repr(ex.get("input", {{}}))[:120], "passed": False,
                    "expected": repr(ex.get("expected_output"))[:120], "actual": "", "error": ""}}
            try:
                out = fn(**ex.get("input", {{}}))
                exp = ex.get("expected_output")
                cmp = ex.get("comparison", "exact")
                if cmp == "contains":
                    rec["passed"] = bool(out is not None and exp in out)
                elif cmp == "type_only":
                    rec["passed"] = type(out).__name__ == type(exp).__name__
                else:
                    rec["passed"] = out == exp
                rec["actual"] = repr(out)[:120]
            except Exception as exc:
                rec["error"] = f"{{type(exc).__name__}}: {{exc}}"
            verdict["example_results"].append(rec)
except Exception:
    verdict["error"] = traceback.format_exc(limit=3)
print(json.dumps(verdict))
"""


def run_module_function(
    source: str,
    func_name: str,
    inputs: dict[str, object] | None = None,
    *,
    examples: list[dict[str, object]] | None = None,
    time_limit: int = 5,
    mem_limit_mb: int = 256,
) -> SandboxRunResult:
    """Import ``source`` and smoke-call ``func_name`` in a resource-limited subprocess.

    Passing means the module imported cleanly and the function is defined and
    callable. A call exception (likely from approximate synthetic inputs) is
    recorded in ``call_error`` but does not fail validation. Import errors, a
    missing function, a timeout, or a crashing exit *do* fail.
    """
    inputs = inputs or {}
    with tempfile.TemporaryDirectory(prefix="mage_sandbox_") as tmp:
        mod_path = Path(tmp) / "synth_tool.py"
        mod_path.write_text(source, encoding="utf-8")
        runner = _RUNNER_TEMPLATE.format(
            module_path=str(mod_path), func_name=func_name, inputs=inputs, examples_json=json.dumps(examples or [])
        )

        preexec = None
        if sys.platform != "win32":
            preexec = lambda: _set_limits(time_limit, mem_limit_mb)  # noqa: E731

        try:
            proc = subprocess.run(
                [sys.executable, "-I", "-c", runner],  # -I: isolated mode (ignore env/user site)
                capture_output=True,
                text=True,
                timeout=time_limit,
                cwd=tmp,
                env={"PATH": "", "PYTHONHASHSEED": "0"},
                preexec_fn=preexec,
            )
        except subprocess.TimeoutExpired:
            return SandboxRunResult(passed=False, error=f"timeout after {time_limit}s")
        except Exception as exc:  # pragma: no cover - spawn failure
            return SandboxRunResult(passed=False, error=f"subprocess failed: {exc}")

        if proc.returncode != 0:
            # A negative return code means the child was killed by a signal.
            # `_set_limits` sets RLIMIT_CPU; the kernel escalates SIGXCPU (-24)
            # to SIGKILL (-9) when the CPU budget is exhausted, so on Linux a
            # runaway loop is killed by the CPU limit *before* the wall-clock
            # `timeout` fires. Surface that as a timeout — matching the
            # TimeoutExpired path that wins on platforms where the wall-clock
            # limit fires first (e.g. macOS) — instead of a generic exit code.
            if proc.returncode in (-9, -24):  # SIGKILL / SIGXCPU — CPU-limit kill
                return SandboxRunResult(
                    passed=False,
                    error=f"timeout after {time_limit}s (cpu limit; killed by signal {-proc.returncode})",
                )
            return SandboxRunResult(passed=False, error=f"nonzero exit {proc.returncode}: {proc.stderr[-300:]}")

        try:
            verdict = json.loads(proc.stdout.strip().splitlines()[-1])
        except (ValueError, IndexError):
            return SandboxRunResult(passed=False, error=f"unparseable runner output: {proc.stdout[-200:]}")

        if not verdict.get("imported"):
            return SandboxRunResult(passed=False, error=verdict.get("error", "import failed"))
        if not verdict.get("callable"):
            return SandboxRunResult(passed=False, error=f"function {func_name!r} is not defined/callable")

        example_failures = [
            f"input={r.get('input')}: expected {r.get('expected')}, got "
            f"{r.get('actual') or r.get('error') or 'None'}"
            for r in verdict.get("example_results", [])
            if not r.get("passed")
        ]
        return SandboxRunResult(
            passed=True,
            call_error=verdict.get("call_error", ""),
            returned_repr=verdict.get("returned", ""),
            example_failures=example_failures,
        )


# --------------------------------------------------------------------------- #
# Combined validation
# --------------------------------------------------------------------------- #
@dataclass
class SandboxReport:
    """Combined static + dynamic verdict for a synthesized tool's source."""

    ok: bool
    static_violations: list[StaticViolation] = field(default_factory=list)
    run: SandboxRunResult | None = None

    def feedback(self) -> str:
        """A compact failure description to feed back to the implementation generator."""
        parts: list[str] = []
        if self.static_violations:
            parts.append("Static analysis rejected the code: " + "; ".join(str(v) for v in self.static_violations))
        if self.run is not None and not self.run.passed:
            parts.append(f"Sandbox run failed: {self.run.error}")
        if self.run is not None and self.run.example_failures:
            parts.append("Example cases failed: " + "; ".join(self.run.example_failures))
        return " | ".join(parts) or "ok"


def validate_tool_source(
    source: str,
    func_name: str,
    *,
    allowed_imports: set[str] | None = None,
    input_schema: dict[str, object] | None = None,
    examples: list[dict[str, object]] | None = None,
    time_limit: int = 5,
    mem_limit_mb: int = 256,
) -> SandboxReport:
    """Run static analysis, then (only if clean) the dynamic smoke run + examples.

    Static failures short-circuit — banned code is never executed. When
    ``examples`` are supplied (WP-8) they run as unit tests in the same isolated
    subprocess; any failing case makes the report not-ok so the repair loop sees
    the expected-vs-actual diff.
    """
    violations = scan_source(source, allowed_imports)
    if violations:
        return SandboxReport(ok=False, static_violations=violations)

    run = run_module_function(
        source,
        func_name,
        synth_inputs(input_schema or {}),
        examples=examples,
        time_limit=time_limit,
        mem_limit_mb=mem_limit_mb,
    )
    return SandboxReport(ok=run.passed and not run.example_failures, run=run)
