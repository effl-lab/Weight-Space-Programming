"""Static checks and execution of a writer's build_weights.py.

`build(script)` runs the script in a plain subprocess; it is what tools/check.py uses inside the
writer sandbox, where the sandbox already bounds what the script can see.
`build(script, isolate=True)` is what the harness uses on the host (final weights and brokered
forward calls): the script runs in a bwrap with no network, no data and a clean environment.
"""
import ast
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

from wsp import paths

ALLOWED_IMPORTS = {"numpy", "math"}
FORBIDDEN_NAMES = {"open", "eval", "exec", "compile", "__import__", "input", "globals", "locals",
                   "getattr", "setattr", "delattr", "vars", "breakpoint", "memoryview"}
FORBIDDEN_ATTRS = {"load", "loadtxt", "genfromtxt", "fromfile", "fromregex", "memmap", "lstsq", "pinv",
                   "solve", "inv", "svd", "eig", "eigh", "DataSource", "lib", "ctypeslib", "ctypes",
                   "tofile", "savetxt", "save", "testing", "f2py", "distutils", "os", "sys"}
BUILD_TIMEOUT_S = 600
VENV = paths.VENV


def static_problems(source: str) -> list[str]:
    try:
        tree = ast.parse(source)
    except SyntaxError as e:
        return [f"syntax error: {e}"]
    problems = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            problems += [f"import {a.name} not allowed" for a in node.names
                         if a.name.split(".")[0] not in ALLOWED_IMPORTS or "." in a.name]
        elif isinstance(node, ast.ImportFrom):
            if (node.module or "") not in ALLOWED_IMPORTS:
                problems.append(f"from {node.module} import not allowed (line {node.lineno})")
            problems += [f"import of name {a.name} not allowed (line {node.lineno})" for a in node.names
                         if a.name in FORBIDDEN_ATTRS or a.name in FORBIDDEN_NAMES or a.name == "*"]
        elif isinstance(node, ast.Name) and (node.id in FORBIDDEN_NAMES
                                             or (node.id.startswith("__") and node.id != "__name__")):
            problems.append(f"name {node.id} not allowed (line {node.lineno})")
        elif isinstance(node, ast.Attribute) and (node.attr in FORBIDDEN_ATTRS or node.attr.startswith("__")):
            problems.append(f"attribute .{node.attr} not allowed (line {node.lineno})")
    return problems


def literal_stats(source: str) -> dict:
    floats = [n.value for n in ast.walk(ast.parse(source))
              if isinstance(n, ast.Constant) and isinstance(n.value, float)]
    return {"lines": source.count("\n") + 1, "chars": len(source), "float_literals": len(floats)}


def _isolated_argv(build_dir: Path) -> list[str]:
    return ["bwrap", "--die-with-parent", "--unshare-all", "--clearenv",
            "--ro-bind", "/usr", "/usr", "--ro-bind", "/lib", "/lib", "--ro-bind", "/lib64", "/lib64",
            "--ro-bind", "/bin", "/bin", "--ro-bind", str(VENV), str(VENV),
            "--proc", "/proc", "--dev", "/dev", "--tmpfs", "/tmp",
            "--bind", str(build_dir), "/build", "--chdir", "/build",
            "--setenv", "PATH", f"{VENV}/bin:/usr/bin:/bin",
            f"{VENV}/bin/python", "-I", "/build/build_weights.py"]


def build(script: Path, isolate: bool = False) -> tuple[dict[str, np.ndarray] | None, list[str]]:
    """Returns (weights, []) or (None, problems). A build that fails is never an exception."""
    source = script.read_text()
    problems = static_problems(source)
    if problems:
        return None, problems
    tmp = Path(tempfile.mkdtemp(prefix="wsp_build_"))
    try:
        (tmp / "build_weights.py").write_text(source)
        argv = _isolated_argv(tmp) if isolate else [sys.executable, "-I", str(tmp / "build_weights.py")]
        try:
            proc = subprocess.run(argv, cwd=tmp, capture_output=True, text=True,
                                  timeout=BUILD_TIMEOUT_S, env={"PATH": "/usr/bin:/bin"})
        except subprocess.TimeoutExpired:
            return None, [f"build exceeded {BUILD_TIMEOUT_S} s"]
        if proc.returncode != 0:
            return None, [f"build exited {proc.returncode}: {proc.stderr[-2000:]}"]
        out = tmp / "weights.npz"
        if not out.exists():
            return None, ["build did not write weights.npz in its working directory"]
        try:
            with np.load(out, allow_pickle=False) as z:
                return {k: z[k] for k in z.files}, []
        except (ValueError, OSError, EOFError) as e:
            return None, [f"weights.npz unreadable without pickle: {e}"]
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
