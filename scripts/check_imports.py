#!/usr/bin/env python3
"""Validate all imports resolve correctly.

Usage: python scripts/check_imports.py
Exit code 0 = all imports valid, 1 = broken imports
"""
import ast
import importlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"

INTERNAL_PREFIXES = ("memory.", "proxy.", "skills.", "clients.", "commands.", "enhancer.")


def resolve_module_path(module_name):
    """Convert a dotted module name to a file path under src/."""
    parts = module_name.split(".")
    base = SRC / Path(*parts)
    if (base / "__init__.py").exists():
        return base / "__init__.py"
    py_file = base.with_suffix(".py")
    if py_file.exists():
        return py_file
    return None


def get_module_exports(filepath):
    """Get all module-level names defined in a Python file.

    Includes names bound inside top-level compound statements (try/except,
    if, with) — a common pattern for optional imports and fallbacks — but
    not names local to function/class bodies.
    """
    try:
        source = filepath.read_text()
    except (OSError, UnicodeDecodeError):
        return None
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return None

    names = set()

    def _add_target(target):
        if isinstance(target, ast.Name):
            names.add(target.id)
        elif isinstance(target, ast.Tuple):
            for elt in target.elts:
                if isinstance(elt, ast.Name):
                    names.add(elt.id)

    def _scan(stmts):
        for node in stmts:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                names.add(node.name)
                # PEP 562: a module-level __getattr__ lazily exports names it
                # compares against (e.g. `if name == "SOTAMode":`). Collect
                # those literals as exports.
                if isinstance(node, ast.FunctionDef) and node.name == "__getattr__":
                    for sub in ast.walk(node):
                        if isinstance(sub, ast.Compare):
                            for side in [sub.left, *sub.comparators]:
                                if isinstance(side, ast.Constant) and isinstance(side.value, str):
                                    names.add(side.value)
                # do NOT descend into other bodies: those names are not exports
            elif isinstance(node, ast.Assign):
                for target in node.targets:
                    _add_target(target)
            elif isinstance(node, ast.AnnAssign) and node.target is not None:
                _add_target(node.target)
            elif isinstance(node, ast.ImportFrom):
                for alias in node.names:
                    names.add(alias.asname or alias.name)
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    names.add(alias.asname or alias.name.split(".")[0])
            elif isinstance(node, ast.Try):
                _scan(node.body)
                for handler in node.handlers:
                    _scan(handler.body)
                _scan(node.orelse)
                _scan(node.finalbody)
            elif isinstance(node, (ast.If, ast.With)):
                _scan(node.body)
                if isinstance(node, ast.If):
                    _scan(node.orelse)

    _scan(tree.body)
    return names


def _negative_import_ranges(tree):
    """Line ranges of `with assertRaises(ImportError)` / `pytest.raises(ImportError)`
    blocks. Imports inside are intentional failure tests and must not be flagged."""
    ranges = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.With):
            continue
        for item in node.items:
            call = item.context_expr
            if not isinstance(call, ast.Call) or not call.func:
                continue
            func = call.func
            fname = None
            if isinstance(func, ast.Attribute):
                fname = func.attr
            elif isinstance(func, ast.Name):
                fname = func.id
            if fname not in ("assertRaises", "raises"):
                continue
            exc_names = []
            for arg in call.args:
                if isinstance(arg, ast.Name):
                    exc_names.append(arg.id)
                elif isinstance(arg, ast.Attribute):
                    exc_names.append(arg.attr)
            if any(e in ("ImportError", "ModuleNotFoundError") for e in exc_names):
                end = getattr(node, "end_lineno", node.lineno)
                ranges.append((node.lineno, end))
    return ranges


def check_file_imports(py_file):
    """Check all internal imports in a single file."""
    broken = []
    try:
        source = py_file.read_text()
    except (OSError, UnicodeDecodeError):
        return broken
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return broken

    neg_ranges = _negative_import_ranges(tree)

    for node in ast.walk(tree):
        if not isinstance(node, ast.ImportFrom):
            continue
        if any(start <= node.lineno <= end for start, end in neg_ranges):
            continue
        module = node.module
        if not module:
            continue
        if not module.startswith(INTERNAL_PREFIXES):
            continue

        mod_path = resolve_module_path(module)
        if mod_path is None:
            rel = py_file.relative_to(ROOT)
            broken.append((str(rel), node.lineno, module, "*", f"module not found: {module}"))
            continue

        exports = get_module_exports(mod_path)
        if exports is None:
            continue

        for alias in node.names:
            name = alias.name
            if name == "*":
                continue
            if name in exports:
                continue
            # `from pkg import sub` may import a submodule/subpackage, not a
            # name exported by pkg/__init__.py. Accept it if that submodule
            # exists under the package directory.
            if mod_path.name == "__init__.py":
                pkg_dir = mod_path.parent
                if (pkg_dir / f"{name}.py").exists() or (pkg_dir / name / "__init__.py").exists():
                    continue
            rel = py_file.relative_to(ROOT)
            broken.append((str(rel), node.lineno, module, name, f"'{name}' not in {module}"))

    return broken


def main():
    all_broken = []

    for directory in [SRC, ROOT / "tests"]:
        if not directory.exists():
            continue
        for py_file in sorted(directory.rglob("*.py")):
            broken = check_file_imports(py_file)
            all_broken.extend(broken)

    if all_broken:
        print(f"Found {len(all_broken)} broken import(s):\n")
        for filepath, lineno, module, name, reason in all_broken:
            print(f"  {filepath}:{lineno}  from {module} import {name}")
            print(f"    -> {reason}")
        print(f"\nRESULT: {len(all_broken)} broken import(s) detected.")
        sys.exit(1)
    else:
        print("RESULT: All internal imports resolve correctly.")
        sys.exit(0)


if __name__ == "__main__":
    main()
