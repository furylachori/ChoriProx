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
    """Get all top-level names defined in a Python file."""
    try:
        source = filepath.read_text()
    except (OSError, UnicodeDecodeError):
        return None
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return None

    names = set()
    for node in ast.iter_child_nodes(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    names.add(target.id)
                elif isinstance(target, ast.Tuple):
                    for elt in target.elts:
                        if isinstance(elt, ast.Name):
                            names.add(elt.id)
        elif isinstance(node, ast.ImportFrom):
            for alias in node.names:
                names.add(alias.asname or alias.name)
        elif isinstance(node, ast.Import):
            for alias in node.names:
                names.add(alias.asname or alias.name.split(".")[0])
    return names


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

    for node in ast.walk(tree):
        if not isinstance(node, ast.ImportFrom):
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
            if name not in exports:
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
