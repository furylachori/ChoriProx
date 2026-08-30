#!/usr/bin/env python3
"""Check that test mocks match production Agent API.

Usage: python scripts/check_mock_drift.py
Exit code 0 = no drift, 1 = drift detected
"""
import ast
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

MOCK_LOCATIONS = [
    ("tests/conftest.py", "MockAgent"),
    ("tests/proxy/test_skill_mode.py", "_MockAgent"),
    ("tests/unit/test_mode_resolution.py", "_MockAgent"),
    ("tests/skills/test_workspace_skill.py", "MockAgent"),
]

PROD_FILE = "src/proxy/agent/agent.py"
PROD_CLASS = "Agent"


def get_class_methods(filepath, class_name):
    """Extract public method/property names from a class in a Python file."""
    source = (ROOT / filepath).read_text()
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            names = set()
            for item in node.body:
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    if not item.name.startswith("_"):
                        names.add(item.name)
                elif isinstance(item, ast.Assign):
                    for target in item.targets:
                        if isinstance(target, ast.Name) and not target.id.startswith("_"):
                            names.add(target.id)
            return names
    return set()


def find_mock_classes(filepath):
    """Find all mock agent classes in a file by scanning for class definitions."""
    source = (ROOT / filepath).read_text()
    tree = ast.parse(source)
    mocks = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            if "MockAgent" in node.name or "_MockAgent" in node.name:
                mocks.append(node.name)
    return mocks


def main():
    prod_methods = get_class_methods(PROD_FILE, PROD_CLASS)
    if not prod_methods:
        print(f"ERROR: Could not find class {PROD_CLASS} in {PROD_FILE}")
        sys.exit(1)

    print(f"Production {PROD_CLASS} public API ({len(prod_methods)} methods):")
    for m in sorted(prod_methods):
        print(f"  - {m}")
    print()

    drift_found = False

    for filepath, class_name in MOCK_LOCATIONS:
        full_path = ROOT / filepath
        if not full_path.exists():
            print(f"SKIP: {filepath} does not exist")
            continue

        mock_methods = get_class_methods(filepath, class_name)
        if not mock_methods:
            actual_mocks = find_mock_classes(filepath)
            if actual_mocks:
                print(f"WARN: Expected {class_name} in {filepath}, found: {actual_mocks}")
            else:
                print(f"SKIP: {filepath}::{class_name} not found")
            continue

        missing = prod_methods - mock_methods
        extra = mock_methods - prod_methods

        if missing:
            print(f"DRIFT: {filepath}::{class_name}")
            print(f"  Missing from mock: {sorted(missing)}")
            drift_found = True
        else:
            print(f"OK:    {filepath}::{class_name} ({len(mock_methods)} methods, all match)")

        if extra:
            print(f"  INFO: Mock has extra (non-production): {sorted(extra)}")

    print()
    if drift_found:
        print("RESULT: Mock drift detected! Update mocks to include missing methods.")
        sys.exit(1)
    else:
        print("RESULT: No mock drift detected.")
        sys.exit(0)


if __name__ == "__main__":
    main()
