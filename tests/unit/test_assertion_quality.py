"""Self-policing test: every test_* function in the suite must contain at least one assertion.

The previous implementation used ``subprocess.run(["grep", ...])`` which
had two failure modes:
- false-negatives on tests that use ``pytest.fail()`` or ``raise``
  instead of bare ``assert``
- slow: spawned a grep subprocess for every test file

The AST-based walk is faster and catches ``ast.Assert``, ``ast.Raise``,
and explicit ``pytest.fail``/``pytest.xfail``/``pytest.skip`` calls —
all of which are ways a test can assert "this must happen" or
"this is the contract".
"""
import ast
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
TARGET_DIRS = [REPO_ROOT / "tests" / "unit", REPO_ROOT / "tests" / "integration"]


def _function_has_assertion(node: ast.FunctionDef) -> bool:
    """Return True if the function body contains at least one
    assertion-like statement.

    Considered assertion-like:
    - ``ast.Assert`` (the ``assert ...`` statement)
    - ``ast.Raise`` (covers ``raise SomeException`` and the
      ``pytest.raises(...)`` context manager)
    - ``ast.Call`` to ``pytest.fail``, ``pytest.xfail``, ``pytest.skip``,
      ``pytest.raises`` (explicit assertion / skip markers)
    """
    for child in ast.walk(node):
        if isinstance(child, ast.Assert):
            return True
        if isinstance(child, ast.Raise):
            return True
        if isinstance(child, ast.Call):
            func = child.func
            if isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name):
                if (
                    func.value.id == "pytest"
                    and func.attr in {"fail", "xfail", "skip", "raises"}
                ):
                    return True
    return False


def _collect_assertion_dead_functions() -> list[str]:
    dead: list[str] = []
    for target in TARGET_DIRS:
        if not target.exists():
            continue
        for path in sorted(target.rglob("test_*.py")):
            try:
                source = path.read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError):
                continue
            try:
                tree = ast.parse(source, filename=str(path))
            except SyntaxError:
                continue
            for node in ast.walk(tree):
                if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    continue
                if not node.name.startswith("test_"):
                    continue
                if not _function_has_assertion(node):
                    dead.append(f"{path}::{node.name}")
    return dead


def test_no_assertion_dead_tests():
    """Every test_* function in unit/ and integration/ must contain at least one assertion-like statement."""
    dead = _collect_assertion_dead_functions()
    assert not dead, (
        "Assertion-dead test functions found (no assert/raise/pytest.fail):\n  "
        + "\n  ".join(dead)
    )
