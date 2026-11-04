"""The checks do not execute the notebooks, so what they use from the package is tested here."""

import ast
import importlib
import json
from pathlib import Path
from types import ModuleType

import pytest

from hotel_cancellation import config

NOTEBOOKS = sorted((config.PROJECT_ROOT / "notebooks").glob("*.ipynb"))
_PACKAGE = "hotel_cancellation"


def _code(path: Path) -> ast.Module:
    cells = json.loads(path.read_text(encoding="utf-8"))["cells"]
    lines = [
        line
        for cell in cells
        if cell["cell_type"] == "code"
        for line in "".join(cell["source"]).splitlines()
        if not line.startswith("%")
    ]
    return ast.parse("\n".join(lines))


def _imported(module: str, name: str) -> object:
    try:
        return importlib.import_module(f"{module}.{name}")
    except ModuleNotFoundError:
        return getattr(importlib.import_module(module), name)


def test_there_are_notebooks_to_check() -> None:
    assert NOTEBOOKS


@pytest.mark.slow
@pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda path: path.stem)
def test_what_a_notebook_uses_from_the_package_exists(path: Path) -> None:
    tree = _code(path)
    imported: dict[str, object] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module and node.module.startswith(_PACKAGE):
            for alias in node.names:
                imported[alias.asname or alias.name] = _imported(node.module, alias.name)
    assert imported
    modules = {name: value for name, value in imported.items() if isinstance(value, ModuleType)}
    missing = [
        f"{node.value.id}.{node.attr}"
        for node in ast.walk(tree)
        if isinstance(node, ast.Attribute)
        and isinstance(node.value, ast.Name)
        and node.value.id in modules
        and not hasattr(modules[node.value.id], node.attr)
    ]
    assert not missing
