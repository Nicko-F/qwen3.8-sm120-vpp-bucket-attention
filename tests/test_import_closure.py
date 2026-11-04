"""Check lazy package imports without importing optional GPU dependencies."""

import ast
from pathlib import Path


def test_every_relative_import_resolves_inside_the_installed_package():
    package = Path(__file__).resolve().parents[1] / "src/qwen_infra"
    errors = []
    for path in package.rglob("*.py"):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if not isinstance(node, ast.ImportFrom):
                continue
            if not node.level:
                if (node.module or "").startswith(("src.", "ops.")):
                    errors.append(f"{path.name}:{node.lineno}: research import {node.module}")
                continue
            base = path.parent
            for _ in range(node.level - 1):
                base = base.parent
            target = base.joinpath(*(node.module or "").split("."))
            if not target.with_suffix(".py").is_file() and not (target / "__init__.py").is_file():
                errors.append(f"{path.name}:{node.lineno}: missing relative module {node.module}")
    assert not errors, "\n".join(errors)
