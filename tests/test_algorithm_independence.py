"""Architecture regression tests for independent OPD algorithm modules."""

import ast
from pathlib import Path


ALGORITHM_FILES = (
    "gkd_opd.py",
    "pg_opd.py",
    "random_gkd_opd.py",
    "random_pg_opd.py",
    "topgap_gkd_opd.py",
    "topgap_pg_opd.py",
    "ps_opd.py",
    "cal_opd.py",
)

CANONICAL_ALGORITHM_NAMES = {
    "pg_opd",
    "random_pg_opd",
    "topgap_pg_opd",
    "gkd_opd",
    "random_gkd_opd",
    "topgap_gkd_opd",
    "ps_opd",
    "cal_opd",
}


def test_concrete_algorithms_do_not_import_other_algorithms():
    algorithm_dir = Path(__file__).resolve().parents[1] / "algorithms"
    violations = []
    for filename in ALGORITHM_FILES:
        path = algorithm_dir / filename
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and (node.module or "").startswith(
                "algorithms"
            ):
                violations.append(f"{filename}:{node.lineno}: from {node.module}")
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name.startswith("algorithms"):
                        violations.append(
                            f"{filename}:{node.lineno}: import {alias.name}"
                        )
    assert not violations, "Cross-algorithm imports found: " + ", ".join(violations)


def test_registry_contains_only_explicit_canonical_names():
    from algorithms import ALGORITHM_REGISTRY

    assert set(ALGORITHM_REGISTRY) == CANONICAL_ALGORITHM_NAMES
    assert "opd" not in ALGORITHM_REGISTRY
