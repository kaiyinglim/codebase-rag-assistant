"""Tests for deterministic Python dependency analysis."""

from pathlib import Path

import pytest

from dependency_analyzer import (
    build_dependency_map,
    find_dependents,
    get_imports,
)


def test_get_imports_reads_both_import_styles(tmp_path: Path):
    """Checks that standard and from-import statements are detected."""
    test_file = tmp_path / "example.py"

    # Create a small Python file with both supported import styles.
    test_file.write_text(
        "import requests\n"
        "from httpie.status import ExitStatus\n",
        encoding="utf-8",
    )

    # Parse the imports from the temporary source file.
    imports = get_imports(test_file)

    assert "requests" in imports
    assert "httpie.status.ExitStatus" in imports


def test_build_dependency_map_maps_files_to_imports(tmp_path: Path):
    """Checks that repository files are mapped to their imports."""
    first_file = tmp_path / "first.py"
    second_file = tmp_path / "second.py"

    # Create a tiny temporary repository for the test.
    first_file.write_text(
        "import requests\n",
        encoding="utf-8",
    )
    second_file.write_text(
        "from httpie.status import ExitStatus\n",
        encoding="utf-8",
    )

    # Build the dependency map from the temporary repository.
    dependency_map = build_dependency_map(str(tmp_path))

    assert dependency_map["first.py"] == ["requests"]
    assert dependency_map["second.py"] == [
        "httpie.status.ExitStatus"
    ]


def test_find_dependents_matches_module_symbols(tmp_path: Path):
    """Checks that a module search finds symbols imported from that module."""
    matching_file = tmp_path / "uses_status.py"
    other_file = tmp_path / "other.py"

    # Only one file imports something from httpie.status.
    matching_file.write_text(
        "from httpie.status import ExitStatus\n",
        encoding="utf-8",
    )
    other_file.write_text(
        "import requests\n",
        encoding="utf-8",
    )

    # Searching the parent module should still match ExitStatus.
    dependents = find_dependents(
        str(tmp_path),
        "httpie.status",
    )

    assert dependents == ["uses_status.py"]


def test_build_dependency_map_rejects_missing_repo(tmp_path: Path):
    """Checks that a missing repository path is rejected."""
    missing_repo = tmp_path / "does_not_exist"

    # The analyzer should fail clearly instead of returning an empty map.
    with pytest.raises(
        ValueError,
        match="Repository path does not exist",
    ):
        build_dependency_map(str(missing_repo))


def test_find_dependents_rejects_empty_dependency(tmp_path: Path):
    """Checks that an empty dependency search is rejected."""
    test_file = tmp_path / "example.py"
    test_file.write_text(
        "import requests\n",
        encoding="utf-8",
    )

    # Whitespace alone should not count as a dependency name.
    with pytest.raises(
        ValueError,
        match="Dependency cannot be empty",
    ):
        find_dependents(str(tmp_path), "   ")