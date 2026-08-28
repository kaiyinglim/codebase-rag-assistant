"""dependency_analyzer.py — analyzes Python import dependencies in a repository.

Parses Python files with the AST and records imported modules so dependency
questions can be answered deterministically rather than through semantic
retrieval.
"""

import ast
from pathlib import Path

from chunker import get_python_files


def get_imports(file_path: Path) -> set[str]:
    """Extracts imported module names from a Python source file.

    Args:
        file_path: Path to the Python file to inspect.

    Returns:
        A set containing the module names imported by the file.
        Returns an empty set if the file cannot be parsed.
    """
    # Read file into a string
    source = file_path.read_text(encoding="utf-8", errors="ignore")

    try:
        tree = ast.parse(source)
    except SyntaxError:
        return set()

    imports = set()

    # Walk the AST and collect both `import x` and `from x import y`.
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.add(alias.name)

        elif isinstance(node, ast.ImportFrom):
            if node.module:
                # Combine the source module with each imported name.
                for alias in node.names:
                    imports.add(
                        f"{node.module}.{alias.name}"
                    )

    return imports


def build_dependency_map(repo_path: str) -> dict[str, list[str]]:
    """Builds a file-to-imported-modules dependency map for a repository.

    Args:
        repo_path: Filesystem path to the Python repository.

    Returns:
        A dictionary mapping each Python file to the modules it imports.

    Raises:
        ValueError: If the repository path does not exist or is not a
            directory.
    """
    repo_root = Path(repo_path)

    # Reject a mistyped or missing repository instead of returning no results.
    if not repo_root.exists():
        raise ValueError(
            f"Repository path does not exist: {repo_path}"
        )

    # A repository must be a directory.
    if not repo_root.is_dir():
        raise ValueError(
            f"Repository path is not a directory: {repo_path}"
        )

    dependency_map = {}

    # Analyze every Python source file discovered in the repository.
    for file_path in get_python_files(repo_path):
        relative_path = str(file_path.relative_to(repo_root))

        imports = sorted(get_imports(file_path))

        dependency_map[relative_path] = imports

    return dependency_map


def find_dependents(repo_path: str, dependency: str) -> list[str]:
    """Finds files that depend on a given module or imported symbol.

    A dependency matches either the exact imported path or any symbol
    imported from that path. 

    Args:
        repo_path: Filesystem path to the Python repository.
        dependency: Module or symbol path to search for.

    Returns:
        A sorted list of repository-relative file paths that import the
        requested dependency.

    Raises:
        ValueError: If the dependency name is empty.
    """
    dependency = dependency.strip()

    if not dependency:
        raise ValueError("Dependency cannot be empty.")

    dependency_map = build_dependency_map(repo_path)
    dependents = []

    for file_path, imports in dependency_map.items():
        # Match the module itself or symbols imported from that module.
        if any(
            imported == dependency
            or imported.startswith(f"{dependency}.")
            for imported in imports
        ):
            # Each matching file only needs to appear once in the result.
            dependents.append(file_path)

    return sorted(dependents)


if __name__ == "__main__":
    import sys

    if len(sys.argv) not in (2, 3):
        print(
            "Usage: python dependency_analyzer.py "
            "<path_to_target_repo> [dependency]"
        )
        sys.exit(1)

    repo_path = sys.argv[1]

    if len(sys.argv) == 3:
        dependency = sys.argv[2]
        dependents = find_dependents(repo_path, dependency)

        print(f"\nFiles depending on {dependency}:")

        if not dependents:
            print("  No dependencies found.")
        else:
            for file_path in dependents:
                print(f"  -> {file_path}")

    else:
        dependencies = build_dependency_map(repo_path)

        # Print a small sample when no dependency is supplied.
        for file_path, imports in list(dependencies.items())[:10]:
            print(f"\n{file_path}")

            for imported_module in imports:
                print(f"  -> {imported_module}")