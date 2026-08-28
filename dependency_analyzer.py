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
    """
    repo_root = Path(repo_path)
    dependency_map = {}

    for file_path in get_python_files(repo_path):
        relative_path = str(file_path.relative_to(repo_root))
        imports = sorted(get_imports(file_path))

        dependency_map[relative_path] = imports

    return dependency_map


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 2:
        print("Usage: python dependency_analyzer.py <path_to_target_repo>")
        sys.exit(1)

    dependencies = build_dependency_map(sys.argv[1])

    for file, imports in list(dependencies.items())[:10]:
        print(f"\n{file}")

        for imported_module in imports:
            print(f"  -> {imported_module}")