"""chunker.py — parses Python source files into function/class-level chunks.

Rather than splitting text at a fixed character count (which risks cutting
a function in half), this walks each file's Abstract Syntax Tree (AST) and
extracts one chunk per function and class definition — including methods
nested inside classes — preserving exact file/line metadata for citation.
"""

import ast
import json
import sys
from pathlib import Path


def get_python_files(repo_path: str) -> list[Path]:
    """Recursively finds all Python files under a repo path.

    Skips common non-source directories (virtual envs, git internals,
    build output) so dependency or generated code isn't parsed as part
    of the target project.

    Args:
        repo_path: Filesystem path to the root of the repo to scan.

    Returns:
        A list of Path objects, one per .py file found.
    """
    skip_dirs = {"venv", ".venv", "__pycache__", ".git", "build", "dist", "node_modules"}
    repo = Path(repo_path)

    return [
        p for p in repo.rglob("*.py") # recursively search for .py files
        if not any(part in skip_dirs for part in p.parts) # p.parts splits the path into pieces; skip if any piece matches skip_dirs
    ]

def chunk_file(file_path: Path, repo_root: Path) -> list[dict]:
    """Parses a Python file into function, method, and class-level chunks.

    Functions and methods keep their complete source code. Classes are
    represented only by their declaration and optional docstring because
    their methods are already extracted separately. This avoids duplicating
    an entire large class in a single embedding chunk.

    Args:
        file_path: Path to the .py file to parse.
        repo_root: Root of the repository, used for relative file paths.

    Returns:
        A list of chunk dictionaries containing source text and metadata.
        Returns an empty list if the file cannot be parsed.
    """
    source = file_path.read_text(encoding="utf-8", errors="ignore")
    source_lines = source.splitlines(keepends=True)

    try:
        # parse the file into an AST — a tree representing its structure
        tree = ast.parse(source)
    except SyntaxError:
        # skip rather than crash the whole indexing run when file won't parse.
        return []

    chunks = []

    # visits every node at every depth
    for node in ast.walk(tree):
        # only keep functions, async functions, and classes
        if not isinstance(
            node,
            (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef),
        ):
            continue

        # Include decorators by starting from the earliest decorator line.
        start_line = (
            min(decorator.lineno for decorator in node.decorator_list)
            if node.decorator_list
            else node.lineno
        )

        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            # pull the exact original source text, including decorators
            text = "".join(
                source_lines[start_line - 1:node.end_lineno]
            )

            chunks.append({
                "name": node.name,
                "type": "function",
                "file": str(file_path.relative_to(repo_root)),
                "start_line": start_line,
                "end_line": node.end_lineno,
                "text": text,  # the actual source code
            })

        elif isinstance(node, ast.ClassDef):
            # Methods are extracted separately, so only keep the class
            # declaration and optional docstring instead of the whole body.
            first_body_line = node.body[0].lineno
            end_line = first_body_line - 1

            # Include the class docstring if one exists.
            if (
                isinstance(node.body[0], ast.Expr)
                and isinstance(node.body[0].value, ast.Constant)
                and isinstance(node.body[0].value.value, str)
            ):
                end_line = node.body[0].end_lineno

            text = "".join(
                source_lines[start_line - 1:end_line]
            ).rstrip()

            chunks.append({
                "name": node.name,
                "type": "class",
                "file": str(file_path.relative_to(repo_root)),
                "start_line": start_line,
                "end_line": end_line,
                "text": text,  # class declaration and optional docstring
            })

    return chunks

def chunk_repo(repo_path: str) -> list[dict]:
    """Chunks every Python file in a repo into one combined list.

    This is the main entry point the /index endpoint will call to
    produce the full set of chunks for embedding.

    Args:
        repo_path: Filesystem path to the root of the repo to chunk.

    Returns:
        A flat list of chunk dicts (see chunk_file for the schema),
        combined across every Python file in the repo.
    """
    repo_root = Path(repo_path)
    all_chunks = []

    for file_path in get_python_files(repo_path):
        # merges each file's chunk list into one flat list
        all_chunks.extend(chunk_file(file_path, repo_root))
    return all_chunks


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python chunker.py <path_to_target_repo>")
        sys.exit(1)

    chunks = chunk_repo(sys.argv[1])
    print(f"Extracted {len(chunks)} chunks from {sys.argv[1]}")

    for c in chunks[:3]:    # first three chunks
        print(f"\n--- {c['type']}: {c['name']} ({c['file']}:{c['start_line']}-{c['end_line']}) ---")
        print(c["text"][:200], "...")

    with open("chunks.json", "w") as f:
        json.dump(chunks, f, indent=2)
    print("\nSaved to chunks.json")