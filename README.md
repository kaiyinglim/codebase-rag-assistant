# Codebase RAG Assistant

A FastAPI service that indexes a Python repository, answers natural-language questions about it with retrieval-augmented generation (RAG), and finds import dependents via AST analysis.

## How it works

```
Python repo
    │
    ▼
chunker.py          AST → function / class chunks
    │
    ▼
embedder.py         Jina embeddings → ChromaDB
    │
    ▼
retriever.py        semantic search over indexed chunks
    │
    ▼
rag.py + llm.py     DeepSeek answer grounded in retrieved code
```

| Stage | Module | Role |
| --- | --- | --- |
| Chunk | [`chunker.py`](chunker.py) | Walks each file’s AST and extracts one chunk per function, async function, and class (class chunks keep the declaration + docstring only; methods are separate). |
| Embed | [`embedder.py`](embedder.py) | Embeds chunks with `jinaai/jina-embeddings-v2-base-code` and stores them into a persistent ChromaDB collection (`./chroma_db`, collection `codebase_chunks`). |
| Retrieve | [`retriever.py`](retriever.py) | Embeds the question with the same model and returns the top-k nearest code chunks. |
| Answer | [`rag.py`](rag.py), [`llm.py`](llm.py) | Formats retrieved chunks as context and asks DeepSeek to answer using only that context. |
| Dependencies | [`dependency_analyzer.py`](dependency_analyzer.py) | Deterministic import graph from AST (`import` / `from … import`), not RAG. |

The HTTP API is defined in [`main.py`](main.py).

## Requirements

- Python 3.11+
- A [DeepSeek](https://platform.deepseek.com/) API key (for `/query` only)

## Setup

```bash
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Create a `.env` in the project root (gitignored):

```env
DEEPSEEK_API_KEY=your_key_here
```

First run downloads the local embedding model via `sentence-transformers`.

## Run

```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

Interactive docs: http://localhost:8000/docs

## API

### `GET /health`

Liveness check.

```json
{ "status": "ok" }
```

### `POST /index`

Clear any previous index, then chunk and embed a Python repository into ChromaDB.

```bash
curl -X POST http://localhost:8000/index \
  -H "Content-Type: application/json" \
  -d '{"repo_path": "/path/to/python/repo"}'
```

```json
{ "status": "success", "chunks_indexed": 1234 }
```

Invalid paths return `400`.

### `POST /query`

Ask a question about the currently indexed codebase. `top_k` defaults to `5` (range 1–20).

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "How are command line arguments parsed?", "top_k": 5}'
```

```json
{
  "answer": "...",
  "sources": [
    {
      "file": "httpie/cli/argparser.py",
      "name": "parse_args",
      "start_line": 42,
      "end_line": 88
    }
  ]
}
```

Typical errors:

| Status | When |
| --- | --- |
| `422` | Empty question or invalid `top_k` |
| `400` | Nothing indexed yet (`Run /index first`) |
| `502` | DeepSeek request failed |

### `GET /dependencies`

Find files that import a module or symbol (AST-based; does not require indexing).

```bash
curl "http://localhost:8000/dependencies?repo_path=/path/to/repo&dependency=requests"
```

```json
{
  "dependency": "requests",
  "dependents": ["pkg/client.py", "pkg/utils.py"]
}
```

Matches exact imports and prefixes (e.g. `httpie.status` matches `httpie.status.ExitStatus`).

## Tests
The current test suite contains 10 tests covering API contracts and dependency analysis.
```bash
python -m pytest -v
```

- [`tests/test_api.py`](tests/test_api.py) — FastAPI contracts (`/health`, `/query` validation and response shape; RAG/DeepSeek mocked)
- [`tests/test_dependency_analyzer.py`](tests/test_dependency_analyzer.py) — import parsing and dependent lookup

## Project layout

```
main.py                   FastAPI app and endpoints
chunker.py                AST chunking
embedder.py               Embeddings + ChromaDB persistence
retriever.py              Semantic retrieval
rag.py                    Retrieval + LLM orchestration
llm.py                    DeepSeek client
dependency_analyzer.py    Import dependency analysis
requirements.txt
tests/
```

## Current limitations

- Indexing currently supports Python source files only.
- Dependency analysis tracks imports, not runtime function calls or a full call graph.
- The service supports one indexed repository at a time; indexing a new repository replaces the previous ChromaDB collection.
- Retrieval quality depends on the embedding model and selected `top_k`.
- DeepSeek can only use the code chunks retrieved for a query; it cannot
  independently browse the repository.

## Notes

- Indexing is Python-only (`.py` files); directories such as `venv`, `.git`, and `__pycache__` are skipped ([`chunker.get_python_files`](chunker.py)).
- Vectors persist under `./chroma_db/` (gitignored). Each `/index` request clears the previous collection before indexing the new repository.
- Answers are constrained to retrieved context; the system prompt in [`llm.py`](llm.py) instructs the model not to invent implementation details.
