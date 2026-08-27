from fastapi import FastAPI
from pydantic import BaseModel

from chunker import chunk_repo
from embedder import embed_and_store_chunks
from rag import answer_question

app = FastAPI()


# --- Request/response schemas -----------------------------------------
# These define the exact shape of data each endpoint accepts and returns.
# FastAPI uses them to automatically validate incoming requests and to
# generate the interactive docs at /docs.

class IndexRequest(BaseModel):
    """Input for POST /index — the repo to chunk and embed."""
    repo_path: str  # filesystem path to the target repo, e.g. "../httpie"


class IndexResponse(BaseModel):
    """Output of POST /index — confirms indexing completed."""
    status: str
    chunks_indexed: int  # number of function/class chunks stored


class QueryRequest(BaseModel):
    """Input for POST /query — a natural language question about the repo."""
    question: str
    top_k: int = 5  # number of most-similar chunks to retrieve as context


class QueryResponse(BaseModel):
    """Output of POST /query — the answer plus the chunks used to produce it."""
    answer: str
    sources: list[dict]  # e.g. [{"file": ..., "start_line": ..., "end_line": ...}]


# --- Endpoints -----------------------------------------------------------

@app.get("/health")
def health():
    """
    Function Description:
        Basic liveness check — confirms the service is running.
    """
    return {"status": "ok"}

@app.post("/index", response_model=IndexResponse)
def index_repo(request: IndexRequest):
    """Indexes a Python repository for semantic code retrieval.

    Parses the repository into function/class-level chunks, generates
    embeddings for those chunks, and stores them in the persistent
    ChromaDB collection.

    Args:
        request: Request containing the filesystem path of the repository
            to index.

    Returns:
        An IndexResponse reporting how many chunks were indexed.
    """
    # Parse the target repository into source-code chunks.
    chunks = chunk_repo(request.repo_path)

    # Generate local Jina embeddings and persist them in ChromaDB.
    stored_count = embed_and_store_chunks(chunks)

    return IndexResponse(
        status="success",
        chunks_indexed=stored_count,
    )

@app.post("/query", response_model=QueryResponse)
def query_codebase(request: QueryRequest):
    """Answers a question about the indexed codebase.

    Retrieves the most relevant source-code chunks from ChromaDB and
    provides them to DeepSeek as grounding context for the answer.

    Args:
        request: Request containing the question and number of chunks
            to retrieve.

    Returns:
        A QueryResponse containing the generated answer and source
        metadata for the retrieved chunks.
    """
    # Run retrieval and grounded LLM generation through the RAG pipeline.
    result = answer_question(
        question=request.question,
        top_k=request.top_k,
    )

    return QueryResponse(
        answer=result["answer"],
        sources=result["sources"],
    )